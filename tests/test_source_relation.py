"""La relation à la source utilise les déplacements, sans changer SELF ordinaire."""
from pathlib import Path
import ast
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'engine'), str(ROOT / 'translate')]
import cards
import core
import obs
import translate
import jax
import jax.numpy as jnp
import numpy as np


class SourceRelationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = core.make_bank(np.array([1]), [1000], [1000], [4])
        deck = jnp.zeros(40, jnp.int32)
        cls.state = jax.jit(core.new_game)(cls.bank, deck, deck)
        cls.pure = staticmethod(jax.jit(core._pure))

    def test_copy_keeps_other_cards_xyz_materials(self):
        s = self.state._replace(zone=self.state.zone.at[0].set(core.Z_OVERLAY),
                                ov_par=self.state.ov_par.at[0].set(1))
        source = s.reg.at[0, 0].set(True)
        ins = jnp.asarray((core.OP_FILTER, 1, 0, 1 | (1 << 20)), jnp.int32)
        reg, _ = self.pure(self.bank, s, source, jnp.bool_(True), jnp.int32(0),
                           jnp.int32(0), jnp.int32(0), jnp.int32(0), ins)
        self.assertTrue(bool(reg[1, 0]))

    def test_self_requires_source_relation_only_when_requested(self):
        for moved, cost_moved, expected in ((False, False, True), (True, False, False),
                                             (False, True, False)):
            s = self.state._replace(mv=self.state.mv.at[0].set(2 if moved else 1),
                                    lk_t=self.state.lk_t.at[0].set(1),
                                    lk_flags=self.state.lk_flags.at[0].set(
                                        core.CF_SRC_UNRELATED if cost_moved else 0))
            for related in (False, True):
                ins = jnp.asarray(cards.SELF(0, related=related), jnp.int32)
                reg, _ = self.pure(self.bank, s, s.reg, jnp.bool_(True), jnp.int32(0),
                                   jnp.int32(0), jnp.int32(0), jnp.int32(0), ins)
                self.assertEqual(bool(reg[0, 0]), expected if related else True)

    def test_observation_shows_relation_on_the_source_token(self):
        s = self.state._replace(zone=self.state.zone.at[0].set(core.Z_SZONE),
                                pos=self.state.pos.at[0].set(core.P_SFU),
                                n_chain=jnp.int32(1), lk_src=self.state.lk_src.at[0].set(0),
                                lk_t=self.state.lk_t.at[0].set(1), mv=self.state.mv.at[0].set(1))
        mask = jnp.zeros(core.NV * s.cid.shape[0] + core.N_SPECIAL, bool)
        observe = jax.jit(lambda state: obs.observe(self.bank, state, m=mask))
        view = observe(s)
        token = int(jnp.argmax(view.slot == 0))
        self.assertTrue(bool(view.source_related[token]))
        lost = observe(s._replace(mv=s.mv.at[0].set(2)))
        self.assertFalse(bool(lost.source_related[token]))
        lost_cost = observe(s._replace(lk_flags=s.lk_flags.at[0].set(core.CF_SRC_UNRELATED)))
        self.assertFalse(bool(lost_cost.source_related[token]))

    def test_single_target_relation_uses_existing_target_register(self):
        source = (ROOT / 'tests' / 'target_relation_oracle_check.py').read_text()
        module = ast.parse(source)
        script = next(ast.literal_eval(node.value) for node in module.body
                      if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                      and node.targets[0].id == 'script')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, 'c600930051.lua')
            path.write_text(script)
            old = translate.OFFICIAL
            try:
                translate.OFFICIAL = folder
                effects, *_ = translate.translate(600930051)
                self.assertEqual(effects[0][6][0], cards.REQ(0, 1, related=True))
                path.write_text(script.replace('LOCATION_SZONE,0,1,1,e:GetHandler()',
                                               'LOCATION_SZONE,0,1,2,e:GetHandler()'))
                multiple, *_ = translate.translate(600930051)
                ops = multiple[0][6]
                self.assertTrue(any(ins[0] == core.OP_MINATK and ins[3] == 3 for ins in ops))
                self.assertTrue(any(ins[0] == core.OP_REQ and ins[3] == 2 for ins in ops))
            finally:
                translate.OFFICIAL = old

    def test_single_target_filter_guards_independent_operation(self):
        module = ast.parse((ROOT / 'tests' / 'target_condition_oracle_check.py').read_text())
        script = next(ast.literal_eval(node.value) for node in module.body
                      if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                      and node.targets[0].id == 'script')
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'c600930051.lua').write_text(script)
            old = translate.OFFICIAL
            try:
                translate.OFFICIAL = folder
                effects, *_ = translate.translate(600930051)
            finally:
                translate.OFFICIAL = old
        operations = [ins[0] for ins in effects[0][6]]
        self.assertEqual(operations.count(core.OP_REQ), 2)
        self.assertIn(core.OP_FILTER, operations)

    def test_first_target_is_selected_before_relation_is_checked(self):
        s = self.state._replace(uid=self.state.uid.at[0].set(5).at[1].set(2),
                                mv=self.state.mv.at[0].set(1).at[1].set(2),
                                lk_t=self.state.lk_t.at[0].set(1), lk_tgt_t=self.state.lk_tgt_t.at[0].set(1),
                                lk_reg=self.state.lk_reg.at[0, 0, 0].set(True).at[0, 0, 1].set(True),
                                reg=self.state.reg.at[0, 0].set(True).at[0, 1].set(False))
        args = (self.bank, s, s.reg, jnp.bool_(True), jnp.int32(0), jnp.int32(0),
                jnp.int32(0), jnp.int32(0))
        reg, _ = self.pure(*args, jnp.asarray(cards.FIRST(1, 0, original=True), jnp.int32))
        self.assertFalse(bool(reg[1, 0]))
        self.assertTrue(bool(reg[1, 1]))
        _, related = self.pure(self.bank, s, reg, jnp.bool_(True), jnp.int32(0), jnp.int32(0),
                               jnp.int32(0), jnp.int32(0), jnp.asarray(cards.REQ(1, 1, related=True)))
        self.assertFalse(bool(related))
        current, _ = self.pure(*args, jnp.asarray(cards.FIRST(1, 0), jnp.int32))
        self.assertTrue(bool(current[1, 0]))
        self.assertFalse(bool(current[1, 1]))

    def test_target_selection_captures_its_own_clock(self):
        ins = jnp.asarray(cards.CHOOSE(0, 0, 1, tgt=True), jnp.int32)
        bank = self.bank._replace(prog=self.bank.prog.at[0, 0, core.PG_TARGET, 0].set(ins))
        s = self.state._replace(ch_src=jnp.int32(0), ch_eff=jnp.int32(0), pg=jnp.int32(core.PG_TARGET),
                                pc=jnp.int32(0), clk=jnp.int32(12), ch_tgt_t=jnp.int32(3),
                                st=self.state.st.at[0].set(core.ST_SEL))
        result = core.sel_finish(bank, s)
        self.assertEqual(int(result.ch_tgt_t), 12)
        ordinary = bank._replace(prog=bank.prog.at[0, 0, core.PG_TARGET, 0].set(
            jnp.asarray(cards.CHOOSE(0, 0, 1), jnp.int32)))
        self.assertEqual(int(core.sel_finish(ordinary, s).ch_tgt_t), 3)

    def test_target_relation_uses_selection_time_not_chain_completion(self):
        s = self.state._replace(mv=self.state.mv.at[0].set(2),
                                lk_t=self.state.lk_t.at[0].set(2),
                                lk_tgt_t=self.state.lk_tgt_t.at[0].set(1),
                                reg=self.state.reg.at[0, 0].set(True))
        _, related = self.pure(self.bank, s, s.reg, jnp.bool_(True), jnp.int32(0),
                               jnp.int32(0), jnp.int32(0), jnp.int32(0),
                               jnp.asarray(cards.REQ(0, 1, related=True)))
        self.assertFalse(bool(related))

    def test_first_in_static_filter_is_refused_without_internal_exception(self):
        with self.assertRaisesRegex(translate.Unsupported, 'GetFirst sans programme'):
            translate.translate(50951359)

    def test_translator_emits_a_guard_for_handler(self):
        script = '''local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE) e:SetCode(EVENT_FREE_CHAIN)
 e:SetOperation(function(e,tp)
  if e:GetHandler():IsRelateToEffect(e) then Duel.Recover(tp,1234,REASON_EFFECT) end
 end)
 c:RegisterEffect(e)
end'''
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'c600930031.lua').write_text(script)
            old = translate.OFFICIAL
            try:
                translate.OFFICIAL = folder
                effects, *_ = translate.translate(600930031)
            finally:
                translate.OFFICIAL = old
        self.assertEqual(effects[0][6][:2], [cards.SELF(0, related=True), cards.REQ(0, 1)])


if __name__ == '__main__':
    unittest.main()
