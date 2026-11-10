"""Nature des maillons et compteurs lors de l'activation d'un piège continu."""
from pathlib import Path
import sys, unittest, gzip, pickle
import jax
import jax.numpy as jnp
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'engine'), str(ROOT/'translate'), str(ROOT/'train'), str(ROOT/'gen')]
import core, cards, translate, obs, model, cardgen


class InlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        activation = (*cards.act()[:7], {'inline': 2})
        quick = cards.quick(rng=core.Z_SZONE, opt=core.OPT_CARD,
                            resolve=[cards.DRAW(cards.ME, 1)])
        cls.bank = core.make_bank(np.array([1, 2]), [0, 1000], [0, 1000], [0, 4],
            ctype=[core.TYPE_TRAP | core.TYPE_CONTINUOUS, core.TYPE_MONSTER | core.TYPE_NORMAL],
            **cards.assemble([1, 2], cards={1: [activation, quick]}, ritual={}, fusion={}, synchro={}, xyz={}, link={}))
        d = jnp.ones(40, jnp.int32)
        cls.state = jax.jit(core.new_game)(cls.bank, d, d)
        cls.state = cls.state._replace(cid=cls.state.cid.at[0].set(0), zone=cls.state.zone.at[0].set(core.Z_SZONE),
            pos=cls.state.pos.at[0].set(core.P_SFU), ctrl=cls.state.ctrl.at[0].set(0),
            dec_player=jnp.int32(0), wk=jnp.int32(core.W_QUICK))

    def begin(self, state=None):
        return jax.jit(core.begin_act)(self.bank, self.state if state is None else state, jnp.int32(0), jnp.int32(0))

    def answer(self, state, yes):
        return jax.jit(core.apply)(self.bank, state,
            jnp.int32(core.special(state.cid.shape[0], core.A_YES if yes else core.A_PASS)))

    def test_optional_question_then_single_card_activation_link(self):
        s = self.begin()
        self.assertEqual(int(s.dec), core.D_YESNO)
        self.assertEqual(int(s.zctx), core.ZC_INLINE)
        self.assertEqual(int(s.st[0]) & (core.ST_CNT << 1), 0)
        s = self.answer(s, True)
        self.assertEqual(int(s.ch_eff), 1)
        self.assertTrue(int(s.st[0]) & (core.ST_CNT << 1))
        s = jax.jit(core.push_link)(self.bank, s)
        self.assertEqual(int(s.n_chain), 1)
        self.assertEqual(int(s.lk_eff[0]), 1)
        self.assertEqual(int(s.lk_flags[0]), core.CF_ACT | core.CF_CONT)
        self.assertEqual(int(core.load_link(self.bank, s, jnp.int32(0)).pc), 0)

    def test_declining_keeps_empty_activation_and_unused_counter(self):
        s = self.answer(self.begin(), False)
        s = jax.jit(core.exec_one)(self.bank, s)
        self.assertEqual(int(s.n_chain), 1)
        self.assertEqual(int(s.lk_eff[0]), 0)
        self.assertEqual(int(s.st[0]) & (core.ST_CNT << 1), 0)

    def test_used_counter_and_damage_step_skip_question(self):
        for state in (self.state._replace(st=self.state.st.at[0].set(core.ST_CNT << 1)),
                      self.state._replace(dstep=jnp.int32(core.DS_BEFORE))):
            self.assertEqual(int(self.begin(state).dec), core.D_RUN)

    def test_continuous_source_must_remain_related_and_faceup(self):
        s = self.answer(self.begin(), True)
        s = jax.jit(core.push_link)(self.bank, s)
        for changed in (s._replace(zone=s.zone.at[0].set(core.Z_GRAVE)),
                        s._replace(pos=s.pos.at[0].set(core.P_SFD)),
                        s._replace(mv=s.mv.at[0].set(s.lk_t[0]+1))):
            self.assertEqual(int(core.load_link(self.bank, changed, jnp.int32(0)).pc), core.P-1)

    def test_plain_banks_have_no_inline_table(self):
        bank = core.make_bank(np.array([1]), [1000], [1000], [4])
        self.assertEqual(bank.einline.shape, (1, 0))

    def test_official_translation_carries_the_alternate_index(self):
        with self.assertRaisesRegex(translate.Unsupported, 'cibles multiples'):
            translate.translate(1802450)

    def test_observation_exposes_card_activation_after_effect_swap(self):
        s = self.answer(self.begin(), True)
        s = jax.jit(core.push_link)(self.bank, s)
        view = jax.jit(obs.observe)(self.bank, s)
        token = int(jnp.argmax(view.slot == 0))
        self.assertTrue(bool(view.card_activation[token]))
        self.assertEqual(int(view.chain[token]), 1)
        ordinary = s._replace(lk_flags=s.lk_flags.at[0].set(core.CF_CONT))
        self.assertFalse(bool(jax.jit(obs.observe)(self.bank, ordinary).card_activation[token]))

    def test_old_checkpoint_matches_explicit_zero_extension(self):
        with gzip.open(ROOT/'train/ckpt/run3_it120_f16.pkl.gz', 'rb') as handle:
            params = jax.tree.map(lambda x: jnp.asarray(x, jnp.float32), pickle.load(handle))
        migrated = dict(params)
        migrated['cards'] = dict(params['cards'])
        for field, width in (('head', model.F_HEAD), ('eff', model.F_EFF), ('ins', model.F_INS), ('filt', model.F_FILT), ('matf', model.F_FILT)):
            old = params['cards'][field]
            migrated['cards'][field] = dict(old, w=jnp.pad(old['w'], ((0, width-old['w'].shape[0]), (0, 0))))
        old = params['dyn']
        migrated['dyn'] = dict(old, w=jnp.pad(old['w'], ((0, model.F_DYN-old['w'].shape[0]), (0, 0))))
        feature_bank = self.bank._replace(
            filt=self.bank.filt.at[0, 1, 0, core.F_REASON].set(core.REASON_LINK),
            prog=self.bank.prog.at[0, 1, core.PG_RESOLVE, 0, 3].set(core.MOVE_REASON_EXPLICIT | (core.REASON_LINK << 1)))
        tokens, mask = model.card_tokens(feature_bank)
        encode = jax.jit(model.encode_cards)
        actual = encode(params['cards'], tokens, mask)
        expected = encode(migrated['cards'], tokens, mask)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        s = jax.jit(core.push_link)(self.bank, self.answer(self.begin(), True))
        s = s._replace(zone=s.zone.at[1].set(core.Z_GRAVE), pos=s.pos.at[1].set(core.P_GRAVE),
                       reason=s.reason.at[1].set(core.REASON_LINK | core.REASON_MATERIAL))
        view = jax.jit(obs.observe)(self.bank, s)
        self.assertTrue(bool(jnp.any(view.reason == (core.REASON_LINK | core.REASON_MATERIAL))))
        infer = jax.jit(model.policy)
        for a, b in zip(infer(params, actual, view), infer(migrated, expected, view)):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)

    def test_generated_banks_have_fixed_inline_shapes(self):
        banks = [cardgen.to_bank(cardgen.Gen(seed).make_set(80, 10)) for seed in (0, 1)]
        self.assertEqual(banks[0].einline.shape, (90, core.E))
        self.assertEqual(jax.tree.map(lambda x: x.shape, banks[0]), jax.tree.map(lambda x: x.shape, banks[1]))

    def test_refund_distinguishes_activation_negation_from_effect_disabling(self):
        bank = self.bank._replace(eopt=self.bank.eopt.at[0, 1].set(core.OPT_NAMEK+2),
                                 erefund=jnp.zeros_like(self.bank.ekind).at[0, 1].set(1))
        state = jax.jit(core.begin_act)(bank, self.state, jnp.int32(0), jnp.int32(0))
        state = jax.jit(core.apply)(bank, state, jnp.int32(core.special(80, core.A_YES)))
        state = jax.jit(core.push_link)(bank, state)
        key = core.opt_key(jnp.int32(0), jnp.int32(core.OPT_NAMEK+2), 1)
        self.assertTrue(bool(core.name_in(state.opt_cid[0], key)))
        disabled = state._replace(lk_neg=state.lk_neg.at[0].set(True))
        load = jax.jit(core.load_link)
        self.assertTrue(bool(core.name_in(load(bank, disabled, jnp.int32(0)).opt_cid[0], key)))
        negated = disabled._replace(lk_flags=disabled.lk_flags.at[0].set(disabled.lk_flags[0] | core.CF_ACT_NEG))
        self.assertFalse(bool(core.name_in(load(bank, negated, jnp.int32(0)).opt_cid[0], key)))
        nonrefundable = bank._replace(erefund=jnp.zeros_like(bank.erefund))
        self.assertTrue(bool(core.name_in(load(nonrefundable, negated, jnp.int32(0)).opt_cid[0], key)))


if __name__ == '__main__':
    unittest.main()
