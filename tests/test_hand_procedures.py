"""Procédures inhérentes : conditions, absence de chaîne, limites d'invocation."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import tempfile
import numpy as np
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'engine'), str(ROOT/'translate')]
import core, cards, translate as T


class HandProceduresTests(unittest.TestCase):
    def test_material_controller_without_context_is_refused(self):
        # Les filtres Fossil attendent un tp fourni par leur procédure spéciale.
        # Le retrait du nil de contexte ne doit pas devenir un IndexError.
        for code in (10040267,45409943,57157964,83656563,95095116):
            with self.subTest(code=code),self.assertRaisesRegex(T.Unsupported,'IsControler'):
                T.translate(code)

    @classmethod
    def setUpClass(cls):
        cls.effects = T.translate(70095154)[0]
        cls.bank = core.make_bank(np.array([70095154, 46986414]), [2100,2500], [1600,2100], [5,7],
                                 ctype=[33,17], **cards.assemble([70095154,46986414], cards={70095154:cls.effects}))
        d = jnp.ones(40,jnp.int32).at[0].set(0)
        s = jax.jit(core.new_game)(cls.bank,d,d)
        cls.state = s._replace(zone=jnp.full(80,core.Z_DECK,jnp.int32).at[0].set(core.Z_HAND),
                              seq=jnp.tile(jnp.arange(40,dtype=jnp.int32),2),
                              tp=jnp.int32(0),dec_player=jnp.int32(0),dec=jnp.int32(core.D_IDLE),
                              ns_used=jnp.bool_(False),n_chain=jnp.int32(0),phase=jnp.int32(core.PH_MAIN1))
        cls.legal = staticmethod(jax.jit(core.legal))
        cls.apply = staticmethod(jax.jit(core.apply))

    def test_cyber_condition_and_global_prohibition(self):
        b,s = self.bank,self.state
        action = core.V_ACT0*80
        self.assertFalse(bool(self.legal(b,s)[action]))
        opponent = s._replace(zone=s.zone.at[40].set(core.Z_MZONE),seq=s.seq.at[40].set(0))
        self.assertTrue(bool(self.legal(b,opponent)[action]))
        own = opponent._replace(zone=opponent.zone.at[1].set(core.Z_MZONE),seq=opponent.seq.at[1].set(0))
        self.assertFalse(bool(self.legal(b,own)[action]))
        locked = opponent._replace(fl_c=opponent.fl_c.at[0].set(core.FL_NO_SS))
        self.assertFalse(bool(self.legal(b,locked)[action]))
        # Une interdiction d'activation par nom ne concerne pas les procédures.
        locked = opponent._replace(lock_cid=opponent.lock_cid.at[0,0].set(0))
        self.assertTrue(bool(self.legal(b,locked)[action]))

    def test_procedure_preserves_normal_summon_and_marks_completion(self):
        s = self.state._replace(zone=self.state.zone.at[40].set(core.Z_MZONE),seq=self.state.seq.at[40].set(0))
        s = self.apply(self.bank,s,jnp.int32(core.V_ACT0*80))
        self.assertEqual(int(s.dec),core.D_ZONE)
        self.assertEqual(int(s.zctx),core.ZC_PROC)
        mask=self.legal(self.bank,s)[core.special(80,core.A_ZONE0):core.special(80,core.A_ZONE0)+7]
        np.testing.assert_array_equal(np.asarray(mask),[True]*5+[False]*2)
        s = self.apply(self.bank,s,jnp.int32(core.special(80,core.A_ZONE0)))
        s = self.apply(self.bank,s,jnp.int32(core.special(80,core.A_POS_ATK)))
        self.assertEqual(int(s.zone[0]),core.Z_MZONE)
        self.assertTrue(bool(s.proc[0]))
        self.assertFalse(bool(s.ns_used))
        self.assertEqual(int(s.n_chain),0)
        self.assertTrue(bool(s.st[0] & core.ST_SPSUM))
        self.assertTrue(bool(s.ev_g[core.EV_SPSUMMON,0]))
        self.assertEqual(int(s.ev_p[core.EV_SPSUMMON]),0)

    def test_full_field_and_nomi_effect_summon_restrictions(self):
        s = self.state
        zone=s.zone.at[1:6].set(core.Z_MZONE).at[40].set(core.Z_MZONE)
        s=s._replace(zone=zone,seq=s.seq.at[1:6].set(jnp.arange(5)).at[40].set(0))
        self.assertFalse(bool(self.legal(self.bank,s)[core.V_ACT0*80]))
        # erl=2 refuse la réanimation même si proc est vrai ; erl=1 l'autorise.
        for limit,proper,expected in ((1,False,False),(1,True,True),(2,False,False),(2,True,False)):
            bank=self.bank._replace(erl=self.bank.erl.at[0].set(limit),ctype=self.bank.ctype.at[0].set(core.TYPE_MONSTER|core.TYPE_SPSUMMON))
            state=self.state._replace(proc=self.state.proc.at[0].set(proper))
            self.assertFalse(bool(self.legal(bank,state)[0]))
            f=jnp.asarray(cards.F(me=core.Z_HAND,spsum=1))
            self.assertEqual(bool(core.filter_mask(bank,state,f,jnp.int32(0),jnp.int32(0))[0]),expected)

    def test_count_is_consumed_without_building_a_chain(self):
        bank=self.bank._replace(eopt=self.bank.eopt.at[0,0].set(core.OPT_NAME))
        s=self.state._replace(zone=self.state.zone.at[40].set(core.Z_MZONE),seq=self.state.seq.at[40].set(0))
        self.assertTrue(bool(self.legal(bank,s)[core.V_ACT0*80]))
        used=self.apply(bank,s,jnp.int32(core.V_ACT0*80))
        self.assertTrue(bool(core.name_in(used.opt_cid[0],jnp.int32(0))))
        self.assertEqual(int(used.n_chain),0)
        repeated=used._replace(dec=jnp.int32(core.D_IDLE))
        self.assertFalse(bool(self.legal(bank,repeated)[core.V_ACT0*80]))

    def test_real_nomi_restriction_and_exclusive_trigger_clones(self):
        effects,_,_,_=T.translate(51808422)  # XX-Saber Faultroll
        self.assertEqual(effects[0][-1]['revive'],2)
        effects,_,_,descs=T.translate(336601)  # Moon Gardna
        self.assertEqual(len(effects),2)
        self.assertEqual(len(descs),2)
        self.assertEqual(effects[0][2],core.OPT_NAME)
        self.assertEqual(effects[1][1],(1<<core.EV_SUMMON)|(1<<core.EV_SPSUMMON))
        self.assertEqual(effects[1][2],core.OPT_NAMEK)

    def test_unsupported_procedure_is_refused(self):
        source='''local s,id=GetID()
function s.initial_effect(c)
local e=Effect.CreateEffect(c)
e:SetType(EFFECT_TYPE_FIELD)
e:SetCode(EFFECT_SPSUMMON_PROC)
e:SetRange(LOCATION_HAND)
e:SetOperation(function() end)
c:RegisterEffect(e)
end'''
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,'c999999999.lua').write_text(source)
            with patch.object(T,'OFFICIAL',tmp),self.assertRaises(T.Unsupported):
                T.translate(999999999)

    def test_clones_with_separate_effect_counters_remain_refused(self):
        source='''local s,id=GetID()
function s.initial_effect(c)
local e=Effect.CreateEffect(c)
e:SetType(EFFECT_TYPE_SINGLE+EFFECT_TYPE_TRIGGER_O)
e:SetCode(EVENT_SUMMON_SUCCESS)
e:SetCountLimit(1)
e:SetOperation(s.operation)
c:RegisterEffect(e)
local e2=e:Clone()
e2:SetCode(EVENT_FLIP_SUMMON_SUCCESS)
c:RegisterEffect(e2)
end
function s.operation(e,tp)
Duel.Recover(tp,100,REASON_EFFECT)
end'''
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,'c999999999.lua').write_text(source)
            with patch.object(T,'OFFICIAL',tmp),self.assertRaisesRegex(T.Unsupported,'limite par carte'):
                T.translate(999999999)

    def test_free_zone_count_alias_compiles_to_the_same_program(self):
        template='''local s,id=GetID()
function s.initial_effect(c)
local e=Effect.CreateEffect(c)
e:SetType(EFFECT_TYPE_ACTIVATE)
e:SetCode(EVENT_FREE_CHAIN)
e:SetTarget(s.target)
e:SetOperation(s.operation)
c:RegisterEffect(e)
end
function s.target(e,tp,eg,ep,ev,re,r,rp,chk)
ALIAS
if chk==0 then return VALUE>0 end
end
function s.operation(e,tp)
ALIAS
if VALUE<=0 then return end
Duel.Draw(tp,1,REASON_EFFECT)
end'''
        values=[]
        with tempfile.TemporaryDirectory() as tmp:
            for alias,value in (('', 'Duel.GetLocationCount(tp,LOCATION_MZONE)'),
                                ('local ft=Duel.GetLocationCount(tp,LOCATION_MZONE)','ft')):
                Path(tmp,'c999999999.lua').write_text(template.replace('ALIAS',alias).replace('VALUE',value))
                with patch.object(T,'OFFICIAL',tmp): values.append(T.translate(999999999)[0])
        self.assertEqual(values[0],values[1])

    def test_existing_group_predicate_keeps_battle_destroyed_self(self):
        s=self.state._replace(zone=self.state.zone.at[0].set(core.Z_MZONE),
                              pos=self.state.pos.at[0].set(core.P_FU_DEF),
                              st=self.state.st.at[0].set(core.ST_BDES))
        reg=jnp.zeros((core.R,80),bool).at[0,0].set(True)
        # Le filtre de Cyber Dragon est le terrain allié : on le réutilise comme prédicat simple.
        for flags,expected in ((1,True),(0,True),(9,False),(8,False)):
            out,ok=core._pure(self.bank,s,reg,jnp.bool_(True),jnp.int32(0),jnp.int32(0),jnp.int32(0),
                              jnp.int32(0),jnp.array([core.OP_FILTER,1,0,flags],jnp.int32))
            self.assertEqual(bool(out[1,0]),expected)


if __name__=='__main__': unittest.main()
