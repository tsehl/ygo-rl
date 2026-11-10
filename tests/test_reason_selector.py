from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'translate')]
import core,cards,jax,translate
import jax.numpy as jnp
import numpy as np
class ReasonSelectorTests(unittest.TestCase):
    def test_any_and_all_reason_masks(self):
        actual=jnp.int32(core.REASON_EFFECT|core.REASON_DESTROY)
        cases=[(core.REASON_EFFECT,True),(core.REASON_DESTROY,True),(core.REASON_BATTLE,False),
               (core.REASON_EFFECT|core.REASON_COST,True),(core.REASON_BATTLE|core.REASON_COST,False),
               (-(core.REASON_EFFECT|core.REASON_DESTROY),True),(-(core.REASON_EFFECT|core.REASON_BATTLE),False)]
        for selector,expected in cases:
            self.assertEqual(bool(core.reason_matches(actual,jnp.int32(selector))),expected,selector)
    def test_handler_condition_uses_reason_not_summon_history(self):
        bank=core.make_bank(np.array([1]),[500],[500],[4])
        deck=jnp.zeros(40,jnp.int32);state=jax.jit(core.new_game)(bank,deck,deck)
        state=state._replace(reason=state.reason.at[0].set(core.REASON_EFFECT|core.REASON_DESTROY))
        run=jax.jit(core._pure)
        for mask,expected in ((core.REASON_EFFECT,True),(core.REASON_DESTROY,True),(core.REASON_COST,False)):
            ins=jnp.asarray(cards.COND_REASON(mask),jnp.int32)
            _,ok=run(bank,state,state.reg,jnp.bool_(True),jnp.int32(0),jnp.int32(0),jnp.int32(0),jnp.int32(0),ins)
            self.assertEqual(bool(ok),expected,mask)
    def test_filter_combination_respects_conjunction_and_rejects_unsupported_shapes(self):
        fl=translate.Filt()
        translate.apply_method(fl,'IsReason',[core.REASON_EFFECT])
        translate.apply_method(fl,'IsReason',[core.REASON_DESTROY])
        self.assertEqual(fl.build(core.Z_GRAVE,0)[core.F_REASON],-(core.REASON_EFFECT|core.REASON_DESTROY))
        with self.assertRaisesRegex(translate.Unsupported,'hors zones suivies'):
            fl.build(core.Z_HAND,0)
        mixed=translate.Filt()
        translate.apply_method(mixed,'IsReason',[core.REASON_EFFECT|core.REASON_DESTROY])
        with self.assertRaisesRegex(translate.Unsupported,'conjonction complexe'):
            translate.apply_method(mixed,'IsReason',[core.REASON_COST|core.REASON_BATTLE])
    def test_refresh_detects_metadata_changes_without_a_zone_move(self):
        bank=core.make_bank(np.array([1]),[500],[500],[4])
        deck=jnp.zeros(40,jnp.int32);state=jax.jit(core.new_game)(bank,deck,deck)
        self.assertTrue(bool(core.changed(state,state._replace(reason=state.reason.at[1].set(core.REASON_COST)))))
        self.assertTrue(bool(core.changed(state,state._replace(suminfo=state.suminfo.at[1].set(core.SUMMON_SPECIAL)))))
    def test_continuous_effect_refreshes_when_summon_or_reason_changes(self):
        filt=cards.F(me=core.Z_GRAVE,st=core.summon_mask(core.SUMMON_FUSION),reason=core.REASON_COST)
        effect=cards.cont(core.Z_MZONE,core.MC_ATK,100,per=filt)
        codes=np.array([1,2])
        bank=core.make_bank(codes,[500,500],[500,500],[4,4],
            **cards.assemble(codes,cards={1:[effect],2:[]},ritual={},fusion={},synchro={},xyz={},link={}))
        deck=jnp.arange(40,dtype=jnp.int32)%2
        state=jax.jit(core.new_game)(bank,deck,deck)
        state=state._replace(zone=state.zone.at[0].set(core.Z_MZONE).at[1].set(core.Z_GRAVE),
            pos=state.pos.at[0].set(core.P_FU_ATK),seq=state.seq.at[0].set(0).at[1].set(0),
            suminfo=state.suminfo.at[1].set(core.SUMMON_FUSION),reason=state.reason.at[1].set(core.REASON_COST))
        state=jax.jit(lambda s:core.refresh(bank,s,fus=False))(state)
        self.assertEqual(int(state.atk_c[0]),600)
        check=jax.jit(lambda old,new:core.refresh_if(bank,old,new))
        changed_type=state._replace(suminfo=state.suminfo.at[1].set(core.SUMMON_SPECIAL))
        self.assertEqual(int(check(state,changed_type).atk_c[0]),500)
        changed_reason=state._replace(reason=state.reason.at[1].set(core.REASON_EFFECT))
        self.assertEqual(int(check(state,changed_reason).atk_c[0]),500)
if __name__=='__main__':unittest.main()
