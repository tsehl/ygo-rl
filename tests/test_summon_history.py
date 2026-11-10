from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'translate'),str(ROOT/'train')]
import core,cards,translate,obs,model
import jax,jax.numpy as jnp,numpy as np
class SummonHistoryTests(unittest.TestCase):
    def test_native_type_masks_distinguish_generic_and_proper_summons(self):
        for player in (0,1):
            info=core.summon_info(core.SUMMON_FUSION,jnp.int32(core.Z_EXTRA),jnp.int32(player))
            self.assertTrue(bool(core.summon_matches(info,core.summon_mask(core.SUMMON_SPECIAL))))
            self.assertTrue(bool(core.summon_matches(info,core.summon_mask(core.SUMMON_FUSION))))
            self.assertFalse(bool(core.summon_matches(info,core.summon_mask(core.SUMMON_RITUAL))))
            self.assertFalse(bool(core.summon_matches(info,core.summon_mask(core.SUMMON_FUSION,True))))
    def test_effect_summon_records_origin_type_player_and_completion(self):
        for proper,typ in ((0,core.SUMMON_SPECIAL),(1,core.SUMMON_RITUAL),(2,core.SUMMON_FUSION),(-152,core.SUMMON_SPECIAL|151)):
            effect=cards.act(resolve=[cards.SPSUMMON(0,core.P_FU_ATK,proper)])
            bank=core.make_bank(np.array([1]),[500],[500],[4],ctype=[core.TYPE_MONSTER|core.TYPE_FUSION],
                **cards.assemble([1],cards={1:[effect]},ritual={},fusion={},synchro={},xyz={},link={}))
            deck=jnp.zeros(40,jnp.int32);extra=jnp.zeros(1,jnp.int32)
            s=jax.jit(core.new_game)(bank,deck,deck,None,extra,extra)
            s=s._replace(dec=jnp.int32(core.D_POS),zctx=jnp.int32(core.ZC_SPSUM),zsel=jnp.int32(0),
                ch_src=jnp.int32(0),ch_eff=jnp.int32(0),ch_p=jnp.int32(1),pg=jnp.int32(core.PG_RESOLVE),pc=jnp.int32(0),
                reg=s.reg.at[0,80].set(True))
            s=jax.jit(core.apply)(bank,s,jnp.int32(core.special(82,core.A_POS_ATK)))
            expected=core.summon_info(typ,jnp.int32(core.Z_EXTRA),jnp.int32(1))
            self.assertEqual(int(s.suminfo[80]),int(expected))
            self.assertEqual(bool(s.proc[80]),proper>0)
            moved=core.group_send(s,jnp.arange(82)==80,core.Z_GRAVE,core.P_GRAVE)
            self.assertEqual(int(moved.suminfo[80]),int(expected))
    def test_hidden_cards_do_not_expose_summon_history(self):
        bank=core.make_bank(np.array([1]),[500],[500],[4])
        deck=jnp.zeros(40,jnp.int32)
        state=jax.jit(core.new_game)(bank,deck,deck)
        state=state._replace(zone=state.zone.at[40].set(core.Z_MZONE),seq=state.seq.at[40].set(0),
            pos=state.pos.at[40].set(core.P_FD_DEF),suminfo=state.suminfo.at[40].set(core.SUMMON_NORMAL))
        view=obs.observe(bank,state,p=0,m=jnp.zeros(core.n_actions(80),bool))
        hidden=(np.asarray(view.slot)==40)
        self.assertTrue(hidden.any())
        self.assertFalse(np.asarray(view.suminfo)[hidden].any())
    def test_both_legacy_dynamic_prefixes_match_zero_extension_with_history(self):
        bank=core.make_bank(np.array([1]),[500],[500],[4])
        deck=jnp.zeros(40,jnp.int32)
        state=jax.jit(core.new_game)(bank,deck,deck)
        state=state._replace(zone=state.zone.at[0].set(core.Z_MZONE),seq=state.seq.at[0].set(0),
            pos=state.pos.at[0].set(core.P_FU_ATK),suminfo=state.suminfo.at[0].set(core.SUMMON_FUSION))
        view=obs.observe(bank,state,p=0,m=jnp.zeros(core.n_actions(80),bool))
        params=model.init(jax.random.PRNGKey(0),d=16,card_layers=1,layers=1)
        embeddings=jnp.ones((1,16))
        infer=jax.jit(model.policy)
        for width in (model.F_DYN-1,model.F_DYN-33,model.F_DYN-65,model.F_DYN-66):
            legacy=dict(params,dyn=dict(params['dyn'],w=params['dyn']['w'][:width]))
            migrated=dict(legacy,dyn=dict(legacy['dyn'],w=jnp.pad(legacy['dyn']['w'],((0,model.F_DYN-width),(0,0)))))
            for actual,expected in zip(infer(legacy,embeddings,view),infer(migrated,embeddings,view)):
                np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-5)
if __name__=='__main__':unittest.main()
