from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'translate')]
import core,cards,translate
import jax,jax.numpy as jnp,numpy as np
class ExtraEffectTests(unittest.TestCase):
    def test_cyber_stein_preserves_cost_and_summons_without_completion(self):
        effects,*_=translate.translate(69015963)
        self.assertEqual(effects[0][4],[cards.PAY_LP(5000)])
        self.assertEqual(effects[0][6][-1],cards.SPSUMMON(0,core.P_FU_ATK))
        filters=[x[2] for x in effects[0][5] if x[0]==core.OP_FILTER]
        self.assertTrue(filters[0][core.F_ZX]&core.ZX_EXTRA)
    def test_generic_special_summon_allows_uncompleted_facedown_extra_only(self):
        effects=cards.Effects([],revive=1)
        bank=core.make_bank(np.array([1]),[1000],[1000],[6],ctype=[core.TYPE_MONSTER|core.TYPE_FUSION],
             **cards.assemble([1],cards={1:effects},ritual={},fusion={},synchro={},xyz={},link={}))
        deck=jnp.zeros(40,jnp.int32);extra=jnp.zeros(1,jnp.int32)
        s=jax.jit(core.new_game)(bank,deck,deck,None,extra,extra)
        run=jax.jit(lambda s:core.filter_mask(bank,s,jnp.array(cards.F(me=core.Z_EXTRA|core.Z_HAND|core.Z_GRAVE,spsum=1)),0,0))
        self.assertFalse(bool(run(s)[0]));self.assertTrue(bool(run(s)[80]))
        s=s._replace(zone=s.zone.at[80].set(core.Z_GRAVE))
        self.assertFalse(bool(run(s)[80]))
        self.assertTrue(bool(run(s._replace(proc=s.proc.at[80].set(True)))[80]))
if __name__=='__main__':unittest.main()
