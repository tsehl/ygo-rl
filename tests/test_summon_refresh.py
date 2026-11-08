from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'engine'))
import core,cards,jax
import jax.numpy as jnp
import numpy as np
class SummonRefreshTests(unittest.TestCase):
    def test_continuous_effect_recomputes_when_attempt_changes_summon_type(self):
        codes=np.array([1,2]);f=cards.F(me=core.Z_GRAVE,st=core.summon_mask(core.SUMMON_FUSION))
        bank=core.make_bank(codes,[500,500],[500,500],[4,4],**cards.assemble(codes,
            cards={1:[cards.cont(core.Z_MZONE,core.MC_ATK,100,per=f)],2:[]},
            ritual={},fusion={},synchro={},xyz={},link={}))
        deck=jnp.arange(40,dtype=jnp.int32)%2
        s=jax.jit(core.new_game)(bank,deck,deck)
        s=s._replace(zone=s.zone.at[0].set(core.Z_MZONE).at[1].set(core.Z_GRAVE),
            pos=s.pos.at[0].set(core.P_FU_ATK),seq=s.seq.at[0].set(0).at[1].set(0),
            suminfo=s.suminfo.at[1].set(core.SUMMON_FUSION))
        s=jax.jit(lambda x:core.refresh(bank,x,fus=False))(s)
        self.assertEqual(int(s.atk_c[0]),600)
        changed=s._replace(suminfo=s.suminfo.at[1].set(core.SUMMON_SPECIAL))
        actual=jax.jit(lambda old,new:core.refresh_if(bank,old,new))(s,changed)
        expected=jax.jit(lambda x:core.refresh(bank,x,fus=False))(changed)
        self.assertEqual(int(actual.atk_c[0]),int(expected.atk_c[0]))
        self.assertEqual(int(actual.atk_c[0]),500)
if __name__=='__main__':unittest.main()
