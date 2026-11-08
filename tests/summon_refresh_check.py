import sys
sys.path[:0]=['engine']
import core,cards,jax,jax.numpy as jnp,numpy as np
codes=np.array([1,2]);f=cards.F(me=core.Z_GRAVE,st=core.summon_mask(core.SUMMON_FUSION))
bank=core.make_bank(codes,[500,500],[500,500],[4,4],**cards.assemble(codes,cards={1:[cards.cont(core.Z_MZONE,core.MC_ATK,100,per=f)],2:[]},ritual={},fusion={},synchro={},xyz={},link={}))
d=jnp.arange(40,dtype=jnp.int32)%2;s=jax.jit(core.new_game)(bank,d,d)
s=s._replace(zone=s.zone.at[0].set(core.Z_MZONE).at[1].set(core.Z_GRAVE),pos=s.pos.at[0].set(core.P_FU_ATK),seq=s.seq.at[0].set(0).at[1].set(0),suminfo=s.suminfo.at[1].set(core.SUMMON_FUSION))
s=jax.jit(lambda x:core.refresh(bank,x,fus=False))(s)
changed=s._replace(suminfo=s.suminfo.at[1].set(core.SUMMON_SPECIAL))
a=jax.jit(lambda x,y:core.refresh_if(bank,x,y))(s,changed)
print('ATK après changement du type sans déplacement :',int(a.atk_c[0]),'; attendu : 500',flush=True)
assert int(a.atk_c[0])==500
