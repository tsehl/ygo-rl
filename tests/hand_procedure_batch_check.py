"""Lot mêlant procédures de main et invocations normales : step_batch == vmap(step)."""
import jax
import jax.numpy as jnp
import numpy as np
from oracle_bank import build
import core,obs

CYBER,JUNK,ELF=70095154,56410040,44287299
_,bank,cmap=build([CYBER,JUNK,ELF])
b,n=4,80
deck=jnp.array([cmap[c] for c in [CYBER,JUNK,ELF]*13+[ELF]],jnp.int32)
states=jax.jit(jax.vmap(lambda key:core.new_game(bank,deck,deck,key)))(jax.random.split(jax.random.PRNGKey(71),b))
legal=jax.jit(jax.vmap(lambda s:core.legal(bank,s)))
step=jax.jit(jax.vmap(lambda s,a:core.step(bank,s,a)))
batch=jax.jit(lambda s,a:core.step_batch(bank,s,a,0.0))
observe=jax.jit(jax.vmap(lambda s,m:obs.observe(bank,s,m=m)))
sa=sb=states;m=legal(sa);uses=0;normal=0
for turn in range(40):
    verbs=m[:,:core.NV*n].reshape(b,core.NV,n)
    proc=verbs[:,core.V_ACT0:] & (bank.ekind[sa.cid].transpose(0,2,1)==core.K_PROC)
    has_proc=proc.reshape(b,-1).any(1)
    has_normal=verbs[:,core.V_SUMMON].any(1)
    # Deux politiques distinctes pour exercer les deux branches dans le même lot.
    prefer_proc=(jnp.arange(b)%2==0) | ~has_normal
    random=jax.random.categorical(jax.random.PRNGKey(turn),jnp.where(m,0.0,-1e9)).astype(jnp.int32)
    action=jnp.where(has_normal,verbs[:,core.V_SUMMON].argmax(1),random)
    action=jnp.where(has_proc&prefer_proc,proc.reshape(b,-1).argmax(1)+core.V_ACT0*n,action).astype(jnp.int32)
    uses+=int((has_proc&prefer_proc&~sa.done).sum())
    normal+=int((has_normal&~(has_proc&prefer_proc)&~sa.done).sum())
    active=~sa.done
    next_state=step(sa,jnp.where(active,action,0))
    sa=jax.tree.map(lambda old,new:jnp.where(active.reshape((-1,)+(1,)*(old.ndim-1)),new,old),sa,next_state)
    sb,mask,forced=batch(sb,jnp.where(sb.done,-1,action))
    assert not bool((forced>=0).any())
    for name,x,y in zip(core.State._fields,sa,sb):
        assert np.array_equal(np.asarray(x),np.asarray(y)),(turn,name)
    m=legal(sa)
    assert np.array_equal(np.asarray(m&~sa.done[:,None]),np.asarray(mask&~sb.done[:,None]))
    o=observe(sa,m)
    # Chaque action sur une carte, procédures comprises, garde un token visible.
    np.testing.assert_array_equal(np.asarray(o.act.sum((1,2))),np.asarray(m[:,:core.NV*n].sum(1)))
assert uses>0 and normal>0,(uses,normal)
print(f'4 parties × 40 décisions : step_batch == vmap(step), {uses} procédures et {normal} invocations normales, actions tokenisées',flush=True)
