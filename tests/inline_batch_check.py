"""Lots mêlant oui/non, compteurs remboursables et décisions ordinaires."""
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'engine'))
import core, cards

activation = (*cards.act()[:7], dict(inline=2))
quick = cards.quick(core.Z_SZONE, cost=[cards.PAY_LP(500)], resolve=[cards.DRAW(0, 1)], opt=core.OPT_NAME)
refundable = (*quick[:7], dict(quick[7], refund=1))
bank = core.make_bank(np.array([1,2,3]), [500,0,0], [500,0,0], [4,0,0],
    ctype=[17,core.TYPE_TRAP|core.TYPE_CONTINUOUS,core.TYPE_TRAP|core.TYPE_CONTINUOUS],
    **cards.assemble([1,2,3], cards={2:[activation,refundable],3:[activation,quick]},
                     ritual={},fusion={},synchro={},xyz={},link={}))
normal = jnp.zeros(40,jnp.int32)
initialize = jax.jit(core.new_game)
begin, apply, execute = jax.jit(core.begin_act), jax.jit(core.apply), jax.jit(core.exec_one)
def question(cid):
    state = initialize(bank,normal.at[0].set(cid),normal)
    state = core.move(state,jnp.int32(0),core.Z_SZONE,jnp.int32(0),jnp.int32(0),core.P_SFU)
    return begin(bank,state._replace(dec_player=jnp.int32(0),wk=jnp.int32(core.W_QUICK)),jnp.int32(0),jnp.int32(0))
def negated(cid):
    state = apply(bank,question(cid),jnp.int32(core.special(80,core.A_YES)))
    for _ in range(3):state=execute(bank,state)
    assert int(state.n_chain)==1
    return state._replace(lk_neg=state.lk_neg.at[0].set(True),
        lk_flags=state.lk_flags.at[0].set(state.lk_flags[0]|core.CF_ACT_NEG),
        st=state.st.at[0].set(state.st[0]|core.ST_LEAVE|core.ST_ACTDIS))
states=[question(1),question(1),question(2),question(2),negated(1),negated(2),initialize(bank,normal,normal),initialize(bank,normal,normal)]
state=jax.tree.map(lambda *values:jnp.stack(values),*states)
vstep=jax.jit(jax.vmap(core.step,in_axes=(None,0,0)))
legal=jax.jit(jax.vmap(core.legal,in_axes=(None,0)))
bstep=jax.jit(lambda bank,state,actions:core.step_batch(bank,state,actions,0.0))
mask=legal(bank,state)
for tick in range(12):
    actions=jax.random.categorical(jax.random.PRNGKey(1700+tick),jnp.where(mask,0.,-1e9)).astype(jnp.int32)
    if tick==0:
        choices=[core.A_YES,core.A_PASS,core.A_YES,core.A_PASS,core.A_PASS,core.A_PASS]
        actions=actions.at[:6].set(jnp.asarray([core.special(80,k) for k in choices],jnp.int32))
    expected=vstep(bank,state,actions)
    actual,actual_mask,forced=bstep(bank,state,actions)
    for field,a,b in zip(core.State._fields,expected,actual):
        np.testing.assert_array_equal(a,b,err_msg=f'{tick}: {field}')
    np.testing.assert_array_equal(legal(bank,expected),actual_mask)
    assert not np.asarray(forced>=0).any()
    if tick==0:
        assert not np.asarray(actual.opt_cid[4]).any(), 'compteur OATH non rendu'
        assert np.asarray(actual.opt_cid[5]).any(), 'compteur sans OATH rendu'
    state,mask=actual,actual_mask
print('8 parties × 12 décisions : step_batch == vmap(step), oui/non et deux modes de compteur exercés',flush=True)
