import sys, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "engine"); sys.path.insert(0, "train")
import core, ppo, realcards, behavior, model, optax
path = sys.argv.pop(1); a = ppo.parse(); model.set_compute(False); params = behavior.load_params(path); fns = ppo.make_fns(a, optax.adam(1e-4)); tok, msk = model.card_tokens(realcards.load()[0]);  env = ppo.Env(a.b, a.frac)
bank, names, main_, extra = realcards.load(); cn = realcards.card_names()
nd = len(names)
cfg = np.array([(i, j) for i in range(nd) for j in range(nd)])[np.arange(env.b) % (nd * nd)]
D, X = np.stack([main_[cfg[:, 0]], main_[cfg[:, 1]]], 1), np.stack([extra[cfg[:, 0]], extra[cfg[:, 1]]], 1)
@jax.jit
def tick(s, m, f, key):
    cemb = fns[5](params, tok, msk)
    a_ = jnp.where(s.dec_player == 0, fns[0](params, cemb, bank, s, m, f, key, jnp.asarray(True))[0], ppo.heuristic(bank, s, m, f, key))
    fl, _ = jax.vmap(lambda s, m, a: behavior.probe(bank, s, m, a))(s, m, a_)
    s2, m2, f2 = core.step_batch(bank, s, a_, a.frac)
    return s, m, a_, fl, s2, m2, f2
key = jax.random.PRNGKey(5)
s, m = env.new(bank, jnp.asarray(D), jnp.asarray(X), jax.random.split(key, env.b)); f = jnp.full(env.b, -1, jnp.int32)
K = behavior.PROBES.index(("reborn_cible", behavior.PROBES[[p[0] for p in behavior.PROBES].index("reborn_cible")][1]))
n = s.cid.shape[1]; nva = core.NV * n; shown = 0
for t in range(600):
    key, k = jax.random.split(key)
    s0, m0, a0, fl, s, m, f = tick(s, m, f, k)
    fl = np.asarray(fl)
    for g in np.nonzero(fl[:, K, 0] & (np.asarray(s0.dec_player) == 0))[0]:
        S0 = jax.tree.map(lambda x: np.asarray(x[g]), s0); M = np.asarray(m0[g]); A = int(a0[g])
        mv = M[:nva].reshape(core.NV, n)[core.V_SELECT]
        print(f"t{t} g{g} err={fl[g,K,1]} dec={S0.dec} need={S0.need} picks={S0.picks} smin={S0.smin} a={A} verb={A//n if A<nva else -1} slot={A%n} sp={A-nva}")
        for i in np.nonzero(mv)[0]:
            print(f"   cand slot {i} {cn[S0.cid[i]][:24]:24s} zone={S0.zone[i]} ctrl={S0.ctrl[i]} batk={int(bank.atk[S0.cid[i]])}")
        print(f"   choisi {cn[S0.cid[A%n]][:24]} zone={S0.zone[A%n]} batk={int(bank.atk[S0.cid[A%n]])} ; spec légales={np.nonzero(M[nva:])[0].tolist()}")
        shown += 1
    if shown > 25 or np.asarray(s.done).all(): break
