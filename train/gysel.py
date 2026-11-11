"""Sonde de sélection au cimetière sur une banque procédurale : quand le réseau (déterministe, joueur 0) choisit
une carte parmi des monstres de son cimetière ou du cimetière adverse (D_SELECT), prend-il la plus forte ATK de base ?
Compare au taux attendu d'un choix au hasard parmi les mêmes candidats. Usage : gysel.py CKPT --b 128 [--eval-bank S]"""
import sys
import numpy as np
import jax
import jax.numpy as jnp
import optax

sys.path.insert(0, "engine"); sys.path.insert(0, "train"); sys.path.insert(0, "gen")
import core, ppo, model, cardgen, behavior

path = sys.argv.pop(1)
a = ppo.parse()
model.set_compute(False)
params = behavior.load_params(path)
fns = ppo.make_fns(a, optax.adam(1e-4))
env = ppo.Env(a.b, a.frac, "structured")


@jax.jit
def tick(bank, tok, msk, s, m, f, key):
    cemb = fns[5](params, tok, msk)
    k1, k2 = jax.random.split(key)
    mine = fns[0](params, cemb, bank, s, m, f, k1, jnp.asarray(True))[0]
    a_ = jnp.where(s.dec_player == 0, mine, ppo.heuristic(bank, s, m, f, k2))
    s2, m2, f2 = core.step_batch(bank, s, a_, a.frac)
    return a_, s2, m2, f2


from collections import defaultdict
st = defaultdict(lambda: [0, 0, 0.0])                  # zone d'arrivée -> occasions, erreurs, hasard attendu
for bi in range(int(__import__("os").environ.get("GYSEL_BANKS", "4"))):
  cset, bank = ppo.bank_of(a.eval_bank + bi, a)
  tok, msk = model.card_tokens(bank)
  key = jax.random.PRNGKey(7 + bi)
  s, m = env.games(bank, cset, np.random.default_rng(1 + bi), key)
  f = jnp.full(env.b, -1, jnp.int32)
  N = s.cid.shape[1]
  nva = core.NV * N
  pend = []                                            # (partie, slot choisi, erreur, hasard) en attente de la zone d'arrivée
  for t in range(a.eval_steps):
      key, k = jax.random.split(key)
      S0, M0 = s, m
      a_, s, m, f = tick(bank, tok, msk, s, m, f, k)
      dec, dp, ff = np.asarray(S0.dec), np.asarray(S0.dec_player), np.asarray(f)
      cand = np.nonzero((dec == core.D_SELECT) & (dp == 0) & ~np.asarray(S0.done))[0]
      zn = np.asarray(s.zone)
      for g, sl, e, x in pend:
          r = st[int(zn[g, sl])]; r[0] += 1; r[1] += e; r[2] += x
      pend = []
      if len(cand) == 0:
          continue
      M = np.asarray(M0[cand])[:, :nva].reshape(len(cand), core.NV, N)[:, core.V_SELECT]
      zone, cid = np.asarray(S0.zone[cand]), np.asarray(S0.cid[cand])
      A = np.asarray(a_[cand])
      batk = np.asarray(bank.atk)[cid]
      ismon = (np.asarray(bank.ctype)[cid] & core.TYPE_MONSTER) != 0
      for j in range(len(cand)):
          c = M[j]
          if c.sum() < 2 or not (zone[j][c] == core.Z_GRAVE).all() or not ismon[j][c].all() or A[j] >= nva:
              continue
          v = batk[j][c]
          if v.max() == v.min():
              continue
          pend.append((cand[j], A[j] % N, int(batk[j][A[j] % N] < v.max()), 1 - (v == v.max()).mean()))
      if np.asarray(s.done).all():
          break
ZN = {v: k for k, v in vars(core).items() if k.startswith("Z_") and isinstance(v, int)}
for z, (o, e, x) in sorted(st.items()):
    print(f"{path} : arrivée {ZN.get(z, z)} : {o} occasions, erreur {e / o:.3f}, hasard attendu {x / o:.3f}")
