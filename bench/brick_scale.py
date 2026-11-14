"""Part fixe et part linéaire du coût de chaque brique du moteur selon la taille du lot (GPU).

Des états de milieu de partie (B0 parties, `--calls` appels de step_batch, politique aléatoire) sont recopiés ou
coupés à chaque taille de `--sizes`, puis chaque brique vmappée est chronométrée seule, dans une boucle compilée. Si le
temps est proportionnel à B, le GPU est saturé (bande passante) : un sous-lot des parties actives et un état plus petit
paient. S'il est plat, c'est le nombre d'opérations lancées par tour qui coûte : il faut moins de tours.

Usage : python bench/brick_scale.py [--b0 1024] [--calls 100] [--sizes 64,128,256,512,1024,2048,4096] [--proc 0]"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "engine"), os.path.join(HERE, "..", "gen")]
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import cardgen  # noqa: E402
import core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--b0", type=int, default=1024)
ap.add_argument("--calls", type=int, default=100)
ap.add_argument("--sizes", default="64,128,256,512,1024,2048,4096")
ap.add_argument("--proc", type=int, default=0)
ap.add_argument("--frac", type=float, default=0.1)
ap.add_argument("--n", type=int, default=5)
a = ap.parse_args()

cset = cardgen.Gen(a.proc).make_set(400, 80)
bank = cardgen.to_bank(cset)
d, x = cardgen.random_decks(cset, np.random.default_rng(1), a.b0)
ng = jax.jit(jax.vmap(lambda bank, d, x, k: core.new_game(bank, d[0], d[1], k, x[0], x[1]), in_axes=(None, 0, 0, 0)))
lg = jax.jit(jax.vmap(lambda bank, s: core.legal(bank, s), in_axes=(None, 0)))
s = ng(bank, jnp.asarray(d), jnp.asarray(x), jax.random.split(jax.random.PRNGKey(0), a.b0))
m = lg(bank, s)
print(f"jax {jax.__version__} {jax.devices()[0].device_kind} ; B0={a.b0}", flush=True)
if a.calls:
    sb = jax.jit(lambda bank, s, x: core.step_batch(bank, s, x, a.frac))
    pol = jax.jit(lambda k, m, f: jnp.where(f >= 0, f, jax.random.categorical(k, jnp.where(m, 0.0, -1e9))).astype(jnp.int32))
    f = jnp.full(a.b0, -1, jnp.int32)
    key = jax.random.PRNGKey(1)
    t0 = time.time()
    for t in range(a.calls):
        key, k = jax.random.split(key)
        s, m, f = sb(bank, s, pol(k, m, f))
        if t == 0:
            jax.block_until_ready(f)
            print(f"compilation de step_batch {time.time() - t0:.0f} s", flush=True)
            t0 = time.time()
    jax.block_until_ready(f)
    live = ~np.asarray(s.done)
    print(f"{a.calls} appels en {time.time() - t0:.0f} s ; {int(live.sum())} parties en cours, tour moyen "
          f"{float(np.asarray(s.turn)[live].mean()):.1f}", flush=True)

# Chaque brique tourne K fois dans une boucle compilée (lax.fori_loop), comme dans step_batch : un appel Python par
# brique paie ~1 à 2 ms de passage des 111 tableaux de l'état, ce qui se confond avec une part fixe du GPU.
K = 20
V = lambda fn: (lambda bk, s: jax.vmap(fn, in_axes=(None, 0))(bk, s))
# legal et auto_pass ne sont pas mesurés ici : ils ne modifient pas l'état, et XLA sort de la boucle un calcul qui
# ne dépend pas du tour (0,13 ms sur CPU, non crédible).
ref = V(lambda b, s: core.refresh(b, s, fus=False))


def where_all(bk, s):
    m = (s.turn + s.clk) % 2 == 0
    return jax.tree.map(lambda x: jnp.where(m.reshape((-1,) + (1,) * (x.ndim - 1)), x, ~x if x.dtype == jnp.bool_ else x + 1), s)


bricks = dict(vide=lambda bk, s: s._replace(clk=s.clk + 1), where_etat=where_all, exec_one=V(core.exec_one), refresh=ref,
              exec_refresh=lambda bk, s: ref(bk, V(core.exec_one)(bk, s)), pass_priority=V(core.pass_priority),
              trig_eval=V(core.trig_eval))
sizes = [int(v) for v in a.sizes.split(",")]
res = {k: [] for k in bricks}
for b in sizes:
    idx = jnp.arange(b) % a.b0
    sx = jax.tree.map(lambda x: x[idx], s)
    for k, body in bricks.items():
        fn = jax.jit(lambda bk, s, body=body: jax.lax.fori_loop(0, K, lambda i, s: body(bk, s), s))
        jax.block_until_ready(fn(bank, sx))
        t1 = time.time()
        for _ in range(a.n):
            r = fn(bank, sx)
        jax.block_until_ready(r)
        res[k].append(1000 * (time.time() - t1) / a.n / K)
    print(f"B={b} fait", flush=True)
print(f"\nms par tour{'':9s}" + "".join(f"{b:>9d}" for b in sizes) + "   µs / partie au plus grand B")
for k, v in res.items():
    print(f"{k:20s}" + "".join(f"{t:9.3f}" for t in v) + f"{1000 * v[-1] / sizes[-1]:14.2f}")
