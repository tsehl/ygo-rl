"""Diagnostic du débit de step_batch au fil des parties : tours de chaque boucle par appel et temps par appel, selon
l'avancement des parties (sans remplacement des parties finies). Politique aléatoire, banque procédurale.

Usage : python bench/loop_stats.py [--b 64] [--calls 300] [--every 20] [--proc 0] [--frac 0.05]
Compteurs (core.step_batch(stats=True)) : ext = tours de la boucle externe (rnd), settle = tours de settle_b,
run = tours de run_b, trig / pass / pass2 / fus = branches lax.cond prises (trig_eval, passes, seconde passe, legal
avec fusion). Tous sont des totaux par appel, au niveau du lot."""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
sys.path.insert(0, os.path.join(HERE, "..", "gen"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import cardgen  # noqa: E402
import core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--b", type=int, default=64)
ap.add_argument("--calls", type=int, default=300)
ap.add_argument("--every", type=int, default=20)
ap.add_argument("--proc", type=int, default=0)
ap.add_argument("--frac", type=float, default=0.05)
ap.add_argument("--prof-at", default="", help="appels (ex. 0,100) où chronométrer chaque brique du moteur sur tout le lot")
a = ap.parse_args()

cset = cardgen.Gen(a.proc).make_set(400, 80)
bank = cardgen.to_bank(cset)
d, x = cardgen.random_decks(cset, np.random.default_rng(1), a.b)
ng = jax.jit(jax.vmap(lambda bank, d, x, k: core.new_game(bank, d[0], d[1], k, x[0], x[1]), in_axes=(None, 0, 0, 0)))
lg = jax.jit(jax.vmap(lambda bank, s: core.legal(bank, s), in_axes=(None, 0)))
sb = jax.jit(lambda bank, s, x: core.step_batch(bank, s, x, a.frac, stats=True))
pol = jax.jit(lambda k, m, f: jnp.where(f >= 0, f, jax.random.categorical(k, jnp.where(m, 0.0, -1e9))).astype(jnp.int32))

s = ng(bank, jnp.asarray(d), jnp.asarray(x), jax.random.split(jax.random.PRNGKey(0), a.b))
m = lg(bank, s)
f = jnp.full(a.b, -1, jnp.int32)
key = jax.random.PRNGKey(1)
t0 = time.time()
jax.block_until_ready(sb(bank, s, pol(key, m, f)))
print(f"jax {jax.__version__} {jax.devices()[0].device_kind} ; B={a.b} ; compilation {time.time() - t0:.0f} s", flush=True)
names = ["ext", "settle", "run", "trig", "pass", "pass2", "fus", "n_run", "n_trig", "n_pass", "n_pas2"]
print(f"{'appels':>9} {'vivantes':>8} {'tour':>5} {'ms/appel':>8} " + " ".join(f"{n:>6}" for n in names), flush=True)
acc, dt = np.zeros(11), 0.0
prof_at = {int(x) for x in a.prof_at.split(",") if x}
V = lambda fn: jax.jit(jax.vmap(fn, in_axes=(None, 0)))
bricks = dict(
    auto_pass=V(core.auto_pass), act_cand=V(lambda b, s: core.act_cand(b, s)), pass_priority=V(core.pass_priority),
    exec_one=V(core.exec_one), trig_eval=V(core.trig_eval), refresh=V(lambda b, s: core.refresh(b, s, fus=False)),
    refresh_fus=V(core.refresh_fus), refresh_rit=V(core.refresh_rit), legal=V(lambda b, s: core.legal(b, s, fus=False)),
    legal_fus=V(core.legal), refresh_syn=V(core.refresh_syn), refresh_xyz=V(core.refresh_xyz),
    refresh_link=V(core.refresh_link), changed=jax.jit(jax.vmap(lambda b, s: core.changed(s, s), in_axes=(None, 0))))
ap_ = jax.jit(jax.vmap(core.apply, in_axes=(None, 0, 0)))
fm_ = jax.jit(jax.vmap(core.forced_m, in_axes=(None, 0, 0)))


def prof(s, m, f, n=5):
    def tm(name, fn, *xs):
        jax.block_until_ready(fn(*xs))
        t1 = time.time()
        for _ in range(n):
            r = fn(*xs)
        jax.block_until_ready(r)
        print(f"  {name:15s} {1000 * (time.time() - t1) / n:8.2f} ms", flush=True)
    for k, fn in bricks.items():
        tm(k, fn, bank, s)
    tm("apply", ap_, bank, s, jnp.zeros(a.b, jnp.int32))
    tm("forced_m", fm_, bank, s, m)


for t in range(a.calls):
    if t in prof_at:
        print(f"briques (B={a.b}, appel {t}, sur tout le lot) :", flush=True)
        prof(s, m, f)
    key, k = jax.random.split(key)
    act = pol(k, m, f)
    jax.block_until_ready(act)
    t1 = time.time()
    s, m, f, c = sb(bank, s, act)
    jax.block_until_ready(c)
    dt += time.time() - t1
    acc += np.asarray(c)
    if (t + 1) % a.every == 0:
        live = ~np.asarray(s.done)
        tour = float(np.asarray(s.turn)[live].mean()) if live.any() else float("nan")
        print(f"{t + 1 - a.every:4d}-{t:<4d} {int(live.sum()):8d} {tour:5.1f} {1000 * dt / a.every:8.1f} "
              + " ".join(f"{v / a.every:6.1f}" for v in acc), flush=True)
        acc, dt = np.zeros(11), 0.0
        if not live.any():
            break
