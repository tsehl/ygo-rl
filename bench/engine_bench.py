"""Débit du moteur sous step_batch (décisions de la politique par seconde), politique aléatoire.

Usage : python engine_bench.py DOSSIER_ENGINE [--b 1024,4096] [--frac 0.05] [--calls 60] [--old-cards] [--extra X]
DOSSIER_ENGINE : dossier contenant core.py et cards.py (moteur actuel, ou ancien moteur extrait de git).
--old-cards : les cartes à effet des decks sont tirées parmi celles d'avant le jalon 4 (isole le coût du moteur).
Une décision = une action choisie par la politique (les actions forcées rendues par step_batch ne comptent pas).
Les parties finies sont remplacées par des parties neuves (réserve pré-générée), pour garder le lot plein.
"""
import argparse
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("engine")
ap.add_argument("--b", default="1024,4096")
ap.add_argument("--frac", type=float, default=0.05)
ap.add_argument("--calls", type=int, default=60)
ap.add_argument("--old-cards", action="store_true")
ap.add_argument("--extra", type=int, default=0, help="Extra Deck de X monstres fusion par joueur (cards.test_decks)")
args = ap.parse_args()
sys.path.insert(0, args.engine)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import cards  # noqa: E402
import core  # noqa: E402

D, C = 40, 670
key = jax.random.PRNGKey(0)
bank = cards.test_bank(key, C)
codes = np.asarray(bank.code)
new = set()
for name in ("J4", "J4B", "J4C", "J4D"):
    new |= getattr(cards, name, set())
fus = set(getattr(cards, "FUSION", {})) | set(getattr(cards, "SYNCHRO", {})) | set(getattr(cards, "XYZ", {}))  # Extra Deck : jamais dans le deck principal
eff = np.array([i for i in range(C, len(codes)) if not (args.old_cards and int(codes[i]) in new) and int(codes[i]) not in fus])
print(f"jax {jax.__version__} {jax.devices()[0].device_kind} ; banque {len(codes)} cartes, {len(eff)} à effet tirables",
      flush=True)


def decks(k, b):
    if args.extra:
        return cards.test_decks(k, bank, b, D, C, 0.3, args.extra)
    k1, k2, k3 = jax.random.split(k, 3)
    normal = jax.random.randint(k1, (b, 2, D), 0, C)
    effc = jnp.asarray(eff)[jax.random.randint(k2, (b, 2, D), 0, len(eff))]
    return jnp.where(jax.random.uniform(k3, (b, 2, D)) < 0.3, effc, normal)  # 30 % de cartes à effet


def run(b):
    if args.extra:
        ng0 = jax.jit(jax.vmap(lambda d, x, k: core.new_game(bank, d[0], d[1], k, x[0], x[1])))
        ng = lambda dx, k: ng0(dx[0], dx[1], k)
    else:
        ng = jax.jit(jax.vmap(lambda d, k: core.new_game(bank, d[0], d[1], k)))
    legal = jax.jit(jax.vmap(lambda s: core.legal(bank, s)))
    sb = jax.jit(lambda s, a: core.step_batch(bank, s, a, args.frac))
    t0 = time.time()
    s = ng(decks(jax.random.fold_in(key, 1), b), jax.random.split(jax.random.fold_in(key, 2), b))
    pool = ng(decks(jax.random.fold_in(key, 3), b), jax.random.split(jax.random.fold_in(key, 4), b))
    m = legal(s)
    pm = legal(pool)
    jax.block_until_ready(pm)
    t_new = time.time() - t0
    f = jnp.full(b, -1, jnp.int32)

    @jax.jit
    def policy(k, m, f, done):
        a = jax.random.categorical(k, jnp.where(m, 0.0, -1e9))
        return jnp.where(f >= 0, f, a).astype(jnp.int32), ((f < 0) & ~done).sum()

    @jax.jit
    def refill(s, m, f, pool, pm):
        d = s.done
        w = lambda x, y: jnp.where(d.reshape(d.shape + (1,) * (x.ndim - 1)), y, x)
        return jax.tree.map(w, s, pool), w(m, pm), jnp.where(d, -1, f)

    kk = jax.random.fold_in(key, 5)
    times, dec = [], 0
    for i in range(args.calls + 3):
        a, nd = policy(jax.random.fold_in(kk, i), m, f, s.done)
        t1 = time.time()
        s, m, f = sb(s, a)
        jax.block_until_ready(f)
        dt = time.time() - t1
        if i == 0:
            print(f"  B={b} : compilation step_batch + 1er appel {dt:.0f} s (new_game + legal {t_new:.0f} s)", flush=True)
        if i >= 3:
            times.append(dt)
            dec += int(nd)
        s, m, f = refill(s, m, f, pool, pm)
    tot = sum(times)
    print(f"  B={b} : {dec / tot:,.0f} décisions/s ; {1000 * tot / len(times):.1f} ms/appel ; "
          f"{dec / len(times):.0f} décisions/appel", flush=True)


for b in map(int, args.b.split(",")):
    run(b)
