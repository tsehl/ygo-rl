"""Heuristique à style (ppo.heuristic_style) contre l'heuristique de référence (ppo.heuristic) : taux de victoire de
quelques styles, sur une banque procédurale jamais vue et sur les vraies cartes (decks par paires). Vérifie aussi
que STYLE_REF redonne exactement heuristic.

Usage : python bench/heur_styles.py [--b 1024 --eval-steps 600] (mêmes options que ppo.py)"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", d) for d in ("engine", "gen", "train")]
import numpy as np
import jax
import jax.numpy as jnp

import cardgen
import ppo
import realcards

R = ppo.STYLE_REF
STYLES = {   # agressivité, choix, prudence, zèle, température
    "référence": R,
    "choix": (1.0, 1.0, 0.0, 1.0, 0.01),
    "pas_suicide": (0.0, 0.0, 0.0, 1.0, 0.01),
    "prudent": (1.0, 0.0, 1.0, 1.0, 0.01),
    "calme": (1.0, 0.0, 0.0, 0.0, 0.01),
    "choix+pas_suicide": (0.0, 1.0, 0.0, 1.0, 0.01),
    "tout": (0.0, 1.0, 1.0, 0.0, 0.01),
    "bruité": (1.0, 0.0, 0.0, 1.0, 0.5),
}


def play(a, env, bank, d, x, side, key, sty):
    """Style sty [B, N_STYLE] (joueur side) contre la référence : (finies, résultat)."""
    pick = jax.jit(lambda bank, s, m, f, k, side, sty: jnp.where(
        s.dec_player == side, ppo.heuristic_style(bank, s, m, f, k, sty), ppo.heuristic(bank, s, m, f, k)))
    s, m = env.new(bank, jnp.asarray(d), jnp.asarray(x), jax.random.split(key, env.b))
    f = jnp.full(env.b, -1, jnp.int32)
    sd = np.asarray(side)
    side = jnp.asarray(side)
    fin, res = np.zeros(env.b, bool), np.zeros(env.b, int)
    same = True
    for t in range(a.eval_steps):
        key, k = jax.random.split(key)
        ac = pick(bank, s, m, f, k, side, sty)
        if t < 20 and np.allclose(np.asarray(sty), R):      # STYLE_REF == heuristic
            same &= bool((ac == ppo.heuristic(bank, s, m, f, k)).all())
        s, m, f = env.step(bank, s, ac)
        dn, w = np.asarray(s.done), np.asarray(s.winner)
        new = dn & ~fin
        res[new] = (w[new] == sd[new]).astype(int) - (w[new] == 1 - sd[new]).astype(int)
        fin |= dn
        if fin.all():
            break
    return fin, res, same


def main():
    a = ppo.parse()
    env = ppo.Env(a.b, a.frac)
    print(f"jax {jax.__version__} {jax.devices()[0].device_kind} ; B={a.b}", flush=True)
    cset, pbank = ppo.bank_of(a.eval_bank, a)
    d, x = cardgen.random_decks(cset, np.random.default_rng(12345), env.b // 4)
    pD, pX, pside = ppo.paired(d, x)
    rbank, names, main_, extra = realcards.load()
    nd = len(names)
    cfg = np.array([(i, j, s) for i in range(nd) for j in range(nd) for s in (0, 1)])[np.arange(env.b) % (2 * nd * nd)]
    p0 = np.where(cfg[:, 2] == 0, cfg[:, 0], cfg[:, 1])
    p1 = np.where(cfg[:, 2] == 0, cfg[:, 1], cfg[:, 0])
    rD, rX, rside = np.stack([main_[p0], main_[p1]], 1), np.stack([extra[p0], extra[p1]], 1), cfg[:, 2]
    styles = {k: jnp.broadcast_to(jnp.asarray(v, jnp.float32), (env.b, ppo.N_STYLE)) for k, v in STYLES.items()}
    styles["tirés (entraînement)"] = ppo.sample_style(jax.random.PRNGKey(5), env.b)
    print(f"{'style':24s}{'procédural':>22s}{'vraies cartes':>22s}", flush=True)
    for nm, sty in styles.items():
        t0 = time.time()
        out = []
        for bank, D, X, side, k in ((pbank, pD, pX, pside, 1), (rbank, rD, rX, rside, 2)):
            fin, res, same = play(a, env, bank, D, X, side, jax.random.PRNGKey(k), sty)
            n = max(int(fin.sum()), 1)
            out.append(f"{(res[fin] == 1).sum() / n:.3f} / {(res[fin] == -1).sum() / n:.3f} ({int(fin.sum())})")
            if nm == "référence":
                print(f"STYLE_REF == heuristic : {same}", flush=True)
        print(f"{nm:24s}" + "".join(f"{o:>22s}" for o in out) + f"   ({time.time() - t0:.0f} s)", flush=True)
    print("FIN", flush=True)


if __name__ == "__main__":
    main()
