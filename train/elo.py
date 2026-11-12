"""Tournoi entre réseaux sauvegardés (et l'heuristique de référence) et classement Elo.

Chaque paire de joueurs est jouée sur deux terrains, avec les fonctions d'évaluation de ppo.py :
- « proc » : banque procédurale d'évaluation à archétypes (jamais vue), decks structurés joués par paires
  (ppo.evaluate struct=True : chaque paire de decks 4 fois, decks et côtés échangés) ;
- « vrai » : vraies cartes, toutes les rencontres de decks des deux côtés (ppo.real_bench).
Les résultats (victoires / défaites / nuls ou parties non finies) sont gardés dans un JSON après chaque match : on
peut relancer après une coupure, les matchs déjà faits ne sont pas rejoués.

Elo : modèle de Bradley-Terry (P(i bat j) = 1 / (1 + 10^((Rj - Ri) / 400))), maximum de vraisemblance, un nul compte
une demi-victoire de chaque côté ; référence fixée à 0 (l'heuristique si elle joue, sinon le premier réseau).
Erreur type par l'information de Fisher. Un écart de 100 Elo ≈ 64 % de victoires, 200 ≈ 76 %.

Usage : python train/elo.py run3=train/ckpt/run3_it120_f16.pkl.gz run5=train/ckpt/run5_it360_f16.pkl.gz ... \
            [--heur] [--out elo.json] [--terrains proc,vrai] [options de ppo.py : --b 1024 --bf16 ...]
ELO_GREEDY=1 : réseaux déterministes (argmax).
"""
import json
import os
import sys
import time

import numpy as np

import ppo
import jax
import optax

import model


def split_args(argv):
    players, rest, extra = [], [], dict(heur=False, out="elo.json", terrains="proc,vrai")
    it = iter(argv)
    for x in it:
        if x == "--heur":
            extra["heur"] = True
        elif x in ("--out", "--terrains"):
            extra[x[2:]] = next(it)
        elif "=" in x and not x.startswith("-"):
            name, path = x.split("=", 1)
            players.append((name, path))
        else:
            rest.append(x)
    return players, rest, extra


def fit_elo(names, games, ref):
    """games : liste (i, j, victoires de i, défaites de i, nuls). Rend (elo [n], erreur type [n])."""
    n = len(names)
    W = np.zeros((n, n))                     # W[i, j] : points de i contre j (nul = 1/2)
    for i, j, w, l, d in games:
        W[i, j] += w + d / 2
        W[j, i] += l + d / 2
    N = W + W.T
    r = np.zeros(n)
    free = np.array([k != ref for k in range(n)])
    for _ in range(2000):                    # Newton sur les log-forces (convexe)
        p = 1 / (1 + np.exp(r[None, :] - r[:, None]))             # P(i bat j)
        g = (W - N * p).sum(1)
        H = -np.diag((N * p * (1 - p)).sum(1)) + N * p * (1 - p)
        Hf = H[np.ix_(free, free)]
        step = np.linalg.solve(Hf - 1e-9 * np.eye(Hf.shape[0]), g[free])
        r[free] -= step
        if np.abs(step).max() < 1e-10:
            break
    cov = np.linalg.inv(-Hf + 1e-9 * np.eye(Hf.shape[0]))
    se = np.zeros(n)
    se[free] = np.sqrt(np.clip(np.diag(cov), 0, None))
    k = 400 / np.log(10)
    return r * k, se * k


def main():
    players, rest, ex = split_args(sys.argv[1:])
    sys.argv = sys.argv[:1] + rest
    a = ppo.parse()
    model.set_compute(a.bf16)
    greedy = bool(os.environ.get("ELO_GREEDY"))
    names = [n for n, _ in players] + (["heur"] if ex["heur"] else [])
    params = {n: ppo.load_params(p) for n, p in players}
    terrains = ex["terrains"].split(",")
    env = ppo.Env(a.b, a.frac)
    fns = ppo.make_fns(a, optax.adam(1e-4))
    res = json.load(open(ex["out"])) if os.path.exists(ex["out"]) else {}
    print(f"jax {jax.__version__} {jax.devices()[0].device_kind} ; B={a.b} ; joueurs {names} ; greedy {greedy}", flush=True)
    key = jax.random.PRNGKey(11)
    for ti, ter in enumerate(terrains):
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ni, nj = names[i], names[j]
                tag = f"{ter}|{ni}|{nj}"
                if tag in res:
                    continue
                if ni == "heur":                  # l'heuristique ne joue que comme adversaire : on retourne la paire
                    ni, nj = nj, ni
                kind, op = (2, None) if nj == "heur" else (1, params[nj])
                k = jax.random.fold_in(key, ti)  # même graine pour tous les matchs d'un terrain : mêmes donnes
                t0 = time.time()
                if ter == "proc":
                    ev = ppo.evaluate(a, env, fns, params[ni], k, kind, op, pre="m", greedy=greedy, struct=True)
                else:
                    ev = ppo.real_bench(a, env, fns, params[ni], k, kind, op, pre="m", greedy=greedy)
                g = ev["m_games"]
                w, l = round(ev["m_win"] * g), round(ev["m_loss"] * g)
                res[tag] = dict(i=ni, j=nj, win=w, loss=l, draw=g - w - l, unfinished=a.b - g)
                json.dump(res, open(ex["out"], "w"), indent=1)
                print(f"{ter} {ni} contre {nj} : {w}-{l}-{g - w - l} ({w / max(g, 1):.1%} ; {a.b - g} non finies ; "
                      f"{time.time() - t0:.0f} s)", flush=True)
    ref = names.index("heur") if "heur" in names else 0
    for sel in terrains + ["total"]:
        games = [(names.index(v["i"]), names.index(v["j"]), v["win"], v["loss"], v["draw"])
                 for t, v in res.items() if (sel == "total" or t.split("|")[0] == sel)
                 and v["i"] in names and v["j"] in names]
        elo, se = fit_elo(names, games, ref)
        print(f"\nElo {sel} (référence {names[ref]} = 0) :")
        for k in np.argsort(-elo):
            print(f"  {names[k]:>12s} {elo[k]:7.0f} ± {se[k]:.0f}")
    print("FIN", flush=True)


if __name__ == "__main__":
    main()
