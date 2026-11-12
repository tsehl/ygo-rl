"""Couverture des mécaniques à l'entraînement : en jeu aléatoire, combien d'occasions (action légale) et d'actions
jouées par mécanique, banques procédurales (decks aléatoires, comme ppo.py) contre vraies cartes (4 decks de
référence), avec decks aléatoires et structurés (cardgen.structured_decks). Une mécanique sans occasion en procédural ne peut pas être apprise.

Usage : python bench/coverage.py [--b 64] [--calls 200] [--banks 0,1]"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", d) for d in ("engine", "gen", "train")]
import numpy as np
import jax
import jax.numpy as jnp

import core
import cardgen
import ppo
import realcards

ROWS = ["synchro", "xyz", "link", "pendule_invoc", "pendule_echelle", "fusion", "rituel", "jeu_rapide_ou_piege",
        "chaine_avec_activation", "select_2plus"]


def kinds_proc(cset):
    """Drapeaux par index de banque : (sort de fusion, sort rituel)."""
    k0 = lambda c: c["effects"][0]["kind"] if c["effects"] else ""
    return (np.array([k0(c) == "fusion_spell" for c in cset]), np.array([k0(c) == "ritual" for c in cset]))


def make_count(bank, fus, rit):
    n_ = None
    fus, rit = jnp.asarray(fus), jnp.asarray(rit)

    def one(s, m, a):
        n = s.cid.shape[0]
        nva = core.NV * n
        mv = m[:nva].reshape(core.NV, n)
        verb, slot = jnp.where(a < nva, a // n, -1), a % n
        ct = bank.ctype[s.cid]
        mine = s.ctrl == s.dec_player
        ex = (s.zone == core.Z_EXTRA) & mine
        idle = s.dec == core.D_IDLE
        act_l = mv[core.V_ACT0:].any(0)
        act = verb >= core.V_ACT0
        summ = idle & mv[core.V_SUMMON]
        rows = []
        for t in (core.TYPE_SYNCHRO, core.TYPE_XYZ, core.TYPE_LINK):
            o = summ & ex & ((ct & t) != 0)
            rows.append((o.any(), idle & (verb == core.V_SUMMON) & o[slot]))
        o = summ & (s.zone == core.Z_SZONE)                         # invocation pendule : V_SUMMON sur l'échelle
        rows.append((o.any(), idle & (verb == core.V_SUMMON) & o[slot]))
        pm = (ct & core.TYPE_PENDULUM) != 0
        o = idle & act_l & pm & (s.zone == core.Z_HAND) & mine     # activer un pendule de la main (échelle)
        rows.append((o.any(), act & o[slot]))
        for fl in (fus[s.cid], rit[s.cid]):
            o = act_l & fl & mine
            rows.append((o.any(), act & o[slot]))
        o = (s.dec == core.D_CHAIN) & act_l & mine & ((ct & (core.TYPE_TRAP | core.TYPE_QUICKPLAY)) != 0)
        rows.append((o.any(), act & o[slot]))
        o = (s.dec == core.D_CHAIN) & act_l
        rows.append((o.any(), act & o[slot]))
        o = s.dec == core.D_SELECT
        rows.append((o & (mv[core.V_SELECT].sum() >= 2), o & (mv[core.V_SELECT].sum() >= 2)))
        return jnp.array([[x, y] for x, y in rows], jnp.int32)

    return jax.jit(jax.vmap(one))


def run(env, bank, D, X, fus, rit, calls, key):
    count = make_count(bank, fus, rit)
    s, m = env.new(bank, jnp.asarray(D), jnp.asarray(X), jax.random.split(key, env.b))
    f = jnp.full(env.b, -1, jnp.int32)
    tot = np.zeros((len(ROWS), 2), np.int64)
    ndec = 0
    fin = np.zeros(env.b, bool)
    for t in range(calls):
        key, k = jax.random.split(key)
        a = jnp.where(f >= 0, f, jax.random.categorical(k, jnp.where(m, 0.0, -1e9)).astype(jnp.int32))
        dec = np.asarray((f < 0) & ~s.done)
        c = np.asarray(count(s, m, a))
        tot += (c * dec[:, None, None]).sum(0)
        ndec += int(dec.sum())
        s, m, f = env.step(bank, s, a)
        fin |= np.asarray(s.done)
    return tot, ndec, int(fin.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=64)
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--banks", default="0,1")
    ap.add_argument("--main", type=int, default=400)
    ap.add_argument("--extra", type=int, default=80)
    ap.add_argument("--arch", type=float, default=0.5, help="colonne en plus : banque Gen(arch), decks structurés (0 : pas de colonne)")
    a = ap.parse_args()
    env = ppo.Env(a.b, 0.1)
    res = {}
    for sd in [int(x) for x in a.banks.split(",")]:
        cset, bank = ppo.bank_of(sd, a)
        d, x = cardgen.random_decks(cset, np.random.default_rng(sd), a.b)
        res[f"proc{sd}"] = run(env, bank, d, x, *kinds_proc(cset), a.calls, jax.random.PRNGKey(sd))
        d, x = cardgen.structured_decks(cset, np.random.default_rng(sd), a.b)
        res[f"proc{sd}_struct"] = run(env, bank, d, x, *kinds_proc(cset), a.calls, jax.random.PRNGKey(sd))
        if a.arch:                                 # banque à archétypes, decks structurés (entraînement depuis le 2026-11-12)
            cset, bank = ppo.bank_of(sd, a, arch=a.arch)
            d, x = cardgen.structured_decks(cset, np.random.default_rng(sd), a.b)
            res[f"proc{sd}_arch"] = run(env, bank, d, x, *kinds_proc(cset), a.calls, jax.random.PRNGKey(sd))
        print(f"banque {sd} faite", flush=True)
    bank, names, main_, extra = realcards.load()
    cn = np.array(realcards.card_names())
    nd = len(names)
    cfg = np.array([(i, j) for i in range(nd) for j in range(nd)])[np.arange(a.b) % (nd * nd)]
    D, X = np.stack([main_[cfg[:, 0]], main_[cfg[:, 1]]], 1), np.stack([extra[cfg[:, 0]], extra[cfg[:, 1]]], 1)
    rit = (np.asarray(bank.ctype) & core.TYPE_SPELL != 0) & (np.asarray(bank.ctype) & core.TYPE_RITUAL != 0)
    res["vraies"] = run(env, bank, D, X, cn == "Polymerization", rit & (cn != ""), a.calls, jax.random.PRNGKey(9))
    print(f"\npour 1000 décisions : occasions / actions jouées (jeu aléatoire, B={a.b}, {a.calls} appels)")
    print(f"{'':24s}" + "".join(f"{k:>20s}" for k in res))
    for i, r in enumerate(ROWS):
        print(f"{r:24s}" + "".join(f"{1000 * t[i, 0] / nd_:11.2f} /{1000 * t[i, 1] / nd_:7.2f}"
                                   for t, nd_, _ in res.values()))
    print(f"{'décisions, parties finies':24s}" + "".join(f"{nd_:>13d}, {g:5d}" for _, nd_, g in res.values()))


if __name__ == "__main__":
    main()
