"""Test de robustesse du moteur sur des banques procédurales (jalon 6, étape 1).

Joue B parties aléatoires (actions légales tirées uniformément) par banque, pour plusieurs graines de banque, et
vérifie après chaque décision des invariants de l'état. Une seule compilation pour toutes les graines : la banque
est un argument de la fonction compilée (cardgen.to_bank donne les mêmes formes à toutes les banques).

Usage : python fuzz.py [--seeds 0-3] [--b 64] [--t 600] [--main 400] [--extra 80] [--dump]
--dump : pour chaque violation, rejoue la partie fautive seule et affiche ses cartes et la décision fautive.

Violations (bits de `check`) : erreurs du moteur. Avertissements (bits de `warn`) : limites des tables du moteur
atteintes (comportement alors approché, pas une erreur de code) ; à surveiller pour dimensionner KM, TM et les candidats.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cardgen  # noqa: E402  (ajoute engine/ au chemin)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import core  # noqa: E402
from core import (Z_DECK, Z_HAND, Z_MZONE, Z_SZONE, Z_GRAVE, Z_REMOVED, Z_EXTRA, Z_OVERLAY, TYPE_EXTRA, TYPE_XYZ,  # noqa: E402
                  D_RUN, D_TRIG, D_NONE, CH, TM, KC)

VIOL = ["pas d'action légale", "case de terrain occupée deux fois", "case de terrain hors limites",
        "zone ordonnée non contiguë", "matériau Xyz incohérent", "monstre de l'Extra Deck en main ou dans le deck",
        "LP <= 0 sans fin de partie", "chaîne hors limites", "position invalide", "décision interne non résolue",
        "contrôleur / propriétaire invalide", "zone inconnue"]
WARN = ["plus de KM effets continus actifs", "plus de TM modificateurs temporaires", "plus de EXK monstres fusion",
        "table « une fois par tour » pleine", "table des noms interdits pleine",
        "arène des candidats à l'activation dépassée"]
ZONES = (Z_DECK, Z_HAND, Z_GRAVE, Z_REMOVED, Z_EXTRA, Z_MZONE, Z_SZONE)


def check(bank, s, padding_mask=None):
    """(violations, avertissements) : masques de bits pour l'état s (après une décision)."""
    n = s.cid.shape[0]
    zi = jnp.full(n, -1)
    for i, z in enumerate(ZONES):
        zi = jnp.where(s.zone == z, i, zi)
    known = (zi >= 0) | (s.zone == Z_OVERLAY)
    padding_changed = jnp.bool_(False)
    if padding_mask is not None:
        known |= padding_mask & (s.zone == 0)
        padding_changed = (padding_mask & (s.zone != 0)).any()
    fieldz = (s.zone == Z_MZONE) | (s.zone == Z_SZONE)
    grp = s.ctrl * len(ZONES) + jnp.maximum(zi, 0)
    inz = zi >= 0
    bins = jnp.zeros((2 * len(ZONES), n + 8), jnp.int32).at[grp, jnp.clip(s.seq, 0, n + 7)].add(inz.astype(jnp.int32))
    dup = (bins > 1).any()
    cnt = bins.sum(1)
    ordered = inz & ~fieldz
    bad_ord = (ordered & ((s.seq < 0) | (s.seq >= cnt[grp]))).any()
    seq_bad = ((s.zone == Z_MZONE) & ((s.seq < 0) | (s.seq >= 7))).any() | ((s.zone == Z_SZONE) & ((s.seq < 0) | (s.seq >= 8))).any()
    ct = bank.ctype[s.cid]
    par = jnp.maximum(s.ov_par, 0)
    ov = s.zone == Z_OVERLAY
    # pendant l'invocation Xyz, les matériaux sont attachés avant le choix de la case et de la position (Xyz encore
    # dans l'Extra Deck)
    placing = ((s.dec == core.D_ZONE) | (s.dec == core.D_POS)) & (s.ov_par == s.pend)
    par_ok = (s.zone[par] == Z_MZONE) | (placing & (s.zone[par] == Z_EXTRA))
    ov_bad = (ov & ((s.ov_par < 0) | ~par_ok | ((ct[par] & TYPE_XYZ) == 0) | (s.ctrl != s.ctrl[par]))).any() \
        | (~ov & (s.ov_par >= 0)).any()
    ex_bad = (((ct & TYPE_EXTRA) != 0) & ((s.zone == Z_HAND) | (s.zone == Z_DECK))).any()
    live = ~s.done
    lp_bad = live & (s.lp <= 0).any()
    ch_bad = (s.n_chain < 0) | (s.n_chain > CH)
    pos_bad = ((s.zone == Z_MZONE) & ~jnp.isin(s.pos, jnp.array([1, 2, 4, 8]))).any() \
        | ((s.zone == Z_SZONE) & ~jnp.isin(s.pos, jnp.array([5, 10]))).any()
    stuck = live & ((s.dec == D_RUN) | (s.dec == D_TRIG) | (s.dec == D_NONE))
    who_bad = ((s.ctrl < 0) | (s.ctrl > 1) | (s.owner < 0) | (s.owner > 1)).any()
    nolegal = live & ~core.legal(bank, s).any()
    v = [nolegal, dup, seq_bad, bad_ord, ov_bad, ex_bad, lp_bad, ch_bad, pos_bad, stuck, who_bad, ~known.all() | padding_changed]
    ncand = jnp.where(live, core.act_cand(bank, s).sum(), 0)
    w = [(s.ovf & 1) != 0, (s.ovf & 2) != 0, (s.ovf & 4) != 0, (s.opt_cid != 0).all(-1).any(),
         (s.lock_cid != 0).all(-1).any(), ncand > core.ACT_ROUNDS * KC]
    bits = lambda xs: sum(jnp.where(x, 1 << i, 0) for i, x in enumerate(xs)).astype(jnp.int32)
    return bits(v), bits(w)


def rollout(bank, d, x, k, t):
    """Une partie aléatoire de t décisions au plus. Rend l'état final, les violations / avertissements cumulés, la
    première décision fautive, le plus grand nombre de décisions dans un même tour, le nombre d'activations d'effets
    choisies par la politique (parmi plusieurs actions légales)."""
    k0, k1 = jax.random.split(k)
    s = core.new_game(bank, d[0], d[1], k0, x[0], x[1])
    v0, w0 = check(bank, s)

    def body(c, kt):
        s, v, w, first, i, last_turn, run, mx, nact = c
        m = core.legal(bank, s)
        a = jax.random.categorical(kt, jnp.where(m, 0.0, -1e9))
        n = s.cid.shape[0]
        nact = nact + ((a // n >= core.V_ACT0) & (a < core.NV * n) & ~s.done & (m.sum() > 1)).astype(jnp.int32)
        s2 = core.step(bank, s, a)
        s2 = jax.tree.map(lambda p, q: jnp.where(s.done, p, q), s, s2)
        vv, ww = check(bank, s2)
        vv, ww = jnp.where(s.done, 0, vv), jnp.where(s.done, 0, ww)
        first = jnp.where((first < 0) & (vv != 0), i, first)
        run = jnp.where(s2.turn == last_turn, run + (~s.done).astype(jnp.int32), 0)
        return (s2, v | vv, w | ww, first, i + 1, s2.turn, run, jnp.maximum(mx, run), nact), None
    c0 = (s, v0, w0, jnp.where(v0 != 0, 0, -1), jnp.int32(1), s.turn, jnp.int32(0), jnp.int32(0), jnp.int32(0))
    (s, v, w, first, _, _, _, mx, nact), _ = jax.lax.scan(body, c0, jax.random.split(k1, t))
    return s, v, w, first, mx, nact


def bits_str(x, names):
    return ", ".join(nm for i, nm in enumerate(names) if (x >> i) & 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-3")
    ap.add_argument("--b", type=int, default=64)
    ap.add_argument("--t", type=int, default=600)
    ap.add_argument("--main", type=int, default=400)
    ap.add_argument("--extra", type=int, default=80)
    ap.add_argument("--dump", action="store_true")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    seeds = range(int(lo), int(hi or lo) + 1)
    run = jax.jit(jax.vmap(lambda bank, d, x, k: rollout(bank, d, x, k, a.t), in_axes=(None, 0, 0, 0)))
    total_v = 0
    for seed in seeds:
        t0 = time.time()
        cset = cardgen.Gen(seed).make_set(a.main, a.extra)
        bank = cardgen.to_bank(cset)
        rng = np.random.default_rng(1000 + seed)
        d, x = cardgen.random_decks(cset, rng, a.b)
        keys = jax.random.split(jax.random.PRNGKey(seed), a.b)
        s, v, w, first, mx, nact = jax.device_get(run(bank, jnp.asarray(d), jnp.asarray(x), keys))
        dt = time.time() - t0
        nv = int((v != 0).sum())
        total_v += nv
        wsum = [int(((w >> i) & 1).sum()) for i in range(len(WARN))]
        print(f"graine {seed} : {int(s.done.sum())}/{a.b} parties finies, tour moyen {s.turn.mean():.1f}, "
              f"max {int(mx.max())} décisions dans un tour, {nact.mean():.1f} activations d'effets par partie, "
              f"{nv} parties avec violation ({dt:.0f} s)", flush=True)
        if any(wsum):
            print("  avertissements : " + "; ".join(f"{WARN[i]} ×{c}" for i, c in enumerate(wsum) if c))
        for g in np.nonzero(v)[0][:5]:
            print(f"  partie {g} : {bits_str(int(v[g]), VIOL)} (décision {int(first[g])})")
            if a.dump:
                dump(bank, cset, d[g], x[g], keys[g], int(first[g]))
    print(f"total : {total_v} parties avec violation")
    return total_v


def dump(bank, cset, d, x, k, upto):
    """Rejoue la partie jusqu'à la décision fautive et affiche l'état et les cartes en jeu."""
    k0, k1 = jax.random.split(k)
    s = core.new_game(bank, jnp.asarray(d[0]), jnp.asarray(d[1]), k0, jnp.asarray(x[0]), jnp.asarray(x[1]))
    step = jax.jit(lambda s, a: core.step(bank, s, a))
    legal = jax.jit(lambda s: core.legal(bank, s))
    ks = jax.random.split(k1, max(upto, 1))
    for i in range(upto):
        m = legal(s)
        a = jax.random.categorical(ks[i], jnp.where(m, 0.0, -1e9))
        prev, s = s, step(s, a)
        if i == upto - 1:
            print(f"    dernière action {int(a)} ; avant : dec {int(prev.dec)} joueur {int(prev.dec_player)} "
                  f"src {int(prev.ch_src)} ; après : dec {int(s.dec)} phase {int(s.phase)} tour {int(s.turn)}")
            print(f"    violations {bits_str(int(check(bank, s)[0]), VIOL)}")
            z, par, cid = np.asarray(s.zone), np.asarray(s.ov_par), np.asarray(s.cid)
            for c in np.nonzero((z == Z_OVERLAY) | (par >= 0))[0]:
                q = int(par[c])
                print(f"    carte {c} ({cset[int(cid[c])]['code']}) zone {int(z[c])} ctrl {int(s.ctrl[c])} ov_par {q} : "
                      f"zone du Xyz {int(z[q])}, ctrl {int(s.ctrl[q])}, code {cset[int(cid[q])]['code']}, "
                      f"type {cset[int(cid[q])]['ctype']:#x}, prev {int(s.pzone[q])}")
    used = sorted(set(np.asarray(s.cid)[(np.asarray(s.zone) & (Z_MZONE | Z_SZONE | Z_GRAVE | Z_HAND | Z_REMOVED | Z_OVERLAY)) != 0]))
    for c in used:
        print("    " + cardgen.describe(cset[int(c)]).replace("\n", "\n    "))


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
