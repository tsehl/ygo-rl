"""Comparaison moteur JAX ↔ oracle ygopro-core sur des cartes procédurales (jalon 6, étape 2 ; ETAPE3.md).

Pour chaque graine de banque : gen/cardgen.py tire les cartes, gen/tolua.py écrit leurs scripts Lua (bibliothèque
gen/procgen.lua) et leurs lignes de base de cartes, puis on joue des parties aléatoires avec la boucle de
engine/compare.py (mêmes vérifications : état, actions légales, fin de partie), sur des decks tirés au hasard dans
la banque (40 cartes + Extra Deck de 15).
Usage : uv run --with jax python gen/compare_gen.py --banks 0-3 --games 10 [--seed0 0] [--main 400 --extra 80] [--focus]
"""
import argparse
import collections
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
sys.path.insert(0, os.path.join(HERE, "..", "oracle"))

import jax  # noqa: E402

import cardgen  # noqa: E402
import compare  # noqa: E402
import core  # noqa: E402
import driver  # noqa: E402
import tolua  # noqa: E402


def rng_range(s):
    a, _, b = s.partition("-")
    return range(int(a), int(b or a) + 1)


def focus(c):
    """Carte qui utilise une brique ajoutée au générateur le 2026-11-05 (variantes de MODIFY, MINATK, LOCK)."""
    return any(e["kind"] == "hand_proc" for e in c["effects"]) or any(cl["verb"] in ("def", "halve_base", "halve") or cl.get("sel") == "min" or cl.get("lock") or cl.get("rd")
               for e in c["effects"] for cl in e.get("clauses", []))


def decks(cset, rng, d=40, x=15, foc=False):
    """Decks tirés au hasard ; foc : la moitié du deck principal parmi les cartes de focus()."""
    main = [c["code"] for c in cset if not cardgen.is_extra(c)]
    ext = [c["code"] for c in cset if cardgen.is_extra(c)]
    fm = [c["code"] for c in cset if not cardgen.is_extra(c) and focus(c)] if foc else []
    nf = d // 2 if fm else 0
    return ([[rng.choice(fm) for _ in range(nf)] + [rng.choice(main) for _ in range(d - nf)] for _ in range(2)],
            [[rng.choice(ext) for _ in range(x)] for _ in range(2)])


def dump(cset, s):
    """Cartes en jeu (main, terrain, cimetière, bannies, Extra Deck) au dernier état comparé, décrites."""
    import numpy as np
    zn = {v: k for k, v in compare.ZNAME.items() if k != "deck"}
    st = {k: np.asarray(v) for k, v in s._asdict().items() if hasattr(v, "shape")}
    print(f"   LP {[int(x) for x in s.lp]} tour {int(s.turn)} joueur du tour {int(s.tp)} phase {int(s.phase):#x} décide {int(s.dec_player)}")
    seen = set()
    for i in np.argsort(st["zone"] * 1000 + st["ctrl"] * 100 + st["seq"]):
        z = int(st["zone"][i])
        if z not in zn:
            continue
        c = cset[int(st["cid"][i])]
        print(f"   j{int(st['ctrl'][i])} {zn[z]:7} {int(st['seq'][i]):2} pos {int(st['pos'][i]):2} st {int(st['st'][i]):#x} "
              f"atk {int(st['atk_c'][i])} : {c['code']}")
        seen.add(int(st["cid"][i]))
    for i in sorted(seen):
        print("     " + cardgen.describe(cset[i]).replace("\n", "\n     ")[:1500])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", default="0")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--main", type=int, default=400)
    ap.add_argument("--extra", type=int, default=80)
    ap.add_argument("--out", default="/tmp/ygo-gen")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dump", action="store_true", help="décrit les cartes en jeu à chaque écart")
    ap.add_argument("--focus", action="store_true", help="la moitié des decks parmi les cartes à briques récentes")
    a = ap.parse_args()
    if a.games <= 0:
        ap.error("--games doit être > 0")
    step, legal = jax.jit(core.step), jax.jit(core.legal)
    db = driver.CardDB()
    tot, nok, errs = 0, 0, collections.Counter()
    for b in rng_range(a.banks):
        cset = cardgen.Gen(b).make_set(a.main, a.extra)
        bank = cardgen.to_bank(cset)
        cmap = {c["code"]: i for i, c in enumerate(cset)}
        compare.EFFECT_KINDS.clear()
        compare.EFFECT_KINDS.update({c["code"]:[cardgen.bricks(e)[0] for e in c["effects"]] for c in cset})

        db.add(tolua.cdb_rows(cset))
        sdir = tolua.write(cset, os.path.join(a.out, str(b)))
        driver.SCRIPTS[:] = [sdir] + [d for d in driver.SCRIPTS if not d.startswith(a.out)]
        for g in range(a.seed0, a.seed0 + a.games):
            rng = random.Random(1000 * b + g)
            t0 = time.time()
            try:
                r = compare.play(db, bank, cmap, g, step, legal, verbose=a.verbose, mode="gen", given=decks(cset, rng, foc=a.focus))
            except Exception as e:  # plantage du pilote ou de l'oracle (script Lua, message non géré…)
                r = dict(ok=False, err=f"{type(e).__name__}: {e}")
            tot += 1
            nok += r["ok"]
            if not r["ok"]:
                errs[r["err"][:60]] += 1
            print(f"banque {b} partie {g} ({time.time() - t0:.0f} s)", r, flush=True)
            for line in sorted(set(compare.LAST_LOG))[:5]:
                print("   oracle :", line[:300])
            if a.dump and not r["ok"] and compare.LAST_STATE is not None:
                dump(cset, compare.LAST_STATE)
    print(f"{nok}/{tot} parties identiques")
    for k, v in errs.most_common():
        print(f"  {v} × {k}")
    for k, v in sorted(compare.COVER.items(), key=str):
        if "relance" in str(k):
            print(f"  {k} : {v}")

    return 0 if tot > 0 and nok == tot else 1


if __name__ == "__main__":
    sys.exit(main())
