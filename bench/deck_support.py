"""Mesure statique des decks (cardgen.structured_decks / random_decks), CPU, quelques secondes :
- filtres qualifiés (attribut / type / archétype / niveau) sur ses propres cartes sans aucune carte du deck qui y réponde ;
- monstres de l'Extra Deck invocables avec le deck principal (_xsup ; Link toujours) ;
- part des monstres du deck dans ses deux niveaux les plus fréquents, part du niveau 4 ;
- part du deck principal dans l'archétype le plus représenté ; nombre de paquets de mécaniques.
Usage : uv run --with jax python bench/deck_support.py [--arch 0.5] [--random]"""
import argparse, collections, sys, os
import numpy as np
sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", "gen"), os.path.join(os.path.dirname(__file__), "..", "engine")]
import cardgen
from cardgen import TYPE_MONSTER

ap = argparse.ArgumentParser()
ap.add_argument("--arch", type=float, default=0.0)
ap.add_argument("--random", action="store_true")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--b", type=int, default=200)
a = ap.parse_args()
for seed in map(int, a.seeds.split(",")):
    cs = cardgen.Gen(seed, arch=a.arch).make_set(400, 80)
    f = cardgen.random_decks if a.random else cardgen.structured_decks
    main, ext = f(cs, np.random.default_rng(seed), a.b)
    tot = dead = 0; byk = collections.Counter(); byd = collections.Counter(); xs = collections.Counter(); xn = collections.Counter()
    top2 = []; l4 = []; arch = []
    for b in range(a.b):
        for p in range(2):
            deck = [cs[i] for i in main[b, p]]
            for c in set(main[b, p]):
                for fl in cardgen._own_filters(cs[c]["effects"], []):
                    k = "setc" if "setc" in fl else "race" if "race" in fl else "attr" if "attr" in fl else "lv"
                    ok = any(cardgen._fmatch(fl, d) for d in deck if d is not cs[c])
                    tot += 1; dead += not ok; byk[k] += 1; byd[k] += not ok
            for k in ext[b, p]:
                kind = cs[k]["recipe"][0]; xn[kind] += 1; xs[kind] += cardgen._xsup(cs[k], deck)
            lv = [c["lv"] for c in deck if c["ctype"] & TYPE_MONSTER]
            cnt = collections.Counter(lv)
            top2.append(sum(v for _, v in cnt.most_common(2)) / max(1, len(lv))); l4.append(cnt[4] / max(1, len(lv)))
            ar = collections.Counter(c.get("arch") or (c["setc"][0] if c["setc"] else None) for c in deck)
            ar.pop(None, None)
            arch.append(max(ar.values(), default=0) / len(deck))
    print(f"graine {seed} : filtres sans cible {dead / max(1, tot):.0%} ({tot}) ;",
          ", ".join(f"{k} {byd[k] / byk[k]:.0%}" for k in ("setc", "race", "attr", "lv") if byk[k]), "|",
          "Extra invocable", ", ".join(f"{k} {xs[k] / xn[k]:.0%}" for k in ("fusion", "synchro", "xyz") if xn[k]), "|",
          f"2 niveaux {np.mean(top2):.0%}, niv 4 {np.mean(l4):.0%}, archétype principal {np.mean(arch):.0%}")
