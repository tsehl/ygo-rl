"""Mesure statique des decks structurés (cardgen.structured_decks) : part des filtres qualifiés (attribut / type /
archétype / niveau) sur ses propres cartes sans aucune carte du deck qui y réponde. CPU, quelques secondes.
Usage : uv run --with jax python bench/deck_support.py"""
import sys, numpy as np
sys.path[:0] = ["gen", "engine"]
import cardgen
from cardgen import TYPE_MONSTER
def walk(x, out):
    if isinstance(x, dict):
        if ("me" in x or "op" in x) and any(k in x for k in ("attr", "race", "setc", "lv")):
            out.append(x)
        for v in x.values(): walk(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x: walk(v, out)
def match(f, c):
    if not c["ctype"] & TYPE_MONSTER: return False
    if "attr" in f and not (c["attr"] & f["attr"]): return False
    if "race" in f and not (c["race"] & f["race"]): return False
    if "setc" in f and f["setc"] not in c["setc"]: return False
    if "lv" in f and not (f["lv"][0] <= c["lv"] <= f["lv"][1]): return False
    return True
for seed in (0, 1, 2):
    g = cardgen.Gen(seed); cs = g.make_set(400, 80)
    main, ext = cardgen.structured_decks(cs, np.random.default_rng(seed), 200, 40, 15)
    tot = dead = 0; byk = {}
    for b in range(200):
        for p in range(2):
            deck = [cs[i] for i in main[b, p]] + [cs[i] for i in ext[b, p]]
            for c in set(main[b, p]):
                fs = []; walk(cs[c]["effects"], fs)
                for f in fs:
                    if f.get("me", 0) and not f.get("op", 0):   # filtre sur ses propres cartes
                        k = "+".join(sorted(k for k in ("attr", "race", "setc", "lv") if k in f))
                        ok = any(match(f, d) for d in deck if d is not cs[c])
                        tot += 1; dead += not ok
                        a = byk.setdefault(k, [0, 0]); a[0] += 1; a[1] += not ok
    lvs = np.array([c["lv"] for c in cs if c["ctype"] & TYPE_MONSTER and not cardgen.is_extra(c)])
    print(f"seed {seed}: filtres propres qualifiés {tot}, sans cible dans le deck {dead/tot:.0%} ;",
          {k: f"{v[1]/v[0]:.0%} ({v[0]})" for k, v in sorted(byk.items(), key=lambda t: -t[1][0])[:6]},
          f"niv 4 : {(lvs==4).mean():.0%}, niv<=4 : {(lvs<=4).mean():.0%}")
