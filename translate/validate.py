"""Vérification des cartes traduites (translate/translate.py) contre l'oracle ygopro-core, carte par carte.

1. Traduction de tous les scripts officiels (4 processus), cache translate/cache/translated.pkl (refait si
   translate.py ou cards.py ont changé).
2. Une banque fixe : cartes traduites du deck principal + monstres normaux (une seule compilation).
3. Parties aléatoires (boucle de engine/compare.py) : chaque deck = K cartes traduites tirées au hasard (2 exemplaires)
   + des monstres normaux. Une carte est « vérifiée » si elle a été jouée (activée / effet résolu, compté par le
   pilote) dans des parties identiques, et « suspecte » si elle figure dans une partie divergente. Les parties
   divergentes sont rejouées avec une seule carte traduite par deck (--bisect) pour isoler la fautive.

Usage : uv run --with luaparser --with jax python translate/validate.py [--games 20] [--k 8] [--seed0 0] [--bisect]
        [--codes a,b,...] [--retranslate]"""
import argparse
import collections
import hashlib
import json
import multiprocessing
import os
import pickle
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "engine"), os.path.join(HERE, "..", "oracle")]
CACHE = os.path.join(HERE, "cache", "translated.pkl")
RESULTS = os.path.join(HERE, "cache", "validated.json")
EXTRA_TYPES = 0x40 | 0x2000 | 0x800000 | 0x4000000  # FUSION, SYNCHRO, XYZ, LINK
PROC_TYPE = dict(synchro=0x2000, xyz=0x800000, link=0x4000000)
TUNER = 0x1000


def _one(code):
    import translate
    try:
        effs, notes, proc, descs = translate.translate(code)
        return code, ("ok", effs, sorted(notes), proc, descs)
    except translate.Unsupported as e:
        return code, ("no", str(e))
    except Exception as e:  # noqa: BLE001
        return code, ("bug", f"{type(e).__name__}: {e}"[:100])


def _sig():
    h = hashlib.sha1()
    for f in (os.path.join(HERE, "translate.py"), os.path.join(HERE, "..", "engine", "cards.py"),
              os.path.join(HERE, "..", "engine", "core.py")):
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def translate_all(force=False):
    sig = _sig()
    if not force and os.path.exists(CACHE):
        with open(CACHE, "rb") as fh:
            d = pickle.load(fh)
        if d["sig"] == sig:
            return d["res"]
    import translate
    codes = translate.all_codes()
    t0 = time.time()
    with multiprocessing.get_context("fork").Pool(4) as pool:
        res = dict(pool.imap_unordered(_one, codes, chunksize=64))
    print(f"traduction : {sum(v[0] == 'ok' for v in res.values())}/{len(res)} en {time.time() - t0:.0f} s", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(dict(sig=sig, res=res), fh)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--k", type=int, default=8, help="cartes traduites distinctes par deck")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--codes", default="", help="limiter les cartes traduites jouées à celles-ci")
    ap.add_argument("--bisect", action="store_true", help="rejouer chaque partie divergente carte par carte")
    ap.add_argument("--retranslate", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--chosen", default="", help="cartes imposées des deux joueurs : a,b,...;c,d,... (rejouer une partie)")
    ap.add_argument("--nosave", action="store_true", help="ne pas mettre à jour validated.json")
    a = ap.parse_args()
    res = translate_all(a.retranslate)

    import jax
    import numpy as np
    import cards
    import compare
    import core
    import driver
    db = driver.CardDB()
    def usable(c, v):
        if v[0] != "ok" or c not in db.rows:
            return False
        t, proc = db.rows[c][3], v[3]
        if proc is None:
            return not t & EXTRA_TYPES
        # monstre de l'Extra Deck à procédure : type cohérent, rang Xyz = niveau de la procédure
        return t & EXTRA_TYPES == PROC_TYPE[proc[0]] and (proc[0] != "xyz" or proc[3] == db.rows[c][6] & 0xFF)
    ok = {c: v for c, v in res.items() if usable(c, v)}
    tr = {c: v[1] for c, v in ok.items()}
    recipes = dict(synchro={}, xyz={}, link={})
    for c, v in ok.items():
        if v[3] is not None:
            k, f, *rest = v[3]
            recipes[k][c] = (f, rest[0], rest[1], 0) if k == "link" else (f, rest[0])  # synchro (F, F), xyz (F, ct)
    extra_set = {c for c, v in ok.items() if v[3] is not None}
    pool = db.normal_monsters()
    want = 0x1 | 0x10 | TUNER  # MONSTER | NORMAL | TUNER : Syntoniseurs normaux
    tuners = sorted(c for c, x in db.rows.items() if x[3] == want and x[1] == 0 and 0 < (x[6] & 0xFF) <= 12)
    codes = sorted(set(tr) | set(pool) | set(tuners))
    r = [db.rows[c] for c in codes]
    bank = core.make_bank(np.array(codes), [x[4] for x in r], [x[5] for x in r], [x[6] & 0xFF for x in r],
                          lsc=[(x[6] >> 24) & 0xFF for x in r], rsc=[(x[6] >> 16) & 0xFF for x in r],
                          ctype=[x[3] for x in r], race=[x[7] for x in r], attr=[x[8] for x in r],
                          setcode=[cards.setcodes(x[2]) for x in r], **cards.assemble(codes, cards=tr, fusion={}, ritual={}, **recipes))
    cmap = {c: i for i, c in enumerate(codes)}
    compare.DESC.update({c: v[4] for c, v in ok.items()})
    step, legal = jax.jit(core.step), jax.jit(core.legal)
    play_set = sorted(int(x) for x in a.codes.split(",")) if a.codes else sorted(tr)
    print(f"{len(tr) - len(extra_set)} cartes traduites du deck principal + {len(extra_set)} de l'Extra Deck "
          f"({', '.join(f'{k} {len(v)}' for k, v in recipes.items())}), banque de {len(codes)}", flush=True)

    stats = collections.defaultdict(lambda: dict(ok=0, bad=0, used_ok=0))
    if os.path.exists(RESULTS) and not a.codes and not a.nosave:
        with open(RESULTS) as fh:
            old = json.load(fh)
        for k, v in old.get("stats", {}).items():
            stats[int(k)].update(v)
    blame = collections.Counter()
    errs = collections.Counter()

    def run(seed, chosen):
        rng = random.Random(seed)
        decks = []
        for p in range(2):
            d = [c for c in chosen[p] if c not in extra_set for _ in range(2)]
            if any(c in recipes["synchro"] for c in chosen[0] + chosen[1]):  # des Syntoniseurs pour les monstres synchro
                d += [rng.choice(tuners) for _ in range(6)]
            d += [rng.choice(pool) for _ in range(40 - len(d))]
            rng.shuffle(d)
            decks.append(d)
        # Extra Deck : même taille pour les deux joueurs (new_game) : chacun reçoit tous les monstres tirés (15 au plus)
        ex = sorted({c for p in range(2) for c in chosen[p] if c in extra_set})[:15]
        extras = [list(ex), list(ex)]
        t0 = time.time()
        try:
            out = compare.play(db, bank, cmap, seed, step, legal, verbose=a.verbose, mode="tr", given=(decks, extras))
        except Exception as e:  # noqa: BLE001
            out = dict(ok=False, err=f"{type(e).__name__}: {e}")
        return out, time.time() - t0

    nok = 0
    for g in range(a.seed0, a.seed0 + a.games):
        rng = random.Random(10_000 + g)
        chosen = [rng.sample(play_set, min(a.k, len(play_set))) for _ in range(2)]
        if a.chosen:
            chosen = [[int(x) for x in part.split(",")] for part in a.chosen.split(";")]
        out, dt = run(g, chosen)
        nok += out["ok"]
        allc = set(chosen[0]) | set(chosen[1])
        print(f"partie {g} ({dt:.0f} s) {out}", flush=True)
        for c in allc:
            stats[c]["ok" if out["ok"] else "bad"] += 1
        if out["ok"]:
            continue
        errs[out["err"][:60]] += 1
        for line in sorted(set(compare.LAST_LOG))[:5]:
            print("   oracle :", line[:300])
        if a.bisect:
            for c in sorted(allc):
                o2, _ = run(g, [[c], [c]])
                if not o2["ok"]:
                    blame[c] += 1
                    print(f"   seule : {c} {db.names.get(c, '')} → {o2['err'][:1500]}", flush=True)
    print(f"{nok}/{a.games} parties identiques")
    for k, v in errs.most_common():
        print(f"  {v} × {k}")
    if blame:
        print("fautives (bissection) :", ", ".join(f"{c} {db.names.get(c, '')}" for c in blame))
    clean = [c for c, s in stats.items() if s["ok"] and not s["bad"]]
    print(f"cartes vues seulement dans des parties identiques : {len(clean)} / {len(stats)} jouées")
    if not a.codes and not a.nosave:
        with open(RESULTS, "w") as fh:
            json.dump(dict(stats={str(k): v for k, v in stats.items()}, blame={str(k): v for k, v in blame.items()}),
                      fh, indent=0)


if __name__ == "__main__":
    main()
