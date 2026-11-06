"""Vérification des cartes traduites (translate/translate.py) contre l'oracle ygopro-core, carte par carte.

1. Traduction de tous les scripts officiels (4 processus), cache translate/cache/translated.pkl (refait si
   translate.py ou cards.py ont changé).
2. Une banque fixe : cartes traduites du deck principal + monstres normaux (une seule compilation).
3. Parties aléatoires : decks de K cartes traduites distinctes (2 exemplaires) + monstres normaux.
   La présence et l'exécution sont comptées séparément à partir des messages de l'oracle.
   Chaque campagne conserve provenance, decks, graines, traces et couverture dans cache/runs/.
   --bisect rejoue les cartes isolément : ce n'est pas un réducteur du scénario divergent.
   Codes de sortie : 0 identique, 1 divergence, 2 erreur / ambiguïté / sources modifiées.

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
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "engine"), os.path.join(HERE, "..", "oracle")]
CACHE = os.path.join(HERE, "cache", "translated.pkl")
RUNS = os.path.join(HERE, "cache", "runs")
EXTRA_TYPES = 0x40 | 0x2000 | 0x800000 | 0x4000000  # FUSION, SYNCHRO, XYZ, LINK
PROC_TYPE = dict(fusion=0x40, synchro=0x2000, xyz=0x800000, link=0x4000000)
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
    from validation_run import fingerprint, package_versions
    files = fingerprint()
    relevant = {k: v for k, v in files.items()
                if k in ("translate/translate.py", "engine/core.py", "engine/cards.py")
                or k.startswith(("data/", "oracle/ygopro-core"))}
    identity = dict(files=relevant, versions=package_versions(), python=list(sys.version_info[:3]))
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()



def translate_all(force=False):
    sig = _sig()
    if not force and os.path.exists(CACHE):
        try:
            with open(CACHE, "rb") as fh:
                d = pickle.load(fh)
            if isinstance(d, dict) and d.get("sig") == sig and "res" in d:
                return d["res"]
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError):
            pass  # Cache interrompu ou incompatible : repartir des scripts épinglés.
    import translate
    codes = translate.all_codes()
    t0 = time.time()
    with multiprocessing.get_context("fork").Pool(4) as pool:
        res = dict(pool.imap_unordered(_one, codes, chunksize=64))
    print(f"traduction : {sum(v[0] == 'ok' for v in res.values())}/{len(res)} en {time.time() - t0:.0f} s", flush=True)
    if _sig() != sig:
        raise RuntimeError("sources ou données modifiées pendant la traduction ; relancer sur une copie figée")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(CACHE), suffix=".tmp", delete=False) as fh:
        temporary = fh.name
        try:
            pickle.dump(dict(sig=sig, res=res), fh)
            fh.close()
            os.replace(temporary, CACHE)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
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
    ap.add_argument("--nosave", action="store_true", help="ne pas enregistrer la campagne")
    a = ap.parse_args()
    if a.games <= 0 or not 1 <= a.k <= 17:
        ap.error("--games doit être > 0 et --k compris entre 1 et 17")
    from validation_run import provenance, save_run, RunJournal
    manifest = provenance(a)
    res = translate_all(a.retranslate)

    import jax
    import numpy as np
    import cards
    import compare
    import core
    import driver
    games = []
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
    recipes = dict(fusion={}, synchro={}, xyz={}, link={})
    for c, v in ok.items():
        if v[3] is not None:
            k, f, *rest = v[3]
            recipes[k][c] = f if k == "fusion" else ((f, rest[0], rest[1], 0) if k == "link" else (f, rest[0]))  # synchro (F, F), xyz (F, ct)
    extra_set = {c for c, v in ok.items() if v[3] is not None}
    pool = db.normal_monsters()
    want = 0x1 | 0x10 | TUNER  # MONSTER | NORMAL | TUNER : Syntoniseurs normaux
    tuners = sorted(c for c, x in db.rows.items() if x[3] == want and x[1] == 0 and 0 < (x[6] & 0xFF) <= 12)
    codes = sorted(set(tr) | set(pool) | set(tuners))
    r = [db.rows[c] for c in codes]
    bank = core.make_bank(np.array(codes), [x[4] for x in r], [x[5] for x in r], [x[6] & 0xFF for x in r],
                          lsc=[(x[6] >> 24) & 0xFF for x in r], rsc=[(x[6] >> 16) & 0xFF for x in r],
                          ctype=[x[3] for x in r], race=[x[7] for x in r], attr=[x[8] for x in r],
                          setcode=[cards.setcodes(x[2]) for x in r], **cards.assemble(codes, cards=tr, ritual={}, **recipes))
    cmap = {c: i for i, c in enumerate(codes)}
    compare.DESC.clear()
    compare.DESC.update({c: v[4] for c, v in ok.items()})
    compare.EFFECT_KINDS.clear()
    compare.EFFECT_KINDS.update({c: [e[0] for e in v[1]] for c, v in ok.items()})
    step, legal = jax.jit(core.step), jax.jit(core.legal)
    play_set = sorted(int(x) for x in a.codes.split(",")) if a.codes else sorted(tr)
    if not play_set:
        ap.error("aucune carte traduite jouable")
    if set(play_set) - set(tr):
        ap.error(f"cartes non traduites ou non jouables : {sorted(set(play_set) - set(tr))}")
    if a.chosen:
        try:
            fixed = [[int(x) for x in part.split(",")] for part in a.chosen.split(";")]
        except ValueError:
            ap.error("--chosen attend a,b;c,d")
        if len(fixed) != 2 or any(not 1 <= len(deck) <= 17 or set(deck) - set(tr) for deck in fixed):
            ap.error("--chosen attend deux listes de 1 à 17 cartes traduites jouables")
    print(f"{len(tr) - len(extra_set)} cartes traduites du deck principal + {len(extra_set)} de l'Extra Deck "
          f"({', '.join(f'{k} {len(v)}' for k, v in recipes.items())}), banque de {len(codes)}", flush=True)

    stats = collections.defaultdict(lambda: dict(present=0, ok=0, bad=0, inconclusive=0, error=0, activated=0, resolved=0, summoned=0))
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
            out = dict(ok=False, status="error", err=f"{type(e).__name__}: {e}")
        out["decks"], out["extras"] = decks, extras
        return out, time.time() - t0

    journal = None if a.nosave else RunJournal(RUNS, manifest)
    if journal:
        manifest["journal"] = str(journal.path)
    def record(game):
        games.append(game)
        if journal:
            journal.append(game)

    nok = 0
    for g in range(a.seed0, a.seed0 + a.games):
        rng = random.Random(10_000 + g)
        chosen = [rng.sample(play_set, min(a.k, len(play_set))) for _ in range(2)]
        if a.chosen:
            chosen = fixed
        out, dt = run(g, chosen)
        nok += out["ok"]
        allc = ((set(chosen[0]) | set(chosen[1])) - extra_set) | (set(out["extras"][0]) & extra_set)
        record(dict(seed=g, chosen=chosen, result=out, seconds=dt))
        summary = {k: v for k, v in out.items() if k not in ("coverage", "trace", "oracle_log", "decks", "extras")}
        print(f"partie {g} ({dt:.0f} s) {summary}", flush=True)
        for c in allc:
            stats[c]["present"] += 1
            stats[c]["ok" if out["ok"] else {"inconclusive": "inconclusive", "error": "error"}.get(out.get("status"), "bad")] += 1
        for ev in out.get("coverage", []):
            if out["ok"] and ev["code"] in allc and ev["stage"] in ("activated", "resolved", "summoned"):
                stats[ev["code"]][ev["stage"]] += ev["count"]
        if out["ok"]:
            continue
        errs[out["err"][:60]] += 1
        for line in sorted(set(compare.LAST_LOG))[:5]:
            print("   oracle :", line[:300])
        if a.bisect and out.get("status") == "failed":
            for c in sorted(allc):
                o2, dt2 = run(g, [[c], [c]])
                record(dict(seed=g, chosen=[[c], [c]], diagnostic="single-card replay", result=o2, seconds=dt2))
                if o2.get("status") == "failed":
                    blame[c] += 1
                    print(f"   rejeu isolé : {c} {db.names.get(c, '')} → {o2['err'][:1500]}", flush=True)
    print(f"{nok}/{a.games} parties identiques")
    for k, v in errs.most_common():
        print(f"  {v} × {k}")
    if blame:
        print("divergences reproduites en isolation :", ", ".join(f"{c} {db.names.get(c, '')}" for c in blame))
    stale = provenance(a)["signature"] != manifest["signature"]
    manifest["sources_changed_during_run"] = stale
    if stale:
        print("CAMPAGNE INVALIDE : sources ou données modifiées pendant son exécution")
    clean = [c for c, s in stats.items() if s["ok"] and not (s["bad"] or s["inconclusive"] or s["error"]) and (s["resolved"] or s["summoned"])]
    print(f"cartes invoquées ou effets résolus dans des parties identiques uniquement : {len(clean)} / {len(stats)} présentes")
    if not a.nosave:
        manifest["stats"] = dict(stats)
        manifest["blame"] = dict(blame)
        path = save_run(RUNS, manifest, games)
        print(f"campagne enregistrée : {path}")
    # Les rejouages diagnostiques ne changent pas le verdict de la campagne originale.
    if stale:
        return 2
    if nok == a.games:
        return 0
    return 2 if any(g["result"].get("status") in ("inconclusive", "error") for g in games if "diagnostic" not in g) else 1



if __name__ == "__main__":
    sys.exit(main())
