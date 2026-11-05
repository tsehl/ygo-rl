"""Extraction des scripts EDOPro (data/CardScripts/official) : pour chaque carte, les appels d'API et les constantes
utilisés, rangés par fonction englobante (initial_effect, condition, coût, cible, opération, filtres…).

Sortie : translate/cache/extract.json.gz, {code: {"fn": {nom de fonction: [items]}, "proc": [...]}} où un item est
"Duel.Destroy", "c:RegisterEffect" (méthode), "EFFECT_UPDATE_ATTACK" (constante), "aux.Stringid"…
Une carte dont le script ne se parse pas a {"err": message}.

Usage : uv run --with luaparser python translate/extract.py [--jobs 4] [--only 26202165,...]"""
import argparse
import glob
import gzip
import json
import multiprocessing
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "data", "CardScripts", "official")
OUT = os.path.join(HERE, "cache", "extract.json.gz")
CONST = re.compile(r"^(EFFECT|EVENT|CATEGORY|LOCATION|REASON|RESET|RESETS|PHASE|TYPE|EFFECT_FLAG|EFFECT_TYPE|POS|"
                   r"ATTRIBUTE|RACE|SUMMON_TYPE|HINTMSG|CHAININFO|SET|COUNTER|STATUS|PLAYER|ACTIVITY|EFFECT_COUNT)_")


def dotted(node):
    """Nom pointé d'une expression (Duel.Destroy, e:SetCode → ':SetCode'), ou None."""
    from luaparser import astnodes as A
    if isinstance(node, A.Name):
        return node.id
    if isinstance(node, A.Index) and isinstance(node.idx, A.Name) and node.notation.name == "DOT":
        base = dotted(node.value)
        return f"{base}.{node.idx.id}" if base else None
    return None


def fn_name(node):
    from luaparser import astnodes as A
    if isinstance(node, (A.Function, A.LocalFunction)):
        return dotted(node.name) or "?"
    if isinstance(node, A.Method):
        return f"{dotted(node.source)}:{node.name.id}"
    return None


def walk(node, ctx, out):
    """Parcours de l'arbre : appels et constantes, par fonction nommée englobante (les fonctions anonymes héritent du
    contexte)."""
    from luaparser import astnodes as A
    if isinstance(node, list):
        for x in node:
            walk(x, ctx, out)
        return
    if not isinstance(node, A.Node):
        return
    nm = fn_name(node)
    if nm is not None:
        ctx = nm.split(".")[-1].split(":")[-1]
    if isinstance(node, A.Call):
        d = dotted(node.func)
        if d:
            out.setdefault(ctx, set()).add(d)
    elif isinstance(node, A.Invoke):
        out.setdefault(ctx, set()).add(":" + node.func.id)
    elif isinstance(node, A.Name) and CONST.match(node.id):
        out.setdefault(ctx, set()).add(node.id)
    for k, v in node.__dict__.items():
        if k in ("comments", "parent") or k.startswith("_"):
            continue
        if isinstance(v, (A.Node, list)):
            walk(v, ctx, out)


def one(path):
    from luaparser import ast
    code = os.path.basename(path)[1:-4]
    src = open(path, encoding="utf-8", errors="replace").read()
    try:
        tree = ast.parse(src)
    except Exception as e:  # noqa: BLE001
        return code, {"err": f"{type(e).__name__}: {str(e)[:200]}"}
    out = {}
    walk(tree, "<top>", out)
    return code, {"fn": {k: sorted(v) for k, v in out.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(SCRIPTS, "c*.lua")))
    if a.only:
        keep = set(a.only.split(","))
        paths = [p for p in paths if os.path.basename(p)[1:-4] in keep]
    with multiprocessing.Pool(a.jobs) as pool:
        res = dict(pool.imap_unordered(one, paths, chunksize=32))
    if a.only:
        print(json.dumps(res, indent=1, ensure_ascii=False))
        return
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with gzip.open(OUT, "wt") as fh:
        json.dump(res, fh)
    err = sum("err" in v for v in res.values())
    print(f"{len(res)} scripts, {err} non parsés → {OUT}")


if __name__ == "__main__":
    main()
