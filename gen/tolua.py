"""Compilateur spécification de carte (gen/cardgen.py) → script Lua EDOPro (jalon 6, étape 2 ; ETAPE3.md).

Chaque carte devient c<code>.lua : la spécification en table Lua, interprétée par la bibliothèque commune
gen/procgen.lua (chargée une fois par duel par Duel.LoadScript). `cdb_rows` donne les lignes de base de cartes que le
pilote de l'oracle (oracle/driver.py, CardDB.add) ajoute à cards.cdb.
Usage : python gen/tolua.py --seed 0 --out /tmp/ygo-gen/0   (écrit les scripts d'une banque)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cardgen  # noqa: E402
from cardgen import TYPE_LINK, CODE0  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FILTER_KEYS = ("f", "tgt", "per")


def norm_f(f):
    """Filtre (dict de cards.F) : clés à 0 retirées (0 = sans contrainte), code en paire."""
    out = {}
    for k, v in f.items():
        if not isinstance(v, (tuple, list)) and v == 0:
            continue
        if k == "code" and isinstance(v, int):
            v = (v, v)
        out[k] = v
    return out


def norm_effect(e, lv_of):
    e = dict(e)
    if "clauses" in e:
        e["clauses"] = [dict(c, f=norm_f(c["f"])) if "f" in c else c for c in e["clauses"]]
    if e["kind"] == "cont":
        c = dict(e["cont"])
        for k in ("tgt", "per"):
            if c.get(k) is not None:
                c[k] = norm_f(c[k])
        if c.get("cnt") is not None:
            c["cnt"] = (norm_f(c["cnt"][0]), c["cnt"][1], c["cnt"][2])
        e["cont"] = c
    if e["kind"] == "fusion_spell":
        e["f"] = norm_f(e["f"])
    if e["kind"] == "ritual":
        e["lv"] = lv_of[e["code"]]
    return e


def norm_recipe(r):
    if r is None:
        return None
    kind, x = r
    if kind == "fusion":
        return (kind, [norm_f(m) for m in x])
    if kind in ("synchro",):
        return (kind, (norm_f(x[0]), norm_f(x[1])))
    if kind == "xyz":
        return (kind, (norm_f(x[0]), x[1]))
    if kind == "link":
        return (kind, (norm_f(x[0]), x[1], x[2]))
    return (kind,)


def lua(v, ind=""):
    """Valeur Python → littéral Lua (dict → table à clés, liste / tuple → tableau, None omis dans les dicts)."""
    if v is None:
        return "nil"
    if hasattr(v, "item") and not isinstance(v, (list, tuple, dict)):  # scalaire numpy
        v = v.item()
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(int(v))
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, dict):
        items = [f"{k}={lua(x, ind)}" for k, x in v.items() if x is not None]
        return "{" + ",".join(items) + "}"
    if isinstance(v, (list, tuple)):
        return "{" + ",".join(lua(x, ind) for x in v) + "}"
    raise TypeError(type(v))


def spec_table(c, lv_of):
    effects = [norm_effect(e, lv_of) for e in c["effects"]]
    return dict(lv=c["lv"], recipe=norm_recipe(c["recipe"]), effects=effects)


def script(c, lv_of):
    head = "\n".join("--" + line for line in cardgen.describe(c).splitlines())
    return (f"{head}\nlocal s,id=GetID()\nDuel.LoadScript(\"procgen.lua\")\ns.spec={lua(spec_table(c, lv_of))}\n"
            f"function s.initial_effect(c)\n\tPG.init(c,s.spec,id)\nend\n")


def write(cset, out):
    """Écrit les scripts de la banque cset dans le dossier out (plus procgen.lua) ; rend out."""
    os.makedirs(out, exist_ok=True)
    lv_of = {c["code"]: c["lv"] for c in cset}
    for c in cset:
        with open(os.path.join(out, f"c{c['code']}.lua"), "w") as fh:
            fh.write(script(c, lv_of))
    with open(os.path.join(HERE, "procgen.lua")) as src, open(os.path.join(out, "procgen.lua"), "w") as dst:
        dst.write(src.read())
    return out


def cdb_rows(cset):
    """{code: ((alias, setcode, type, atk, def ou flèches, level, race, attribute), nom)} pour CardDB.add."""
    rows = {}
    for c in cset:
        assert c["code"] >= CODE0
        setcode = 0
        for i, sc in enumerate(c["setc"][:4]):
            setcode |= sc << (16 * i)
        level = c["lv"] | (c["lsc"] << 24) | (c["rsc"] << 16)
        rows[c["code"]] = ((0, setcode, c["ctype"], c["atk"], c["df"], level, c["race"], c["attr"]),
                           f"Gen{'Link' if c['ctype'] & TYPE_LINK else ''} {c['code'] - CODE0}")
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Écrit les scripts Lua d'une banque procédurale")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--main", type=int, default=400)
    ap.add_argument("--extra", type=int, default=80)
    ap.add_argument("--out", default="/tmp/ygo-gen/0")
    a = ap.parse_args()
    cs = cardgen.Gen(a.seed).make_set(a.main, a.extra)
    print(write(cs, a.out), len(cs), "cartes")
