"""Compteurs déterministes du coût du moteur (aucune mesure de temps : identiques d'une passe à l'autre, sur CPU).

À relever après chaque ajout au moteur, à la place d'un débit CPU trop bruité :
- octets d'état par partie (détail des plus gros champs) : sous vmap, chaque `switch` / `cond` / `_where` recopie l'état ;
- par brique (exec_one, refresh, legal, apply, trig_eval, pass_priority, auto_pass, refresh_ex, forced_m, observe),
  tracée pour UNE partie, banque en argument : nombre d'opérations StableHLO et octets écrits (somme des tailles des
  résultats de toutes les opérations, toutes branches comprises : c'est ce que vmap paie par partie et par appel ;
  le corps d'une boucle est compté une fois).

Usage : python bench/cost_stats.py [--proc 0] [--extra 15] [--top 12] [--json sortie.json] [--ref ancien.json]"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--engine", default=os.path.join(HERE, "..", "engine"))
ap.add_argument("--proc", type=int, default=0)
ap.add_argument("--extra", type=int, default=15)
ap.add_argument("--top", type=int, default=12)
ap.add_argument("--json", default="")
ap.add_argument("--ref", default="")
args = ap.parse_args()
sys.path[:0] = [args.engine, os.path.join(HERE, "..", "gen")]

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import cardgen  # noqa: E402
import core  # noqa: E402
import obs as obsm  # noqa: E402

cset = cardgen.Gen(args.proc).make_set(400, 80)
bank = cardgen.to_bank(cset)
d, x = cardgen.random_decks(cset, np.random.default_rng(0), 1, 40, args.extra)
ng = lambda bank, d, x, k: core.new_game(bank, d[0], d[1], k, x[0], x[1])
S = jax.eval_shape(ng, bank, jnp.asarray(d[0]), jnp.asarray(x[0]), jax.random.PRNGKey(0))
N = S.cid.shape[0]
NA = core.n_actions(N)
out = {}

size = {k: int(np.prod(v.shape)) * v.dtype.itemsize for k, v in S._asdict().items()}
tot = sum(size.values())
out["etat_octets"] = tot
print(f"état : {tot} octets par partie ({len(size)} champs, N = {N} slots, {NA} actions)")
for k, v in sorted(size.items(), key=lambda kv: -kv[1])[:args.top]:
    f = getattr(S, k)
    print(f"  {k:10s} {v:6d}  {100 * v / tot:4.1f} %  {tuple(f.shape)} {f.dtype}")
by_dt = {}
for k, v in size.items():
    by_dt[str(getattr(S, k).dtype)] = by_dt.get(str(getattr(S, k).dtype), 0) + v
print("  par type : " + ", ".join(f"{k} {v}" for k, v in sorted(by_dt.items(), key=lambda kv: -kv[1])))

BYTES = dict(i1=1, i8=1, ui8=1, i16=2, ui16=2, i32=4, ui32=4, i64=8, ui64=8, f32=4, f64=8, f16=2, bf16=2)
TY = re.compile(r"tensor<([0-9x]*?)x?(u?i\d+|b?f\d+)>")


def count(name, fn, *shapes):
    txt = jax.jit(fn).lower(*shapes).compiler_ir(dialect="stablehlo").operation.get_asm(large_elements_limit=4)
    ops, byt = 0, 0
    for line in txt.splitlines():
        if "stablehlo." not in line or "stablehlo.constant" in line:
            continue
        ops += 1
        ty = TY.findall(line.split("->")[-1] if "->" in line else line.rsplit(":", 1)[-1])
        if ty:
            dims, dt = ty[-1]
            byt += int(np.prod([int(v) for v in dims.split("x") if v] or [1])) * BYTES.get(dt, 4)
    out[name] = dict(ops=ops, octets=byt)
    return ops, byt


a = jax.ShapeDtypeStruct((), jnp.int32)
m = jax.ShapeDtypeStruct((NA,), jnp.bool_)
bricks = [
    ("exec_one", lambda bk, s: core.exec_one(bk, s), (bank, S)),
    ("refresh", lambda bk, s: core.refresh(bk, s), (bank, S)),
    ("refresh_ex", lambda bk, s: core.refresh_ex(bk, s), (bank, S)),
    ("legal", lambda bk, s: core.legal(bk, s), (bank, S)),
    ("legal sans fusion", lambda bk, s: core.legal(bk, s, fus=False), (bank, S)),
    ("apply", lambda bk, s, a: core.apply(bk, s, a), (bank, S, a)),
    ("trig_eval", lambda bk, s: core.trig_eval(bk, s), (bank, S)),
    ("pass_priority", lambda bk, s: core.pass_priority(bk, s), (bank, S)),
    ("auto_pass", lambda bk, s: core.auto_pass(bk, s), (bank, S)),
    ("forced_m", lambda bk, s, m: core.forced_m(bk, s, m), (bank, S, m)),
    ("observe", lambda bk, s, m: obsm.observe(bk, s, None, m), (bank, S, m)),
]
ref = json.load(open(args.ref)) if args.ref else {}
print(f"\n{'brique':20s}{'opérations':>12s}{'octets écrits':>16s}{'× état':>8s}" + ("   écart ops / octets" if ref else ""))
for name, fn, sh in bricks:
    ops, byt = count(name, fn, *sh)
    extra = ""
    if name in ref:
        extra = f"   {100 * (ops / ref[name]['ops'] - 1):+.1f} % / {100 * (byt / ref[name]['octets'] - 1):+.1f} %"
    print(f"{name:20s}{ops:12d}{byt:16d}{byt / tot:8.1f}{extra}", flush=True)
if ref:
    print(f"état : {100 * (tot / ref['etat_octets'] - 1):+.1f} %")
if args.json:
    json.dump(out, open(args.json, "w"), indent=1)
