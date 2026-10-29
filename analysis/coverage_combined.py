#!/usr/bin/env python3
"""Couverture combinée : une carte est couverte seulement si TOUT ce qu'elle utilise l'est
(fonctions Duel.*, méthodes Card/Group, aux.*, codes EFFECT_*/EVENT_*).
Ordre des briques : par nombre de cartes qui les utilisent (même convention que count_primitives.py)."""
import json, os, re, statistics, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
from count_primitives import DATA_DIR, DUEL_RE, AUX_RE, METHOD_RE, EXCLUDED_METHODS, strip_comments

CODE_RE = re.compile(r"\b((?:EFFECT|EVENT)_[A-Z0-9_]+)\b")
FUNC_RE = re.compile(r"\bfunction\b")
LOOP_RE = re.compile(r"\b(for|while|repeat)\b")

cards = {}
loc, nfunc = {}, {}
for n in sorted(os.listdir(DATA_DIR)):
    if not (n.startswith("c") and n.endswith(".lua")):
        continue
    t = strip_comments(open(os.path.join(DATA_DIR, n), encoding="utf-8", errors="replace").read())
    f = {("Duel", m) for m in DUEL_RE.findall(t)} | {("aux", m) for m in AUX_RE.findall(t)}
    f |= {("meth", m) for m in METHOD_RE.findall(t) if m not in EXCLUDED_METHODS}
    f |= {("code", m) for m in CODE_RE.findall(t)}
    cards[n] = f
    loc[n] = sum(1 for l in t.splitlines() if l.strip())
    nfunc[n] = len(FUNC_RE.findall(t))

N = len(cards)
freq = Counter(x for f in cards.values() for x in f)
order = [x for x, _ in freq.most_common()]
rank = {x: i + 1 for i, x in enumerate(order)}
need = sorted(max((rank[x] for x in f), default=0) for f in cards.values())
kinds = Counter(k for k, _ in order)

def cov(n):  # % de cartes couvertes avec les n briques les plus fréquentes
    import bisect
    return 100 * bisect.bisect_right(need, n) / N

def n_for(p):
    return need[min(N - 1, int(p / 100 * N + 0.999) - 1)]

out = {"cards": N, "features_total": len(order), "features_by_kind": dict(kinds),
       "coverage": {n: round(cov(n), 2) for n in (50, 100, 150, 200, 300, 400, 600, 800, 1000, 1500)},
       "features_for": {p: n_for(p) for p in (50, 80, 90, 95, 99, 99.9, 100)},
       "singletons": sum(1 for v in freq.values() if v == 1),
       "loc": {q: statistics.quantiles(loc.values(), n=100)[q - 1] for q in (50, 90, 99)} | {"max": max(loc.values())},
       "functions_per_script": {q: statistics.quantiles(nfunc.values(), n=100)[q - 1] for q in (50, 90, 99)},
       "scripts_with_loops": round(100 * sum(1 for n in cards if LOOP_RE.search(open(os.path.join(DATA_DIR, n), errors="replace").read())) / N, 1)}
# composition des briques nécessaires à 95 % / 99 %
for p in (95, 99):
    out[f"kinds_at_{p}"] = dict(Counter(k for k, _ in order[:n_for(p)]))
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "coverage_combined.json"), "w"), indent=1, ensure_ascii=False)
