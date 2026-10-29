#!/usr/bin/env python3
"""Usage statistics over Yu-Gi-Oh card scripts (stdlib only).

Reads every official CardScripts c*.lua file (read-only), strips Lua comments,
and computes:
  1. Duel API call frequency
  2. aux helper references
  3. Card method calls (capitalised method names, minus effect setters)
  4. Constants passed to effect setters (per setter)
  5. Effect.CreateEffect count per card (histogram + mean)
  6. Coverage curve of top-N Duel functions

Writes primitives_stats.json and primitives_stats.md into the analysis folder.
"""
import fnmatch
import json
import os
import re
import sys
from collections import Counter

DATA_DIR = "/Users/thomas/Documents/collab/ygo/data/CardScripts/official"
OUT_DIR = "/Users/thomas/Documents/collab/ygo/analysis"
JSON_OUT = os.path.join(OUT_DIR, "primitives_stats.json")
MD_OUT = os.path.join(OUT_DIR, "primitives_stats.md")

DUEL_RE = re.compile(r"\bDuel\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")
AUX_RE = re.compile(r"\baux\.([A-Za-z_][A-Za-z0-9_]*)\b")
METHOD_RE = re.compile(r":([A-Z]\w*)\s*\(")
CONST_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
CREATE_RE = re.compile(r"\bEffect\.CreateEffect\b")

SETTERS = ["SetCategory", "SetType", "SetCode", "SetProperty", "SetRange", "SetCountLimit"]
SETTER_RES = {name: re.compile(r":(%s)\s*\(" % name) for name in SETTERS}

EXCLUDED_METHODS = {
    "SetDescription", "SetCategory", "SetType", "SetCode", "SetProperty",
    "SetRange", "SetTargetRange", "SetCountLimit", "SetCondition", "SetCost",
    "SetTarget", "SetOperation", "SetValue", "SetLabel", "SetLabelObject",
    "SetReset", "SetHintTiming", "SetAbsoluteRange", "SetOwnerPlayer",
    "Clone", "RegisterEffect",
}

# One pass over the text: block comments and line comments are matched so they
# can be removed; string literals are matched so that "--" inside a string is
# not treated as a comment start.
LEX_RE = re.compile(
    r"--\[(=*)\[.*?\]\1\]"        # block comment --[[ ... ]] / --[==[ ... ]==]
    r"|--[^\n]*"                  # line comment
    r"|\[(=*)\[.*?\]\2\]"         # long string (kept)
    r"|\"(?:\\.|[^\"\\\n])*\""    # double-quoted string (kept)
    r"|'(?:\\.|[^'\\\n])*'",      # single-quoted string (kept)
    re.S,
)


def strip_comments(text):
    def repl(m):
        s = m.group(0)
        if s.startswith("--"):
            return "\n" * s.count("\n") if "\n" in s else " "
        return s
    return LEX_RE.sub(repl, text)


def extract_args(text, open_idx):
    """Return the text between the '(' at open_idx and its matching ')'."""
    depth = 0
    for j in range(open_idx, len(text)):
        ch = text[j]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:j]
    return None  # unbalanced


def sorted_rows(occ, files):
    rows = [{"token": t, "files": files[t], "occurrences": occ[t]} for t in files]
    rows.sort(key=lambda r: (-r["files"], -r["occurrences"], r["token"]))
    return rows


def md_table(rows, limit=None):
    shown = rows if limit is None else rows[:limit]
    out = ["| token | files | occurrences |", "|---|---:|---:|"]
    for r in shown:
        out.append("| %s | %d | %d |" % (r["token"], r["files"], r["occurrences"]))
    return "\n".join(out)


def main():
    names = sorted(
        n for n in os.listdir(DATA_DIR)
        if fnmatch.fnmatchcase(n, "c*.lua") and os.path.isfile(os.path.join(DATA_DIR, n))
    )
    if not names:
        sys.exit("No c*.lua files found in %s" % DATA_DIR)

    duel_occ, duel_files = Counter(), Counter()
    aux_occ, aux_files = Counter(), Counter()
    meth_occ, meth_files = Counter(), Counter()
    set_occ = {s: Counter() for s in SETTERS}
    set_files = {s: Counter() for s in SETTERS}
    per_file_duel = {}          # file -> set of Duel function names
    effect_counts = []          # per file, Effect.CreateEffect count
    failures = []               # (file, error)
    processed = 0

    for name in names:
        path = os.path.join(DATA_DIR, name)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                raw = fh.read()
            code = strip_comments(raw)

            duel = DUEL_RE.findall(code)
            aux = AUX_RE.findall(code)
            meth = [m for m in METHOD_RE.findall(code) if m not in EXCLUDED_METHODS]

            duel_occ.update(duel)
            duel_files.update(set(duel))
            aux_occ.update(aux)
            aux_files.update(set(aux))
            meth_occ.update(meth)
            meth_files.update(set(meth))
            per_file_duel[name] = set(duel)

            for setter in SETTERS:
                consts_in_file = []
                for m in SETTER_RES[setter].finditer(code):
                    args = extract_args(code, m.end() - 1)
                    if args is None:
                        continue
                    consts = CONST_RE.findall(args)
                    set_occ[setter].update(consts)
                    consts_in_file.extend(consts)
                if consts_in_file:
                    set_files[setter].update(set(consts_in_file))

            effect_counts.append(len(CREATE_RE.findall(code)))
            processed += 1
        except Exception as exc:  # record and continue
            failures.append((name, "%s: %s" % (type(exc).__name__, exc)))

    # ---- stat 1-4 tables -------------------------------------------------
    duel_rows = sorted_rows(duel_occ, duel_files)
    aux_rows = sorted_rows(aux_occ, aux_files)
    meth_rows = sorted_rows(meth_occ, meth_files)
    setter_rows = {s: sorted_rows(set_occ[s], set_files[s]) for s in SETTERS}

    # ---- stat 5 -----------------------------------------------------------
    hist = Counter(min(c, 10) for c in effect_counts)
    histogram = []
    for k in range(0, 11):
        label = "10+" if k == 10 else str(k)
        histogram.append({"effects": label, "files": hist.get(k, 0)})
    mean_effects = (sum(effect_counts) / float(len(effect_counts))) if effect_counts else 0.0
    effects_stat = {
        "files": len(effect_counts),
        "total_effects": sum(effect_counts),
        "mean": round(mean_effects, 4),
        "max": max(effect_counts) if effect_counts else 0,
        "histogram": histogram,
    }

    # ---- stat 6 -----------------------------------------------------------
    ranking = sorted(duel_files, key=lambda t: (-duel_files[t], -duel_occ[t], t))
    all_files = len(per_file_duel)
    with_duel = [s for s in per_file_duel.values() if s]
    coverage = []
    for n_top in [10, 25, 50, 100, 150, 200, 300]:
        top = set(ranking[:n_top])
        covered_all = sum(1 for s in per_file_duel.values() if s <= top)
        covered_duel = sum(1 for s in with_duel if s <= top)
        coverage.append({
            "N": n_top,
            "distinct_duel_functions_available": min(n_top, len(ranking)),
            "covered_files": covered_all,
            "pct_of_all_files": round(100.0 * covered_all / all_files, 2) if all_files else 0.0,
            "covered_files_with_duel_calls": covered_duel,
            "pct_of_files_with_duel_calls": round(100.0 * covered_duel / len(with_duel), 2) if with_duel else 0.0,
        })
    coverage_stat = {
        "files_total": all_files,
        "files_with_zero_duel_calls": all_files - len(with_duel),
        "files_with_duel_calls": len(with_duel),
        "distinct_duel_functions": len(ranking),
        "note": ("pct_of_all_files counts files with zero Duel calls as covered "
                 "(empty set is a subset of any top-N set); pct_of_files_with_duel_calls "
                 "excludes them."),
        "table": coverage,
    }

    meta = {
        "data_dir": DATA_DIR,
        "files_matched": len(names),
        "files_processed": processed,
        "files_failed": len(failures),
        "failed_files": [{"file": f, "error": e} for f, e in failures],
    }

    full = {
        "meta": meta,
        "duel_api_calls": duel_rows,
        "aux_helpers": aux_rows,
        "card_methods": meth_rows,
        "effect_setters": {s: setter_rows[s] for s in SETTERS},
        "effects_per_card": effects_stat,
        "duel_coverage": coverage_stat,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(JSON_OUT, "w", encoding="utf-8") as fh:
        json.dump(full, fh, indent=2, ensure_ascii=False)

    # ---- markdown ---------------------------------------------------------
    md = []
    md.append("# Card script primitive statistics\n")
    md.append("- Source: `%s`" % DATA_DIR)
    md.append("- Total card files matched: **%d**" % len(names))
    md.append("- Files processed: **%d**; failed to read/parse: **%d**" % (processed, len(failures)))
    md.append("- Comments (`--` and `--[[ ]]`) stripped before matching.")
    md.append("- Full tables: `%s`\n" % JSON_OUT)

    md.append("## 1. Duel API calls (top 80 of %d distinct)\n" % len(duel_rows))
    md.append(md_table(duel_rows, 80) + "\n")

    md.append("## 2. aux helper references (top 80 of %d distinct)\n" % len(aux_rows))
    md.append(md_table(aux_rows, 80) + "\n")

    md.append("## 3. Card method calls, uppercase names (top 80 of %d distinct)\n" % len(meth_rows))
    md.append("Excludes effect setters and `Clone`/`RegisterEffect` (see stat 4).\n")
    md.append(md_table(meth_rows, 80) + "\n")

    md.append("## 4. Constants passed to effect setters\n")
    for s in SETTERS:
        rows = setter_rows[s]
        md.append("### %s (top 50 of %d distinct)\n" % (s, len(rows)))
        md.append(md_table(rows, 50) + "\n")

    md.append("## 5. Effects per card (`Effect.CreateEffect`)\n")
    md.append("- Files: %d" % effects_stat["files"])
    md.append("- Total effects: %d" % effects_stat["total_effects"])
    md.append("- **Mean effects per card: %.4f**" % effects_stat["mean"])
    md.append("- Max: %d\n" % effects_stat["max"])
    md.append("| effects | files |")
    md.append("|---|---:|")
    for h in histogram:
        md.append("| %s | %d |" % (h["effects"], h["files"]))
    md.append("")

    md.append("## 6. Duel API coverage curve\n")
    md.append("Duel functions ranked by file count; for each N, the share of card files whose "
              "entire set of `Duel.*` calls lies within the top-N set.\n")
    md.append("- Files total: %d (with zero Duel calls: %d)" %
              (coverage_stat["files_total"], coverage_stat["files_with_zero_duel_calls"]))
    md.append("- Distinct Duel functions: %d\n" % coverage_stat["distinct_duel_functions"])
    md.append("| N | covered files | % of all files | % of files with Duel calls |")
    md.append("|---:|---:|---:|---:|")
    for c in coverage:
        md.append("| %d | %d | %.2f%% | %.2f%% |" % (
            c["N"], c["covered_files"], c["pct_of_all_files"], c["pct_of_files_with_duel_calls"]))
    md.append("\n*Files with zero Duel calls count as covered in the first column (vacuously).*\n")

    if failures:
        md.append("## Files that could not be read/parsed\n")
        for f, e in failures:
            md.append("- `%s`: %s" % (f, e))
        md.append("")

    with open(MD_OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))

    # ---- console summary --------------------------------------------------
    print("files matched:   %d" % len(names))
    print("files processed: %d" % processed)
    print("files failed:    %d" % len(failures))
    print("mean effects/card: %.4f" % effects_stat["mean"])
    print("wrote %s" % JSON_OUT)
    print("wrote %s" % MD_OUT)


if __name__ == "__main__":
    main()
