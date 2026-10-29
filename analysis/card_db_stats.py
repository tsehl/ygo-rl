#!/usr/bin/env python3
"""Compute statistics over the Yu-Gi-Oh card database (read-only).

Stdlib only. The database is opened with mode=ro and never written to.
Run with: python3 -I card_db_stats.py
"""
import json
import math
import os
import re
import sqlite3
import statistics
from collections import Counter

DB_PATH = "/Users/thomas/Documents/collab/ygo/data/BabelCDB/cards.cdb"
CONST_PATH = "/Users/thomas/Documents/collab/ygo/data/CardScripts/constant.lua"
OUT_DIR = "/Users/thomas/Documents/collab/ygo/analysis"
JSON_PATH = os.path.join(OUT_DIR, "card_db_stats.json")
MD_PATH = os.path.join(OUT_DIR, "card_db_stats.md")

U32 = (1 << 32) - 1
U64 = (1 << 64) - 1

NAME_RE = re.compile(r"^(TYPE|RACE|ATTRIBUTE)_[A-Z0-9_]+$")
LITERAL_RE = re.compile(r"^(0[xX][0-9a-fA-F]+|\d+)$")

# Constants that the script relies on by name. Missing any of these is fatal.
REQUIRED = [
    "TYPE_MONSTER", "TYPE_SPELL", "TYPE_TRAP", "TYPE_TOKEN", "TYPE_NORMAL",
    "TYPE_EFFECT", "TYPE_FUSION", "TYPE_SYNCHRO", "TYPE_XYZ", "TYPE_LINK",
    "TYPE_PENDULUM", "TYPE_RITUAL", "TYPE_QUICKPLAY", "TYPE_CONTINUOUS",
    "TYPE_EQUIP", "TYPE_FIELD", "TYPE_COUNTER",
]


def parse_constants(path):
    """Return (values, skipped). values maps name -> int for literal definitions.

    Accepts `NAME = 0x..`, `NAME=0x..`, decimal literals and trailing `--` comments.
    Composite expressions (e.g. TYPE_EXTRA = A|B) are skipped and reported.
    """
    values = {}
    skipped = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, 1):
            code = raw.split("--", 1)[0].strip()
            if "=" not in code:
                continue
            name, _, rhs = code.partition("=")
            name, rhs = name.strip(), rhs.strip()
            if not NAME_RE.match(name):
                continue
            if LITERAL_RE.match(rhs):
                val = int(rhs, 16) if rhs[:2].lower() == "0x" else int(rhs, 10)
                if name in values and values[name] != val:
                    raise SystemExit(f"conflicting definitions for {name} (line {lineno})")
                values[name] = val
            else:
                skipped.append({"line": lineno, "text": code})
    return values, skipped


def single_bits(values, prefix):
    """Single-bit constants with the given prefix, ordered by value."""
    out = {}
    for name, v in sorted(values.items(), key=lambda kv: kv[1]):
        if name.startswith(prefix) and v > 0 and (v & (v - 1)) == 0:
            out[name[len(prefix):]] = v
    return out


def hexs(v):
    return hex(v)


def atk_bucket(atk):
    if atk == -2:
        return "?"
    if atk < 0:
        return "negative_other"
    if atk >= 3000:
        return "3000+"
    lo = (atk // 500) * 500
    return f"{lo}-{lo + 499}"


ATK_LABELS = [f"{lo}-{lo + 499}" for lo in range(0, 3000, 500)] + ["3000+"]


def text_stats(lengths):
    if not lengths:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None}
    s = sorted(lengths)
    n = len(s)
    p90 = s[max(0, math.ceil(0.9 * n) - 1)]  # nearest-rank percentile
    return {
        "count": n,
        "min": s[0],
        "median": statistics.median(s),
        "p90": p90,
        "max": s[-1],
    }


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def pct(part, whole):
    return f"{100.0 * part / whole:.1f}%" if whole else "0.0%"


def main():
    values, skipped = parse_constants(CONST_PATH)

    missing = [n for n in REQUIRED if n not in values]
    if missing:
        raise SystemExit("could not parse required constants: " + ", ".join(missing))

    print("Parsed TYPE_ constants:")
    for name, v in sorted(values.items(), key=lambda kv: kv[1]):
        if name.startswith("TYPE_"):
            print(f"  {name} = {hexs(v)}")
    print(f"Parsed {len(values)} literal constants; {len(skipped)} composite lines skipped:")
    for s in skipped:
        print(f"  line {s['line']}: {s['text']}")

    TYPE_BITS = single_bits(values, "TYPE_")
    RACE_BITS = single_bits(values, "RACE_")
    ATTR_BITS = single_bits(values, "ATTRIBUTE_")

    T = values
    MON, SPELL, TRAP, TOKEN = T["TYPE_MONSTER"], T["TYPE_SPELL"], T["TYPE_TRAP"], T["TYPE_TOKEN"]
    EFFECT, PENDULUM, LINK = T["TYPE_EFFECT"], T["TYPE_PENDULUM"], T["TYPE_LINK"]
    EXTRA_MASK = T["TYPE_FUSION"] | T["TYPE_SYNCHRO"] | T["TYPE_XYZ"] | T["TYPE_LINK"]
    SPELL_SUB = {k: T["TYPE_" + k] for k in ("QUICKPLAY", "CONTINUOUS", "EQUIP", "FIELD", "RITUAL")}
    TRAP_SUB = {k: T["TYPE_" + k] for k in ("CONTINUOUS", "COUNTER")}
    SPELL_SUB_MASK = 0
    for v in SPELL_SUB.values():
        SPELL_SUB_MASK |= v
    TRAP_SUB_MASK = 0
    for v in TRAP_SUB.values():
        TRAP_SUB_MASK |= v
    # Every single-bit TYPE_ constant except the top-level kinds and TOKEN is a monster subtype/flag.
    MON_SUB = {k: v for k, v in TYPE_BITS.items() if k not in ("MONSTER", "SPELL", "TRAP", "TOKEN")}
    KNOWN_TYPE_MASK = 0
    for v in TYPE_BITS.values():
        KNOWN_TYPE_MASK |= v
    KNOWN_RACE_MASK = 0
    for v in RACE_BITS.values():
        KNOWN_RACE_MASK |= v
    KNOWN_ATTR_MASK = 0
    for v in ATTR_BITS.values():
        KNOWN_ATTR_MASK |= v

    con = sqlite3.connect("file:" + DB_PATH + "?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT d.id, d.alias, d.setcode, d.type, d.atk, d.def, d.level, d.race, "
            "d.attribute, t.desc FROM datas d LEFT JOIN texts t ON t.id = d.id ORDER BY d.id"
        ).fetchall()
    finally:
        con.close()

    total_rows = len(rows)
    cards = [r for r in rows if (r[1] or 0) == 0]
    excluded_alias = total_rows - len(cards)

    kind = Counter()
    other_sub = Counter()
    token_rows = 0
    multi_kind_rows = 0
    unmapped_type_bits = Counter()
    deck = Counter()
    mon_sub = Counter()
    spell_sub = Counter()
    trap_sub = Counter()
    main_level = Counter()
    link_rating = Counter()
    pend_scale = Counter()
    atk_buckets = Counter()
    race_counts = Counter()
    attr_counts = Counter()
    unknown_race = Counter()
    unknown_attr = Counter()
    setcode_dist = Counter()
    archetypes = set()
    missing_desc = Counter()
    text_lens = {"effect_monsters": [], "spells": [], "traps": []}

    for cid, alias, setcode, t, atk, deff, level, race, attr, desc in cards:
        t = (t or 0) & U32
        level = level or 0
        atk = atk or 0
        race = (race or 0) & U64
        attr = (attr or 0) & U32

        is_token = bool(t & TOKEN)
        token_rows += is_token
        unmapped = t & ~KNOWN_TYPE_MASK & U32
        for bit in range(32):
            if unmapped & (1 << bit):
                unmapped_type_bits[hexs(1 << bit)] += 1

        has_m, has_s, has_t = bool(t & MON), bool(t & SPELL), bool(t & TRAP)
        if not is_token and has_m + has_s + has_t > 1:
            multi_kind_rows += 1

        if is_token:
            k = "other"
            other_sub["token"] += 1
        elif has_m:
            k = "monster"
        elif has_s:
            k = "spell"
        elif has_t:
            k = "trap"
        else:
            k = "other"
            other_sub["no_mst_bits"] += 1
        kind[k] += 1

        # setcodes: 64-bit value holding up to four 16-bit archetype codes
        sc = (setcode or 0) & U64
        chunks = [(sc >> (16 * i)) & 0xFFFF for i in range(4)]
        nonzero = [c for c in chunks if c]
        setcode_dist[len(nonzero)] += 1
        archetypes.update(nonzero)

        if desc is None:
            missing_desc[k] += 1

        if k == "monster":
            is_extra = bool(t & EXTRA_MASK)
            deck["extra" if is_extra else "main"] += 1
            for name, bit in MON_SUB.items():
                if t & bit:
                    mon_sub[name] += 1
            lvl = level & 0xFF
            if t & LINK:
                link_rating[lvl] += 1
            elif not is_extra:
                main_level[lvl] += 1
            if t & PENDULUM:
                pend_scale[(level >> 24) & 0xFF] += 1
            atk_buckets[atk_bucket(atk)] += 1
            if race == 0:
                race_counts["(none)"] += 1
            for name, bit in RACE_BITS.items():
                if race & bit:
                    race_counts[name] += 1
            for bit in range(64):
                if (race & ~KNOWN_RACE_MASK) & (1 << bit):
                    unknown_race[hexs(1 << bit)] += 1
            if attr == 0:
                attr_counts["(none)"] += 1
            for name, bit in ATTR_BITS.items():
                if attr & bit:
                    attr_counts[name] += 1
            for bit in range(32):
                if (attr & ~KNOWN_ATTR_MASK) & (1 << bit):
                    unknown_attr[hexs(1 << bit)] += 1
            if t & EFFECT:
                text_lens["effect_monsters"].append(len(desc or ""))

        elif k == "spell":
            for name, bit in SPELL_SUB.items():
                if t & bit:
                    spell_sub[name] += 1
            if not (t & SPELL_SUB_MASK):
                spell_sub["Normal"] += 1
            text_lens["spells"].append(len(desc or ""))

        elif k == "trap":
            for name, bit in TRAP_SUB.items():
                if t & bit:
                    trap_sub[name] += 1
            if not (t & TRAP_SUB_MASK):
                trap_sub["Normal"] += 1
            text_lens["traps"].append(len(desc or ""))

    # ---- assemble results -------------------------------------------------
    n_cards = len(cards)
    mon_total = kind["monster"]
    deck_total = deck["main"] + deck["extra"]
    atk_out = {label: atk_buckets.get(label, 0) for label in ATK_LABELS}
    atk_out["?"] = atk_buckets.get("?", 0)
    atk_out["negative_other"] = atk_buckets.get("negative_other", 0)

    setcode_out = {str(k): setcode_dist.get(k, 0) for k in range(5)}

    results = {
        "source": {"database": DB_PATH, "constants": CONST_PATH, "table": "datas/texts"},
        "filter": {
            "rows_total": total_rows,
            "excluded_alias_rows": excluded_alias,
            "alias_rule": "rows with alias != 0 excluded (alternate artworks)",
            "cards_after_filter": n_cards,
        },
        "constants": {
            "TYPE": {k: hexs(v) for k, v in TYPE_BITS.items()},
            "RACE": {k: hexs(v) for k, v in RACE_BITS.items()},
            "ATTRIBUTE": {k: hexs(v) for k, v in ATTR_BITS.items()},
            "composite_lines_skipped": skipped,
        },
        "top_level_kind": {
            "Monster": kind["monster"],
            "Spell": kind["spell"],
            "Trap": kind["trap"],
            "other": kind["other"],
            "other_breakdown": dict(other_sub),
            "token_rows_with_TYPE_TOKEN": token_rows,
            "anomalies_multiple_top_kind_bits": multi_kind_rows,
        },
        "monster": {
            "deck_split": {"main": deck["main"], "extra": deck["extra"],
                           "extra_definition": "TYPE_FUSION|TYPE_SYNCHRO|TYPE_XYZ|TYPE_LINK"},
            "subtype_counts": {k: mon_sub.get(k, 0) for k in MON_SUB},
            "level_rank_main_deck": {str(k): main_level[k] for k in sorted(main_level)},
            "link_rating": {str(k): link_rating[k] for k in sorted(link_rating)},
            "pendulum_left_scale": {str(k): pend_scale[k] for k in sorted(pend_scale)},
            "atk_buckets": atk_out,
            "race_counts": dict(sorted(race_counts.items(), key=lambda kv: kv[0])),
            "attribute_counts": dict(sorted(attr_counts.items(), key=lambda kv: kv[0])),
            "unknown_race_bits": dict(unknown_race),
            "unknown_attribute_bits": dict(unknown_attr),
        },
        "spell": {
            "subtype_counts": {"Normal": spell_sub.get("Normal", 0),
                               **{k: spell_sub.get(k, 0) for k in SPELL_SUB}},
            "note": "subtype flags are not mutually exclusive; Normal = no subtype flag set",
        },
        "trap": {
            "subtype_counts": {"Normal": trap_sub.get("Normal", 0),
                               **{k: trap_sub.get(k, 0) for k in TRAP_SUB}},
            "note": "subtype flags are not mutually exclusive; Normal = no subtype flag set",
        },
        "setcodes": {
            "distinct_setcodes_per_card": setcode_out,
            "max_distinct_setcodes_on_one_card": max(k for k, v in setcode_dist.items() if v),
            "distinct_archetype_codes": len(archetypes),
        },
        "effect_text_length_chars": {
            "effect_monsters": text_stats(text_lens["effect_monsters"]),
            "spells": text_stats(text_lens["spells"]),
            "traps": text_stats(text_lens["traps"]),
            "missing_desc_rows_by_kind": dict(missing_desc),
        },
        "data_quality": {
            "unmapped_type_bits": dict(unmapped_type_bits),
        },
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    write_markdown(results, mon_total, deck_total, n_cards, kind)

    print()
    print(f"Rows total: {total_rows}; excluded alias rows: {excluded_alias}; cards after filter: {n_cards}")
    print(f"Monster {kind['monster']}, Spell {kind['spell']}, Trap {kind['trap']}, other {kind['other']} "
          f"(tokens {token_rows})")
    print(f"Distinct archetype codes: {len(archetypes)}")
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {MD_PATH}")


def write_markdown(r, mon_total, deck_total, n_cards, kind):
    f = r["filter"]
    tk = r["top_level_kind"]
    m = r["monster"]
    out = []
    out.append("# Yu-Gi-Oh card database statistics\n")
    out.append(f"Source: `{r['source']['database']}` (table `datas`, texts from `texts`).\n")
    out.append("## Filter\n")
    out.append(md_table(["Metric", "Value"], [
        ["Rows in `datas`", f["rows_total"]],
        ["Excluded (alias != 0, alternate artworks)", f["excluded_alias_rows"]],
        ["Cards after filter", n_cards],
    ]))
    out.append("\n## Top-level kind\n")
    out.append(md_table(["Kind", "Count", "Share"], [
        [k, tk[k], pct(tk[k], n_cards)] for k in ("Monster", "Spell", "Trap", "other")
    ]))
    out.append(f"\nTokens (TYPE_TOKEN) are counted in `other`: {tk['token_rows_with_TYPE_TOKEN']}. "
               f"Multi-kind anomalies: {tk['anomalies_multiple_top_kind_bits']}.\n")

    out.append("## Monsters\n")
    out.append(md_table(["Deck", "Count", "Share"], [
        ["Main", m["deck_split"]["main"], pct(m["deck_split"]["main"], mon_total)],
        ["Extra (Fusion/Synchro/Xyz/Link)", m["deck_split"]["extra"], pct(m["deck_split"]["extra"], mon_total)],
    ]))
    out.append("\n### Subtype flags (every TYPE_ bit; zero rows omitted)\n")
    sub_rows = [[k, v, pct(v, mon_total)] for k, v in m["subtype_counts"].items() if v]
    out.append(md_table(["Flag", "Count", "Share of monsters"], sub_rows))
    out.append("\n### Level / rank (main deck, level & 0xFF)\n")
    out.append(md_table(["Level/Rank", "Count"], list(m["level_rank_main_deck"].items())))
    out.append("\n### Link rating (Link monsters)\n")
    out.append(md_table(["Link rating", "Count"], list(m["link_rating"].items())))
    out.append("\n### Pendulum left scale\n")
    out.append(md_table(["Scale", "Count"], list(m["pendulum_left_scale"].items())))
    out.append("\n### ATK buckets (monsters)\n")
    out.append(md_table(["ATK", "Count"], list(m["atk_buckets"].items())))
    out.append("\n### Race (monsters; a card may have one race)\n")
    out.append(md_table(["Race", "Count"], list(m["race_counts"].items())))
    out.append("\n### Attribute (monsters)\n")
    out.append(md_table(["Attribute", "Count"], list(m["attribute_counts"].items())))

    out.append("\n## Spells\n")
    s_total = tk["Spell"]
    out.append(md_table(["Subtype", "Count", "Share of spells"],
                        [[k, v, pct(v, s_total)] for k, v in r["spell"]["subtype_counts"].items()]))
    out.append("\n## Traps\n")
    t_total = tk["Trap"]
    out.append(md_table(["Subtype", "Count", "Share of traps"],
                        [[k, v, pct(v, t_total)] for k, v in r["trap"]["subtype_counts"].items()]))

    out.append("\n## Setcodes (archetypes)\n")
    sc = r["setcodes"]
    out.append(md_table(["Distinct setcodes on one card", "Cards"],
                        list(sc["distinct_setcodes_per_card"].items())))
    out.append(f"\nMax distinct setcodes on one card: {sc['max_distinct_setcodes_on_one_card']}. "
               f"Distinct archetype codes (nonzero 16-bit chunks): {sc['distinct_archetype_codes']}.\n")

    out.append("## Effect text length (characters)\n")
    rows = []
    for key, label in (("effect_monsters", "Effect monsters"), ("spells", "Spells"), ("traps", "Traps")):
        s = r["effect_text_length_chars"][key]
        rows.append([label, s["count"], s["min"], s["median"], s["p90"], s["max"]])
    out.append(md_table(["Group", "Count", "Min", "Median", "p90", "Max"], rows))
    out.append("\np90 uses nearest-rank.\n")

    out.append("## Data quality notes\n")
    dq = r["data_quality"]
    out.append(f"- Composite constant lines skipped: {len(r['constants']['composite_lines_skipped'])}")
    out.append(f"- Unmapped type bits seen: {dq['unmapped_type_bits'] or 'none'}")
    out.append(f"- Unknown race bits seen: {m['unknown_race_bits'] or 'none'}")
    out.append(f"- Unknown attribute bits seen: {m['unknown_attribute_bits'] or 'none'}")
    out.append(f"- Rows with no description text: {r['effect_text_length_chars']['missing_desc_rows_by_kind'] or 'none'}")

    with open(MD_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
