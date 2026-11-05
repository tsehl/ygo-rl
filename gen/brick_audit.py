"""Audit des briques : quelles caractéristiques d'effet des vraies cartes du benchmark (train/realbank.npz) le
générateur procédural ne produit jamais ? Caractéristiques par effet : genre (K_*), opcodes par partie du programme
(coût / cible / résolution ; les conditions COND_* sans partie) avec leurs arguments significatifs, registres retirés, événements de
déclenchement, modificateurs continus (MC_*), damage step, limites par tour, filtres par nom / archétype.

Usage : python gen/brick_audit.py [--banks 10]"""
import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", d) for d in ("engine", "gen", "train")]
import numpy as np

import core
import cardgen
import realcards

OPS = ("END FILTER SELF REQ REQ_FREE MINATK CHOOSE MOVE SPSUMMON DRAW DAMAGE RECOVER SHUFFLE_HAND COND_CHAIN CHAINCARD "
       "NEGATE PAY_LP EVENT COND_TURN COND_EVP COND_SELF LOCK MODIFY COND_ATK ATTACKER COND_PHASE FUSMAT RITMAT LABEL "
       "COND_LABEL YESNO").split()
KINDS = "NONE ACTIVATE IGNITION TRIG_F TRIG_O QUICK CONT".split()
PARTS = ("coût", "cible", "résol")
EVS = "SUMMON FLIP_SUMMON SPSUMMON ATTACK TO_GRAVE BATTLE_DESTROYED FLIP PRE_DMG".split()
# arguments significatifs par opcode (les numéros de registre sont des détails d'assemblage : on les retire)
ARGS = {"MOVE": lambda a, b, c: (b, c), "SPSUMMON": lambda a, b, c: (b, c), "MODIFY": lambda a, b, c: (b,),
        "EVENT": lambda a, b, c: (b,), "COND_PHASE": lambda a, b, c: (a,), "COND_TURN": lambda a, b, c: (a,),
        "COND_EVP": lambda a, b, c: (a, b), "COND_SELF": lambda a, b, c: (a, b, c), "COND_CHAIN": lambda a, b, c: (a, b),
        "COND_ATK": lambda a, b, c: (a,)}


def features(bank, c):
    """Ensemble de caractéristiques (chaînes) de la carte d'index c."""
    out = set()
    ek, prog, etrig = np.asarray(bank.ekind[c]), np.asarray(bank.prog[c]), np.asarray(bank.etrig[c])
    emod, edmg, eopt = np.asarray(bank.emod[c]), np.asarray(bank.edmg[c]), np.asarray(bank.eopt[c])
    filt = np.asarray(bank.filt[c])
    for e in range(ek.shape[0]):
        if ek[e] == core.K_NONE:
            continue
        k = KINDS[ek[e]]
        out.add(f"genre {k}")
        if k == "CONT":
            out.add(f"continu MC={int(emod[e, core.M_CODE])}")
        for pt in range(3):
            for op, a1, b1, c1 in prog[e, pt]:
                if op == core.OP_END:
                    continue
                nm = OPS[op]
                part = "condition" if nm.startswith("COND_") else PARTS[pt]   # coût ou cible : vérifiée à l'activation
                out.add(f"{part} {nm}" + (f" {ARGS[nm](int(a1), int(b1), int(c1))}" if nm in ARGS else ""))
        for i, ev in enumerate(EVS):
            if etrig[e] >> i & 1:
                out.add(f"déclencheur {ev}")
        if edmg[e]:
            out.add(f"damage step {int(edmg[e])}")
        if eopt[e]:
            out.add(f"limite par tour {int(eopt[e])}")
        for f in filt[e]:
            if f[core.F_CODE1] or f[core.F_CODE2]:
                out.add("filtre par nom")
            if f[core.F_SET]:
                out.add("filtre par archétype")
    if np.asarray(bank.erl[c]):
        out.add("EnableReviveLimit")
    if np.asarray(bank.ldiff[c]):
        out.add("Link : noms différents")
    if np.asarray(bank.lneed[c]):
        out.add("Link : matériau Link requis")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", type=int, default=10)
    a = ap.parse_args()
    seen = collections.Counter()
    ncards = 0
    for sd in range(a.banks):
        cset = cardgen.Gen(sd).make_set(400, 80)
        bank = cardgen.to_bank(cset)
        for c in range(len(cset)):
            for ft in features(bank, c):
                seen[ft] += 1
        ncards += len(cset)
    rbank, *_ = realcards.load()
    names = realcards.card_names()
    missing = collections.defaultdict(list)
    for c, nm in enumerate(names):
        if not nm:
            continue
        for ft in features(rbank, c):
            if seen[ft] == 0:
                missing[ft].append(nm)
    print(f"{ncards} cartes procédurales ({a.banks} banques), {sum(1 for n in names if n)} vraies cartes\n")
    print("caractéristiques des vraies cartes jamais produites par le générateur :")
    for ft, nms in sorted(missing.items(), key=lambda x: -len(x[1])):
        print(f"  {ft:42s} {len(nms):3d}  {', '.join(sorted(set(nms)))}")
    full = [nm for c, nm in enumerate(names) if nm and all(seen[ft] for ft in features(rbank, c))]
    print(f"\nvraies cartes entièrement couvertes : {len(full)} / {sum(1 for n in names if n)}")
    print("\ncaractéristiques rares à l'entraînement (< 0,2 % des cartes) mais présentes dans les vraies cartes :")
    real = collections.Counter(ft for c, nm in enumerate(names) if nm for ft in features(rbank, c))
    for ft, n in sorted(real.items(), key=lambda x: seen[x[0]]):
        if 0 < seen[ft] < 0.002 * ncards:
            print(f"  {ft:42s} générateur {seen[ft]:4d} / {ncards}, vraies cartes {n}")


if __name__ == "__main__":
    main()
