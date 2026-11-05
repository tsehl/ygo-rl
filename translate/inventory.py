"""Inventaire : quelle part des 13 541 cartes officielles le moteur sait-il jouer, et quelles fonctionnalités manquent,
classées par nombre de cartes débloquées ?

Entrée : translate/cache/extract.json.gz (translate/extract.py). Chaque carte est un ensemble d'items (appels d'API,
méthodes, constantes). Un item est :
- « ignoré » : affichage, plomberie Lua, données génériques (Duel.Hint, aux.Stringid, s.filter, RACE_*, SET_*…) ;
- « couvert » : présent dans le script d'au moins une carte que le moteur implémente (engine/cards.py CARDS,
  vérifiées contre l'oracle). Approximation : une carte couverte peut n'utiliser qu'une partie d'une fonctionnalité
  (p. ex. EFFECT_UPDATE_ATTACK oui, mais pas avec toutes ses valeurs) ; la vraie mesure viendra du traducteur ;
- « manquant » sinon.
Une carte est jouable (au sens de cet inventaire) si elle n'a aucun item manquant.

Classement : glouton. À chaque pas, on ajoute la fonctionnalité qui rend jouables le plus de cartes (en comptant
les cartes qu'elle débloque seule), ce qui donne la courbe de couverture. Avec --popular, les cartes sont pondérées
par leur présence dans les decks (pas encore de données : à venir).

Usage : uv run --with jax python translate/inventory.py [--steps 150] [--out translate/INVENTAIRE.md]"""
import argparse
import collections
import gzip
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "engine")]
IN = os.path.join(HERE, "cache", "extract.json.gz")

# affichage, plomberie Lua et données génériques (le moteur les a sous forme de champs de carte, ou elles ne changent
# rien à la partie)
IGNORE = re.compile(
    r"^(GetID|aux\.Stringid|Duel\.Hint|Duel\.HintSelection|Duel\.ConfirmCards|:SetDescription|:SetHintTiming|"
    r"HINTMSG_.*|aux\.RegisterClientHint|EFFECT_FLAG_CLIENT_HINT|Duel\.BreakEffect|"
    r"s\..*|[a-z_]+\.[a-z_]+|math\..*|table\..*|string\..*|ipairs|pairs|type|tostring|tonumber|select|next|print|"
    r"RACE_.*|ATTRIBUTE_.*|SET_.*|Effect\.CreateEffect|:RegisterEffect|:SetType|:SetCode|:SetOperation|:SetTarget|"
    r":SetCondition|:SetCost|:SetCategory|:GetHandler|:GetHandlerPlayer|:Clone|CATEGORY_.*|Duel\.SetOperationInfo|"
    r"Duel\.SetPossibleOperationInfo|:GetFirst|:GetNext|:Iter|aux\.Next|:Filter|:FilterCount|:IsExists|:Select|"
    r":FilterSelect|:GetCount|:Merge|:AddCard|:RemoveCard|:Sub|:Clear|:KeepAlive|:DeleteGroup|Group\.CreateGroup|"
    r"Group\.FromCards|:IsContains|:Match|:GetClassCount|:GetSum|:SetLabel|:GetLabel|:SetLabelObject|"
    r":GetLabelObject|aux\.FilterBoolFunction|aux\.FilterBoolFunctionEx|aux\.FaceupFilter|aux\.AND|aux\.OR|aux\.NOT|"
    r"aux\.TargetBoolFunction|Duel\.ShuffleHand|Duel\.ShuffleDeck|:GetOwner|:GetControler|:GetCode|:IsCode|"
    r":IsSetCard|:IsRace|:IsAttribute|:GetRace|:GetAttribute|:IsType|:IsMonster|:IsSpell|:IsTrap|:IsSpellTrap|"
    r":IsFaceup|:IsFacedown|:IsControler|:IsLocation|:GetLocation|:GetAttack|:GetDefense|:GetLevel|"
    r":IsLevelBelow|:IsLevelAbove|:IsAttackBelow|:IsAttackAbove|:IsDefenseBelow|:IsDefenseAbove|"
    r"TYPE_.*|POS_.*|LOCATION_.*|PLAYER_.*|REASON_EFFECT|RESET_EVENT|RESETS_STANDARD|RESET_PHASE|PHASE_END)$")

PLUMBING_NOTE = ("plomberie et données génériques (groupes, filtres de base, affichage) : le traducteur doit les "
                 "comprendre, mais ce ne sont pas des fonctionnalités du moteur")


def load():
    with gzip.open(IN, "rt") as fh:
        d = json.load(fh)
    cards = {}
    for code, v in d.items():
        if "err" in v:
            continue
        s = set()
        for items in v["fn"].values():
            s.update(items)
        cards[int(code)] = s
    return cards


def supported_codes():
    import cards as engine_cards
    return set(engine_cards.CARDS)


def greedy(missing, steps):
    """missing : {code: set d'items manquants}. Rend [(item, cartes débloquées à ce pas, total jouables)]."""
    left = {c: set(m) for c, m in missing.items() if m}
    done = sum(1 for m in missing.values() if not m)
    out = []
    added = set()
    for _ in range(steps):
        # score d'un item : cartes dont c'est le seul manque (après les ajouts précédents) ; à égalité, nombre de
        # cartes qui l'utilisent
        alone = collections.Counter()
        uses = collections.Counter()
        for m in left.values():
            uses.update(m)
            if len(m) == 1:
                alone.update(m)
        if not uses:
            break
        best = max(uses, key=lambda k: (alone[k], uses[k]))
        added.add(best)
        freed = []
        for c in list(left):
            left[c].discard(best)
            if not left[c]:
                freed.append(c)
                del left[c]
        done += len(freed)
        out.append((best, len(freed), uses[best], done))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--out", default=os.path.join(HERE, "INVENTAIRE.md"))
    a = ap.parse_args()
    cards = load()
    sup = supported_codes()
    known = [c for c in sup if c in cards]
    ok = set().union(*(cards[c] for c in known))
    ok = {i for i in ok if not IGNORE.match(i)}
    missing = {c: {i for i in s if not IGNORE.match(i) and i not in ok} for c, s in cards.items()}
    n = len(cards)
    playable = sum(1 for m in missing.values() if not m)
    freq = collections.Counter()
    for m in missing.values():
        freq.update(m)
    nmiss = collections.Counter(min(len(m), 20) for m in missing.values())
    steps = greedy(missing, a.steps)

    L = [f"# Inventaire des fonctionnalités (généré par translate/inventory.py)\n",
         f"{n} scripts officiels ; {len(known)} cartes implémentées dans le moteur (engine/cards.py) ; "
         f"{len(ok)} items couverts (présents dans leurs scripts), "
         f"{len({i for s in cards.values() for i in s if not IGNORE.match(i)})} items non ignorés au total.\n",
         f"Cartes sans aucun item manquant : **{playable} ({100 * playable / n:.1f} %)**. Ignorés : {PLUMBING_NOTE}.\n",
         "Approximation : « couvert » = vu dans une carte implémentée ; la mesure exacte viendra du traducteur "
         "(cartes traduites et vérifiées contre l'oracle).\n",
         "## Nombre d'items manquants par carte\n", "| manquants | cartes | cumul |", "|---|---|---|"]
    cum = 0
    for k in range(21):
        cum += nmiss[k]
        L.append(f"| {k}{'+' if k == 20 else ''} | {nmiss[k]} | {100 * cum / n:.1f} % |")
    L += ["\n## Courbe gloutonne (fonctionnalité qui débloque le plus de cartes à chaque pas)\n",
          "| pas | item | débloquées | utilisé par (restantes) | jouables | % |", "|---|---|---|---|---|---|"]
    for i, (item, fr, us, done) in enumerate(steps):
        L.append(f"| {i + 1} | `{item}` | {fr} | {us} | {done} | {100 * done / n:.1f} % |")
    L += ["\n## Items manquants les plus fréquents (toutes cartes)\n", "| item | cartes |", "|---|---|"]
    for item, k in freq.most_common(200):
        L.append(f"| `{item}` | {k} |")
    with open(a.out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"{n} cartes, {len(known)} implémentées, jouables {playable} ({100 * playable / n:.1f} %) → {a.out}")
    for i, (item, fr, us, done) in enumerate(steps[:40]):
        print(f"{i + 1:3d} {item:45s} +{fr:5d} (utilisé {us:5d}) → {done:5d} ({100 * done / n:.1f} %)")


if __name__ == "__main__":
    main()
