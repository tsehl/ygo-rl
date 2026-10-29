# Étape 1 — Briques d'effets (analyse EDOPro)

Données : `data/CardScripts` (scripts Lua EDOPro), `data/BabelCDB/cards.cdb`. Stats détaillées : `analysis/primitives_stats.md`, `analysis/card_db_stats.md`.

## Chiffres clés

- 14 321 cartes : 9 224 monstres (7 067 Main Deck, 2 157 Extra Deck), 2 838 magies, 2 059 pièges. 616 archétypes.
- 13 541 scripts officiels, **2,55 effets par carte en moyenne** (max 11).
- 237 fonctions moteur `Duel.*` au total. Le top 100 couvre 86 % des cartes, **le top 150 en couvre 96 %**, le top 200 en couvre 99,7 %.
- **Attention, ce chiffre ne compte que les `Duel.*`.** En couverture combinée (une carte n'est couverte que si toutes ses fonctions `Duel.*`, méthodes Card/Group, `aux.*` et codes `EFFECT_*`/`EVENT_*` le sont ; `analysis/coverage_combined.py`), il y a 1 140 éléments d'API distincts. Couverture selon le nombre d'éléments : 400 → 52 %, 600 → 79 %, 800 → 94 %, 1 000 → 99 %, 1 140 → 100 %. Le top 150 ne couvre que 10 % des cartes. Mais un élément d'API n'est pas une brique : 459 sont des requêtes `Is*`/`Get*` (des champs de filtre), 41 des types ou drapeaux d'effet, 41 de la plomberie. Les scripts restent courts : 57 lignes en médiane, 125 au 99e centile, 183 au maximum.

## Lecture

La plupart des fonctions `Duel.*` les plus utilisées ne sont pas des effets mais de la plomberie : affichage (`Hint`, `ConfirmCards`), annonces à la chaîne (`SetOperationInfo`), requêtes (`IsExistingMatchingCard`, `GetLocationCount`) et sélections (`SelectTarget`, `SelectMatchingCard`). Les vraies actions sont peu nombreuses :

- **Déplacements de carte** : `SpecialSummon`, `SendtoHand`, `Destroy`, `SendtoGrave`, `Remove`, `SendtoDeck`, `Draw`. Ce sont toutes des variantes de « déplacer N cartes de la zone A vers la zone B, pour la raison R ».
- **Modificateurs continus** (`SetCode EFFECT_*`) : ATK/DEF (`UPDATE_ATTACK`…), interdictions (`CANNOT_SPECIAL_SUMMON`…), protections (`INDESTRUCTABLE_*`), négation (`DISABLE*`).
- **Déclencheurs** (`SetCode EVENT_*`) : `FREE_CHAIN` (activable quand on veut), invoqué, envoyé au cimetière, phase, chaîne, détruit, attaque.

## Ébauche du langage de briques

```
Effet = {
  type        : Activation | Ignition | Déclenché | Rapide | Continu
  déclencheur : Événement(type, filtre)       # pour Déclenché / Rapide
  condition   : Prédicat sur l'état
  limite      : 1 fois par tour (soft / hard / nom), …
  coût        : [Action]
  cible       : Sélecteur
  résolution  : [Action]
}
Sélecteur = (joueur, zones, filtre(type, attribut, race, niveau, ATK/DEF, archétype), nombre, ciblé ?)
Action    = Déplacer(sélecteur, zone, position, raison)
          | Modifier(sélecteur, stat, valeur, durée)
          | Accorder / Interdire(sélecteur, drapeau, durée)
          | Annuler(maillon de chaîne) | Dégâts / Soin(joueur, n) | …
```

## Existant à exploiter

- Le moteur de règles complet d'EDOPro est en C++ ([ygopro-core](https://github.com/ProjectIgnis/ygopro-core)). Les scripts Lua ne décrivent que les cartes.
- [ygo-agent](https://github.com/sbl1996/ygo-agent) entraîne déjà par renforcement sur ce moteur (ygoenv = ygopro-core + envpool sur CPU, entraînement en JAX). Pas de cartes procédurales.
