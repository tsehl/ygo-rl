# Étape 3 — Cartes et decks procéduraux pour l'entraînement (conception)

Décidé le 2026-10-29. Document de travail : il évoluera avec le moteur.

## Principe

- L'agent s'entraîne **uniquement sur des cartes et des decks générés**. Le vrai jeu (vraies cartes encodées dans le DSL, vrais decks) n'entre jamais dans l'entraînement. Il sert exclusivement de banc d'évaluation.
- Le réseau lit l'arbre d'effets de chaque carte (le DSL), jamais un identifiant. Il doit donc généraliser à des cartes qu'il n'a jamais vues.
- Des cartes et des decks purement aléatoires ne suffisent pas. Le vrai Yu-Gi-Oh repose sur des synergies (chercheurs, moteurs, protections, ratios 3/2/1). Un agent entraîné sur des decks sans structure apprendrait l'avantage de cartes, mais pas à exécuter un plan, et les vrais decks seraient hors distribution par leur structure.
- Le risque inverse : un générateur trop régulier enseigne ses propres tics. On ne généralise que sur l'axe où on a mis de la diversité. D'où un mélange, et une variété structurelle délibérée dans le générateur.

## 1. Générateur de cartes

- Chaque carte = stats (niveau, ATK/DEF, attribut, type) + effets écrits avec les briques du moteur (genres, conditions, coûts, cibles, résolutions).
- **Budget de puissance** : un coût estimé par brique, ajusté selon le contexte (un coût réduit la valeur, une restriction aussi, la vitesse l'augmente). Une carte générée doit tenir dans une fourchette de budget selon sa rareté de rôle. La calibration initiale sera grossière ; on la corrigera avec les taux de victoire observés (une carte toujours jouée ou jamais jouée signale un budget faux).
- Les vraies cartes servent à calibrer les distributions (fréquence des briques, taille des programmes, statistiques de `analysis/`), pas à être copiées.

## 2. Archétypes procéduraux

- Un archétype = une étiquette (setcode, comme ygopro) + un ensemble de cartes générées qui se référencent : « ajoute 1 monstre X depuis ton deck », « invoque 1 X depuis ton cimetière », « si tu contrôles un X… ».
- Rôles typiques à tirer (sans les imposer tous) : chercheur, extendeur, boss, protection, recyclage, carte de terrain ou continue. La forme d'un archétype doit varier : nombre de cartes, rôles présents, dépendance interne forte ou faible.
- Budget appliqué à la carte **et** à l'archétype (un archétype très cohérent paie sa cohérence par des cartes individuellement plus faibles).
- Conséquence pour le moteur : champ `setcode` dans la banque et critère « archétype X » dans les filtres (jalon 4).

## 3. Decks

- 40 cartes, construites comme de vrais decks : 1 à 3 archétypes + cartes génériques (équivalents des magies/pièges universels), ratios 3/2/1.
- Mélange d'entraînement (proportions à régler) :
  - majorité de decks structurés ;
  - une part de decks aléatoires, pour la robustesse et pour que l'agent ne suppose jamais qu'un combo existe forcément.

## 4. Sélection des decks par évolution (plus tard)

- Une fois qu'un premier agent tient debout : population de decks, les decks qui gagnent survivent et mutent (échange de cartes, d'archétypes, de ratios), sur le modèle de la league d'AlphaStar.
- But : faire émerger un métagame au lieu d'un échantillon uniforme, ce qui est le vrai moteur de stratégies émergentes.
- Garde-fou : garder de la diversité dans la population (sinon effondrement sur un seul deck dominant).

## 5. Évaluation

- Banc fixe de vrais decks avec de vraies cartes, jamais vus à l'entraînement, joués contre des adversaires de référence (agent précédent, politique aléatoire, éventuellement heuristiques simples).
- Mesures : taux de victoire sur ce banc, écart entre decks générés et vrais decks (mesure directe de la généralisation), et ablations (avec / sans structure d'archétype) pour vérifier que la structure aide plutôt qu'elle n'enferme.

## Questions ouvertes

- Formule du budget de puissance et sa calibration.
- Proportions du mélange structuré / aléatoire.
- Quand introduire l'évolution des decks (dès le début, ou après un agent de base).
- Taille du banc d'évaluation, et choix des vrais decks qui le composent (limités aux cartes que le moteur sait jouer à ce moment-là).
