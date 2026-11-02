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

## Implémentation, étape 1 : générateur minimal et robustesse (2026-11-02)

Ordre de travail retenu (2026-11-02) : 1) générateur minimal et fuzz du moteur ; 2) compilateur vers Lua pour
vérifier les cartes générées contre ygopro-core ; 3) encodage de l'arbre d'effets, réseau, PPO, premier
entraînement court sur des decks aléatoires (test de bout en bout) ; 4) archétypes et decks structurés ;
5) évolution des decks et banc d'évaluation.

### Générateur (`gen/cardgen.py`)

- Une carte est d'abord une **spécification** (dict Python), traduite ensuite en briques (`bricks`, puis
  `cards.assemble` et `core.make_bank` dans `to_bank`). L'étape 2 ajoutera une seconde traduction, vers Lua, à
  partir de la même spécification : c'est pour ça que le générateur ne tire pas directement des instructions.
- Effet = genre (`act` pour les magies / pièges, `ignition`, `trigger`, `quick`, `cont`), événements, une fois par
  tour, conditions, coût, puis 1 ou 2 clauses.
- Clause = **verbe × sélecteur** (`destroy`, `banish`, `bounce`, `tograve`, `tohand`, `ss`, `atk`, `disable`, sur
  `target` : cible à l'activation ; `all` : tout le filtre ; `pick` : choix à la résolution ; `event` : cartes de
  l'événement ; `self` : la carte elle-même), ou action sans carte (`draw`, `damage`, `recover`, `burn_per`, `negate`,
  `chain_destroy`). Une seconde clause (« puis ») ne s'applique que si la première a touché une carte.
- Les gabarits de briques sont ceux des vraies cartes vérifiées (Pot of Greed, MST, Trap Hole, Monster Reborn,
  Emergency Call, Back-Up Rider, Effect Veiler, Magic Jammer, Ghost Ogre…) ; seules leurs combinaisons sont
  nouvelles. Registres : coût dans 2-3, sélection de la première clause dans 0-1 (la cible reste dans 1 jusqu'à la
  résolution), refiltrage à la résolution dans 2.
- Cadres : monstres normaux (dont Syntoniseurs et pendules normaux), monstres à effet, rituels avec leur sort,
  magies normales / jeu-rapide / continues, sorts de fusion (gabarit de Polymerization, éventuellement limité à un
  archétype), pièges normaux / continus / contre-pièges, monstres fusion / synchro / Xyz / Link (recettes par
  filtres ; fusion parfois par nom de carte du deck principal).
- Garde-fous : un effet d'ignition ou rapide sans coût qui consomme une carte est toujours « une fois par tour »
  (sinon boucle infinie offerte à l'agent) ; un effet doit tenir dans les tableaux du moteur (`fits` : PT, P, NF).
- **Budget grossier** (`value`, en cartes d'avantage) : valeur par clause × vitesse (rapide ×1,25, répétable ×1,3)
  − coût − conditions ; une carte dépasse rarement `CARD_CAP` = 2,6 ; les stats d'un monstre à effet perdent
  900 ATK+DEF par point de valeur (référence : `STAT_BUDGET` par niveau). À recalibrer sur les taux de victoire.
- Archétypes : pour l'instant, 30 % des monstres reçoivent un setcode tiré parmi 8 (`SETS`), sans structure ;
  l'étape 4 fera de vrais archétypes. Codes des cartes générées : `CODE0` = 600 000 000 et plus.
- `to_bank` complète `tprog` à 2 × PT instructions : toutes les banques procédurales ont les mêmes formes, une seule
  compilation sert pour toutes les graines (la banque est un argument de la fonction compilée).
- `python gen/cardgen.py --seed S` affiche une banque en clair (`describe`).

### Robustesse (`gen/fuzz.py`)

- Parties aléatoires (actions légales uniformes, `vmap(step)`), B parties par graine de banque, decks de 40 cartes
  + Extra Deck de 15 tirés sans structure (`random_decks`).
- Après chaque décision, `check` vérifie : une action légale existe ; aucune case de terrain occupée deux fois ;
  zones ordonnées contiguës ; matériaux Xyz rattachés à un Xyz du même contrôleur (exception : pendant l'invocation
  Xyz, les matériaux sont attachés avant le choix de la case et de la position) ; pas de monstre de l'Extra Deck en
  main ou dans le deck ; LP > 0 si la partie continue ; chaîne dans [0, CH] ; positions valides ; pas de décision
  interne (D_RUN, D_TRIG) rendue à la politique ; contrôleur et propriétaire valides.
- Avertissements (limites des tables, comportement alors approché) : KM, TM, EXK, tables par nom KO pleines,
  plus de KC effets candidats.
- Le fuzz ne trouve que les erreurs de structure (plantages, états incohérents, blocages). Les erreurs de règle
  (le moteur joue autrement qu'ygopro) demandent l'étape 2 : compiler les cartes générées en Lua et lancer
  `compare.py` dessus.

Résultats (2026-11-02, CPU, 32 parties × 800 décisions par graine, banques de 400 + 80 cartes) :
- graines 0 à 9 : environ 50 tours par partie, au plus ~25 décisions dans un tour, 16 à 25 activations d'effets par
  partie, aucun avertissement ;
- un seul bug trouvé (graine 7) : un effet qui invoque spécialement en défense un monstre Link (Link remis en jeu
  depuis le cimetière) laissait la décision de position sans action légale. ygopro place toujours un Link en attaque,
  quelle que soit la position demandée (`special_summon_step`, operations.cpp). Corrigé dans `legal`.
- Durée : ~350 s par graine sur les 4 cœurs (~750 s quand la non-régression tourne en même temps).
