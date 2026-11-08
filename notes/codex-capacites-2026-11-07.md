# Registres par nom et candidats d’activation — 7 novembre 2026

Deux limites provoquaient des erreurs de règles. Le registre de quatre noms écrasait son dernier élément lorsqu’un cinquième nom était consommé ou interdit. `legal` ne testait que les 24 premiers candidats syntaxiques : une main remplie de magies interdites pouvait donc masquer l’effet légal d’un monstre. `trig_eval` utilisait également cette troncature.

## Implémentation

Les registres sont dimensionnés à partir des emplacements du duel : `N*E` clés de compteur et `N` noms interdits par joueur. Ce sont des majorants du nombre de clés présentes dans ce moteur, où les identifiants de carte ne changent pas pendant une partie. Ajouter une clé déjà présente ne consomme plus une nouvelle entrée. Le remboursement OATH laisse une case libre réutilisable.

Les candidats et les déclencheurs passent par blocs de 24, avec arrêt dès que tous les candidats ont été testés. L’arène accepte au plus 192 emplacements, borne contrôlée avant allocation ; elle couvre les deux decks de 60 cartes et Extra Decks de 15. Les 16 blocs maximaux évitent une matrice candidats × cartes non bornée. Les petites fenêtres gardent un seul bloc. Les programmes et mutations des compteurs inactifs utilisent une borne de boucle nulle sous vectorisation.

L’état et les compteurs restent de forme fixe pour une taille de duel donnée. Aucune modification des observations, du modèle ou des formats de banque.

## Mesures intermédiaires

CPU, quatre vCPU effectifs, JAX 0.12.0.1, banque passée en argument, B=64, Extra Deck 15, 20 appels par répétition. Les compilations et un échauffement sont exclus. Les sources de référence sont celles du lot pièges continus. Banque identique et empreinte de l’état sémantique final identique pour les trois répétitions, après normalisation des registres dont la taille change.

- Registres élargis et activations par blocs : médianes 315 → 305 décisions/s (−3,2 %).
- Variante de recherche limitée au préfixe usuel, suivie d’une recherche conditionnelle dans le reste : 321 → 282 (−12,1 %). Variante écartée.

La mesure finale inclut également le parcours complet des déclencheurs : 323 / 323 / 323 → 280 / 294 / 298 décisions/s, soit des médianes de 323 → 294 (−9,0 %). L’empreinte sémantique finale est identique pour les six mesures. Une variante déroulant directement le premier bloc de déclencheurs a également été écartée : elle n’améliore pas le débit. Pas de mesure GPU ni d’entraînement CPU.

## Vérifications

Les échecs initiaux contre l’oracle sont conservés : réactivation d’un nom oublié après cinq consommations ; effet du chercheur absent du masque après accumulation d’une grande main verrouillée.

Les 78 tests unitaires passent, dont cinq nouveaux scénarios : cinq noms distincts, insertions idempotentes, 35 activations proposées, refus explicite d’une arène trop grande et déclencheur légal au-delà d’un premier bloc verrouillé. Les deux parties de cinq compteurs distincts passent contre l’oracle. Les deux parties d’interdictions passent également, avec cinq noms persistants et jusqu’à 25 magies en main candidates. Le scénario vide ensuite la main par effet pour éviter l’énumération combinatoire des choix de défausse dans le pilote oracle.

Les deux variantes de compteur passent leurs quatre parties (24 réponses pour chacune ; plafond observé de 1 activation par tour sans remboursement et de 3 avec OATH). Les quatre parties de réponses à l’effet initial passent. Le lot mixte de 8 parties × 12 décisions est identique entre `step_batch` et `vmap(step)`, tous les champs et masques comparés.

Les dix parties générées passent (banques 0–1, cinq parties par banque, main 80 / Extra Deck 10, sélection focus). La régression historique complète reste en cours ; son résultat doit être ajouté avant le commit moteur.

Preuves locales ignorées par Git : `translate/cache/runs/verification-capacites-20261107/`, dont les empreintes des sources au début de la validation.

Les dix premiers modes historiques terminés avant interruption ont été conservés après comparaison des empreintes des sources. Les huit autres et obs/vmap/sb sont relancés avec le même moteur ; le résumé final regroupera les dix journaux conservés et les nouveaux.

La régression historique regroupée est terminée : **1 640 / 1 640 parties identiques** dans les 18 modes ; `obs_check`, `vmap_check` et `sb_check` passent. Les 61 empreintes de référence ont été contrôlées à la reprise : seul `tests/lock_capacity_oracle_check.py` diffère, car ce test ciblé a été renforcé après l'empreinte initiale ; les sources moteur, générateur, oracle et scripts de régression étaient inchangées. Les dix modes conservés et les huit modes relancés sont archivés avec les trois contrôles dans `translate/cache/runs/verification-capacites-20261107/regression-complete/`.
