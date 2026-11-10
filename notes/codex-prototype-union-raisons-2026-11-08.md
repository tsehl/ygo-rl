# Prototype : union de raisons

La disjonction de filtres `IsReason(A) or IsReason(B)` est compilée en une union des bits, réutilisant le champ et l'opération de filtre existants. Le test ciblé vérifie l'union, sa simplification lorsqu'elle est combinée à une autre condition, ainsi que le refus des disjonctions mélangeant une raison et un type, ou utilisant une négation non représentable.

Cette copie inclut les travaux en cours sur l'historique d'invocation, les raisons, les compteurs OATH et les exclusions de types. Un scénario contre l'oracle reste à exécuter pour l'union ; aucune intégration ni validation complète n'est encore faite. Sources locales : `translate/cache/runs/prototype-union-raisons/sources/`.

Le scénario généré contre l'oracle passe **4 / 4 parties** : le filtre `DESTROY | COST` est vrai après destruction (dix effets résolus par partie) et faux après un simple envoi par effet (zéro effet). L'empreinte du moteur, du générateur et du modèle est identique à celle du prototype de type ayant passé 112 tests unitaires, 4 parties de fusion et 10 parties générées ; seul le traducteur diffère pour cette union. La série unitaire complète et une campagne de scripts officiels restent à relancer sur le lot combiné.

Un troisième cas est ajouté au scénario : une libération en coût doit aussi satisfaire l'union `DESTROY | COST`. Il est relancé pour vérifier les deux branches positives, en plus du cas négatif.

Sur le traducteur combiné, la série complète passe **112 tests unitaires**. La campagne de 30 parties sur les cartes officielles est lancée ; le scénario complémentaire de libération en coût reste en cours.

Le scénario étendu passe **6 / 6 parties** : les deux raisons possibles sont réellement exercées. La destruction et la libération en coût permettent la recherche ; l'envoi simple ne la permet pas. Journal : `translate/cache/runs/prototype-union-raisons/oracle-cost.log`.

Le catalogue reconstruit sur les sources combinées contient **784 / 13 541 scripts traduits** : 67 gagnés par rapport aux 717 de l'historique, aucune perte et aucune erreur interne. La liste reproductible des codes est dans `translate/cache/runs/prototype-union-raisons/comparaison-catalogue.json`. La campagne oracle est encore en cours : ce décompte statique n'est pas une preuve que les effets de ces scripts ont été exercés.

Parmi les 67 scripts nouvellement traduits, les premiers refus précédents étaient `condition : c:IsReason` pour 62, `filtre : not c:IsType` pour 3 et `filtre : c:IsReason` pour 2. Ce classement décrit l'ancien premier obstacle, pas une attribution causale indépendante de chaque changement.

Campagne officielle des graines 286–315 : **30 / 30 parties identiques**. Parmi les 365 cartes présentes, 191 ont été invoquées ou ont eu au moins un effet résolu. Les 104 fichiers archivés du prototype ont été revérifiés sans différence après la campagne. Rapport et journal : `translate/cache/runs/prototype-union-raisons/campagne-286-315/`.

La comparaison intermédiaire donne **780 scripts** avec raisons/OATH, puis **784** avec le lot combiné : quatre scripts supplémentaires, aucun perdu, zéro erreur interne. Les premiers refus de ces quatre scripts concernaient tous `not c:IsType`. Ce constat n'isole pas nécessairement tous les changements rencontrés ensuite dans leurs scripts.

Parmi les 67 scripts gagnés, 34 étaient présents dans la campagne ; 28 ont été invoqués ou ont eu un effet résolu, et 5 ont effectivement résolu un effet. L'inspection des programmes et des descriptions relie quatre de ces résolutions à une condition `IsReason` et une à un filtre d'exclusion de type exécuté lors de l'activation. Cela mesure des usages concrets des nouvelles capacités, mais ne valide pas les autres effets des cartes restantes.

La graine 261 du lot raisons/OATH a été rejouée avec les mêmes cartes imposées sur le lot combiné : elle reproduit la divergence. Malgré les 30 parties conformes de la campagne 286–315, ce lot n'est donc pas validé. Le diagnostic montre que l'oracle propose l'effet déclenché de Number 37 à la déclaration de l'attaque alors que le moteur JAX passe au combat. Une correction isolée de ce point d'événement est en test dans `/tmp/ygo-seed261-debug` ; journal de l'échec combiné : `translate/cache/runs/prototype-union-raisons/rejeu-graine-261-echec.log`.

Diagnostic de la graine 261 : l'oracle est arrêté sur une proposition de chaîne pour Number 37 après déclaration d'attaque ; JAX a déjà terminé le combat. Sur l'état divergent, les tests d'activabilité (`test_ok` et `act_ok`) de cet effet sont vrais : le problème n'est donc pas son coût ni son compteur. La déclaration ouvre une fenêtre rapide sans appeler le point de déclenchement ; une correction isolée est en cours de rejeu et devra aussi être mesurée et testée contre les autres modes de combat.

Le candidat isolé qui traite les déclencheurs immédiatement après déclaration de l'attaque rejoue correctement la graine 261 jusqu'à la fin : **1 / 1**, onze cartes sur seize présentes invoquées ou avec effet résolu. Il est reporté dans une copie combinée distincte (`/tmp/ygo-combined-attack-workbench`, sources et empreintes archivées dans `translate/cache/runs/prototype-attaque-combine/`). Le rejeu sur cette copie, des scénarios de chaînes/Knightmare et une mesure CPU sont lancés. La correction n'est pas encore intégrée à la branche.

Sur le candidat de fenêtre d'attaque, les deux tests unitaires « attaque directe » et « attaque avec cible » passent, et les comparaisons historiques ciblées passent **10 / 10 chaînes** et **10 / 10 Knightmare**. Journaux dans `translate/cache/runs/prototype-attaque-combine/`. La validation complète et la mesure CPU restent nécessaires.

Sur la copie combinée figée intégrant le point de déclenchement d'attaque : **114 tests unitaires** passent, et la graine 261 rejouée avec les cartes exactes est identique jusqu'à la fin (281 décisions, 11 cartes sur 16 invoquées ou avec effet résolu). Les 106 sources archivées ont été contrôlées sans différence. Journaux et rapport : `translate/cache/runs/prototype-attaque-combine/`. Les 30 parties officielles et les parties générées doivent être rejouées sur cette version finale ; la régression historique est en cours.

Le scénario natif d'attaque directe passe **2 / 2**, avec huit résolutions effectives du déclencheur dans chaque partie. Il complète la graine 261, qui couvre l'attaque avec cible. Journal : `translate/cache/runs/prototype-attaque-combine/attaque-directe-oracle.log`.

Première mesure CPU de la correction d'attaque : médianes **404 → 350 décisions/s** avec banque passée en argument et table OATH allouée. Les résultats de la version modifiée remontent au fil des trois répétitions (310 / 350 / 364), comme lors de mesures sensibles à l'ordre ; une mesure inversée est en cours. Une variante distincte regroupe le point de déclenchement en une seule occurrence après l'action, avec une boucle bornée à zéro ou un passage ; les deux tests unitaires d'attaque passent, mais son rejeu natif et sa mesure restent en cours. Elle n'est pas intégrée.

La campagne officielle sur le lot avec correction de la fenêtre d'attaque passe **30 / 30** ; 191 / 365 cartes présentes ont été invoquées ou ont eu un effet résolu. Les 106 fichiers archivés sont inchangés après la campagne. Les parties générées et la régression historique continuent.

Mesure CPU répétée avec l'ordre inversé : **361 / 375 / 400 décisions/s après**, puis **367 / 399 / 387 avant**, médianes 375 contre 387. La différence est plus faible que dans la première mesure et la dispersion reste importante. Une variante isolée à point unique est en cours d'évaluation ; elle ne remplace pas le candidat testé.

La variante à point unique passe aussi le rejeu natif de la graine 261 (**1 / 1**, onze cartes sur seize invoquées ou avec effet résolu). L'attaque directe et la mesure comparative sont en cours ; elle n'a pas remplacé le candidat en régression.

Les deux banques procédurales ont été rejouées sur le lot avec correction d'attaque : **10 / 10 parties identiques**, sources archivées inchangées. Le journal est conservé dans `translate/cache/runs/prototype-attaque-combine/generated-10-games.log`.

La variante à point unique passe également **2 / 2** parties natives avec attaque directe et huit déclenchements résolus par partie. La mesure comparative et la régression complète nécessaires à une éventuelle adoption restent en cours ou à refaire.

Première comparaison CPU du candidat à point unique contre le candidat actuel : **404 / 410 / 406 décisions/s actuelles**, puis **327 / 353 / 357 à point unique**, avec les mêmes états communs. La variante ne montre pas de gain dans cet ordre ; la série inversée est lancée car les mesures précédentes dépendaient fortement de l'ordre. Aucun changement n'a été porté dans la copie en régression.

Mesure inversée du point unique : **413 / 403 / 362 décisions/s à point unique**, puis **381 / 394 / 398 pour le candidat actuel**, états communs identiques. L'écart s'inverse selon l'ordre et ne démontre pas un avantage reproductible. La variante à point unique est donc écartée à ce stade ; la régression continue sur le candidat initial, sans changement de ses sources.

Reprise après interruption de la VM : les huit modes déjà terminés ont été conservés et les dix autres relancés avec les mêmes sources, graines et commandes. Les **18 modes totalisent 1640 / 1640 parties identiques** ; observation (**800 permutations**), `vmap` et `step_batch` passent. Les 106 empreintes des sources du candidat ont été revérifiées avant intégration ; journaux : `translate/cache/runs/prototype-attaque-combine/regression-reprise/`. La campagne officielle finale est à **30 / 30**, la campagne générée à **10 / 10**, et les 114 tests unitaires passent. Les mesures CPU restent sensibles à l'ordre (404 → 350 puis 387 → 375 décisions/s) ; aucune conclusion GPU. La divergence de relation d'effet découverte ensuite préexistait à ce lot et reste suivie dans `notes/codex-relation-effet-2026-11-09.md`.
