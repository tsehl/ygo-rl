# Audit initial des refus du traducteur

## Méthode

L'échantillon reproductible comprend 80 scripts tirés au hasard parmi les 12 840 refusés par le cache du lot à 701 cartes, avec la graine 20261107. Le rapport détaillé est conservé dans `translate/cache/runs/audit-refus-20261107.json` ; il contient les codes, premières raisons et chemins des scripts. Chaque script a également été retraduit avec la version de travail v5. Aucun des 80 n'est entièrement traduit par cette version. Un échantillon aussi petit ne permet pas de conclure à une proportion précise pour l'ensemble du catalogue.

Le classement des refus indique **le premier obstacle rencontré**, pas le nombre de cartes qui deviendraient traduisibles si cet obstacle disparaissait. Plusieurs obstacles peuvent coexister ; il faut continuer l'analyse au-delà du premier et tester la carte finale.

## Premiers examens manuels

- **Garlandolf (30646525)** : la première version refuse `IsRitualSummoned`. La version v5 franchit cette condition, puis refuse `GetAttack` sur la carte source. Le script comporte ensuite un filtre de DEF comparée à cette ATK et une augmentation dépendant du nombre de destructions. Ajouter le seul type d'invocation ne suffit donc pas à traduire cette carte. Elle combine des calculs et filtres paramétrés avec des opérations de destruction et modification déjà présentes en partie.
- **Poison of the Old Man (8842266)** et **Pyramid Energy (76754619)** : premier refus `Duel.SelectOption`. Il s'agit de choix entre branches, dont le résultat est conservé dans un label et réutilisé à la résolution. Dommages, soin et modifications de stats existent déjà ; le choix, sa mémorisation et le contrôle de flux doivent être représentés correctement pour éviter une brique propre à chaque carte.
- **Petiteranodon (82946847)** : premier refus `IsReason` ; il faut distinguer destruction et raison d'effet. Le script utilise aussi `SpecialSummonStep` / `SpecialSummonComplete`, puis applique une interdiction d'attaque. La raison et le déroulement des invocations sont des règles transversales ; ajouter le premier test seul ne garantit pas la traduction complète.
- **Pilica (71175527)** : premier refus sur un `if` suivi d'instructions. Le script enchaîne une invocation par étapes, une annulation des effets du monstre et une restriction d'invocation jusqu'à la fin du tour. L'obstacle n'est pas seulement la syntaxe du `if` : la restriction de joueur doit aussi être représentée.
- **Blackwing – Aurora (4068622)** : premier refus `not c:IsType`. Le script contient aussi une sélection de deux matériaux avec contraintes de groupes, un paiement par bannissement, puis copie d'effet et changement de nom. Représenter la négation d'un type ne suffirait pas pour cette carte.

Ces exemples montrent à la fois des mécanismes à concevoir et des possibilités de composer des opérations existantes. Ils ne permettent pas encore de quantifier la part respective de chaque cause. Suite : inspecter davantage de l'échantillon, noter tous les blocages de chaque carte et identifier les représentations communes avant d'élargir l'API du moteur.

## Présence de motifs dans l'échantillon

Une recherche textuelle dans les 80 scripts trouve `IsReason` dans 17 scripts, des modifications de stats dans 14, les invocations par étapes dans 4, une API d'équipement dans 4, `SelectOption` dans 3 et une négation directe de `IsType` dans 3. Ces groupes se recoupent ; une occurrence peut être dans une fonction non atteinte, un commentaire ou un effet secondaire. Ce ne sont ni des comptes de blocages ni des gains de cartes. Ils servent à choisir les scripts à lire pour comprendre quelles sémantiques doivent réellement être composées.

## Piste technique : raison d'un déplacement

Le moteur partage déjà `OP_MOVE` pour envoyer, détruire, payer et bannir, mais ne conserve pas de raison par carte. Le noyau natif utilise un masque de raisons combinables : par exemple `REASON_EFFECT | REASON_DESTROY` pour une destruction par effet et `REASON_SUMMON | REASON_MATERIAL | REASON_RELEASE` pour une libération de matériaux d'invocation normale. `IsReason` teste l'intersection avec ce masque. Le seul champ `c == 1` de `OP_MOVE` indique actuellement la destruction, sans permettre de reconstruire toutes les autres raisons.

Une direction à vérifier est de conserver une raison par carte et de paramétrer le déplacement existant avec les bonnes raisons, plutôt que créer une instruction distincte par cause. Le contexte coût / résolution ne suffit pas à lui seul : libération, défausse, matériaux et envoi par règle ajoutent des bits différents. Il faut aussi prendre en compte les matériaux Xyz envoyés lors du départ de leur parent et la redirection des Pendules. Les tests doivent comparer séparément destruction par effet, destruction au combat, coût, libération et envoi simple contre l'oracle, ainsi que les combinaisons de bits ; les observations et le générateur doivent exposer la même information.

Les quatre premiers refus `IsReason` de l'échantillon ne sont pas équivalents : Arnis et Kryuel vérifient `REASON_BATTLE`, Petiteranodon vérifie les deux bits `REASON_EFFECT` et `REASON_DESTROY`, tandis que Big Belly Knight combine `REASON_COST` avec la nature de l'effet responsable et la zone Overlay précédente. Un masque de raison sera donc utile mais ne suffira pas à lui seul à traduire toutes ces cartes.

La raison est un argument explicite de plusieurs appels Lua (`Destroy`, `SendtoGrave`, `Remove`, `Release`) : le traducteur l'ignore actuellement. Il faudra conserver cet argument plutôt que l'inférer systématiquement de la phase coût/résolution ; le noyau natif ajoute par ailleurs certains bits, comme `DESTROY` ou `RELEASE`.

Une sonde native reproductible, `tests/reason_oracle_reference_check.py`, vérifie maintenant deux parties par cas et relève les effets effectivement résolus. Les combinaisons observées sont : destruction par effet = EFFECT + DESTROY ; envoi simple par effet = EFFECT ; libération en coût = COST + RELEASE ; destruction au combat = BATTLE + DESTROY. Cette sonde établit la référence native ; elle ne valide pas encore une implémentation JAX de `IsReason`.
