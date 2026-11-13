# Traducteur Lua → briques

But : couvrir tout le jeu en traduisant les 13 541 scripts officiels (`data/CardScripts/official`) vers les briques du
moteur, et vérifier chaque carte traduite contre l'oracle. Les vraies cartes restent un benchmark : elles ne servent
qu'à mesurer ce que le moteur sait jouer, jamais à l'entraînement.

## État et reprise (10 novembre 2026)

La branche `codex` intègre **784 scripts sur 13 541**. Le dernier lot intégré est validé par 1 640 parties historiques, les contrôles d'observation et de vectorisation, 30 parties officielles et 10 parties générées. Une partie identique peut résoudre un effet sans que toutes ses branches soient exercées : les campagnes enregistrent séparément présence, activation et résolution.

Le moteur gère notamment les procédures de fusion, rituel, synchro, Xyz et Link prises en charge par le traducteur, les historiques d'invocation, les raisons de déplacement composables, les exclusions de type, les compteurs OATH et les déclencheurs à la déclaration d'attaque. Les règles détaillées et leurs limites sont dans les tests et dans les notes du dépôt.

La branche `codex-relation-wip` sauvegarde la correction expérimentale de la relation entre une carte et un effet. Elle couvre la source déplacée en coût, une cible unique et la création de la relation pendant le ciblage. Les cibles multiples et les `GetFirst` sur groupes potentiellement multiples sont refusés par le traducteur ; les invocations groupées restent expérimentales. La validation de ce sous-ensemble est en cours : ne pas compter sa couverture comme intégrée avant son report sur `codex`.

Pour reprendre : exécuter les tests ciblés dans `tests/`, `engine/regress.sh`, puis `translate/validate.py --games 30 --k 8`. Comparer les signatures de sources des campagnes avant de réutiliser leurs résultats. Après intégration de la relation, mesurer `main` contre `codex` au repos avec `bench/engine_bench.py --b 64 --calls 20 --extra 15`, trois alternances dans chaque ordre ; vérifier toute perte supérieure à 10 % avant un nouveau lot. Ensuite classer les refus avec `translate/translate.py --top` et limiter chaque lot à une fonctionnalité livrable rapidement.

## Fichiers

- `extract.py`, `inventory.py` → `INVENTAIRE.md` (étape 1, 2026-11-05) : appels d'API et constantes de chaque script,
  approximation de la couverture.
- `translate.py` : un interpréteur symbolique sur l'arbre de luaparser. Il lit `initial_effect`, puis les fonctions
  de condition, de coût, de cible et d'opération, et les traduit motif par motif. Tout ce qui sort des motifs connus
  lève `Unsupported(raison)`. `translate.py --top 60` classe les raisons par nombre de cartes : ce classement dit quoi
  ajouter ensuite, au traducteur ou au moteur. `translate(code)` renvoie `(effets, notes, procédure, descriptions)` :
  - `procédure` vaut `("synchro", F1, F2)`, `("xyz", F, ct, niveau)`, `("link", F, min, max)` ou None ;
  - `descriptions` donne l'`aux.Stringid` de chaque effet, celle que l'oracle affiche dans ses options.
- `validate.py` : vérification carte par carte (voir plus bas). Cache `cache/translated.pkl`, refait quand
  `translate.py`, `cards.py` ou `core.py` changent. Résultats cumulés dans `cache/validated.json`. Le dossier `cache/`
  n'est pas versionné.

## Couverture historique (2026-11-05)

441 / 13 541 scripts traduits et jouables par `validate.py` (3,3 %) :
- 420 du deck principal ;
- 21 de l'Extra Deck : synchro 6, Xyz 12, Link 3.

La fusion est partiellement traduite (`Fusion.AddProcMix` et `MixN` à quantités fixes ; voir plus bas).

Le tableau ci-dessous donne les premières raisons de refus, avec un comptage antérieur aux derniers ajouts.

| raison | cartes | remarque |
|---|---|---|
| EnableReviveLimit sans procédure reconnue | 372 | |
| Fusion.AddProcMix | 344 | |
| SPSUMMON_PROC (invocation spéciale par procédure) | 336 | |
| procédure d'équipement | 231 | |
| GetLocationCount hors comparaison | 168 | surtout `local ft=…` |
| GlobalCheck | 167 | |
| SetLabel sur un effet | 147 | |
| EnableCounterPermit | 147 | |
| SetUniqueOnField | 142 | |
| propriétés 0x14000 | 140 | |
| AddCustomActivityCounter | 140 | |
| IsReason | 138 | |
| condition Link sur le groupe de matériaux | 128 | |
| EVENT_PHASE | 124 | |

Les sélections de taille variable (`Select*(mn, mx)` avec mn < mx) sont traduites depuis le 2026-11-05 (voir plus
bas). Elles n'ont débloqué aucune carte à elles seules : chaque carte ne compte que sa première raison de refus, et
les ~378 scripts concernés butent tous sur un autre manque (IsReason passe de 138 à 177 cartes, par exemple).

## Vérification (`validate.py`)

Une seule banque contient les cartes traduites, les monstres normaux et des Syntoniseurs normaux : elle n'est compilée
qu'une fois. Ensuite, on joue des parties aléatoires avec la boucle de `engine/compare.py`, en mode `tr` :
- chaque joueur reçoit K cartes traduites tirées au hasard, à 2 exemplaires, complétées par des monstres normaux, plus
  6 Syntoniseurs si une carte synchro a été tirée ;
- les deux joueurs reçoivent le même Extra Deck, l'union des monstres de l'Extra Deck tirés (15 au plus), parce que
  `new_game` impose deux Extra Decks de même taille.

Les campagnes distinguent la présence d’une carte et l’exécution de ses effets (voir « Validation traçable »). Avec `--bisect`, chaque partie
divergente est rejouée avec une seule carte traduite par deck, pour isoler la fautive. Deux options servent au
débogage :
- `--chosen a,b;c,d` rejoue une partie avec ces cartes imposées ;
- `--nosave` désactive l’enregistrement de la campagne.

`compare.translate` fait correspondre les options de l'oracle pour les cartes traduites (« activer l'effet n de
cette carte ») grâce aux descriptions (`compare.DESC`). Pour la description 221 (déclencheur dans la chaîne), il exige désormais un unique effet
déclenché optionnel dans les métadonnées statiques ; une ambiguïté reste non concluante.

## Règles du moteur corrigées ou ajoutées pendant la traduction

- OPT_NAMEK : limite « une fois par tour » partagée par nom (count limit à code).
- TRIG_WHEN : déclencheurs « quand » (1<<30), qui manquent leur moment si autre chose s'est passé (`tew`, `brk()`).
- TRIG_FIELD : déclencheurs de type FIELD depuis une zone (1<<29), avec `teg` et `in_range`.
- Effets d'ignition et effets rapides depuis la main, le cimetière ou une zone magie / piège (`inr`). Les zones
  pendule sont lues comme la zone magie / piège.
- Champs du filtre : F_CODE1 négatif (« pas ce nom »). Un filtre d'invocation spéciale exige TYPE_MONSTER.
- Damage step : un déclencheur FIELD TRIGGER_O sans la propriété DAMAGE_STEP ne réagit qu'aux événements de combat
  (EV_BATTLE_DESTROYED, EV_PRE_DMG). C'est la règle d'`effect::is_activateable` dans ygopro-core : les codes 1132-1149
  sont exemptés, de même que SINGLE TRIGGER_O, TRIGGER_F et FLIP.
- Sélection de taille variable : `CHOOSE(..., mn=k)` (bit 9 de l'argument) entre en D_SELECT avec `smin = min(k,
  besoin)` ; dès que `picks >= smin`, A_PASS termine la sélection (`sel_finish`) ; la sélection n'est alors plus
  jouée d'office (`forced_m`). Le traducteur prend au moins une carte (mn = 0 → 1) : le pilote de l'oracle ne propose
  l'ensemble vide qu'avec une annulation possible. `compare.translate` ajoute A_PASS aux ensembles plus petits que le
  maximum.
- `cards.act(opt=)` transmet la limite d'activation. Avant ce correctif, une magie ou un piège avec
  EFFECT_COUNT_CODE_OATH pouvait s'activer deux fois par tour.

## Validations

| run | parties identiques | cause des divergences |
|---|---|---|
| 2 | 7 / 12 | |
| 4 | 10 / 12 | The Cheerful Coffin (DiscardHand(1,3) traduit en « exactement 3 », désormais refusé) ; Evoltile Lagosucho (option de l'oracle mal traduite, corrigé par DESC) |
| 5 (graines 12-41) | 28 / 30 | deux cartes à deux effets ACTIVATE sans description (Hidden Soldiers…) : `compare.translate` prend désormais l'effet légal dans le moteur |
| 6 (graines 42-71) | 29 / 30 | Gogogo Goliath : la correspondance par description s'appliquait aussi aux attaques (corrigé ; la partie rejouée est identique) |

Divergences anciennes non élucidées :
- Constellar Alrescha ;
- un monstre absent de la zone monstre (val3, partie 8) ;
- une sélection de Gren Maju (val2, partie 9).

Usage : `uv run --with luaparser --with jax python translate/validate.py --games 30 --k 8 --seed0 12 --bisect`

Débit (CPU, `bench/engine_bench.py ../engine --b 64 --calls 20 --extra 15`, machine au repos, après les règles
ci-dessus et la sélection de taille variable) : 301 décisions/s, contre 258-267 avant. Pas de coût mesurable.

## Reprise (pour la session suivante)

Pistes, par ordre de rendement probable :
1. Fusion : sous-ensemble fixe et substituts simples réalisés (bilan du 2026-11-06 ci-dessous).
   Restent les quantités variables, alternatives complexes et conditions non représentées.
2. SPSUMMON_PROC : procédures de main sans coût et à conditions représentables réalisées ;
   restent les sélections, coûts, opérations et restrictions complexes.
3. IsReason (177), EVENT_PHASE (129, déclencheurs de phase : fonction du moteur à écrire, voir
   `process_phase_event` dans `oracle/ygopro-core/processor.cpp`), équipement (231).
4. Divergences anciennes non élucidées (voir plus haut) : les rejouer avec `--chosen` quand elles réapparaissent.

Méthode qui a marché :
- `translate.py --top 40` après chaque ajout, pour voir ce qui se débloque. Chaque carte n'affiche que sa première
  raison de refus : un ajout peut ne rien débloquer et seulement déplacer les cartes vers la raison suivante.
- `validate.py --games 30 --k 8 --seed0 S --bisect` sur des graines neuves à chaque fois (12-41 et 42-71 déjà
  jouées). La première exécution après un changement de `translate.py`, `cards.py` ou `core.py` retraduit tout, ce
  qui prend ~4 min.
- Rejouer une partie divergente : `--codes X --seed0 G --games 1 --k 1 --nosave` pour une carte seule (c'est ce que
  fait la bissection), ou `--chosen` avec les cartes des deux decks.
- Non-régression sans bloquer le travail : copier le dépôt dans le scratchpad, avec des liens symboliques vers
  `oracle/ygopro-core` et `data`, puis lancer `JOBS=3 engine/regress.sh LOGS` depuis la copie (~45 min). Garder
  cette copie figée ; toute nouvelle version du moteur doit être vérifiée dans une nouvelle campagne.

## Validation traçable (branche codex)

`validate.py` renvoie **0** si toutes les parties sont identiques, **1** pour une divergence,
**2** si la campagne contient une erreur du pilote ou une correspondance d'effet ambiguë.
Un changement des sources ou des données pendant la campagne l'invalide également (code 2).
`compare.py` renvoie un code non nul dès qu'une partie diverge ; `engine/regress.sh` propage
les échecs tout en exécutant les autres contrôles. `--games 0` n'est pas une validation.

Chaque campagne est indépendante et enregistrée dans `translate/cache/runs/` :
- un JSON final avec commit, empreintes des fichiers Python, commits et modifications des
  dépendances, empreintes de la bibliothèque oracle et de la base de cartes, versions Python/JAX,
  arguments, graines, decks exacts, choix joués et messages du pilote ;
- un journal JSONL écrit après chaque partie, pour conserver les parties terminées si le processus
  est interrompu. L'absence de JSON final indique une campagne incomplète.

Les anciennes statistiques de `validated.json` sont conservées sur disque mais ne sont plus
fusionnées avec les nouvelles campagnes. `--codes` enregistre également ses résultats ;
`--nosave` désactive l'enregistrement. Ne pas comparer des compteurs issus de signatures différentes
comme s'ils décrivaient le même programme.

La présence dans un deck, une tentative d'invocation, une invocation réussie, une activation,
une annulation et une résolution non annulée sont distinctes. Les messages de chaîne de l'oracle
fournissent la couverture par code de carte et description d'effet ; ils ne prouvent pas qu'un effet
continu a été exercé, ni qu'une opération a changé l'état (elle peut résoudre sans effet).
Les compteurs agrégés d'exécution portent uniquement sur les parties identiques. Le détail par effet
reste dans `games[].result.coverage`. Une carte invoquée n'a pas nécessairement exercé tous ses effets.
Les cartes retirées par la limite d'Extra Deck ne sont pas comptées comme présentes.

Une description ambiguë n'est plus départagée à partir des actions légales du moteur testé.
L'inspecteur natif lit le code d'événement enregistré sur l'effet proposé par l'oracle ; si cela
ne suffit pas à l'identifier, le résultat reste « inconclusive ».

`--bisect` reste un **rejeu carte par carte**, avec d'autres decks, et non une réduction du scénario
original. Il peut manquer un défaut d'interaction. Le scénario original et sa trace sont conservés ;
un réducteur préservant la divergence reste à implémenter. L'absence de divergence dans les rejouages
isolés ne résout pas la divergence originale.

Pour la non-régression, utiliser une copie **figée** du code avec des liens vers les dépendances
épinglées. Ne pas y recopier des fichiers pendant la campagne : recommencer une campagne pour
vérifier une nouvelle version. `YGO_PYTHON=/chemin/vers/.venv/bin/python` permet d'utiliser les
versions déjà installées plutôt qu'une résolution ad hoc de JAX.

## Fusion : premier sous-ensemble

Le traducteur reconnaît `Fusion.AddProcMix` et `Fusion.AddProcMixN` : deux à quatre matériaux,
par code ou par filtre représentable dans `MAT_FIELDS`. Les quantités de `MixN` doivent être fixes
et positives. L'effet par défaut `Fusion.RegisterSummonEff(c)` (Polymerization) est également traduit ;
les variantes avec options sont refusées.

Les substituts de matériaux nommés sont pris en charge : `sub=true` autorise un seul remplacement,
uniquement aux emplacements donnés par un argument numérique dans le script. Un filtre fonctionnel
`Card.IsCode` ne donne pas cette permission. Les recettes à matériaux variables, tables d'alternatives et filtres
hors des champs du moteur restent refusés. L'option `insf` concerne les invocations sans matériaux,
comme Instant Fusion, dont les effets ne sont pas pris en charge par cette traduction.
Une recette reconnue ne suffit pas à rendre toute la carte traduisible : ses autres effets doivent l'être.

Contrôles ciblés :

```sh
.venv/bin/python -m unittest discover -s tests -v
YGO_JAX_CACHE=/workspace/.cache/ygo-jax .venv/bin/python tests/fusion_oracle_check.py
```

Les cartes fictives de `tests/fixtures/fusion/` isolent les recettes (matériaux distincts, répétés,
filtres qui se recouvrent, matériau absent ou quantité insuffisante). Elles ne servent jamais à
l'entraînement. Le contrôle vérifie les parties contre l'oracle et exige que les fusions positives
aient effectivement été invoquées, et que Polymerization n'ait pas pu résoudre dans les cas négatifs.

### Correspondance des descriptions corrigée

Le codage de `aux.Stringid(code, index)` doit suivre la version épinglée de
`data/CardScripts/utility.lua` : `(code << 20) | (index & 0xfffff)`. L'ancien calcul
`code * 16 + index` était incorrect ; le repli sur l'effet 0 masquait cette erreur.
Les types statiques des effets permettent également d'écarter un effet continu d'une
activation, ou un effet d'ignition d'une réponse en chaîne, sans consulter le masque légal JAX.
Pour les effets du même type et de même description, `oracle/effect_inspector.cpp` lit, sans
modifier l'oracle, le code d'événement enregistré sur l'effet de `select_chains`. L'indice de
l'option est conservé même si plusieurs tuples d'options sont identiques. Si l'événement ne
suffit toujours pas, le résultat reste non concluant. L'inspecteur est construit par `setup.sh`
ou à la première ambiguïté, contre les en-têtes et le binaire épinglés de ygopro-core.

Les JSON finaux et le cache de traduction sont publiés atomiquement : une interruption ne doit
pas laisser un fichier final partiellement écrit. Un cache tronqué ou incompatible est reconstruit
à partir des scripts épinglés. Le journal JSONL reste le support de récupération d'une campagne
interrompue (ignorer une éventuelle dernière ligne incomplète).

La clé du cache comprend aussi les versions de Python, JAX/jaxlib, NumPy et luaparser :
un changement de runtime impose une nouvelle traduction, même si les scripts sont inchangés.

## Substituts : validation et coût (2026-11-06)

Le jalon des substituts traduisait **507 / 13 541 cartes** : 426 du deck principal et 81 de
l'Extra Deck, dont 60 fusions. Les effets passifs `EFFECT_FUSION_SUBSTITUTE` simples sont
reconnus ; les callbacks de valeur et conditions arbitraires restent refusés.

L'affectation exacte repose sur les sous-ensembles des quatre emplacements de recette,
un histogramme de profils et des transformées zêta. Aucun tableau carte × carte n'est créé.
Les banques sans substitut gardent le chemin de calcul précédent ; les métadonnées sont
partagées dans la banque, sans nouvel état par partie. L'encodeur lit les permissions par
emplacement pour les nouveaux modèles. Les anciens checkpoints restent utilisables et
conservent leur calcul précédent en ignorant ces nouveaux bits.

```sh
YGO_JAX_CACHE=/workspace/.cache/ygo-jax .venv/bin/python tests/substitute_oracle_check.py
YGO_JAX_CACHE=/workspace/.cache/ygo-jax .venv/bin/python tests/effect_identity_check.py
.venv/bin/python bench/fusion_match_bench.py --b 32 --n 110 --calls 50
```

Les scénarios ciblés vérifient notamment les substituts face cachée, le refus de deux
remplacements, les filtres non substituables et le cas valide de deux cartes substituts
dont une utilisée normalement. Les cinq anciens scénarios ambigus sont rejoués avec leurs
decks complets et doivent dépasser l'ancien point de blocage en consultant l'inspecteur natif.
Résultats, provenance et limites : [bilan](../notes/codex-substituts-2026-11-06.md).

## Procédures de main et poursuite du moteur (2026-11-06)

Le catalogue atteint **572 / 13 541 cartes utilisables par le compilateur** : 490 du
Deck principal et 82 de l'Extra Deck (60 fusions, 6 synchros, 13 Xyz, 3 Links).
Ce total ne signifie pas que tous les effets de ces cartes ont été exercés.

Les procédures simples `EFFECT_SPSUMMON_PROC` depuis la main se compilent en
`K_PROC` : conditions et compteurs existants, action propre à l'emplacement de
l'effet, placement dans une zone principale. Elles ne forment pas de chaîne et
conservent l'invocation normale. Les monstres `TYPE_SPSUMMON` ne peuvent pas être
invoqués normalement ni posés. `EnableReviveLimit` et l'interdiction permanente
par `EFFECT_SPSUMMON_CONDITION` à valeur `aux.FALSE` sont distingués, y compris
pour les invocations par effet et pendule. Les procédures à coût/opération et
les restrictions conditionnelles non représentables restent refusées.

Des clones SINGLE d'un même déclencheur peuvent partager un emplacement pour
les événements d'invocation normale, flip et spéciale, s'ils ont les mêmes
programmes, description et une limite partagée par nom (ou aucune limite).
Les compteurs par effet et les déclencheurs FIELD ne sont pas regroupés.
Les requêtes de nombre de cartes acceptent aussi les bornes supérieures ; les
alias locaux de `GetLocationCount` restent symboliques, sans ajout de bytecode.

Le générateur produit également ces procédures et leur version Lua indépendante,
avec les formes fixes des banques et les dimensions actuelles du modèle.

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tests/hand_procedure_oracle_check.py
.venv/bin/python tests/flip_self_oracle_check.py
.venv/bin/python tests/effect_identity_check.py
.venv/bin/python gen/compare_gen.py --banks 0-1 --games 5 --main 80 --extra 10 --focus
```

Le comparateur de cartes générées renvoie un code non nul si une partie échoue.
Le test d'identité déplie volontairement trois clones optimisés pour continuer à
exercer l'inspecteur natif sur les cinq anciens scénarios ambigus.

Les nouvelles campagnes ont aussi révélé une erreur de filtre sur les monstres
marqués détruits au combat mais encore sur le terrain : les requêtes de groupe
et prédicats sur la carte elle-même doivent les voir, tandis que le ciblage les
exclut. Une fixture minimale vérifie que l'effet FLIP s'active réellement.
[Bilan, mesures et suite](../notes/codex-procedures-2026-11-06.md).

## Paiements de procédures et registres virtuels (2026-11-06)

Le compilateur peut maintenant conserver un groupe entre `SetTarget` et
`SetOperation` d'une procédure de main, puis payer son coût avant le placement,
sans créer de chaîne. Les sélections de groupe simples et de taille variable
utilisent les décisions existantes. Les contraintes supplémentaires de groupe,
les annulations et les paiements qui libèrent une zone pleine restent des
chantiers distincts : les formes non représentées sont refusées.

Les groupes du compilateur ont des registres virtuels. Une passe de liveness
alloue les quatre registres physiques, élimine les requêtes inutilisées et
partage les requêtes identiques entre deux mutations. La résolution commence une
nouvelle époque, car des réponses ont pu modifier le terrain. Les instructions
qui modifient leur registre conservent son emplacement physique et ne partagent
pas ce groupe avec une requête indépendante. Les limites de longueur et de
nombre de filtres sont vérifiées après cette passe.

`GetMatchingGroup` capture désormais ses références au moment de l'appel.
`Filter`, `FilterSelect`, `FilterCount` et `Clone` travaillent sur ces groupes,
y compris après un déplacement. Les tests de compte à la résolution sont
réellement exécutés. Un test de non-vacuité peut être supprimé seulement si son
bloc ne fait que déplacer le groupe testé : déplacer un groupe vide est déjà
sans effet.

La validation réserve 15 emplacements d'Extra Deck par joueur, en conservant la
taille réelle dans `new_game(..., extra_counts=...)`. Les autres emplacements
ont `zone=0` dès l'initialisation ; ils ne constituent aucune carte en jeu. Une
seule forme JAX sert ainsi aux différentes tailles et aux Extra Decks
asymétriques. Le comportement historique sans `extra_counts` est conservé.

```sh
.venv/bin/python tests/procedure_cost_oracle_check.py
.venv/bin/python tests/group_semantics_oracle_check.py
.venv/bin/python tests/procedure_payment_generation_oracle_check.py
.venv/bin/python tests/procedure_cost_batch_check.py
.venv/bin/python tests/padded_extra_oracle_check.py
```

Les procédures rituelles usuelles `AddProcGreaterCode`, `AddProcEqualCode`,
`AddProcGreater` et `AddProcEqual` réutilisent les instructions rituelles du
moteur. Un niveau littéral est accepté seulement s'il correspond au niveau
de tous les monstres nommés dans la base épinglée. Les opérations, matériaux et
zones personnalisés restent refusés. La limite de réanimation d'un monstre
rituel sans effet est portée par la carte, sans occuper un emplacement d'effet.

Les protections temporaires contre la destruction au combat sont prises en
compte dans le cache de statistiques. Le cas qui a révélé le défaut, `Draw
Muscle`, est rejoué avec les mêmes decks et les mêmes 98 premières décisions
oracle que dans la campagne divergente. Les scénarios ciblés comptent aussi les
attaques réellement dirigées vers un monstre protégé.

Deux effets sans description peuvent être distingués par leur portée déclarée
si leurs types et zones natives suffisent. Le comparateur conserve le refus
d'ambiguïté lorsque ces métadonnées se recouvrent, sans consulter la légalité JAX.

```sh
.venv/bin/python tests/translated_ritual_oracle_check.py
.venv/bin/python tests/draw_muscle_oracle_check.py
.venv/bin/python tests/replay_draw_muscle_failure.py
.venv/bin/python tests/temporary_protection_generation_oracle_check.py
.venv/bin/python tests/disjoint_ranges_oracle_check.py
```

Catalogue de ce lot : **700 / 13 541**, dont 611 cartes du deck principal et
89 de l'Extra Deck. Cela décrit la traduction acceptée ; les campagnes mesurent
séparément les cartes présentes, invoquées et les effets réellement résolus.
[Bilan et provenance](../notes/codex-couts-registres-2026-11-06.md).

## Activation initiale des pièges continus et compteurs (2026-11-07)

Le moteur représente la question native 94 : accepter l'effet initial d'un
piège continu remplace l'effet du même maillon, en conservant la nature
d'activation de carte. Refuser ne consomme pas le compteur de l'autre effet.
La traduction accepte une activation sans callbacks et un seul effet rapide
libre depuis la zone magie/piège. Les autres compositions sont refusées.

Les compteurs `EFFECT_COUNT_CODE_OATH` sont remis lorsque l'activation est niée,
au chargement du maillon concerné ; une simple annulation d'effet ne les remet
pas. Les métadonnées et la nature du maillon sont visibles par le réseau.
Les anciens checkpoints sont compatibles avec une extension nulle des poids.

Le programme de test dérivé fusionne les bornes de compte avec les filtres et
ignore les candidats inactifs par une borne de boucle nulle. Le bytecode de
résolution et les quatre registres physiques sont conservés. Les nouvelles
variantes de pièges sont aussi produites par le générateur et sa bibliothèque
Lua, à formes de banques fixes.

```sh
.venv/bin/python tests/inline_activation_oracle_check.py
.venv/bin/python tests/inline_responses_oracle_check.py
.venv/bin/python tests/inline_generation_oracle_check.py
.venv/bin/python tests/refundable_count_oracle_check.py
.venv/bin/python tests/inline_batch_check.py
```

Catalogue : **701 / 13 541**, avec Sealing Ceremony of Mokuton et aucune perte.
[Scénarios, coût et validation](../notes/codex-pieges-compteurs-2026-11-07.md).

## Équipement (2026-11-13)

Moteur : relation d'équipement `eq_t` (cible, -1 sinon) et `eq_mv` (horloge de
l'équipement, pour reconnaître une cible qui a bougé puis est revenue). Brique
`OP_EQUIP` (Duel.Equip : cible recto en zone monstre, équipement encore en zone
magie/piège et non déplacé depuis le chargement du maillon). Les effets
`EFFECT_TYPE_EQUIP` sont des effets continus en zone magie/piège dont la cible
est le monstre équipé (`cont(..., tgt=EQUIPPED)`, `M_SELF = M_EQUIP`).
`drop_equips` (appelé par `refresh` et `dmg_send`, comme `drop_overlays`)
envoie au cimetière, raison RULE + LOST_TARGET + DESTROY, un équipement dont la
cible a quitté le terrain, est passée face verso ou a bougé. Une magie
d'équipement activée restée sans cible (cible partie avant la résolution) part
au cimetière en fin de chaîne (ST_LEAVE). Banque sans équipement : largeur 0,
chemin éliminé au traçage. `compare.py` compare la cible d'équipement
(QUERY_EQUIP_CARD côté oracle).

Traducteur : `aux.AddEquipProcedure(c[, p[, f]])` (aussi sous la forme
`local e=aux.AddEquipProcedure(...)`, l'effet rendu restant inutilisable), filtre
sur des propriétés fixes seulement, et effets `EFFECT_TYPE_EQUIP` continus.
`aux.FilterBoolFunction` / `FaceupFilter` / `TargetBoolFunction` n'appliquent
plus les arguments d'appel au filtre interne (le Lua les ignore).

Limites connues : la limite d'équipement n'est pas revérifiée à l'ajustement
(sans objet tant que le filtre est fixe), `EVENT_EQUIP` n'est pas levé, la
relation n'est pas visible par le réseau (seulement ses effets sur l'ATK).

Validation : 12/12 (graines 5000-5011, les 6 cartes d'équipement traduites) et
20/20 (graines 5100-5119, avec Typhon, Trou Noir, Renaissance, Force Miroir,
Raigeki, Prison Dimensionnelle, Trappe). Catalogue **773 / 13 541** ; seulement
6 / 250 magies d'équipement : refus suivants « plus de 2 effets » (31),
arguments supplémentaires de la procédure (26), `GetEquipTarget` (13),
`SetUniqueOnField` (12).

## E = 4 et règles de chaîne trouvées par le générateur multi-effets (2026-11-13)

`core.E` passe de 2 à 4 (86 % des scripts ont au plus 4 effets). Les bits de
compteurs par effet (`ST_CNT`, `ST_CNTN`) sont déplacés au-dessus du bit 17 et
dimensionnés par E. Les branches d'activation par effet sont regroupées en une
seule branche à effet dynamique : leur coût ne grandit plus avec E. Les banques
créées avec moins d'effets (`train/realbank.npz`) sont complétées par des effets
vides au chargement. Catalogue à E=4 : **815 / 13 541**.

Le générateur tire maintenant 1 à 4 effets par monstre à effet (poids
0,45 / 0,30 / 0,17 / 0,08). `compare_gen` a ainsi trouvé deux règles qui
manquaient au moteur :

- Une carte qui ira au cimetière en fin de chaîne (`ST_LEAVE`, soit
  STATUS_LEAVE_CONFIRMED : magie / piège résolu ou activation niée) ne peut
  plus être renvoyée en main (`card::is_capable_send_to_hand`). Exemple : un
  renvoi en main qui cible le contre-piège qui lui répond.
- `Duel.NegateRelatedChain(tc, RESET_TURN_SET)`, appelé par les annulations à
  la Effect Veiler, annule aussi les maillons en cours de `tc` encore liés à
  leur carte, avec un EFFECT_DISABLE_CHAIN qui reste même si `tc` quitte ensuite
  le terrain. C'est le bit `nrc` de `MODIFY(..., MC_DISABLE, nrc=True)` (bit 14
  de l'opérande). Il est posé par le générateur, Effect Veiler et Forbidden
  Chalice.

Le traducteur accepte `Duel.NegateRelatedChain(tc, RESET_TURN_SET)` quand un
EFFECT_DISABLE sur la même carte suit dans la même fonction : le MODIFY
MC_DISABLE de cette carte reçoit `nrc`. Sinon la carte est refusée. Aucune des
101 cartes concernées n'est débloquée pour autant : chacune bute d'abord sur un
autre refus (en tête : « DISABLE avec valeur », 6 cartes ; puis DAMAGE_STEP sans
StatChange et activation combinée, 4 chacune). Catalogue inchangé : 815.
