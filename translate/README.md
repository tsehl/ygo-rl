# Traducteur Lua → briques

But : couvrir tout le jeu en traduisant les 13 541 scripts officiels (`data/CardScripts/official`) vers les briques du
moteur, et vérifier chaque carte traduite contre l'oracle. Les vraies cartes restent un benchmark : elles ne servent
qu'à mesurer ce que le moteur sait jouer, jamais à l'entraînement.

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

## Couverture (2026-11-05)

441 / 13 541 scripts traduits et jouables par `validate.py` (3,3 %) :
- 420 du deck principal ;
- 21 de l'Extra Deck : synchro 6, Xyz 12, Link 3.

La fusion n'est pas encore traduite (`Fusion.AddProcMix`, 344 cartes).

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
1. Fusion : traduire `Fusion.AddProcMix` / `AddProcMixN` (494 cartes) vers les recettes que le moteur a déjà
   (`cards.assemble(fusion=...)`, mode `fus` de `compare.py`). Débloque aussi une partie d'EnableReviveLimit.
2. SPSUMMON_PROC (336) : invocation spéciale par procédure depuis la main, nouvelle fonction du moteur.
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

Une description ambiguë n'est plus départagée à partir des actions légales du moteur testé :
le résultat est « inconclusive ». Il faut alors améliorer la correspondance ou ajouter un scénario
ciblé avant de conclure sur cette carte.

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

Les substituts de matériaux nommés (`sub=true` avec un code de matériau) restent explicitement refusés :
le moteur ne les représente pas. Les recettes à matériaux variables, tables d'alternatives et filtres
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
Les véritables ambiguïtés entre plusieurs effets du même type et de même description restent
non concluantes et font échouer la validation.
