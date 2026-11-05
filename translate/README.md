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

Autre manque : les sélections de taille variable (`min < max` pendant la résolution, ~378 scripts) demandent une
fonction du moteur, une sélection min..max avec arrêt anticipé. Aujourd'hui, CHOOSE prend toujours exactement
min(max, nombre).

## Vérification (`validate.py`)

Une seule banque contient les cartes traduites, les monstres normaux et des Syntoniseurs normaux : elle n'est compilée
qu'une fois. Ensuite, on joue des parties aléatoires avec la boucle de `engine/compare.py`, en mode `tr` :
- chaque joueur reçoit K cartes traduites tirées au hasard, à 2 exemplaires, complétées par des monstres normaux, plus
  6 Syntoniseurs si une carte synchro a été tirée ;
- les deux joueurs reçoivent le même Extra Deck, l'union des monstres de l'Extra Deck tirés (15 au plus), parce que
  `new_game` impose deux Extra Decks de même taille.

Une carte est « vérifiée » si elle n'a figuré que dans des parties identiques. Avec `--bisect`, chaque partie
divergente est rejouée avec une seule carte traduite par deck, pour isoler la fautive. Deux options servent au
débogage :
- `--chosen a,b;c,d` rejoue une partie avec ces cartes imposées ;
- `--nosave` laisse `validated.json` intact.

`compare.translate` fait correspondre les options de l'oracle pour les cartes traduites (« activer l'effet n de
cette carte ») grâce aux descriptions (`compare.DESC`). Pour la description 221 (déclencheur dans la chaîne), il passe
par l'effet `tg_e` en attente.

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
- `cards.act(opt=)` transmet la limite d'activation. Avant ce correctif, une magie ou un piège avec
  EFFECT_COUNT_CODE_OATH pouvait s'activer deux fois par tour.

## Validations

| run | parties identiques | cause des divergences |
|---|---|---|
| 2 | 7 / 12 | |
| 4 | 10 / 12 | The Cheerful Coffin (DiscardHand(1,3) traduit en « exactement 3 », désormais refusé) ; Evoltile Lagosucho (option de l'oracle mal traduite, corrigé par DESC) |
| 5 | voir `ETAPE3.md` | |

Divergences anciennes non élucidées :
- Constellar Alrescha ;
- un monstre absent de la zone monstre (val3, partie 8) ;
- une sélection de Gren Maju (val2, partie 9).

Usage : `uv run --with luaparser --with jax python translate/validate.py --games 30 --k 8 --seed0 12 --bisect`
