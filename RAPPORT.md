# Rapport d'état — projet YGO RL (2026-11-03)

État complet après six jours de travail (2026-10-29 → 11-03, 44 commits). Les détails sont dans `ETAPE1.md`
(analyse des scripts), `ETAPE2.md` (moteur, jalon par jalon), `ETAPE3.md` (générateur, réseau, PPO) et
`bench/RESULTATS.md` (faisabilité GPU).

## 1. Résumé

- **Moteur** : il couvre les règles principales du jeu. Il rejoue à l'identique les parties de l'oracle
  ygopro-core, dans 18 modes de test, et aussi sur des cartes procédurales compilées en Lua.
  - Règles couvertes : phases, combat et damage step, chaînes, SEGOC, effets rapides, modificateurs et couches
    ATK/DEF, annulation, protections, tout l'Extra Deck (fusion, rituel, synchro, Xyz, Link, pendule).
  - Débit sur L4 : 10 à 12k décisions/s.
- **Générateur procédural** : il fabrique des banques de 480 cartes, que l'oracle vérifie (banques 0-9 : toutes
  les parties identiques). Le fuzz ne trouve plus aucune violation (graines 0-9).
- **Réseau et PPO self-play** : ils fonctionnent de bout en bout. Le réseau lit les briques des cartes, jamais un
  identifiant.
  - Après 200 itérations (12,5 M de décisions, 70 min de L4), le modèle bat l'aléatoire à 97,6 % sur une banque
    jamais vue.
  - Il ne bat plus sa version d'il y a 60 itérations qu'à 54-60 % : les progrès ralentissent.
- **Pas encore fait** :
  - évaluation sur les vraies cartes ;
  - archétypes et decks structurés (étape 4) ;
  - adversaires variés (pool de versions passées) ;
  - tout run sur A100.
- **Budget Colab** : 185,9 CU restants sur 200 ce mois-ci.

## 2. Ce qui est fait

### 2.1 Moteur (`engine/core.py`, 2 530 lignes)

Il est écrit en JAX pur, avec un état à tailles fixes (struct of arrays). `step(état, action)` applique l'action
puis avance jusqu'à la prochaine décision. `step_batch` traite un lot de parties en évitant le coût des branches
sous vmap.

| Jalon | Contenu | Vérification (`compare.py`) |
|---|---|---|
| 1 | Squelette : pioche, phases, invocation normale / pose, combat complet | `normal` |
| 2 | Interpréteur de briques : filtres, registres, CHOOSE / MOVE / DRAW / DAMAGE, magies et ignitions | `mixed` |
| 3a | Chaînes, pièges, jeux-rapides, contre-pièges, fenêtres de priorité | `chain` |
| 3b | Déclencheurs, SEGOC, damage step découpé comme ygopro, effets rapides de monstre | `trig`, `quick` |
| 4.1 | Setcodes, filtres archétype / race / nom, effets continus de stats, modificateurs temporaires | `cont` |
| 4.2 | Fenêtres du damage step, couches ATK/DEF complètes (SET, SET_FINAL, SET_BASE, ordre des id), limites par carte | `dmg`, `ds` |
| 4.3a | Interdictions et protections (attaque, ciblage, immunités, indestructible, pas d'invocation spéciale…) | `flag` |
| 4.3b | Annulation d'effets (DISABLE, DISABLE_EFFECT, RESET_DISABLE) | `neg` |
| 5a-f | Extra Deck : fusion, rituel, synchro, Xyz, Link (+ Extra Link, co-liens), pendule | `fus`, `rit`, `syn`, `xyz`, `lnk`, `xlk`, `kn`, `pen` |

- **90 vraies cartes** traduites en briques (`engine/cards.py`), dont des cartes difficiles :
  - Ash Blossom / Ghost Ogre ;
  - Magic Jammer ;
  - Knightmare Phoenix / Cerberus ;
  - Polymerization ;
  - des rituels ;
  - des pendules.
- **Non-régression** (`engine/regress.sh`, ~45 min sur CPU), dernier passage complet le 2026-11-02 :
  - 18 modes de `compare.py`, toutes les parties identiques à l'oracle ;
  - `obs_check` : l'observation ne fait pas fuir l'information cachée ;
  - `vmap_check` ;
  - `sb_check` : step_batch == vmap(step).
- **Débit** (L4, B=1024, actions aléatoires) :
  - 12,4k décisions/s avec les vraies cartes et un Extra Deck de 15 ;
  - 10,3k sur une banque procédurale.

  Historique : 15,8k avant le jalon 4, 5,5k après tout l'Extra Deck, puis deux passes d'optimisation (caches
  recalculés au point de décision seulement, arrêt anticipé, double passe, tours forcés en sous-lot).

### 2.2 Générateur procédural (`gen/`)

- **`cardgen.py`** :
  - une carte est une spécification (dict), traduite en briques ;
  - un effet combine un genre (activation, ignition, déclencheur, rapide, continu), une condition, un coût et
    1 à 2 clauses verbe × sélecteur ;
  - tous les cadres sont couverts : normaux, effets, rituels, magies et pièges de tous types, fusion, synchro,
    Xyz, Link, pendules ;
  - budget de puissance grossier, et 30 % de setcodes tirés sans structure ;
  - une seule compilation pour toutes les banques, car la banque est un argument.
- **`tolua.py` + `procgen.lua`** : la même spécification est compilée en Lua, et `compare_gen.py` fait jouer
  l'oracle sur les cartes générées. Cela a trouvé et corrigé quatre vraies règles :
  - l'interdiction d'invocation spéciale s'applique aussi à la fusion, au rituel et aux procédures ;
  - le joueur de l'événement d'une invocation spéciale par effet est PLAYER_NONE ;
  - un point d'événement s'ouvre après la défausse de fin de tour ;
  - le modificateur d'une carte est suspendu tant qu'elle est annulée.
- **`fuzz.py`** : parties aléatoires avec contrôle des invariants d'état après chaque décision. Graines 0-9
  propres (un bug Link corrigé) ; refait le 2026-11-02 sur les graines 0-3 : 0 violation sur 256 parties.

### 2.3 Réseau (`train/model.py`, 1,19 M de paramètres à d=128)

- **Encodeur de cartes** :
  - chaque carte devient une séquence de tokens : en-tête (stats, types, setcodes, flèches, échelles,
    procédures), effets, instructions de ses programmes, filtres, filtres de matériaux ;
  - les entiers sont encodés en symlog plus 16 bits, pour que le réseau puisse comparer des archétypes ou des
    zones sans identifiant appris ;
  - transformer de 2 couches. Les tokens utiles (~8 sur 91) sont regroupés et coupés à 48.
- **Politique** :
  - 64 tokens d'observation (`engine/obs.py`, cartes visibles seulement), chacun = vecteur de sa carte + son
    état ; plus un token global ;
  - transformer de 3 couches ;
  - logits verbe × token (8 verbes) et 14 actions spéciales, valeur en tanh.
- **Option `--bf16`** : calcul en bfloat16, paramètres et pertes en f32.

### 2.4 PPO self-play (`train/ppo.py`)

- La même politique joue des deux côtés. GAE par joueur, récompense ±1 en fin de partie.
- B parties en parallèle. Une partie finie est remplacée par une neuve, tirée d'une réserve à usage unique.
- Nouvelle banque procédurale toutes les 10 itérations, decks de 40 + 15 tirés sans structure.
- `act_step` fait tout un appel du rollout dans une seule fonction compilée : politique, step_batch,
  remplacement.
- Évaluation sur une banque tenue à l'écart : contre l'aléatoire, et contre la politique d'il y a `--old-lag`
  itérations.
- Outils : `train/prof_update.py` (chronométrage de la mise à jour), `bench/engine_bench.py --proc` (moteur
  seul, banque procédurale).

### 2.5 Résultats d'entraînement

**Run 1** (2026-11-02, L4, f32) : la session Colab a disparu vers 75 min et le log et les paramètres sont
perdus. Il en reste ceci : 2,8k décisions/s, et 66 → 81 → 88 % contre l'aléatoire aux itérations 9, 19 et 49.

**Run 2** (2026-11-03, L4, bf16, B=1024, 64 appels par itération, 200 itérations) :

| itération | contre l'aléatoire | contre l'itération −60 | entropie |
|---|---|---|---|
| 19 | 82,5 % | 82,6 % (initialisation) | 0,83 |
| 59 | 89,0 % | 91,1 % (initialisation) | 0,55 |
| 79 | 92,7 % | 70,2 % | 0,56 |
| 99 | 94,7 % | 61,4 % | 0,48 |
| 139 | 97,1 % | 59,6 % | 0,46 |
| 179 | 97,7 % | 60,1 % | 0,50 |
| 199 | 97,6 % | 53,6 % | 0,42 |

- Environ 1000 parties par évaluation.
- Débit global de 3,5 à 4,0k décisions/s, 16-18 s par itération.
- Paramètres finaux : `train/ckpt/run2_it200_f16.pkl.gz`.

## 3. Lecture critique des résultats

- **Battre l'aléatoire ne prouve presque rien.** 97 % contre un joueur uniforme montre que le pipeline apprend
  quelque chose de transférable à des cartes inconnues, mais c'est la barre la plus basse.
- **L'avantage sur les versions passées baisse** (70 → 60 → 54 %). Trois explications possibles, non départagées :
  - la politique atteint ce que permettent des decks aléatoires sans structure ;
  - le self-play pur tourne en rond ;
  - le réseau ou les hyperparamètres saturent.

  Une partie du résultat tient sans doute aux decks : une main faible perd quoi qu'on joue. Il manque une
  référence (rejouer chaque partie avec les decks échangés) pour savoir quelle marge est atteignable.
- **Ce qui manque le plus** :
  - aucune mesure sur les vraies cartes, qui sont le but annoncé (le banc d'évaluation) ;
  - aucune analyse qualitative : le modèle active-t-il ses effets à bon escient, enchaîne-t-il ?
- **Les decks d'entraînement n'ont aucune structure** (cartes tirées au hasard, setcodes sans archétype).
  `ETAPE3.md` le dit dès la conception : un agent entraîné ainsi apprend l'avantage de cartes, pas à exécuter un
  plan.

## 4. Limites connues du moteur (dette)

### 4.1 Tailles fixes de la banque

- **E = 2 effets par carte et P = 12 instructions par programme.** La conception visait E=8 et P=24.
  - Ça suffit pour les 90 cartes traduites et pour le générateur actuel.
  - Mais beaucoup de vraies cartes ont 3 effets ou plus, et des programmes plus longs.
  - C'est la limite la plus structurante pour l'objectif « jeu complet ». L'augmenter coûte du débit et de la
    mémoire de compilation : il faudra la mesurer.
- **Autres tables bornées** :
  - KM = 16 effets continus actifs ;
  - TM = 16 modificateurs temporaires ;
  - CH = 12 maillons de chaîne ;
  - 64 tokens d'observation : jusqu'à 61 sont utilisés en fin de partie, la marge est faible.

### 4.2 Règles non modélisées ou approchées

Notées depuis le jalon 3b, non reprises depuis :
- « manquer le timing » : ygopro efface les déclencheurs optionnels « quand » si l'événement n'est pas la
  dernière chose arrivée ;
- les déclencheurs de terrain (EFFECT_TYPE_FIELD : « quand un monstre adverse… ») ;
- la condition d'un déclencheur n'est évaluée qu'au point d'événement ;
- un monstre détruit au combat est exclu de toutes les sélections, même sans ciblage.

Restrictions du générateur, qui contourne des limites du moteur :
- **un seul déclencheur par carte** : le moteur ne garde qu'un effet déclenché en attente par carte ;
- **pas de « gagne X ATK jusqu'à la fin du tour »** (`temp.attack`) ;
- **pas de filtre de stats dans la cible d'un modificateur continu** ;
- **« ne peut pas attaquer » compte aussi l'invocation par flip.**

Hors règles :
- le deck n'est pas remélangé après une recherche. C'est sans conséquence pour l'agent, qui ne voit pas l'ordre.

### 4.3 Briques absentes

Vérifié dans `engine/core.py` : il n'y a ni compteurs, ni cartes d'équipement, ni jetons, ni changement de
contrôle, ni magies de terrain. Ce sont des pans entiers du jeu réel, absents de toute la chaîne (moteur,
générateur, réseau). Un banc de vrais decks devra s'en passer, ou attendre ces briques.

## 5. Débit : où part le temps (run 2)

| Poste | Temps par itération | Remarque |
|---|---|---|
| Rollout (64 appels) | ~10,6 s | ~165 ms par appel d'`act_step`, contre 88 ms pour le moteur seul (banque procédurale 0) et 13 ms pour `act` |
| Mise à jour (2 époques × 8 minibatchs) | ~7 s | 11 s en f32 ; en f32, un minibatch demande 15 Go |
| Encodeur de cartes | négligeable | 6,5 ms, avant + arrière |

- L'écart du rollout n'est pas expliqué :
  - sur CPU, le choix du réseau ajoute 18 % ;
  - les cartes procédurales coûtent 15-20 % de plus que les vraies ;
  - la fin de partie coûte ~20 % de plus que le début ;
  - fusionner les fonctions n'a rien changé.

  - mesure L4 découpée (2026-11-03) : la banque en argument (87,7 ms contre 88,4 en constante) et les parties neuves (0 s) ne coûtent rien ; `act` prend 13,4 ms. Il reste ~65 ms par appel.

  Prochaine mesure : `act`, `step_batch` et `act_step` séparément, sur les mêmes états de rollout (milieu de partie).
- À ce débit (4k décisions/s), une heure de L4 donne ~14 M de décisions pour ~2 CU. Le budget mensuel restant
  (~186 CU) permet donc ~90 h de L4, ou moins d'A100 (non mesuré).

## 6. Infrastructure et pièges connus

- **Oracle** : ygopro-core 11.0 compilé localement (`setup.sh`, `oracle/BUILD.md`), piloté par ctypes
  (`oracle/driver.py`).
- **Colab depuis la VM cloud** (`colab` CLI) :
  - le code s'envoie en morceaux de 24 Ko ;
  - le travail se lance en arrière-plan, et on interroge le log ;
  - `colab exec` peut rester bloqué : toujours l'envelopper dans `timeout` ;
  - les paramètres se rapatrient en base64 (pickle f16 gzip, ~3 Mo de sortie) ;
  - une session est déjà morte sans prévenir vers 75 min : relever les paramètres régulièrement.
- **Compilation** :
  - `step_batch` compile en ~6 min sur GPU, ~6 min sur CPU (5,8 Go au pic) ;
  - le cache persistant `YGO_JAX_CACHE` évite de recompiler entre deux processus de la même session.
- **Session cloud** (4 cœurs, 15 Go) : non-régression complète ~45 min, fuzz ~13 min par graine.

## 7. Prochaines étapes proposées, par ordre d'utilité

1. **Évaluation sur les vraies cartes.**
   - Construire 4 à 8 decks avec les 90 cartes traduites, en ajoutant des cartes si nécessaire.
   - Y faire jouer le checkpoint contre l'aléatoire et contre l'initialisation.
   - C'est la seule mesure du but annoncé, et elle dira si la limite E=2 bloque déjà.
2. **Référence de marge** : rejouer les parties d'évaluation avec les decks échangés, pour séparer l'effet des
   decks de celui du jeu.
3. **Self-play contre un pool de versions passées** (league simple), pour éviter de tourner en rond. Peu de code :
   `act_step` accepte déjà un adversaire réseau.
4. **Étape 4 : archétypes et decks structurés** (chercheurs, extendeurs, ratios 3/2/1), avec une part de decks
   aléatoires.
5. **Moteur** :
   - lever les limites qui bloquent le générateur (plusieurs déclencheurs par carte, temp.attack, manquer le
     timing, déclencheurs de terrain) ;
   - étudier le coût de E > 2.

   Chacun de ces points demande la non-régression complète et une mesure de débit.
6. **Débit** : mesurer `act` sur GPU. Ensuite, un run plus long sur A100, seulement quand les points 1 à 3
   auront montré quoi entraîner.
