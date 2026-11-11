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
  depuis le cimetière) laissait la décision de position sans action légale. ygopro refuse une invocation spéciale
  sans la position d'attaque pour un Link (`field::is_player_can_spsummon` : `sumpos &= POS_FACEUP_ATTACK`, refus
  si `sumpos == 0`) : le Link reste où il est, et `IsCanBeSpecialSummoned(..., POS_FACEUP_DEFENSE)` est faux.
  Corrigé dans `spsummon` (exec_one) et dans `filter_mask` (F_SPSUM = 2, en défense seulement : jamais un Link).
- Durée : ~350 s par graine sur les 4 cœurs (~750 s quand la non-régression tourne en même temps).

## Implémentation, étape 2 : cartes générées contre l'oracle (2026-11-02)

### Compilation vers Lua (`gen/tolua.py`, `gen/procgen.lua`)

- Chaque carte devient `c<code>.lua` : un en-tête (la description de `describe`), la spécification en table Lua
  (`s.spec`), et `s.initial_effect` qui appelle `PG.init`. Toute la sémantique est dans une bibliothèque commune,
  `gen/procgen.lua` (table `PG`, chargée par `Duel.LoadScript`), qui interprète la spécification comme le fait
  `cardgen.bricks` côté moteur. Les gabarits Lua sont ceux des vraies cartes qui ont servi de modèle aux briques
  (Trap Hole, Torrential Tribute, Magic Jammer, Ghost Ogre, Man-Eater Bug, Penguin Soldier, Banner of Courage,
  Gravity Bind, Skill Drain, Polymerization par `Fusion.RegisterSummonEff`, rituels par
  `Ritual.AddProcGreaterCode` / `AddProcEqualCode`, procédures `Fusion.AddProcMix`, `Synchro.AddProcedure`,
  `Xyz.AddProcedure`, `Link.AddProcedure`, `Pendulum.AddProcedure`).
- Correspondances à respecter :
  - « puis » : la seconde clause ne passe que si la première a touché une carte (`Duel.Destroy(...)>0`, sinon
    `#g>0`) ; une clause sans carte passe toujours ;
  - une fois par tour : OPT_NAME → `SetCountLimit(1,id)` (le nom est partagé par tous les effets de la carte),
    OPT_CARD → `SetCountLimit(1)` ;
  - chaque effet porte la description `aux.Stringid(id, idx)` ; `compare.translate` en tire l'effet du moteur
    (`desc >> 20 == code`, `desc & 0xFFFFF` = idx). Un déclencheur optionnel seul hors chaîne passe par
    SelectEffectYesNo avec la description 221 (processor.cpp) : traduit vers l'effet K_TRIG_O de la carte ;
  - un déclencheur optionnel a `EFFECT_FLAG_DELAY` (il ne « rate » pas son moment) ; un piège sur événement n'en a
    pas (Trap Hole) ;
  - matériau de fusion sans contrainte : `aux.TRUE` (un `nil` dans la liste variadique d'`AddProcMix` la coupe) ;
  - filtre de matériau synchro appelé sans contexte d'invocation par `Synchro.NonTuner` : `IsAttribute` & co
    sans argument d'invocation dans ce cas.
- `cdb_rows` donne les lignes de `cards.cdb` (setcode packé sur 16 bits, `level | lsc << 24 | rsc << 16`) ;
  `driver.CardDB.add` les ajoute, `driver.SCRIPTS` reçoit le dossier des scripts de la banque.

### Comparaison (`gen/compare_gen.py`)

- Pour chaque graine de banque : génération, écriture des scripts dans `/tmp/ygo-gen/<b>`, puis parties avec la
  boucle de `engine/compare.py` (`play(..., mode="gen", given=decks)`), decks de 40 + Extra Deck de 15 tirés au
  hasard dans la banque. `--dump` décrit les cartes en jeu au moment d'un écart.
- Restriction du générateur : pas de filtre de stats dans la cible d'un modificateur continu d'ATK / DEF. ygopro
  évalue la cible pendant le calcul de l'ATK (`GetAttack` y rend la valeur temporaire), le moteur sur les stats
  de base ; reproduire ce point fixe ne vaut pas le coût.
- Restriction du générateur : au plus un déclencheur par carte. Le moteur ne garde qu'un effet déclenché en
  attente par carte (`tg` / `tg_e`) ; ygopro propose tous ceux dont l'événement est arrivé, y compris deux
  événements simultanés (envoyé au cimetière + détruit au combat). À lever si les archétypes de l'étape 4 en
  ont besoin (il faudrait `tg` en [E, N]).
- Filtre « invoqué ce tour » (Big-Tusked Mammoth) : `st = ST_SUMMON_TURN | ST_FLIP_TURN`, comme la vraie carte
  (le moteur ne pose que ST_FLIP_TURN lors d'une invocation par flip).

Écarts trouvés et corrigés dans le moteur (règles réelles, vérifiées contre ygopro-core) :
- **Interdiction d'invocation spéciale** (EFFECT_CANNOT_SPECIAL_SUMMON sur l'Extra Deck ou la main) : elle
  bloquait les effets « invoquez spécialement » mais pas les sorts de fusion, les rituels ni les procédures
  synchro / Xyz / Link. Ajoutée à F_FUS, F_RIT (`filter_mask`) et à `can_syn` (`legal`).
- **Défausse de fin de tour** : ygopro ouvre un point d'événement après la défausse (PhaseEvent, étapes 21-22 :
  déclencheurs « envoyé au cimetière », réponses à leur chaîne), sans fenêtre libre (`hand_adjusted`), puis
  recontrôle la limite de main. Le moteur passait directement au tour suivant. Nouveau retour de fenêtre R_HAND.
- **Modificateur posé par une carte sur elle-même** (« +300 ATK à vos monstres », la source comprise) : suspendu
  tant que la carte est annulée (`effect::is_available` : owner == handler, sans EFFECT_FLAG_CANNOT_DISABLE ; il
  revient si l'annulation cesse). Nouveau champ `tm_src` de la table des modificateurs temporaires.

- **Joueur de l'événement d'une invocation spéciale par effet** : Duel.SpecialSummon lève EVENT_SPSUMMON_SUCCESS
  avec `ep = PLAYER_NONE` (operations.cpp, `special_summon`, étape 4) ; seules les procédures (synchro / Xyz / Link,
  pendule) donnent le joueur qui invoque. Un piège généré « quand l'adversaire invoque spécialement » (condition
  `evp`, qui lit `ep`) ne répond donc qu'aux procédures. `eff_position` lève l'événement avec le joueur 2.

Fausse piste écartée : vider `ev_g` en fin de chaîne. ygopro vide les événements au début de chaque maillon
(`break_effect` dans `solve_chain`, déjà fait par `load_link`), mais garde ceux du dernier maillon résolu : un
piège « quand » sans DELAY répond à une invocation faite par le maillon 1 (Torrential Tribute après Monster Reborn ;
modes trig et quick de `compare.py`).

## Implémentation, étape 3 : réseau et PPO self-play (2026-11-02)

### Réseau (`train/model.py`)

- Encodeur de cartes : chaque carte de la banque est lue depuis ses briques, jamais depuis un identifiant. Elle
  donne ~91 tokens : en-tête, E effets, E×3×P instructions, E×NF filtres et 8 filtres de matériaux. Les entiers
  sont encodés en symlog plus 16 bits de poids faible, pour comparer archétypes, zones et types entre cartes. Un
  transformer par carte (2 couches) donne un vecteur [d] par carte.
- Politique : les T=64 tokens de l'observation (`engine/obs.py`) reçoivent le vecteur de leur carte plus leur
  état visible, et un token global s'y ajoute. Transformer (3 couches), puis logits verbe × token (NV=8) et 14
  actions spéciales sur le token global. Valeur en tanh. `to_engine` traduit l'indice vers l'action du moteur.
- d=128 : 1,19 M de paramètres. Jax pur, optax (clip 1.0 + adam 3e-4).

### PPO (`train/ppo.py`)

- Self-play : la même politique joue des deux côtés. GAE par joueur, avec des retenues [B,2] dans un scan inverse.
  Récompense ±1 en fin de partie.
- Rollout de `--steps` appels de `step_batch` sur B parties. Une partie finie est remplacée par une partie
  d'une réserve à usage unique, régénérée quand plus de 25 % a servi.
- Banque procédurale neuve toutes les 10 itérations. Mise à jour : 2 époques, minibatchs sur l'axe des parties,
  encodage des cartes recalculé à chaque minibatch.
- Évaluation : réseau contre une politique aléatoire, sur une banque jamais vue en entraînement (graine 1 000 000).
  Le côté du réseau alterne d'une partie à l'autre.

### Premier run (L4, 2026-11-02)

`--b 1024 --steps 64 --mb 8`, d=128, 3 couches.
- Compilation : ~8 min.
- 22,5 s par itération (62,7k décisions) :
  - rollout ~11 s, dont ~10,5 s pour le moteur et la politique ;
  - mise à jour ~11 s.
- Débit global : ~2,8k décisions/s. Le moteur plus la politique seuls font ~6k/s, la moitié des 12,4k du bench
  aléatoire (cartes réelles, Extra Deck 15). Cartes procédurales plus chargées en effets ? À mesurer.
- Mémoire GPU : 21 Go.

| itération | contre l'aléatoire (≈1000 parties) | entropie |
|---|---|---|
| 9 | 66,3 % | 1,07 |
| 19 | 81,1 % | 0,88 |
| 49 | 87,7 % | — |

- Le nombre de parties finies par itération passe de 33 à ~600 : les parties raccourcissent, la politique
  gagne vite.
- La session Colab a disparu vers 75 min, sans erreur dans le log. On a perdu la fin du log et les paramètres :
  `params.pkl` était sur le disque de la VM. Prochain run : relever le log souvent et copier les paramètres
  hors de la VM.

Le pipeline fonctionne de bout en bout : il apprend et généralise à une banque nouvelle contre l'aléatoire.
Mais battre l'aléatoire est un critère faible.

Suite :
1. Réduire le coût de la mise à jour (la moitié du temps).
2. Comprendre l'écart de débit du moteur sur les cartes procédurales.
3. Évaluer contre des versions passées de la politique et sur les vraies cartes (benchmark).

### Débit : où part le temps (2026-11-03)

Mesures de `train/prof_update.py` (observations synthétiques aux formes réelles) et de `bench/engine_bench.py --proc` (moteur seul, banque procédurale) :
- **Encodeur de cartes** : ~8 tokens utiles sur 91 (33 au plus sur 30 banques, 26 pour les vraies cartes).
  - Les tokens utiles sont regroupés en tête et coupés à `LC = 48` : même sortie (écart 1e-6), avant + arrière 1,5 → 0,9 s sur CPU.
  - Sur L4, il ne coûte de toute façon que 6,5 ms (avant + arrière) : la mise à jour, c'est la politique.
- **Politique** :
  - En f32, un minibatch de 128 parties × 64 steps demande 15 Go.
  - En bf16 (`--bf16` : calcul en bf16, paramètres, LayerNorm, softmax et pertes en f32), 400 ms par minibatch.
  - Mise à jour complète : 11 → 7 s par itération.
- **Tokens d'observation** : 37 valides sur 64 en moyenne, mais 56 au 99e centile et 61 au maximum. Compacter ne gagnerait presque rien.
- **Moteur seul, banque procédurale** (L4, B=1024) : 10,3k décisions/s, 96 ms par appel. Les vraies cartes donnent 12,4k : les cartes procédurales ne coûtent que 15-20 % de plus (même écart sur CPU).
- **Rollout PPO** : 169 ms par appel, contre 96 ms pour le moteur seul.
  - Sur CPU, le choix du réseau ajoute 18 % au moteur, et le moteur coûte autant avec les actions du réseau qu'avec des actions aléatoires.
  - Le coût monte de ~20 % entre le tour 4 et le tour 26.
  - Fusionner politique, step_batch et remplacement en une fonction (`act_step`, sans synchronisation hôte) n'a pas changé le temps du rollout.
  - Mesure découpée sur L4 (2026-11-03, B=1024, banque procédurale 0, bf16) :
    | mesure | ms par appel |
    |---|---|
    | moteur seul, banque en constante de compilation (`engine_bench.py --proc 0`) | 88,4 |
    | moteur seul, banque en argument (`--bank-arg`) | 87,7 |
    | `act` seul (observation + politique, `prof_update.py`) | 13,4 |
    | dont `observe` | 0,4 |
    | `act_step` dans le rollout PPO (it 1-5) | 162-170 |
    | parties neuves (`env.games`) pendant le rollout | 0 s par itération |
  - La banque en argument ne coûte rien, et les parties neuves non plus (le chronomètre ne les compte plus dans `act_step`). Le reste du rollout (values, stack, gae) tient en ~0,1 s.
  - act + moteur = ~102 ms. Il reste ~65 ms par appel d'`act_step` sans explication. Candidats : les positions du rollout sont en milieu de partie, alors que le bench part des positions initiales sur 40 appels (+20 % sur CPU entre le tour 4 et le tour 26) ; ou la fusion en une seule fonction compile moins bien que les morceaux séparés.
  - **Explication** (`ppo.py --prof 10`, L4, 2026-11-03) : chaque morceau chronométré seul, 10 fois sur le même état. Deux cas : l'état de fin de rollout après 3 itérations (milieu de partie, tour moyen 33) et la réserve (débuts de partie).
    | ms par appel | milieu de partie | début de partie |
    |---|---|---|
    | `act` | 13,4 | 13,4 |
    | `step_batch`, actions du réseau | 133,1 | 53,4 |
    | `step_batch`, actions aléatoires | 126,6 | 48,3 |
    | `act_step` | 160,2 | 74,8 |
  - L'écart vient de la position : en milieu de partie, le moteur coûte 2,5 fois plus qu'au début. Le bench (40 appels depuis le début de partie, 88 ms) le sous-estimait donc. Les actions du réseau coûtent ~5 % de plus que des actions aléatoires. Le remplacement et la colle autour d'`act_step` coûtent ~14 ms.
  - Pourquoi : les boucles de `step_batch` (exécution, settle, actions forcées) sont des `while_loop` au niveau du lot. Leur nombre de tours dépend de l'état (plus de cartes, de chaînes et d'actions forcées en milieu de partie), alors que le coût d'un tour, lui, est fixe.
  - Les parties sont longues : seules 24-123 parties sur 1024 finissent par itération de 64 appels.
  - Pistes : compter les tours de chaque boucle par appel en milieu de partie, pour savoir laquelle grossit ; faire tourner le bench du moteur sur des états de milieu de partie, sinon il ne représente pas l'entraînement.

### Milieu de partie : tours de boucle (2026-11-04)

Outils :
- `bench/loop_stats.py` : `step_batch(..., stats=True)` rend un compteur int32[11]. Le script affiche, par fenêtre d'appels, les tours de chaque boucle (externe, settle, run, trig, passes) et le nombre de parties actives par tour. `--prof-at` chronomètre chaque brique vmappée sur tout le lot.
- `bench/forced_stats.py` : actions forcées par type de décision, selon l'avancement de la partie.

Diagnostic :
- En milieu de partie, la moitié des actions forcées étaient des passes en fenêtre de chaîne (D_CHAIN hors déclencheurs) sans aucune activation possible.
- `auto_pass` se fiait à `act_cand`, qui surestime ce qui est activable. Ces passes n'étaient donc pas jouées dans settle : chacune coûtait un tour complet de la boucle externe (legal, forced_m, apply).
- Briques sur L4 (lot de 1024) : exec_one 2,7 ms, trig_eval 3,2, legal 2,7, refresh 1,9, pass_priority 1,8.

Corrections :
- `auto_pass` exact : quand il y a au plus `KA = 2` candidats, il les teste avec `act_ok`. Au-delà, pas de passe automatique et `legal` décide.
  - Le résultat de la partie ne change pas : l'action forcée A_PASS appelle le même `pass_priority`.
- Dans `step_batch`, `auto_pass` (devenu plus cher) n'est plus recalculé pour tout le lot à chaque tour de settle. Il est gardé dans la boucle et recalculé seulement pour les parties que run, trig_eval ou les passes viennent de modifier (sous-lot de B/8).

Résultats (L4, B=1024, banque procédurale 0, actions aléatoires) :

| version | début de partie (appels 0-24), ms | milieu de partie (appels 100-124), ms | tours externes / settle / run, milieu de partie |
|---|---|---|---|
| avant | 101 | 129 | 8,3 / 35 / 22,6 |
| auto_pass exact | 93 | 118 | 2,8 / 17,8 / 16,8 |
| + auto_pass en sous-lot | 91 | 113 | 2,8 / 17,8 / 16,8 |

Ce qui reste :
- Surtout la boucle d'exécution, avec ~17 tours par appel.
- Chaque tour paie exec_one et refresh sur tout le lot, pour ~47 parties actives en moyenne sur 1024.
- Un sous-lot pour la traîne de settle n'avait rien gagné, et avait coûté 70 % de compilation en plus.

Réglage de `frac` (arrêt anticipé de `step_batch`), PPO de bout en bout (L4, B=1024, 64 appels, `--mb 8 --bf16`, itérations 2-9) :

| frac | décisions par itération | ms par appel d'`act_step` | décisions/s (rollout + mise à jour) |
|---|---|---|---|
| 0,05 | 64,3k | ~148 | ~3 870 |
| 0,1 | 60,6k | ~125 | ~4 030 |
| 0,2 | 55,1k | ~118 | ~3 770 |

- Un `frac` plus grand rend la main plus tôt : la traîne des longues résolutions continue à l'appel suivant, en même temps que le travail des autres parties.
- Les parties rendues en cours de résolution ne produisent pas de décision à cet appel, mais leur case du tampon coûte quand même dans la mise à jour (~7 s par itération, fixe).
- Optimum vers 0,1 (+4 %) : c'est la nouvelle valeur par défaut de `ppo.py`.
- Les données ne changent pas : une partie suit la même suite de décisions quel que soit `frac` (`sb_check.py`).

### Deuxième run (L4, 2026-11-03)

`--b 1024 --steps 64 --iters 200 --mb 8 --bf16 --eval-every 20 --old-lag 60`, d=128.
- Compilation : ~8 min.
- 17-18 s par itération, puis ~16 s, soit 3,5k → 4,0k décisions/s.
- 12,5 M de décisions, 92 500 parties, ~70 min de calcul, 2,7 CU.
- Paramètres finaux (f16, gzip) : `train/ckpt/run2_it200_f16.pkl.gz`. Ils sont rapatriés par `colab exec` en base64 (2,9 Mo de sortie, ça passe).
- Cette fois, la session a tenu 95 min.

| itération | contre l'aléatoire | contre l'itération −60 | entropie |
|---|---|---|---|
| 19 | 82,5 % | 82,6 % (it 0) | 0,83 |
| 59 | 89,0 % | 91,1 % (it 0) | 0,55 |
| 79 | 92,7 % | 70,2 % (it 20) | 0,56 |
| 99 | 94,7 % | 61,4 % (it 40) | 0,48 |
| 139 | 97,1 % | 59,6 % (it 80) | 0,46 |
| 179 | 97,7 % | 60,1 % (it 120) | 0,50 |
| 199 | 97,6 % | 53,6 % (it 140) | 0,42 |

Environ 1000 parties par évaluation, sur une banque jamais vue.
- Contre la version d'il y a 60 itérations, l'avantage passe de 70 % à ~60 %, puis à 54 % sur la fin : les progrès ralentissent.
- Avec des decks aléatoires sans structure, une partie du résultat tient au tirage. Il faut un plafond de référence : par exemple, rejouer une partie avec les decks échangés.

Pistes :
- Plus de self-play contre des versions passées (pool d'adversaires).
- Des decks structurés (archétypes, étape 4).
- Évaluer sur les vraies cartes (benchmark).
- Mesurer `act` sur GPU.

### Évaluation honnête (2026-11-04)

Outils :
- **Vraies cartes** (`train/realcards.py`, `train/realbank.npz`) : 86 vraies cartes, jamais vues à l'entraînement, et quatre decks de 40 cartes + Extra Deck de 15 construits à la main :
  - contrôle (magies / pièges génériques) ;
  - synchro / Xyz / Link ;
  - rituel / fusion ;
  - pendule.
  - La banque est complétée aux formes procédurales (480 cartes, tprog à 2 × PT) : pas de nouvelle compilation.
  - On la reconstruit avec `python train/realcards.py` (il faut l'oracle).
- **Adversaire heuristique** (`ppo.heuristic`) : active tout ce qui peut l'être, invoque le plus fort, attaque avec tout, pose ses magies / pièges, puis passe. Ses choix (cibles, matériaux, zones) sont faits au hasard.
- **Decks par paires** (`ppo.paired`) : chaque paire de decks est jouée 4 fois (le réseau avec A puis B, en premier puis en second). La force des decks et l'avantage du premier joueur s'annulent.
- **Sur les vraies cartes**, toutes les rencontres (deck du réseau × deck adverse × côté) sont jouées, avec un taux de victoire par deck du réseau.
- `act_step` prend un adversaire aléatoire, réseau ou heuristique (`lax.switch`).
- L'évaluation périodique de `ppo.py` joue contre l'aléatoire, l'heuristique et la version d'il y a `--old-lag` itérations. Le banc des vraies cartes tourne tous les `--bench-every`.
- `train/bench_eval.py CKPT` évalue un checkpoint.

Run 2, itération 200 (L4, B=1024, ~1020 parties finies par ligne) :

| | contre l'aléatoire | contre l'heuristique |
|---|---|---|
| heuristique, banque procédurale | 86,7 % | — |
| heuristique, vraies cartes | 88,2 % | — |
| réseau, banque procédurale | 97,8 % | 66,2 % |
| réseau, vraies cartes | 95,2 % | 64,7 % |
| contre l'heuristique, vraies cartes, par deck du réseau | — | contrôle 66 %, pendule 70 %, rituel / fusion 59 %, synchro 65 % |

Lecture :
- Le réseau généralise aux vraies cartes : il perd seulement ~2 points par rapport à la banque procédurale, alors qu'il n'a jamais vu ces cartes et que certaines de leurs briques (négation, fusion, rituel, Knightmare) sont rares ou absentes des banques d'entraînement.
- Contre l'heuristique, il ne gagne que ~65 % : elle est bien plus discriminante que l'aléatoire, et il y a encore beaucoup de marge.
- Le deck rituel / fusion est le moins bien joué (59 %) : ce sont les procédures les plus longues (sort + matériaux).

### Le réseau maîtrise-t-il le jeu de base ? (2026-11-04)

Deux mesures sur le run 2, itération 200 (L4, B=1024).

**Politique déterministe** (argmax au lieu du tirage ; `ppo.play_eval(..., greedy=True)`, `BENCH_GREEDY=1 python train/bench_eval.py CKPT`). Résultats contre l'heuristique :

| | politique tirée | politique déterministe |
|---|---|---|
| banque procédurale | 66,2 % | 69,1 % |
| vraies cartes | 64,7 % | 65,0 % |

Contre l'aléatoire, en procédural, les deux politiques font ~98 %. Jouer l'argmax ne gagne presque rien : la politique n'est pas « bonne mais bruitée ».

**Tests de comportement** (`train/behavior.py CKPT`). Le joueur testé affronte l'heuristique avec les 4 decks de référence (toutes les rencontres). À chacune de ses décisions, on compte les occasions et les erreurs de jeu de base (`PROBES`). On compte aussi, par carte, les activations et les fins de tour où elle restait activable en main. Les références sont l'heuristique (47 % contre elle-même) et l'aléatoire (11 %). `BEHAV_WHO=reseau,aleatoire` restreint les joueurs testés : sur CPU, chacun recompile `tick`.

Erreurs, en % des occasions :

| sonde | réseau | réseau dét. | heuristique | aléatoire |
|---|---|---|---|---|
| attaque perdue d'avance (tous les monstres adverses recto ATK plus forts) | 17,0 | 17,6 | 29,2 | 31,9 |
| attaque directe ratée (terrain adverse vide, quitte le combat) | 0,8 | 0,1 | 0,0 | 64,2 |
| fin de tour sans invoquer / poser alors que possible | 7,2 | 5,7 | — | 28,0 |
| invoque depuis la main un monstre moins fort qu'un autre invocable | 21,5 | 19,7 | 4,7 | 39,0 |
| fin de tour avec une magie activable en main | 6,3 | 1,2 | — | 18,6 |
| Dark Hole en étant devant en ATK | 22,7 | 17,5 | 46,8 | 44,7 |
| MST sans cible | 26,3 | (1 occasion) | 32,1 | 20,8 |
| Monster Reborn : pas le plus fort | 74,6 | 72,6 | 71,2 | 83,7 |
| passe alors que Mirror Force / Sakuretsu / Trap Hole est activable | 11,0 | 7,7 | 0,0 | 48,5 |
| pose en défense un monstre plus fort que tout le terrain adverse | 5,7 | 5,5 | — | 45,0 |

(Les cibles d'attaque comptent trop peu d'occasions pour conclure : le moteur choisit seul quand il n'y a qu'une cible.)

Lecture :
- **Acquis** : attaquer directement quand le terrain est vide, poser ses magies, Pot of Greed / Upstart / Graceful Charity / Raigeki joués quasi systématiquement, pièges défensifs activés dans ~90 % des cas, Dark Hole mieux utilisé que par l'heuristique.
- **Approximatif** :
  - 17 % d'attaques suicide ;
  - 20 % d'invocations d'un monstre moins fort. Une partie est légitime (Syntoniseur, Mystic Tomato), mais l'écart avec l'heuristique (5 %) est grand ;
  - 6-7 % de fins de tour sans invoquer.
- **Pas appris** :
  - la cible de Monster Reborn : 73 %, autant que l'heuristique, qui choisit ses cibles au hasard ;
  - les effets rapides avec intention : en déterministe, MST n'est joué qu'une fois (304 fois en mode tiré), Rush Recklessly 2 fois (254), Shrink 0 fois (105). Dans chaque fenêtre de chaîne, « passer » a la plus forte probabilité : le réseau n'utilise ces cartes que grâce au bruit du tirage. Et quand il les joue, c'est sans cible une fois sur quatre ;
  - les échelles pendule : Dragong 25 activations pour 110 fins de tour où elle restait jouable, Flash Knight 16 / 11, contre 115 et 130 pour l'heuristique ;
  - les invocations Link : Knightmare Phoenix n'est jamais activé (207 fois chez l'heuristique) ;
  - les effets d'ignition sous-utilisés : Little Swordsman of Aile 7 activations contre 109, Cannon Soldier 29 contre 121, Exiled Force 37 contre 107.
- Bilan : le combat et les magies simples sont acquis ; les choix de cible, le timing des effets rapides et les mécaniques longues (pendule, Link) ne le sont pas. Plus d'entraînement (run A100) devrait en corriger une partie. Pour la cible de Reborn et les effets rapides, il faudra peut-être un adversaire plus fort que le self-play actuel ou un meilleur signal.

### Couverture des mécaniques, decks structurés, adversaires d'entraînement (2026-11-05)

**Couverture** (`bench/coverage.py`, L4, jeu aléatoire, B=512, 300 appels). Le script compte les occasions de chaque mécanique (action légale) pour 1000 décisions. Il compare deux banques procédurales, avec decks aléatoires (comme le run 2) ou structurés, aux 4 decks de vraies cartes.

| mécanique | aléatoires | structurés | vraies cartes |
|---|---|---|---|
| invocation pendule | 0,2 – 0,9 | 11 – 14 | 18 |
| échelle pendule | 10 – 11 | 20 – 25 | 22 |
| rituel | 0,9 – 2,0 | 6,0 – 7,4 | 16 |
| fusion | 4,9 – 8,7 | 8,3 – 10,5 | 6,1 |
| synchro | 2,0 – 3,9 | 2,4 – 2,8 | 6,2 |
| Xyz | 2,7 – 4,8 | 2,0 – 4,0 | 16 |
| Link | 48 – 54 | 39 – 46 | 87 |
| piège / jeu-rapide activable en chaîne | 65 – 72 | 53 – 56 | 72 |
| sélection à ≥ 2 choix | 35 – 37 | 36 – 37 | 40 |

- Avec des decks aléatoires, l'invocation pendule n'arrive quasiment jamais : un deck n'a jamais deux échelles. Le run 2 ne l'a donc jamais apprise. Le rituel est rare aussi : il faut le sort et son monstre dans le même deck.
- Les effets rapides, les choix de cible et le Link sont aussi fréquents qu'avec les vraies cartes. Leurs défauts (voir la section précédente) viennent de l'apprentissage, pas de la couverture.
- **Decks structurés** (`cardgen.structured_decks`, `ppo.py --decks structured`, le nouveau défaut) :
  - cartes en 1 à 3 exemplaires ;
  - paquets tirés avec une certaine probabilité : pendules (25 %), paires rituelles (25 %), fusion avec ses matériaux nommés (30 %), monstres d'un même niveau avec des Xyz de ce rang (30 %), Syntoniseurs (30 %) ;
  - 20 % de decks restent aléatoires ;
  - réserve de 2048 decks tirée une fois par banque (~1 s).
  
  Effet : le pendule et le rituel se rapprochent des vraies cartes. Le Xyz ne bouge pas (le paquet Xyz a été ajouté après la mesure ; un comptage statique montre que les decks aléatoires ont déjà souvent ≥ 4 monstres du rang d'un de leurs Xyz). L'écart vient probablement des niveaux : la plupart des vrais monstres sont niveau 4, alors que les niveaux procéduraux sont étalés (`LV_W`).
- L'évaluation procédurale garde des decks aléatoires, pour rester comparable aux chiffres précédents.

**Heuristique à style** (`ppo.heuristic_style`). Un style par partie, avec cinq paramètres :
- agressivité : attaquer même quand l'attaque est perdue d'avance ;
- choix raisonnés : cible d'attaque = le plus fort monstre détruit sans suicide ; cartes adverses ou de son cimetière / deck = la plus forte ; ses propres cartes sur le terrain ou en main (coûts, libérations, défausses) = la plus faible ;
- prudence : poser en défense un monstre de la main plus faible que le plus fort monstre adverse recto ;
- zèle : activer aussi dans les fenêtres de chaîne de son propre tour ;
- température du bruit.

Les règles ne lisent que l'état (ATK / DEF, positions, zones), jamais l'identité des cartes, donc elles valent pour les cartes procédurales. `STYLE_REF` redonne exactement `ppo.heuristic`, vérifié action par action. Le calcul reste négligeable.

Contre la référence (`bench/heur_styles.py`, L4, B=1024 ; victoire / défaite) :

| style | procédural | vraies cartes |
|---|---|---|
| référence | 0,496 / 0,504 | 0,498 / 0,502 |
| choix raisonnés | 0,510 / 0,490 | 0,529 / 0,471 |
| pas d'attaque perdue d'avance | 0,568 / 0,432 | 0,650 / 0,350 |
| prudent | 0,566 / 0,434 | 0,682 / 0,318 |
| calme (pas de zèle) | 0,479 / 0,521 | 0,492 / 0,508 |
| choix + pas de suicide | 0,574 / 0,426 | 0,676 / 0,324 |
| tout (pas de suicide, choix, prudent, calme) | 0,560 / 0,440 | 0,723 / 0,277 |
| bruité (température 0,5) | 0,478 / 0,522 | 0,400 / 0,600 |
| tirés comme à l'entraînement | 0,562 / 0,438 | 0,609 / 0,391 |

Sur les vraies cartes, le style « tout » bat la référence plus nettement (72 %) que le run 2 (65 %). Les styles tirés donnent un adversaire varié, en moyenne un peu plus fort que la référence. L'évaluation garde la référence pour rester comparable.

**Mélange d'adversaires** (`ppo.py --mix-heur X --mix-old Y`) :
- une part fixe des emplacements du lot joue contre l'heuristique à style (style retiré à chaque nouvelle partie), une autre contre une version de la ligue ; le reste est en self-play ;
- la ligue garde `--league-size` versions, une ajoutée toutes les `--league-every` itérations, et une version est tirée à chaque itération ;
- dans ces parties, le réseau joue le côté `b % 2` ;
- `act_step` accepte un adversaire par partie (`kind [B]`) : le réseau adverse et l'heuristique sont calculés sur tout le lot (~+13 ms par appel sur L4 avec une ligue) ;
- seules les décisions du réseau testé entrent dans le PPO (`tr["need"]`).

### Audit des briques (2026-11-05)

`gen/brick_audit.py` liste les caractéristiques d'effet des 86 vraies cartes du benchmark que le générateur ne produit jamais (10 banques de 480 cartes) : genre, opcodes par partie (coût / cible / résolution) avec leurs arguments significatifs, déclencheurs, MC_*, damage step, limites, filtres.

Première version fausse : elle comparait les arguments bruts, numéros de registre compris, et la partie où se trouvent les conditions. Elle annonçait 59 / 86 cartes couvertes et des manques qui n'en étaient pas : coûts « se sacrifier / se défausser » (MOVE (1,16,0) contre (2,16,0) dans le générateur), COND_SELF de Sangan, COND_TURN / COND_EVP des pièges (en partie coût dans le générateur, cible dans les vraies cartes). Les registres sont maintenant retirés et les COND_* comptées sans partie : **72 / 86**.

Vrais manques, comblés le même jour (`cardgen.card_clause`, `gen/procgen.lua`) :
- variantes de modificateurs de stats : DEF (Castle Walls), ATK de base / 2 (Shrink), ATK / 2 (Reverse Glasses), bonus sur soi effacé si ses effets sont annulés (`rd`, Little Swordsman), ou permanent (`perm`, Number 17) ; ces modificateurs peuvent tous être activés pendant le damage step (jeu-rapide / piège) ;
- verrou après recherche (`lock`, LOCK de Sangan / Witch) ;
- sélection « le monstre adverse au plus faible ATK » (`sel="min"`, MINATK de Fissure), pour détruire, bannir ou renvoyer en main.

Après ajout : **80 / 86**. Restent Knightmare (label, oui / non, noms différents), Berserker (matériau Link requis), Arsenal Bug, Kazejin (damage step du calcul, ATTACKER) et SHUFFLE_HAND (Graceful Charity) : cartes isolées, laissées de côté. Les nouvelles briques changent le tirage de toutes les banques, donc la banque procédurale d'évaluation : les chiffres procéduraux des runs 1-3 ne sont plus directement comparables (les vraies cartes, si).

### Troisième run : mélange d'adversaires et decks structurés (L4, 2026-11-05)

`--b 1024 --steps 64 --iters 120 --mb 8 --bf16 --eval-every 20 --bench-every 20 --old-lag 60 --mix-heur 0.2 --mix-old 0.2 --decks structured`, d=128. 20 % des parties contre l'heuristique à style, 20 % contre la ligue, le reste en self-play.
- ~17,5 s par itération, comme le run 2. Le débit affiché (2,8k décisions/s) ne compte plus que les décisions du réseau entraîné (`tr["need"]`), pas celles des adversaires. Ce n'est pas un ralentissement.
- 5,9 M de décisions entraînées, 76 400 parties, ~35 min de calcul. Paramètres : `train/ckpt/run3_it120_f16.pkl.gz`.

| itération | aléatoire (proc.) | heuristique (proc.) | contre l'itération −60 | aléatoire (vraies) | heuristique (vraies) |
|---|---|---|---|---|---|
| 19 | 80,1 % | 37,9 % | 81,0 % (it 0) | 81,4 % | 40,9 % |
| 39 | 87,3 % | 48,2 % | 85,2 % (it 0) | 87,2 % | 43,4 % |
| 59 | 91,7 % | 54,4 % | 91,5 % (it 0) | 92,4 % | 57,8 % |
| 79 | 92,8 % | 59,3 % | 71,9 % (it 20) | 94,1 % | 58,5 % |
| 99 | 95,1 % | 61,7 % | 64,8 % (it 40) | 95,6 % | 61,1 % |
| 119 | 95,5 % | 60,9 % | 59,2 % (it 60) | 95,2 % | 61,7 % |

À l'itération 119, contre l'heuristique sur les vraies cartes, par deck du réseau : contrôle 56 %, pendule 70 %, rituel / fusion 53 %, synchro 68 %. Le run 2 n'évaluait pas contre l'heuristique pendant l'entraînement : pas de point de comparaison à itération égale. Il faisait 65 % (vraies cartes) à l'itération 200.

**Comportement** (`behavior.py`, contre l'heuristique, vraies cartes) : réseau tiré 58,0 %, **déterministe 62,2 %**. Contrairement au run 2, l'argmax gagne maintenant plus que le tirage.

| sonde (% des occasions) | run 2 dét. | run 3 tiré | run 3 dét. |
|---|---|---|---|
| attaque perdue d'avance | 17,6 | 15,6 | 16,1 |
| invoque un monstre moins fort | 19,7 | 24,8 | 25,6 |
| fin de tour sans invoquer | 5,7 | 8,3 | 8,7 |
| Dark Hole en étant devant | 17,5 | 53,7 | 54,3 |
| MST sans cible | (1 occasion) | 27,6 | 28,6 |
| Monster Reborn : pas le plus fort | 72,6 | 73,3 | 75,3 |
| passe avec un piège défensif activable | 7,7 | 13,1 | 0,1 |

Activations en déterministe (run 2 → run 3) : MST 1 → 413, Shrink 0 → 145, Rush Recklessly 2 → 308, Dragong 25 → 106 (0 fin de tour où elle restait jouable), Little Swordsman 7 → 22, Cannon Soldier 29 → 56, Exiled Force 37 → 90. Knightmare Phoenix : toujours jamais activé.

Lecture :
- **Acquis grâce au run 3** : les effets rapides sont joués avec intention (en déterministe), les échelles pendule sont posées, les effets d'ignition sont bien plus utilisés.
- **Toujours pas appris** : la cible de Monster Reborn (75 %), les attaques suicide (16 %), le Link.
- **Régressions** : Dark Hole joué en étant devant (54 %, comme l'heuristique à 47 %) et invocations d'un monstre moins fort (25 %). Hypothèse : contre des adversaires heuristiques qui remplissent leur terrain, vider le terrain paie plus souvent, et la sonde (ATK totale) ne voit pas ce contexte.
- Résultat chiffré contre l'heuristique un peu en dessous du run 2 (62 % contre 65 %), mais avec 60 % des itérations et un jeu plus complet. Le plateau à ~61 % depuis l'itération 99 suggère qu'un run plus long seul ne suffira pas : la cible de Reborn et le Link demandent autre chose (couverture des briques, signal).

Vérification (2026-11-05) :
- `compare_gen.py --banks 0-3 --games 10 --focus` (moitié des decks parmi les cartes à nouvelles briques) : 40 / 40 parties identiques.
- `fuzz.py --seeds 0-3` : aucune violation d'invariant.
- `compare_gen.py --banks 0-9 --games 10` : 97 / 100. Les trois écarts viennent de règles du moteur que les nouvelles banques font apparaître, pas des nouvelles briques :
  - **relance d'attaque** (banque 6 partie 4, banque 7 partie 9) : un piège déclenché par la déclaration d'attaque invoque spécialement un monstre chez le défenseur. L'oracle redonne la main à l'attaquant (nouvelle cible ou abandon), le moteur poursuit l'attaque sur la cible d'origine (ou en direct). Règle ajoutée au moteur le 2026-11-05 (voir « Relance d'attaque » dans `ETAPE2.md`) ;
  - **LP à 0 par un coût** (banque 0 partie 4) : un effet coûte 800 LP à un joueur qui en a 800. Le moteur finit la partie tout de suite ; l'oracle fait encore choisir la cible avant de constater la défaite. Même vainqueur, quelques décisions de différence.


### Quatrième run : features numériques (L4, 2026-11-11)

Diagnostic : l'encodeur ne voyait ATK, DEF et niveau qu'en `sign·log1p(|x|)/8` (plus 16 bits de poids faible). 1500 et 2000 ATK diffèrent de 0,036 : le réseau ne pouvait presque pas comparer deux monstres. C'est cohérent avec ce qu'il n'apprenait pas (cible de Reborn, attaques suicide, invocation du plus faible).

Changement (`train/model.py`) : ATK et DEF en thermomètre (20 seuils, pas de 250 jusqu'à 5000) plus un terme linéaire x/5000, niveau en un-parmi-14 plus linéaire. 57 features (`NUM`) ajoutées à la fin de l'en-tête de carte (lu depuis la banque) et des features dynamiques (valeurs courantes `atk`, `df`, `level` de l'observation). Les anciens checkpoints se chargent (entrées tronquées). `YGO_NO_NUM=1` met ces features à zéro : témoin à architecture identique.

Expérience : deux runs L4 identiques au run 3 (`--b 1024 --steps 64 --iters 120 --mb 8 --bf16 --mix-heur 0.2 --mix-old 0.2 --decks structured`), A = témoin (`YGO_NO_NUM=1`), B = features numériques. ~19 s par itération pour les deux : coût négligeable.

| itération | A heur. (proc.) | B heur. (proc.) | A heur. (vraies) | B heur. (vraies) |
|---|---|---|---|---|
| 19 | 38,9 % | 38,5 % | 36,6 % | 39,7 % |
| 39 | 46,5 % | 46,3 % | 40,9 % | 47,6 % |
| 59 | 50,5 % | 54,3 % | 47,5 % | 52,8 % |
| 79 | 55,4 % | 56,5 % | 50,1 % | 58,5 % |
| 99 | 59,3 % | 61,8 % | 56,3 % | 66,7 % |
| 119 | 60,2 % | 62,8 % | 56,0 % | **72,6 %** |

Contre l'aléatoire à l'itération 119 : A 95,2 % / 93,6 % (proc. / vraies), B 96,8 % / 96,0 %. Par deck du réseau (vraies cartes, heuristique, it 119) : A contrôle 62 %, pendule 60 %, rituel / fusion 44 %, synchro 58 % ; B 70 %, 74 %, 62 %, 85 %.

`behavior.py` (contre l'heuristique, vraies cartes, B=1024) : A tiré 53,9 %, déterministe 57,5 % ; **B tiré 70,3 %, déterministe 70,7 %** (run 3 : 58,0 / 62,2 %).

| sonde (% des occasions, déterministe) | run 3 | A | B |
|---|---|---|---|
| attaque perdue d'avance | 16,1 | 13,6 | 10,1 |
| invoque un monstre moins fort | 25,6 | 25,3 | 10,2 |
| fin de tour sans invoquer | 8,7 | 0,2 | 2,4 |
| Dark Hole en étant devant | 54,3 | 41,4 | 52,6 |
| MST sans cible | 28,6 | 29,7 | 28,5 |
| Monster Reborn : pas le plus fort | 75,3 | 80,2 | 84,8 |
| pose en défense un monstre dominant | — | 16,2 | 19,4 |

Lecture :
- Le gain sur les vraies cartes (+16,6 points contre l'heuristique à itération égale) dépasse de loin la variance entre runs (le témoin A fait 56 % là où le run 3, même configuration, faisait 61,7 %). Il est plus fort sur les vraies cartes que sur les procédurales (+2,6) : les vraies cartes ont des ATK plus étalées, et les procédurales sont aussi le terrain d'entraînement, où les deux réseaux s'ajustent.
- Les sondes qui dépendent d'une comparaison d'ATK s'améliorent nettement (invocation du plus fort 25 → 10 %, attaques perdues 14 → 10 %).
- La cible de Monster Reborn ne s'améliore pas. Vérification de la sonde (`train/rbdump.py CKPT --b 128`, qui affiche candidats et choix) : elle compte bien ce qu'elle dit, mais elle compare l'ATK de base, sans les effets (Arsenal Bug tombe à 1000 sans autre Insecte : Mystic Tomato est alors le bon choix). Sur 26 occasions du réseau B en déterministe : 77 % d'« erreurs », contre 66 % attendus pour un choix au hasard parmi les mêmes candidats ; l'heuristique fait ~54 % sur ses propres occasions (bruit et styles). Le réseau n'a donc pas de préférence pour le monstre fort, sans biais net contre non plus. Hypothèse : dans les banques procédurales, la cible d'une résurrection pèse peu sur le résultat ; à vérifier par une sonde équivalente sur les cartes procédurales.
- Paramètres : `train/ckpt/run4_it120_f16.pkl.gz` (format bf16 → f16, en-tête 219 entrées).

### Cinquième et sixième runs : run long, banques riches en résurrections (L4, 2026-11-11)

Deux runs L4 lancés en parallèle, même configuration que le run 4 B (features numériques) :
- **run 5** : 360 itérations (`--iters 360 --eval-every 40 --old-lag 120`), pour voir où plafonne la configuration ;
- **run 6** : 120 itérations avec `--gen-revive 4` : les banques d'entraînement tirent quatre fois plus souvent une invocation spéciale depuis le cimetière avec choix de la cible (`Gen(revive)` dans `gen/cardgen.py`, `ss_gy` → `ss` avec `sel=target|pick` sur Z_GRAVE ; 26 → 88 clauses d'invocation spéciale par banque). Banques d'évaluation inchangées. Vérifié avant le run : `compare_gen.py --banks 0-3 --games 10 --revive 4` 39 / 40 (le seul écart est un défaut moteur indépendant des résurrections, `notes/bug-verrou-attaque-atk-phase-2026-11-11.md`), `fuzz.py --revive 4` graines 0-1 sans violation.

Contre l'heuristique (procédural / vraies cartes) :

| itération | run 5 proc. | run 5 vraies | run 6 proc. | run 6 vraies | run 4 B vraies |
|---|---|---|---|---|---|
| 39 | 49,9 % | 53,2 % | 54,5 % | 53,8 % | 47,6 % |
| 79 | 57,6 % | 64,4 % | 59,3 % | 61,8 % | 58,5 % |
| 119 | 62,8 % | 77,2 % | 63,7 % | 75,5 % | 72,6 % |
| 159 | 66,0 % | 79,9 % | | | |
| 199 | 69,4 % | 80,2 % | | | |
| 239 | 69,7 % | 77,4 % | | | |
| 279 | 69,4 % | 78,2 % | | | |
| 319 | 73,8 % | 83,2 % | | | |
| 359 | **74,9 %** | **83,8 %** | | | |

Run 5 à l'it 359 : contre l'aléatoire 98,8 % / 99,7 %, contre lui-même à l'it 240 55,7 %, par deck (vraies, heuristique) contrôle 89 %, pendule 88 %, rituel / fusion 71 %, synchro 87 %. Les deux runs à 120 itérations (run 4 B : 72,6 %, run 6 : 75,5 %, run 5 : 77,2 %) donnent une idée de la variance entre runs : ~5 points.

`behavior.py` (vraies cartes, contre l'heuristique) : run 5 tiré 82,5 %, **déterministe 84,8 %** ; run 6 tiré 73,3 %, déterministe 75,2 %.

| sonde (% des occasions, déterministe) | run 4 B | run 6 | run 5 |
|---|---|---|---|
| attaque perdue d'avance | 10,1 | 4,3 | 3,0 |
| invoque un monstre moins fort | 10,2 | 8,0 | 14,6 |
| fin de tour sans invoquer | 2,4 | 6,4 | 1,6 |
| Dark Hole en étant devant | 52,6 | 44,0 | 6,1 |
| MST sans cible | 28,5 | 31,2 | 14,3 (14 activations) |
| Monster Reborn : pas le plus fort | 84,8 | **96,6** | **43,9** |
| pose en défense un monstre dominant | 19,4 | 12,8 | 5,6 |
| finit le tour avec une magie activable | 1,5 | 0,0 | 34,5 |
| passe avec Mirror Force / Sakuretsu / Trap Hole activable | 0,0 | 0,0 | 38,7 |

Lecture :
- Le run long paie : +11 points sur les vraies cartes entre 120 et 360 itérations, avec un palier entre 160 et 280 puis une reprise. La courbe n'est pas plate à 360 : un run plus long (A100) se justifie.
- Le run 5 a changé de style : il garde ses magies rapides et ses pièges (MST joué 14 fois en déterministe contre 286 en tiré, Rush Recklessly 10 contre 262, Mirror Force non activée dans 39 % des occasions). Les sondes « magie gardée » et « piège passé » le comptent comme une erreur, mais garder un MST sans cible ou un piège pour une meilleure attaque n'en est pas une ; le taux de victoire monte. Ces deux sondes sont à revoir (ne compter que les magies sans valeur de réserve, et les pièges contre une attaque létale).
- **Les résurrections ajoutées au générateur n'aident pas la cible de Reborn** (run 6 : 96,6 % d'« erreurs », pire que le run 4). Le run long, lui, la fait passer à 44 % (hasard ~66 %).
- Sonde équivalente sur les banques procédurales d'évaluation (`train/gysel.py CKPT --b 256`, `GYSEL_BANKS=6` : sélections dans le cimetière du joueur 0, suivi de la carte choisie jusqu'à sa zone d'arrivée, erreur = ne prend pas le plus fort en ATK) :

  | paramètres | vers la main : occasions, erreur (hasard) | dont plus fort de niveau ≤ 4 | vers le terrain | bannie |
  |---|---|---|---|---|
  | run 3 | 753, 3,3 % (50 %) | — | 14, 64 % (63 %) | 47, 72 % (64 %) |
  | run 4 B | 107, 94,4 % (53 %) | 91, 94,5 % | 12, 75 % (62 %) | 34, 68 % (64 %) |
  | run 5 | 471, 2,1 % (50 %) | 20, 50 % | 18, 50 % (65 %) | 46, 54 % (59 %) |
  | run 6 | 102, 18,6 % (52 %) | 11, 82 % | 11, 64 % (54 %) | 28, 86 % (58 %) |

  Les sélections vers la main (récupérations) dominent ; le sens de la préférence change d'un run à l'autre (run 3 et 5 : le plus fort ; run 4 : le plus faible). Quand le plus fort est de niveau ≤ 4, run 5 et 6 ne le prennent plus : la préférence porte plutôt sur le niveau (monstre à sacrifice) que sur l'ATK. Les résurrections vers le terrain restent trop rares sur ces banques pour conclure (11-18 occasions).
- Sonde sur des banques riches en résurrections (`GYSEL_REVIVE=4`) : en cours.
- Paramètres : `train/ckpt/run5_it360_f16.pkl.gz`, `train/ckpt/run6_it120_f16.pkl.gz`. ~1,94 CU/h par session L4 ; 157 CU restantes après les runs 4 à 6.

## Procédures inhérentes depuis la main (2026-11-06)

Le générateur inclut `hand_proc` parmi les effets des monstres du deck principal :
conditions de terrain vide, adversaire sans monstre, terrain vide face à un
adversaire présent, ou présence d'un allié correspondant à un filtre ; limite
optionnelle par nom. Un monstre ne reçoit qu'une telle procédure. Le budget
attribue 0,7 carte d'avantage à la possibilité d'invocation supplémentaire.

Les briques `K_PROC`, `FILTER`, `REQ` (borne inférieure ou supérieure) et
`REQ_FREE` portent ces conditions. Aucun nouvel axe du modèle ni champ de partie ;
le genre 7 utilise un emplacement déjà réservé de l'encodeur. `gen/procgen.lua`
produit indépendamment la procédure `EFFECT_SPSUMMON_PROC` pour l'oracle.
`compare_gen.py --focus` inclut désormais ces cartes et renvoie un code non nul
si une partie échoue. L'audit distingue le genre PROC et le sens des bornes REQ.

Banques 0–1, cinq parties chacune, 80 cartes principales / 10 Extra, focus :
10/10 parties identiques et **44 procédures de main effectivement choisies**.
Un défaut ancien de filtre sur SELF après destruction au combat a été révélé
et corrigé au passage. Bilan complet, provenance et débit :
`notes/codex-procedures-2026-11-06.md`. Aucun entraînement CPU effectué.

## Paiements des procédures et protection au combat (2026-11-07)

Le générateur et sa bibliothèque Lua produisent les coûts LP, défausse et
bannissement du cimetière des procédures, ainsi que la protection temporaire
contre la destruction au combat. Les formes des banques et du modèle restent
inchangées. Les limites d'activation des magies/pièges générés sont transmises
aux briques JAX.

Scénarios de paiements : 6/6 parties identiques, avec quatre paiements LP,
quatre défausses et deux bannissements effectivement joués. Protection générée :
2/2, 40 résolutions et six attaques vers un monstre protégé. Nouvelle passe
banques 0–1, cinq parties chacune, 80 cartes principales / 10 Extra, focus :
10/10. Provenance et détails : `notes/codex-couts-registres-2026-11-06.md`.

## Pièges continus à effet rapide et compteurs OATH (2026-11-07)

Le générateur produit une activation vide de piège continu accompagnée d'un
effet rapide de terrain, avec compteurs par carte ou par nom, éventuellement
OATH. Les banques gardent des formes fixes. Le réseau lit l'effet initial
optionnel, le caractère remboursable du compteur et la nature publique du
maillon. Encodeur et politique des anciens checkpoints sont vérifiés contre
une extension nulle des poids. Scénarios ciblés et mesures CPU :
`notes/codex-pieges-compteurs-2026-11-07.md`. Aucun entraînement CPU.
