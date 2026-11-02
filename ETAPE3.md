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
