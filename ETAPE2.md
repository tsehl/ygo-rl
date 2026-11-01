# Étape 2 — Langage de briques et représentation d'état JAX (conception)

Leçons des benchmarks (`bench/RESULTATS.md`) appliquées dès le départ :
- aucune opération O(N²) dans le moteur : l'ordre (deck, cimetière) est un entier explicite par carte ;
- reset uniquement des parties finies, à partir d'une réserve de parties pré-générées ;
- le réseau ne voit que les cartes visibles (~30-48 tokens), le reste est résumé ;
- les tables d'effets sont partagées (une banque de cartes globale), pas copiées dans chaque partie.

Oracle : `oracle/ygopro-core/bin/release/libocgcore.dylib` (ygopro-core 11.0, `oracle/BUILD.md`). Le flag `DUEL_PSEUDO_SHUFFLE` (0x10) supprime le mélange aléatoire, donc les deux moteurs peuvent jouer exactement la même partie.

## 1. Principe : un moteur qui s'arrête à chaque décision

Au Yu-Gi-Oh, les décisions tombent au milieu des résolutions (choisir une cible, oui/non, choisir un nombre, l'ordre des déclencheurs…). Le moteur JAX est donc une machine à états : `step(état, action)` applique l'action puis **avance jusqu'à la prochaine décision** d'un joueur (ou la fin de la partie). Chaque décision est une action unique de l'agent ; une sélection de plusieurs cartes = plusieurs choix successifs + « terminé ».

`step` = une boucle `lax.while_loop` (bornée) sur une micro-machine : phase de jeu → fenêtre d'action → construction de chaîne → résolution (interpréteur de bytecode) → collecte d'événements → déclencheurs → nouvelle chaîne, etc. La boucle sort dès qu'un état « en attente de décision » est atteint.

Espace d'actions unifié (un seul masque, une seule tête de sortie) :
- pointer une carte (slot) × un effet (jusqu'à E) : activer, invoquer, attaquer, choisir une cible, choisir un matériau…
- quelques actions spéciales : passer/terminé, oui/non, nombre (0..9), zone (0..6), position, option d'effet (0..3), phase suivante.

## 2. Représentation d'état (struct of arrays, tailles fixes)

N = nombre de slots de cartes par partie : 2 × (60 deck + 15 extra) + 10 jetons = 160 (configurable). Toutes les cartes d'une partie existent dès le départ ; un jeton est un slot réservé.

**Par carte (dynamique)** : `card_id` (int32, index dans la banque), `zone` (deck, main, zone monstre, zone magie/piège, terrain, cimetière, bannie, extra, matériel Xyz, hors-jeu), `seq` (position dans la zone : case 0-4/0-6, ou rang dans le deck / cimetière), `controller`, `owner`, `pos` (recto ATK / recto DEF / verso DEF / verso), `status` (bits : invoqué ce tour, a attaqué, a changé de position, effet annulé, activé ce tour, …), `summon_type`, `reason` et `prev_zone/prev_controller/prev_pos` (pour les « si détruite et envoyée au cimetière depuis le terrain »), `overlay_of` (slot du monstre Xyz ou -1), `counters`, `equip_to`, `turn_id` (tour d'arrivée dans la zone).

Piocher = prendre la carte de `seq` minimal du deck : O(N). Mélanger = permuter `seq`.

**Global** : LP[2], tour, phase, joueur du tour, invocation normale utilisée, état du combat (attaquant, cible, étape de dommage), compteurs « une fois par tour » (table [joueur, clé] avec clé = id de carte ou id d'effet), chaîne (pile de profondeur max CH=16), file d'événements, table des modificateurs actifs, pile de décision en attente.

**Maillon de chaîne** : slot source, index d'effet, joueur, cibles (masque booléen [N]), registres sauvegardés (label/nombre choisis au coût), catégories annoncées + zones concernées (pour Ash Blossom & co : « ajoute / invoque / envoie depuis le deck »), annulé (activation / effet).

**Banque de cartes (statique, globale, en mémoire GPU)** : pour chaque carte : type (bits monstre/magie/piège × normal/effet/fusion/synchro/xyz/link/rituel/pendule/jeton, continue/rapide/terrain/équipement/contre…), attribut, type de monstre, niveau/rang/lien, ATK/DEF de base, flèches Link, échelles Pendule, archétypes (jusqu'à 4), matériaux requis (pour les invocations depuis l'Extra Deck), et E=8 effets compilés. Banque de 50-200 k cartes procédurales + les vraies cartes traduites. Une partie ne stocke que des `card_id`.

## 3. Le langage de briques

Un effet = un **en-tête** (données fixes, lues sans exécuter de code) + 4 **programmes** de bytecode.

En-tête :
- `kind` : activation (carte magie/piège), ignition, rapide, déclenché obligatoire / optionnel, continu (modificateur permanent), procédure d'invocation ;
- `range` : zones depuis lesquelles l'effet est utilisable (main, terrain, cimetière…) ;
- `event` : pour les déclenchés/rapides, type d'événement + filtre sur la carte concernée (`SINGLE` = la carte elle-même, ou filtre général) + raison (destruction, combat, effet, coût) ;
- `limit` : 1×/tour soft, 1×/tour hard (par nom), 1×/duel, aucune ;
- `categories` : bits (pioche, recherche, invocation spéciale, destruction, annulation…) — nécessaires pour les cartes qui réagissent à « un effet qui ferait X » ;
- `spell_speed` (déduit du kind pour l'essentiel).

Programmes : `condition`, `cost`, `target`, `resolve`. Chacun = suite d'au plus P=24 instructions `(op, a, b, c)`.

Machine à registres : 8 registres **ensembles** (masque booléen [N]) + 8 registres **entiers**. Les registres de `cost`/`target` sont sauvegardés dans le maillon de chaîne et relus par `resolve` (= le `SetLabel` / `GetTargetCards` des scripts Lua).

Filtre de cartes = **un enregistrement fixe**, pas un arbre d'expression : masques (types, attributs, types de monstre, zones, positions), intervalles (niveau, ATK, DEF), archétype, nom (= ou ≠), propriétaire (moi / adversaire / les deux), `self`/`not self`, recto/verso, « ciblable », « peut être invoqué spécialement ». Évaluer un filtre = une dizaine de comparaisons sur N : bon marché et vectorisé. Couvre la grande majorité des `aux.FilterBoolFunction` / filtres locaux des scripts. Un filtre composé (« OU ») = deux filtres + union de registres.

Instructions (première liste, à compléter pendant la traduction des premières cartes) :
- Requêtes : `SELECT r ← filtre` ; `UNION/INTER/MINUS` ; `COUNT i ← r` ; `SUM_ATK`, `MAX_ATK` ; `EVENT_CARDS r` (les cartes de l'événement déclencheur) ; `CHAIN_LINK i` ; `LP i ← joueur` ; comparaisons et `JUMP_IF`.
- Décisions (**points d'arrêt** du moteur) : `CHOOSE r ← r', joueur, min, max, ciblé ?` ; `YESNO` ; `CHOOSE_OPTION n` ; `ANNOUNCE_NUMBER` ; `CHOOSE_ZONE`.
- Actions : `MOVE r → zone, position, raison` (couvre invoquer spécialement, ajouter à la main, détruire, envoyer au cimetière, bannir, renvoyer au deck, défausser, libérer, attacher comme matériel) ; `DRAW joueur n` ; `DAMAGE / RECOVER joueur n` ; `CHANGE_POS r` ; `NEGATE_ACTIVATION / NEGATE_EFFECT maillon` ; `ADD_COUNTER / REMOVE_COUNTER` ; `SHUFFLE`.
- Modificateurs : `MODIFY r, stat, valeur, durée` (ATK/DEF/niveau/attribut/type…) ; `GRANT r, drapeau, durée` (« ne peut pas être détruit par combat », « ne peut pas invoquer spécialement sauf… », « ne peut pas attaquer »…). Ils créent une entrée dans la table des modificateurs.
- `FAIL` (l'effet ne peut pas être activé : utilisé dans `condition`/`target` en mode test).

Comme dans les scripts Lua (`chk==0`), `cost` et `target` sont exécutés une première fois **en mode test** (sans choix ni effet de bord) pour calculer le masque d'actions légales. Le mode test ne doit contenir aucun `CHOOSE` bloquant : il remplace un choix par « au moins min candidats existent ».

### Modificateurs continus

Une table de M=64 entrées par partie : `(source, effet, code, filtre ou masque cible, valeur, durée, condition d'existence)`.
- Effet continu d'une carte recto sur le terrain (« les Dragons que vous contrôlez gagnent 500 ATK ») : cible = filtre réévalué, existe tant que la source est recto sur le terrain et non annulée.
- Effet temporaire issu d'une résolution (« gagne 500 ATK jusqu'à la fin du tour ») : cible = masque figé, durée = fin de tour / fin de phase / tant que recto.

Les stats courantes (ATK, DEF, niveau, interdictions) sont **recalculées** à partir des stats de base + table, en O(M × N), après chaque action qui modifie le terrain. Les couches de règles d'ygopro (`SET_ATTACK_FINAL` > `SET_ATTACK` > `UPDATE_ATTACK`, ordre d'application) sont reproduites dans ce calcul, vérifiées contre l'oracle.

## 4. Règles de chaîne et déclencheurs

- File d'événements de taille fixe (type, slot, raison, joueur responsable) remplie par les actions.
- Après chaque action ou résolution : on cherche les effets déclenchés dont l'événement correspond (vectorisé sur N × E), avec la distinction « quand » (peut rater le moment) / « si ».
- SEGOC : déclencheurs obligatoires du joueur du tour, puis ceux de l'adversaire, puis les optionnels. Le joueur choisit l'ordre parmi les siens (décision).
- Vitesses de sort 1/2/3 et passage de priorité : chaque ajout à la chaîne ouvre une fenêtre de réponse à l'adversaire, puis au joueur, jusqu'à deux « passer » consécutifs.

## 5. Ce que voit le réseau

- Tokens = cartes visibles par le joueur (sa main, les deux terrains, les deux cimetières, les bannies recto, les matériels Xyz), regroupés par cumsum en O(N), max 64 (48 avant le jalon 4) + un token global (LP, phase, tour, chaîne en cours, décision demandée).
- Son propre deck/extra deck : connus comme multiensemble → un token résumé (moyenne des embeddings).
- Encodage d'une carte = encodeur de ses effets (en-tête + bytecode). Les paramètres ne changent pas pendant un rollout : l'encodage des ~150 cartes de chaque partie est calculé **une fois par rollout**, pas à chaque coup.

## 6. Couverture des vraies cartes : position honnête

On ne reproduit pas les cartes : on reproduit le moteur (toutes les règles : chaîne, timing, toutes les invocations) et les briques. L'entraînement se fait sur des cartes procédurales générées directement dans le langage. Les vraies cartes ne servent qu'à l'évaluation finale : on traduit seulement les decks de test (quelques centaines de cartes au plus), et on ajoute une brique si l'une d'elles l'exige.

## 7. Vérification contre l'oracle

- Pilote Python (ctypes) d'ygopro-core : crée un duel avec `DUEL_PSEUDO_SHUFFLE`, charge les scripts et `cards.cdb`, répond aux demandes de décision.
- Même partie dans le moteur JAX (en local sur CPU) avec les mêmes choix ; comparaison de l'état (zones, positions, ATK/DEF, LP) après chaque décision.
- ygopro-core charge n'importe quel script Lua : un petit compilateur briques → Lua permet de vérifier le moteur sur des **cartes procédurales** aussi, pas seulement sur de vraies cartes. Le générateur de cartes devient alors un générateur de tests.
- Correspondance des décisions entre les deux moteurs : c'est la partie délicate (ygopro-core pose ses questions dans son propre format, `MSG_SELECT_*`). À traiter dès le jalon 1.

## 8. Jalons

1. **Squelette** : état, pioche, phases, invocation normale/pose, combat complet (attaque directe, calcul des dégâts, destruction), monstres normaux seulement, pas de chaîne. Pilote de l'oracle + comparaison sur des parties aléatoires avec decks de monstres normaux.
2. **Interpréteur** : filtres, registres, `SELECT/CHOOSE/MOVE/DRAW/DAMAGE`, magies normales et effets d'ignition. Premières vraies cartes simples (Pot of Greed, Raigeki, Card Trooper…).
3. **Chaîne** : pièges, effets rapides, déclencheurs, SEGOC, file d'événements, annulations (Ash Blossom).
4. **Modificateurs** continus et temporaires, durées, couches ATK/DEF.
5. **Extra Deck** : fusion, synchro, xyz, link, rituel, pendule.
6. Générateur de cartes procédurales, réseau, PPO, premiers entraînements sur A100.

Chaque jalon : moteur JAX vérifié contre l'oracle avant de passer au suivant. Les optimisations de vitesse attendent d'avoir ce moteur réel.

## Avancement

**Jalon 1 fait (2026-10-28).** `engine/core.py` (moteur JAX), `engine/compare.py` (harness contre l'oracle), `engine/vmap_check.py` (lot vmappé = parties jouées une par une).
- 330/330 parties aléatoires identiques à ygopro-core, état et actions légales comparés à chaque décision (graines 0-29 et 1000-1299, ~60 000 décisions).
- Côté oracle, on écarte les options « annuler » et « retirer ». Les décisions forcées (option unique, ou sélection où il faut tout prendre) sont jouées automatiquement des deux côtés.
- Détails trouvés en route :
  - plusieurs cartes envoyées au cimetière ensemble y arrivent dans l'ordre de `card_operation_sort` (contrôleur, zone, puis seq), pas dans l'ordre du choix ;
  - les sélections carte par carte (msg 26) renvoient -1 aussi bien pour « terminer » que pour « annuler ».
- Limite : la politique aléatoire ne couvre pas tout (cas rares : défausse de 2 cartes ou plus, combats à 0 ATK contre 0 ATK).

**Observation et information cachée (2026-10-29).** `engine/obs.py`, tests dans `engine/obs_check.py`.
- `new_game(..., key)` mélange les decks et répartit les cartes au hasard dans les slots. Sans clé, l'ordre reste fixe pour la comparaison avec l'oracle.
- `observe(bank, s, p)` produit 48 tokens au plus (64 depuis le jalon 4) : sa main, tout le terrain (les cartes verso adverses sans identité), les cimetières et les bannies, plus un vecteur global (LP, tour, phase, décision, compteurs dont la main adverse et les decks). L'ordre des tokens ne dépend que d'informations visibles. `slot` sert seulement à reconstruire l'action et n'est pas une entrée du réseau.
- Tests :
  - 800 observations restent identiques quand on change les cartes cachées et qu'on renumérote les slots ;
  - toutes les actions légales ont un token ;
  - contrôle positif : changer une carte visible change bien l'observation.
- Reste à faire : les bits « déjà vue par X » arriveront avec les effets qui révèlent des cartes ou les renvoient en main (jalon 2+). Les cimetières au-delà de 48 tokens sont tronqués, en gardant les cartes les plus récentes et celles visées par une action.

**Jalon 2 : premiers effets (2026-10-29).** `engine/cards.py` (12 vraies cartes écrites en briques + assembleur), interpréteur d'effets dans `engine/core.py`.
- Cartes : Pot of Greed, Raigeki, Dark Hole, Hinotama, Dian Keto, Upstart Goblin, Fissure, Monster Reborn, Graceful Charity, Harpie's Feather Duster, Exiled Force, Cannon Soldier.
- Un effet = genre (activation de magie / effet d'ignition) + 3 programmes (coût, cible, résolution) de 12 instructions au plus, sur 4 registres de cartes. Opcodes : FILTER, SELF, REQ, REQ_FREE, MINATK, CHOOSE, MOVE, SPSUMMON, DRAW, DAMAGE, RECOVER, SHUFFLE_HAND. Une instruction par itération de `step`, et chaque choix du joueur interrompt l'exécution (D_SELECT, D_ZONE, D_POS).
- Nouvelles actions : poser une magie, activer (depuis la main, posée, ou un monstre sur le terrain), choisir une case magie/piège, choisir la position (invocation spéciale).
- Comparaison avec l'oracle, decks de 28 normaux + 12 cartes à effet : 129/130 parties identiques au premier passage. La partie en échec a révélé un bug : en cas de destruction mutuelle au combat, les deux monstres partaient l'un après l'autre et non simultanément. Ça ne se voyait qu'avec un monstre réanimé chez l'adversaire, quand les deux cartes vont dans le même cimetière. C'est corrigé. Après correction : 100/100 parties mixtes (graines 100-199) et 40/40 parties de monstres normaux identiques. Couverture : chaque carte activée des deux façons (depuis la main et posée), 42 mélanges de main, 48 choix de position.
- Graceful Charity mélange la main avec le vrai RNG de l'oracle. Le moteur JAX ne mélange pas, et le harness recopie l'ordre de la main de l'oracle après chaque `shuffle_hand`. C'est sans effet sur l'agent, puisque l'ordre de la main ne porte aucune information.
- `obs_check` passe : les actions légales (activables ou non) ne fuient pas l'information cachée. La perturbation du test garde le type public des cartes posées (monstre en zone monstres, magie en zone magie).
- Limites :
  - chaîne d'un seul maillon (pas de réponse) ;
  - CHOOSE avec min = max seulement ;
  - MOVE vers le cimetière seulement ;
  - pas de pièges ni d'effets déclenchés ;
  - `legal` évaluait `test_ok` sur toutes les cartes (vmap), ce qui coûtait cher. Corrigé au jalon 3b (compaction en KC candidats, voir plus bas).

**Jalon 3a : chaînes, pièges et jeux-rapides, fenêtres de priorité (2026-10-29).**
- Cartes ajoutées : Jar of Greed et Just Desserts (pièges), Mystical Space Typhoon (jeu-rapide), Magic Jammer (contre-piège, coût : défausser 1) et Seven Tools of the Bandit (contre-piège, coût : 1000 LP). Les deux derniers nient l'activation d'une magie ou d'un piège et détruisent la carte.
- Pile de chaîne de 12 maillons au plus. Chaque maillon garde sa carte, son effet, son joueur, ses registres de cibles et un drapeau « nié ». La vitesse d'un maillon (1, 2 ou 3) borne ce qui peut lui répondre. La chaîne se résout du haut vers le bas, puis les magies et pièges résolus partent ensemble au cimetière.
- Fenêtres de priorité, sur le modèle de ygopro-core :
  - fenêtre « rapide » ouverte après chaque action : invocation, pose, activation, déclaration d'attaque, passage de phase ;
  - fenêtres de phase : Draw, Standby, début et fin du combat, End.
  - Deux passes de suite ferment une fenêtre. La suite dépend du point de retour (`ret`) et du fait qu'une chaîne s'est résolue ou non.
- Nouveaux opcodes : COND_CHAIN, CHAINCARD, NEGATE, PAY_LP, et DAMAGE avec multiplicateur. Nouvelle action : passer. Côté réseau, l'observation expose le rang de chaque carte dans la chaîne et son drapeau « nié ».
- Comparaison avec l'oracle, avant les deux correctifs ci-dessous : 39/40 + 98/100 parties mixtes et 57/60 parties « chain » (22 normaux, 6 effets, 12 magies et pièges rapides). Tous les échecs sont corrigés et revérifiés un par un. Après les correctifs : 100/100 parties « chain » (graines 600-699). Couverture : chaque carte de chaîne a été activée, Seven Tools 12 fois au minimum.
- Détails trouvés en route :
  - aller en End Phase depuis le combat passe d'abord par la Main 2, avec une fenêtre pour l'adversaire, car ygopro-core simule un « to_ep » depuis l'idle ;
  - deck vide à la pioche de la Draw Phase : la défaite n'est pas immédiate. Les fenêtres de la Draw Phase s'ouvrent quand même, et la défaite tombe au prochain `adjust`, c'est-à-dire à la prochaine activation ou à la fin de la fenêtre de phase. C'est le champ `odraw`. Les pioches dues à un effet font perdre tout de suite (non revérifié dans le cas d'une décision en cours de résolution) ;
  - MAX_ITER=256 itérations internes par `step`.
- `vmap_check` et `obs_check` passent.
- Limites :
  - pas d'effets déclenchés, donc pas de SEGOC ;
  - pas d'effets rapides de monstre ;
  - CHOOSE avec min = max seulement ;
  - les cibles des maillons ne sont pas dans l'observation.

**Jalon 3b fait : effets déclenchés, SEGOC, effets rapides de monstre (2026-10-29).**
- Volet 1, pièges sur événement : Trap Hole, Torrential Tribute, Sakuretsu Armor, Mirror Force. Les événements (invocation, déclaration d'attaque…) sont accumulés puis exposés au point d'événement suivant, où les pièges peuvent répondre. 10/10 parties « chain » identiques (graines 500-509), puis 60/60 avec le volet 2 en place (graines 1200-1259). Bug trouvé en route : OP_EVENT lisait le masque d'événement dans le mauvais champ de l'instruction.
- Volet 2, effets déclenchés de monstre : Sangan, Witch of the Black Forest (envoyés au cimetière), Man-Eater Bug (retournement), Mystic Tomato (détruit au combat). Nouveaux genres d'effet : déclencheur obligatoire et déclencheur optionnel.
  - Événements : envoi au cimetière, destruction au combat, retournement. Chaque carte garde sa zone précédente (`pzone`) pour les conditions du type « envoyée du terrain au cimetière ».
  - Point d'événement (fin de chaîne, invocation, pose, changement de position, étapes du damage step) : les déclencheurs en attente sont collectés, puis mis en chaîne dans l'ordre SEGOC de l'OCG : obligatoires du joueur du tour, obligatoires de l'adversaire, optionnels du joueur du tour, optionnels de l'adversaire. La fenêtre rapide s'ouvre ensuite, ou se rouvre telle qu'elle était si personne n'a rien activé.
  - Damage step découpé comme dans ygopro-core : calcul des dégâts, fenêtre avec les déclencheurs de retournement, puis envoi au cimetière des monstres détruits au combat et deuxième fenêtre avec les déclencheurs « détruit au combat ». Un monstre détruit au combat reste sur le terrain jusque-là mais ne peut plus être ciblé (`is_capable_be_effect_target`).
  - Nouveaux opcodes : COND_SELF (condition sur la zone actuelle et précédente de la carte source), LOCK. Une fois par tour par nom (`opt`). CHOOSE accepte « jusqu'à n » (min < max), MOVE vers la main, filtres sur la DEF et l'attribut.
  - Défaite par deck vide différée : elle tombe au `break_effect` qui suit le coût et la cible d'une activation, pas au début de l'activation.
- Limites connues :
  - « manquer le timing » n'est pas modélisé : ygopro efface les déclencheurs optionnels sans EFFECT_FLAG_DELAY quand l'événement n'est pas la dernière chose arrivée. Les quatre cartes du volet sont toutes sûres de ce point de vue ;
  - pas de mélange du deck après une recherche (l'oracle tourne en DUEL_PSEUDO_SHUFFLE). Pour l'entraînement, il faudra une clé RNG dans l'état ;
  - la condition d'un déclencheur n'est évaluée qu'au point d'événement ;
  - verrous et « une fois par tour » remis à zéro au début de chaque tour ;
  - pas de déclencheurs de terrain (EFFECT_TYPE_FIELD : « quand un monstre adverse… ») ;
  - un monstre détruit au combat est exclu de toutes les sélections, alors que ygopro ne l'exclut que des sélections avec ciblage. La différence ne compte que pour un effet sans cible résolu pendant le damage step ;
  - le harness suppose que l'effet déclenché d'une carte est son effet 0.
- Volet 3, effets rapides de monstre : D.D. Crow (depuis la main, bannit une carte du cimetière adverse), Ghost Ogre & Snow Rabbit (depuis la main ou le terrain, détruit la carte dont l'effet est activé sur le terrain), Winged Rhynos (retourne en main en réponse à un piège).
  - Nouveau genre d'effet K_QUICK (vitesse 2), avec ses zones d'activation (`erng` : main et/ou zone monstres recto). COND_CHAIN accepte « le maillon précédent est un effet d'une carte sur le terrain », CHAINCARD une zone quelconque, MOVE envoie aussi vers les bannies.
  - Bug trouvé en route : toute activation depuis la main passait par le choix d'une case magie/piège. Seules les activations de carte y passent maintenant.
- Comparaison avec l'oracle (code final) : trig 100/100 (graines 1000-1099), quick 80/80 (1600-1679 : 20 normaux, 9 effets rapides, 11 autres), chain 60/60, mixed 60/60, normal 40/40. Couverture du volet 3 : D.D. Crow activé 394 fois, Ghost Ogre 36 (main et terrain), Winged Rhynos 56. Les modes autres que quick excluent les effets rapides, pour que les anciennes graines rejouent les mêmes decks.

**Moteur sous vmap : premières corrections de coût (2026-10-29).** `vmap_check` (16 parties en lot de 400 pas, CPU Colab, un cœur) prenait 420 s de compilation et plus d'une heure d'exécution.
- `legal` testait coût et cible (`test_ok`) pour chaque carte et chaque effet. Sous vmap, chaque instruction calcule ses 22 branches, dont plusieurs en O(N) : le tout était quadratique en N. Désormais on compacte les (carte, effet) candidats en KC = 24 cases (`jnp.nonzero(size=KC)`) et on ne teste qu'elles. Même chose pour les déclencheurs, testés une seule fois pour les 4 étapes SEGOC au lieu de 4.
- `step` est découpé en deux boucles : `run` exécute les effets sans évaluer `legal`, qui n'est calculé qu'aux points de décision. Avant, un lax.cond le recalculait à chaque instruction (sous vmap, un cond calcule ses deux branches).
- Cache de compilation persistant de JAX (`~/.cache/ygo-jax`, ou la variable YGO_JAX_CACHE).
- Résultat : compilation 241 s, exécution 700 s, lot identique aux parties jouées une par une. Ce sont des chiffres CPU. La vitesse réelle se mesurera sur GPU.
- Fuite mémoire du harness, corrigée : `new_game` appelé hors jit recompilait `step` à chaque partie (lax.cond sur une lambda neuve), soit environ 1,1 Go de plus par partie. `compare.py` l'appelle maintenant via un `jax.jit` unique. Une partie passe de 25 s à 6 s sur Colab.
- Observation : les recherches dans le deck rendaient légal le choix d'une carte que l'observation ne montrait pas (`obs_check` : 34 états sur 8347). Une carte que le joueur peut choisir lui est maintenant montrée, sans sa place dans le deck (tri par identifiant). Pendant une recherche, le test d'invariance considère tout le deck du joueur comme visible. `obs_check` passe : 8347 états, 800 observations invariantes.
- Pistes suivantes, à décider après une mesure sur GPU (traitées plus bas) :
  - les exécutions d'effets (`exec_one`) calculent encore, sous vmap, toutes les branches de fin de chaîne et de déclencheurs ;
  - la boucle de `step` attend la partie la plus lente du lot. L'alternative : une micro-étape par appel, l'enveloppe RL n'interrogeant le réseau qu'aux vraies décisions.

**Moteur sur GPU (L4, 2026-10-29).** Premier test L4 : environ 480 pas/s à B = 1024, inutilisable. Corrections, toutes validées contre l'oracle (non-régression complète) et par `vmap_check` :
- `_switch` / `_cond` remplacent `lax.switch` / `lax.cond` dans l'exécution : sous vmap, toutes les branches sont calculées de toute façon, mais seuls les champs réellement modifiés sont sélectionnés (avant : l'état entier, pour chaque branche). `_pure` ne rend plus qu'une ligne du registre par branche. `test_ok` ne lit que les PT = 6 premières instructions des programmes de coût et de cible (`cards.assemble` le vérifie).
- `trig_eval` est différé (état interne D_TRIG) et traité dans `settle`, au lieu d'être calculé à chaque instruction exécutée.
- `step_batch(bank, s, a, frac)` : version lot pour l'entraînement. Chaque partie s'arrête à sa prochaine décision. Avec frac > 0, l'appel rend la main quand il ne reste plus que cette fraction du lot en route, et les parties en retard reprennent à l'appel suivant. Avec frac = 0, le résultat est identique à vmap(step), champ par champ.
- Horloge de déplacement (`clk`, `mv`, `lk_t`) : une carte déplacée après l'activation sort du registre du maillon au chargement (`load_link`). Avant, chaque déplacement effaçait la carte dans `lk_reg` ([CH, R, N] recopié à chaque `move`).
- Passes automatiques : une fenêtre de réponse où le joueur ne peut rien activer (`auto_pass`) est passée par `settle`, sans aller-retour avec la politique. Les actions forcées tombent de 3,67 à 0,58 par vraie décision. 96 % étaient des passes en D_CHAIN.
- **Bug de coût sous vmap dans `draw_n`.** La boucle de pioche prenait pour borne le b de l'instruction courante, quelle qu'elle soit, puisque le switch évalue toutes les branches. Avec DAMAGE b = 1000, cela faisait des milliers de tours par appel. La borne est maintenant masquée par `op == OP_DRAW`. Ce seul correctif triple le débit.
- Résultat (L4, frac = 0,05) : après les quatre premiers points, 4 865 décisions/s à B = 1024 et 2 801 à B = 4096 ; avec les passes automatiques et le correctif `draw_n`, 14 846 et 14 657. À frac = 0, environ 9 500. Le moteur est maintenant limité par le GPU : environ 60 ms de kernels par appel à B = 1024.
- Profil restant : 58 % du temps GPU passe dans deux fusions élément par élément de forme [B, KC, N], que XLA compile avec Triton : le test des candidats dans `legal` (`act_ok` → scan de `test_ok`) et le même test dans `trig_eval`. Les flags XLA qui désactivent ces fusions n'y changent rien. Les command buffers CUDA n'apportent plus que 2 %.
- Pistes, si le moteur pèse face au réseau : précalculer hors du scan de `test_ok` les `filter_mask` des instructions ; alléger `group_send`.

**Jalon 4, volet 1 : archétypes et modificateurs de stats (2026-10-30).** Première pièce de l'architecture des effets continus.
- Banque : `setcode` (4 archétypes de 16 bits par carte, règle d'ygopro `(sa&0xfff)==(sb&0xfff) and (sa&sb)==sb`). Filtres : archétype, race, nom (un ou deux codes). Plus d'ATK/DEF de base dans les filtres : ils lisent l'ATK/DEF courante.
- **Effets continus** (genre `K_CONT`, en-tête `emod`) : code `MC_*`, cible (la carte elle-même, SINGLE_RANGE, ou le filtre 0 sur les monstres recto, EFFECT_TYPE_FIELD avec des zones relatives au contrôleur de la carte), valeur (× nombre de cartes du filtre 1), condition fixe (joueur du tour, phases). Un effet s'applique tant que sa carte est recto dans ses zones. Une magie/piège continue doit en plus être résolue : bit `ST_ON`, posé quand son maillon se résout sans être nié, comme `enable_field_effect` d'ygopro. Une magie continue reste sur le terrain après résolution.
- **Modificateurs temporaires** (`OP_MODIFY`, RESETS_STANDARD_PHASE_END) : table de TM = 16 entrées par partie (code, masque [N] et valeur par carte, horloge). Une carte déplacée après la création en sort grâce à l'horloge `mv`. La table est vidée au changement de tour, comme le `reset_phase(PHASE_END)` d'ygopro à la fin de la PhaseEvent de l'End Phase (après la défausse de fin de tour).
- **Calcul** (`stats`, d'après `card::get_attack`) : les effets continus actifs sont compactés en KM = 16 cases (`nonzero`, ordre carte puis effet, comme les id d'effet d'ygopro), suivis des modificateurs temporaires dans l'ordre de création. ATK = max(0, base, ou dernier SET_BASE_ATTACK, + somme des UPDATE). Coût O((KM + TM) × N). Hors zone monstres : stats de base.
- **Cache** : `atk_c` / `df_c` dans l'état, recalculés (`refresh`) après chaque micro-pas (`apply`, chaque instruction exécutée, `settle`). Le combat les recalcule lui-même après avoir retourné la cible.
- **Débordements** : champ `ovf` de l'état (bit 1 : plus de KM effets continus actifs, bit 2 : plus de TM modificateurs dans le tour). `compare.py` le traite comme une erreur ; l'entraînement devra le compter. Avec KM = 8, deux parties `cont` sur 100 débordaient (jusqu'à 11 effets actifs, decks très chargés en cartes continues).
- **Observation** : T passe de 48 à 64 tokens. `obs_check` a trouvé une sélection de Monster Reborn à 53 cibles (les deux cimetières), dont 5 sans token. Question ouverte : avec des decks de 60 cartes (N = 160), une sélection sur les cimetières peut dépasser n'importe quel T fixe.
- Cartes : E - Emergency Call (archétype E-HERO), Bladefly (terrain, les deux camps, ±), Banner of Courage (magie continue, condition tour + Battle Phase), Dark Magician Girl (300 × Dark Magician / Magician of Black Chaos dans les cimetières), Back-Up Rider (+1500 ciblé jusqu'à la fin du tour), Deepsea Macrotrema (déclencheur, +500 à tous ses monstres EAU), The Little Swordsman of Aile (ignition, coût : libérer un autre monstre).
- Mode `cont` de `compare.py` : 14 normaux, 5 E-HERO, 2 Dark Magician, 13 cartes du jalon 4, 6 autres. Les cartes du jalon 4 sont exclues des anciens modes, qui rejouent donc les mêmes decks. `engine/regress.sh` lance toute la non-régression (graines ci-dessous) en parallèle, puis `obs_check` et `vmap_check`.
- Comparaison avec l'oracle : cont 100/100 (graines 3000-3099), et toute la non-régression inchangée (trig 100/100, quick 80/80, chain 60/60, mixed 60/60, normal 40/40). `obs_check` et `vmap_check` passent. Couverture sur 100 parties : Back-Up Rider 321 activations, Banner of Courage 308, Emergency Call 229, Little Swordsman of Aile 399. Sur 4 parties, 589 états comparés sur 1 342 ont une ATK modifiée, dont 188 avec Banner actif en Battle Phase. Sur 20 parties, le +500 de Deepsea Macrotrema est actif dans 23 états, mais jamais sur deux monstres à la fois (trop peu de monstres EAU).
- Coût GPU pas encore mesuré : `refresh` ajoute O((KM + TM) × N) à chaque micro-pas.
- Repoussé au volet 2 : les fenêtres du damage step (Rush Recklessly, Shrink, Castle Walls et la plupart des cartes de stats s'y activent), SET_ATTACK / SET_ATTACK_FINAL, les interdictions (Command Knight : « ne peut pas être ciblé pour une attaque »).

**Jalon 4, volet 2 : damage step et couches ATK/DEF complètes (2026-10-30).**
- **Damage step découpé comme le BattleCommand d'ygopro (MR5)**. Fenêtres R_DS1 (début, étape 20), R_DS2 (après le retournement de la cible, étape 22) et R_DS3 (PHASE_DAMAGE_CAL, étape 24, événement `EV_PRE_DMG` = EVENT_PRE_DAMAGE_CALCULATE). Puis calcul des dommages, R_DMG1 (après le calcul) et R_DMG2 (après l'envoi au cimetière), comme avant. La cible verso est retournée à l'étape 21, mais son événement FLIP n'est levé qu'au calcul (`flip_delayed`). `dstep` vaut DS_BEFORE, DS_CAL ou DS_AFTER.
- **Annulation** (adjust, étape 14) : l'attaquant ou la cible a bougé depuis le début du damage step (`ds_t`, horloge `mv`), ou l'attaquant est en défense. On saute alors à la fin (étape 33).
- **Activation au damage step** (`effect::is_activateable`) : en-tête `edmg` avec les bits DMG_STEP (EFFECT_FLAG_DAMAGE_STEP, fenêtres DS_BEFORE), DMG_CAL (EFFECT_FLAG_DAMAGE_CAL, fenêtre DS_CAL) et DMG_STAT (aux.StatChangeDamageStepCondition : rien après le calcul). Exceptions : les contre-pièges, et les effets sur un événement de combat (EV_PRE_DMG). Les déclencheurs ne sont pas concernés.
- **Couches ATK/DEF** (`stats`, pli ordonné de `card::get_attack` / `get_defense`) :
  - UPDATE (`up` pour les SINGLE temporaires, `upc` pour les continus) ;
  - SET (un SET FIELD remet `up` à zéro) ;
  - SET_FINAL (SINGLE temporaire : immédiat et remet `up` / `upc` à zéro ; continu : appliqué en dernier) ;
  - SET_BASE (continus d'abord, temporaires dans l'ordre).
- **Ordre des effets = id d'effet d'ygopro**, tiré du compteur `field_id`, qui sert à plusieurs choses :
  - un modificateur temporaire reçoit son id à la création ;
  - les effets continus d'une carte reçoivent un nouvel id quand elle devient active (`enable_field_effect` : arrivée recto sur le terrain, retournement, résolution d'une magie continue ; `en_t`) ;
  - un effet à condition en reçoit un nouveau quand sa condition redevient vraie (`effect::is_available`, `cav` / `cav_t`). Exemple : un Banner of Courage activé avant un Reverse Glasses repasse après lui à chaque Battle Phase, et son +200 s'ajoute au SET_ATTACK_FINAL.

  Le pli trie les KM + TM entrées selon ces dates (horloge `clk`).
- **`card::add_effect`** : un modificateur temporaire SET_ATTACK_FINAL efface définitivement les UPDATE / SET / SET_FINAL temporaires déjà sur la carte. Un SET_ATTACK ou SET_BASE_ATTACK efface les SET / SET_FINAL / SET_BASE. Même règle pour la DEF (`MC_KILL`). Exemple trouvé par l'oracle : l'ATK à 0 de Kazejin retire pour de bon le +500 de Deepsea Macrotrema.
- Un effet FIELD d'un monstre détruit au combat ne s'applique plus (`is_available`, STATUS_BATTLE_DESTROYED).
- Valeurs « moitié » (Shrink, Reverse Glasses) : Lua calcule `x/2` en flottant, et `effect::get_value` arrondit avec `std::round`, donc moitié supérieure.
- Limites par carte (SetCountLimit(1) sans code) : bits ST_CNT (remis à zéro à chaque tour) et ST_CNTN (EFFECT_FLAG_NO_TURN_RESET), effacés quand la carte bouge (`reset_effect_count`). `eopt` : OPT_NAME, OPT_CARD, OPT_CARD_NR.
- Nouveaux opcodes : COND_ATK (la carte attaque / est attaquée), ATTACKER. OP_MODIFY prend un mode de valeur (MV_CONST, MV_HALF_BASE_ATK, MV_HALF_ATK) et une durée (TD_TURN, TD_DCAL = jusqu'à la fin du calcul des dommages, `reset_phase(PHASE_DAMAGE_CAL)`).
- Cartes : Rush Recklessly, Shrink, Castle Walls, Reverse Glasses, Kazejin (effet rapide sur PRE_DAMAGE_CALCULATE, une fois tant qu'il reste sur le terrain), Arsenal Bug (SET_ATTACK / SET_DEFENSE continus, condition « aucun autre Insecte recto »). Mode `dmg` de `compare.py` : 16 normaux, 14 cartes du volet 2, 6 du volet 1, 4 autres.
- **Deux fenêtres en trop corrigées autour de la déclaration d'attaque.** Elles étaient invisibles tant qu'aucune carte ne pouvait s'activer en Battle Phase hors des fenêtres de chaîne ; le mode `ds` les a révélées.
  - Une chaîne résolue dans la fenêtre de l'étape 9 (R_ATK2) n'est suivie que de la fenêtre ouverte par SolveChain. Cette fenêtre remet `returns` à FALSE, donc l'étape 10 ne rejoue pas l'étape 9 : on entre en damage step. L'analyse précédente supposait l'inverse.
  - Une attaque annulée (attaquant détruit, déplacé ou plus en attaque recto depuis la déclaration, `ds_t`) saute l'étape 9 : pas de fenêtre R_ATK2, retour au choix des attaques.
  - Pas encore modélisé : la reprise d'attaque (`attack_rollback`, quand les monstres adverses changent pendant ces fenêtres), qui demande au joueur s'il rejoue son attaque (décision oui/non).
- Mode `ds` de `compare.py` : les decks du mode `dmg`, mais la politique aléatoire écarte 9 fois sur 10 l'activation des cartes du volet 2 hors damage step. Sans ce biais, ces cartes partent avant la Battle Phase : sur 20 parties `dmg`, aucune décision réelle dans les deux premières fenêtres du damage step. Avec le biais : 160 décisions, et 4 échecs sur 20 qui ont mené aux deux corrections ci-dessus.
- Comparaison avec l'oracle : dmg 100/100 (graines 4000-4099), ds 100/100 (5000-5099), et toute la non-régression inchangée (trig 100/100, quick 80/80, cont 100/100, chain 60/60, mixed 60/60, normal 40/40). `obs_check` et `vmap_check` passent. Couverture `dmg` sur 100 parties : Castle Walls 400 chaînes, Shrink 425, Rush Recklessly 411, Reverse Glasses 366, Kazejin 58 ; Magic Jammer et Seven Tools y répondent. Sur 20 parties `ds` : 160 décisions dans les fenêtres DS_BEFORE, 16 en DS_CAL ; Arsenal Bug à 1000 dans 1 734 états comparés.
- Coût GPU toujours pas mesuré (le tri des KM + TM entrées et le pli s'ajoutent à `refresh`).

**Jalon 4, volet 3a : interdictions et protections (2026-10-30).**
- **Drapeaux par carte** `fl_c`, recalculés par `refresh` à partir des effets continus actifs, comme les stats : FL_NO_ATTACK (EFFECT_CANNOT_ATTACK), FL_NO_BTARGET (CANNOT_BE_BATTLE_TARGET), FL_NO_ETARGET (CANNOT_BE_EFFECT_TARGET), FL_IMM_TRAP (IMMUNE_EFFECT, effets de piège), FL_NO_SS (CANNOT_SPECIAL_SUMMON visant des cartes, dans toutes les zones). S'y ajoutent `ib_c` (INDESTRUCTABLE_BATTLE : ATK minimale du monstre adverse, BIG = aucune) et `pf_c` (interdictions des joueurs : PF_NO_TRAP = CANNOT_ACTIVATE sur les cartes piège).
- **Immunité** (`is_affect_by_effect`), calculée en premier dans `stats` :
  - les effets continus de piège ne s'appliquent pas aux monstres immunisés ;
  - à la résolution d'un effet de piège, MOVE et MODIFY les ignorent (`Duel.Destroy`, et `RegisterEffect` qui ignore un effet non applicable).
  - En revanche l'immunité **n'empêche pas le ciblage** : `is_capable_be_effect_target` ne teste que CANNOT_BE_EFFECT_TARGET, STATUS_BATTLE_DESTROYED et STATUS_SUMMONING. L'oracle l'a montré : Castle Walls peut cibler Elemental HERO Wildheart, puis le +500 ne s'applique pas.
- **Ciblage** : drapeau `tgt` sur FILTER (bit 8 de c) et sur ATTACKER, qui exclut les cartes FL_NO_ETARGET. Il est ajouté aux cartes existantes qui ciblent dans leur script (SelectTarget / SetTargetCard) : Monster Reborn, Exiled Force, Mystical Space Typhoon, Trap Hole, Sakuretsu Armor, Man-Eater Bug, D.D. Crow, Back-Up Rider, Rush Recklessly, Shrink, Castle Walls, Kazejin. L'exclusion des monstres détruits au combat reste appliquée à toutes les sélections (limite connue).
- **Attaques** : un monstre FL_NO_ATTACK n'est pas proposé, ni un monstre sans cible possible quand l'adversaire contrôle des monstres. `get_attack_target` prend `select_target = true` par défaut, y compris pour la liste des attaquants : un premier essai qui l'ignorait a été corrigé grâce à l'oracle. Les monstres FL_NO_BTARGET ne sont pas proposés comme cible. L'attaque directe reste interdite tant que l'adversaire contrôle un monstre, même non ciblable. Si les cibles disparaissent entre la déclaration et le choix, l'attaque compte comme déclarée, sans fenêtre (`attack_announce_failed`, étapes 4-7).
- **Mémoire** : le pli des stats déroulé en Python (32 entrées, deux fois) faisait grossir chaque appel inliné de `refresh`. Résultat : environ 3,8 Go par processus `compare.py`, et l'OOM killer sur 4 processus en parallèle. Le pli est maintenant un `lax.scan` sur les entrées (ATK et DEF ensemble), à 1,75 Go par processus, compilation comprise.
- **Combat** : un monstre n'est pas détruit si l'ATK de son adversaire est >= `ib_c` (la valeur d'Obnoxious Celtic Guard reçoit le monstre adverse).
- **Nouveaux champs de filtre** : statut (F_ST, ex. invoqué ou invoqué par flip ce tour ; nouveau bit ST_FLIP_TURN), niveau min / max, F_SPSUM (IsCanBeSpecialSummoned, sur Monster Reborn et Mystic Tomato).
- Cartes : Gravity Bind (piège continu, les monstres de niveau 4 ou plus ne peuvent pas attaquer), Big-Tusked Mammoth, Command Knight (ne peut pas être ciblé pour une attaque s'il y a un autre monstre de son côté, +400 aux Guerriers), Obnoxious Celtic Guard, Lord of D., Elemental HERO Wildheart, Ally of Justice Quarantine (pas d'invocation spéciale de monstres LUMIÈRE), Mirage Dragon (l'adversaire ne peut pas activer de pièges pendant la Battle Phase). Mode `flag` de `compare.py` : 14 normaux, 14 cartes du volet 3a, 4 des volets 1-2, 8 des anciens modes.
- Couverture sur 20 parties `flag` (états comparés) :
  - 28 choix de cible d'attaque avec Command Knight protégé ;
  - 415 phases de combat avec un attaquant bloqué ;
  - 925 avec un Dragon non ciblable ;
  - 876 avec un monstre LUMIÈRE non invocable au cimetière ;
  - 13 avec Wildheart face à une chaîne de piège ;
  - 167 décisions d'un joueur privé de pièges ;
  - Gravity Bind activé 70 fois.
- Comparaison avec l'oracle : flag 100/100 (graines 6000-6099), toute la non-régression inchangée (trig, quick, cont, dmg, ds, chain, mixed, normal). `obs_check` et `vmap_check` passent.

**Jalon 4, volet 3b : annulation d'effets (2026-10-30).**
- **Statut « effets annulés »** : bit FL_DISABLED de `fl_c` (EFFECT_DISABLE, STATUS_DISABLED). Il vient :
  - des modificateurs temporaires MC_DISABLE (jusqu'à la fin du tour : Forbidden Chalice, Effect Veiler) ;
  - des effets continus MC_DISABLE (Skill Drain : les monstres à effet recto), d'une carte elle-même non annulée.
- **Conséquences** :
  - les effets continus d'une carte annulée ne s'appliquent plus (stats, interdictions, protections, immunité) ;
  - à la résolution, le maillon d'un monstre annulé encore lié au maillon (sur le terrain, pas déplacé depuis l'activation) est annulé (`solve_chain` : STATUS_DISABLED et `is_has_relation`). Cela couvre aussi `NegateRelatedChain`.
- **Une carte annulée peut toujours activer ses effets** : `is_activateable` ne teste pas STATUS_DISABLED. Le coût est payé et l'effet annulé à la résolution. Un premier essai qui interdisait l'activation a été corrigé grâce à l'oracle. Vérifié sur 30 parties `neg` : 54 activations par des monstres annulés (Cannon Soldier, Little Swordsman of Aile, Kazejin, Exiled Force), toutes identiques à l'oracle.
- **Ordre de calcul dans `stats`** :
  1. annulations temporaires ;
  2. immunité des cartes non annulées ;
  3. annulations continues, qu'un monstre immunisé aux pièges ignore ;
  4. reste des effets, sans ceux des cartes annulées ;
  5. immunité finale.
- **Historique** : l'immunité (SINGLE_RANGE) d'une carte ne compte que si cette carte était déjà recto sur le terrain et non annulée au calcul précédent (bit FL_ONF). Dans ygopro, `refresh_disable_status` passe avant STATUS_EFFECT_ENABLED dans `enable_field_effect`, et `is_available` exclut une carte annulée. Un Elemental HERO Wildheart qui arrive sur le terrain sous Skill Drain est donc annulé et le reste ; un Skill Drain activé après Wildheart ne le touche pas. Cas trouvé par l'oracle.
- **EFFECT_DISABLE_EFFECT** (Forbidden Chalice, Effect Veiler ; pas Skill Drain, qui n'a qu'EFFECT_DISABLE) : quand une carte du terrain qui en est affectée active un effet, `AddChain` annule son maillon avant le coût (EFFECT_DISABLE_CHAIN). Le maillon reste annulé même si la carte quitte le terrain, par exemple libérée en coût (`ch_dis`, puis `lk_neg`). Cas trouvé par l'oracle : un Cannon Soldier sous Forbidden Chalice qui se libère lui-même.
- **RESET_DISABLE** : un modificateur temporaire posé par une carte sur elle-même avec ce drapeau (`MODIFY(..., rd=True)`, le +700 de Little Swordsman of Aile) disparaît quand la carte est annulée.
- **Encodage d'OP_MODIFY** : b = code (5 bits) | mode << 5 | durée << 9 | rd << 13. Nouvel opcode COND_PHASE, nouveau champ de filtre F_NODIS (not IsDisabled).
- Cartes : Skill Drain (piège continu, coût 1000 LP), Forbidden Chalice (jeu-rapide, damage step, +400 et annulation), Effect Veiler (effet rapide depuis la main pendant la Main Phase adverse). Mode `neg` de `compare.py` : 14 normaux, 10 cartes du volet 3b, 12 monstres à effet des autres volets, 4 autres.
- Couverture sur 20 parties `neg` (états comparés) : 3 945 avec un monstre annulé sur le terrain, dont 731 Wildheart, 745 Dark Magician Girl, 884 Command Knight. Skill Drain activé 78 fois, Forbidden Chalice 104, Effect Veiler 22.
- Comparaison avec l'oracle : neg 100/100 (graines 7000-7099), toute la non-régression inchangée (trig, quick, cont, dmg, ds, flag, chain, mixed, normal). `obs_check` et `vmap_check` passent.

**Coût du jalon 4 (mesures CPU, 2026-10-30).** Banc `bench/engine_bench.py` (step_batch, frac = 0,05, politique aléatoire, parties finies remplacées).
- CPU, B = 64, mêmes cartes (celles d'avant le jalon 4) : moteur `9e6349a` 482 décisions/s ; moteur actuel 119 décisions/s, soit environ ×4 plus lent.
- Cause : `refresh` (stats, drapeaux, annulations) coûte ~13,5 ms pour 256 parties, contre 3,8 ms pour `exec_one`, et il tourne après chaque instruction exécutée. Le coût croît avec KM (environ 1,4 ms par case à B = 256) ; les filtres n'en sont qu'environ un tiers, le reste est une centaine de petites opérations [K, N].
- Déjà fait : pli des stats sous forme close (dernier index et sommes masquées sur [K, N], plus de `lax.scan` séquentiel, qui sur GPU lançait 32 étapes de petits kernels à chaque instruction) ; un filtre de moins par effet continu (compte et condition fusionnés dans le filtre 1). Aucun gain sur CPU, gain attendu sur GPU, à mesurer.
- **Mesure L4 (2026-10-31)**, `bench/engine_bench.py`, frac = 0,05, 40 appels :

  | moteur | B = 1024 | B = 4096 |
  |---|---|---|
  | `9e6349a` (avant le jalon 4) | 15 782 décisions/s (62 ms/appel) | 14 918 (264 ms/appel) |
  | actuel, toutes les cartes | 7 012 (140 ms) | 4 921 (797 ms) |
  | actuel, cartes d'avant le jalon 4 | 6 647 (148 ms) | 4 712 (834 ms) |

  Le surcoût vient du moteur (stats, drapeaux, annulations, fenêtres), pas des nouvelles cartes. Il est de ×2,2 à B = 1024 et de ×3 à B = 4096. La compilation passe de ~100 s à ~220 s.
- **Composants sur L4** (B = 1024, un appel vmappé, sur l'état initial) :

  | composant | ancien | actuel |
  |---|---|---|
  | `exec_one` | 0,88 ms | 1,22 ms |
  | `refresh` | — | 1,01 ms |
  | `legal` | 4,61 ms | 6,26 ms |
  | `trig_eval` | 4,68 ms | 5,88 ms |
  | `apply` | 0,60 ms | 0,82 ms |

  Une itération de la boucle d'exécution (`exec_one` + `refresh`) passe de 0,88 à 2,2 ms. `legal` et `trig_eval` prennent 26 à 36 % de plus (filtres plus larges, damage step).
- Coût de la mesure : 0,64 CU.

**Optimisation du coût du jalon 4 (2026-10-31).** Compteurs d'itérations sur CPU (B = 256, par appel de step_batch) : 34 pas de `settle` (presque tous avec `pass_priority`), 18 pas d'exécution d'effets, 4 `legal`. Mais seules 3,4 % des parties changent réellement d'état après un `pass_priority`, et 0,6 % après une instruction.
- **`refresh` retiré de `ds_damage`** (et donc de `battle`). Sous vmap, cette branche de `quick_end` était calculée à chaque `pass_priority` et à chaque `apply`, soit deux `refresh` de plus à chaque fois. L'état est déjà à jour en entrant, et `settle` recalcule après.
- **Recalcul seulement si l'état a changé** :
  - `changed(prev, s)` compare les entrées de `stats` qui bougent (zones, positions, statuts, contrôleurs, phase, joueur du tour, tour, horloge, modificateurs temporaires). `refresh` est idempotent sur un état inchangé, y compris l'historique de l'immunité.
  - Chemin d'une partie (`step`) : `refresh_if`, avec un `lax.cond`.
  - Chemin du lot (`step_batch`) : `refresh_b` ne recalcule que les parties modifiées, compactées dans un sous-lot de B/8 (tout le lot s'il y en a plus). La boucle d'exécution des effets passe au niveau du lot (`run_b`). Réécrire seulement les champs produits par `refresh` dans le sous-lot s'est révélé plus lent (330 contre 482 décisions/s sur CPU) : tout l'état est réécrit.
- **Programmes de test compacts** (`bank.tprog`) : coût puis cible sans les OP_END de remplissage, sur la longueur du plus long de la banque (6 instructions aujourd'hui, au lieu de 2 × PT = 12). `test_ok`, appelé par `legal` et `trig_eval` sur KC candidats, coûte deux fois moins.
- **`engine/sb_check.py`** (ajouté à `regress.sh`) : `step_batch` (frac = 0) donne les mêmes états et masques que `vmap(step)`, champ par champ, sur 32 parties × 150 décisions. `compare.py` ne passe que par `step`.
- Résultats (CPU, B = 64, cartes d'avant le jalon 4) : 119 → 180 (sans les refresh de `ds_damage`) → 381 (recalcul des seules parties modifiées) → 482 (programmes de test compacts), comme le moteur `9e6349a`.
- Résultats L4 (`engine_bench.py`, frac = 0,05, toutes les cartes) : **13 497 décisions/s à B = 1024, 12 801 à B = 4096**, contre 7 012 / 4 921 avant et 15 782 / 14 918 avant le jalon 4. Il reste environ 15 % de surcoût pour tout le jalon 4. Coût des deux mesures : 0,31 CU.

**Jalon 5, volet a : Extra Deck et fusion (2026-10-31).** Ce que montre l'oracle pour Polymerization :
1. choix du monstre fusion (msg 15) ;
2. matériaux choisis un par un (msg 26) : seuls sont proposés ceux qui laissent la recette réalisable, et la sélection se termine d'elle-même quand elle est complète ;
3. case (msg 18, 7 cases dont les zones de monstre extra 5 et 6, MR5 + DUEL_FSX_MMZONE) ;
4. position (msg 19).

- **Taille de partie** : `new_game(..., extra0, extra1)` avec X cartes d'Extra Deck par joueur, d'où N = 2 × (D + X). L'Extra Deck est une zone ordonnée (`compact`), verso. Sans Extra Deck (X = 0), rien ne change pour les anciens modes.
- **Banque** :
  - `fmat` [C, FK = 4, FW] : une recette, soit un filtre par matériau (main et terrain) ;
  - `fmn` : nombre de matériaux ;
  - `erl` : EnableReviveLimit.
- **État** :
  - `proc` (STATUS_PROC_COMPLETE) : mis à l'invocation depuis l'Extra Deck, effacé en main, dans le deck ou l'Extra Deck. Monster Reborn (F_SPSUM) refuse une carte à réanimation limitée qui n'a pas été correctement invoquée ;
  - `fz_c` (cache de `refresh`) : les monstres fusion de l'Extra Deck dont la recette est réalisable. Test de Hall sur les 2^FK sous-ensembles d'emplacements, en O(EXK × 2^FK × N). Sans case libre pour le monstre fusion, un matériau doit venir du terrain (`GetLocationCountFromEx` avec les matériaux) : le test de Hall est refait en imposant un emplacement tenu par un monstre du terrain. Cas trouvé par l'oracle.
- **OP_FUSMAT** (D_SELECT, `fsel`) : un matériau n'est proposé que s'il existe une affectation des matériaux déjà choisis et de ce candidat à des emplacements distincts (les FK! = 24 permutations), avec un test de Hall sur les emplacements restants (et la contrainte de terrain ci-dessus). Pas de choix joué d'office tant qu'il reste plus d'une option : l'oracle présente chaque choix.
- **Cases** : un monstre invoqué depuis l'Extra Deck peut aller dans les 5 cases principales libres et dans les zones extra 5 / 6. La règle sans Link : aucune si le joueur en occupe déjà une ; sinon chacune, si l'adversaire n'occupe pas la case symétrique (`is_location_useable` : 11 − seq).
- **Libérations** : sans case principale libre, l'une des libérations doit libérer une case principale. Un monstre en zone extra ne suffit pas (cas trouvé par l'oracle).
- Un monstre de l'Extra Deck renvoyé en main retourne dans l'Extra Deck. L'observation montre son propre Extra Deck ; celui de l'adversaire est caché.
- Cartes : Polymerization ; Flame Swordsman, Gaia the Dragon Champion (deux noms), Blue-Eyes Ultimate Dragon (trois matériaux identiques). Mode `fus` de `compare.py` : 20 normaux, 12 matériaux, 3 Polymerization, 2 Monster Reborn, 3 autres ; Extra Deck de 6 (2 de chaque).
- `sb_check`, `vmap_check` et `obs_check` jouent maintenant avec des Extra Decks (6 monstres fusion, 15 % de matériaux dans les decks).
- Comparaison avec l'oracle : fus 100/100 (graines 8000-8099), toute la non-régression inchangée ; `obs_check`, `vmap_check` et `sb_check` passent (avec Extra Decks).
- **Coût du volet 5a.** La première version faisait tomber le CPU de 482 à 172 décisions/s, même sans Extra Deck : `fusion_ok` prenait 159 ms sur les 174 ms d'un `refresh` (B = 256). Quatre correctifs :
  - chaque carte est résumée par le masque des emplacements qu'elle peut remplir, et le test de Hall (avec ses variantes « un emplacement tenu par le terrain ») se fait sur l'histogramme de ces masques (main / terrain), en O(16 × 16) ;
  - filtre de matériau allégé (`mat_mask`, champs MAT_FIELDS ; `assemble` vérifie qu'une recette n'en utilise pas d'autres) ;
  - `fusion_ok` seulement si une partie du lot a des monstres fusion dans son Extra Deck, et `legal` sans `fus_picks` si aucune partie du lot ne choisit de matériaux ;
  - un seul `group_send` pour un renvoi en main qui envoie des monstres de l'Extra Deck dans l'Extra Deck (destination par carte), au lieu de deux.

  Résultat sur CPU (B = 64, sans Extra Deck) : 420 décisions/s.
- **Mesure L4** (`engine_bench.py`, frac = 0,05) :

  | | B = 1024 | B = 4096 |
  |---|---|---|
  | sans Extra Deck | 12 651 décisions/s (13 497 avant le volet) | 13 046 |
  | Extra Deck de 15 (`--extra 15`, N = 110 au lieu de 80) | 10 699 | — |

  Coût de la mesure : 0,36 CU.

**Jalon 5, volet b : rituel (2026-10-31).** Ce que montre l'oracle (procédure EDOPro `Ritual.AddProcGreaterCode` / `AddProcEqualCode`) :
1. choix du monstre rituel en main (msg 15), parmi ceux pour lesquels un ensemble de libérations valide existe ;
2. libérations :
   - avec une case principale libre, `SelectWithSumGreater` / `SelectWithSumEqual` (msg 23, MSG_SELECT_SUM) : **une seule décision**, le sous-ensemble entier. `driver.py` énumère les sous-ensembles valides (Greater : somme ≥ niveau et somme − plus petit < niveau ; Equal : somme exacte) ;
   - sans case principale libre, `SelectUnselectGroup` (msg 26) : carte par carte, avec la contrainte qu'une libération libère une case principale ; la sélection se termine d'elle-même ;
3. case, puis position. L'invocation est correcte (`CompleteProcedure`) : Monster Reborn peut ensuite réanimer le monstre.

- **Matériaux** (`rit_mats`) : monstres de la main et du terrain du joueur, de niveau > 0, sauf le monstre rituel lui-même.
- **Test d'existence** (`ritual_ok`, cache `rz_c`, bit 0 Greater, bit 1 Equal) : sommes de sous-ensembles par masques de bits sur l'histogramme des niveaux (main / terrain), niveaux de LVMAX = 12 à 1, sommes < SMAX = 26 (`sum_ok`). Une seule fois par niveau de monstre rituel (vmap sur 13 niveaux), pas par carte. Coût indépendant de N. Calculé seulement si une partie du lot a un monstre rituel en main (`has_ritual`).
- **OP_RITMAT** (D_SELECT, `rsel` = mode) : un matériau est proposé s'il existe un ensemble valide qui contient les choix déjà faits et ce candidat (`rit_picks`, calcul par classe niveau × main / terrain). La sélection se termine quand la somme atteint le niveau (`rit_done`).
- **Une décision de l'oracle = plusieurs choix JAX** : avec une case libre, le moteur joue d'office la sélection quand l'union des candidats est elle-même le seul ensemble valide (règle `uniq` de `forced_m`), car l'oracle n'a alors qu'une option. Le harness de `compare.py` saute, dans la suite d'actions d'une réponse msg 23, les cartes déjà choisies d'office. Sans case libre (msg 26), pas de règle `uniq` : l'oracle propose chaque carte.
- `SPSUMMON(r, pos, proc=True)` pose `proc` (invocation correcte). Les monstres rituels ne s'invoquent ni ne se posent normalement (`can_ns`). Les sorts rituels portent aussi TYPE_RITUAL : les tests de monstre rituel demandent TYPE_MONSTER | TYPE_RITUAL.
- Cartes : Black Luster Ritual (Greater, niveau 8) → Black Luster Soldier ; End of the World (Equal) → Demise, King of Armageddon (ignition : payer 2000 LP, détruire toutes les autres cartes du terrain). Ruin n'est pas traduit (son second nom n'est pas dans la recette JAX). Mode `rit` de `compare.py` : 22 normaux, 3 de chaque carte rituelle, 2 Monster Reborn, 4 autres.
- Comparaison avec l'oracle : rit 100/100 (graines 9000-9099 ; sur les 20 premières : 63 sélections par somme, 37 libérations carte par carte sans case libre, 60 activations de Demise). Toute la non-régression reste verte ; `obs_check`, `vmap_check` et `sb_check` passent (les decks de test contiennent maintenant les cartes rituelles).
- **Coût du volet 5b.** CPU (B = 64, `--calls 20`) : à cartes égales (rituels exclus des decks), 405 décisions/s contre 408 pour `662270e`, donc pas de surcoût moteur. Avec les cartes rituelles dans les decks du banc : 322 à 345 selon les mesures (bruit de ± 20). Sans calcul rituel (`has_ritual` forcé à faux) sur les mêmes decks : 345. `ritual_ok` coûte donc environ 5 %, le reste vient des parties elles-mêmes (Demise, sélections plus longues). `sum_ok` traite Greater et Equal en un seul passage, et les effectifs par paquets binaires : 1,4× plus rapide, identique sur 4 000 cas aléatoires.

**Jalon 5, volet c : synchro (2026-10-31).** Ce que montre l'oracle (procédure EDOPro `Synchro.AddProcedure(c, f1, 1, 1, Synchro.NonTuner(f2), 1, 99)`, effet EFFECT_SPSUMMON_PROC de l'Extra Deck) :
1. idle : une option « spsummon » par exemplaire invocable (msg 11) ;
2. matériaux carte par carte (msg 26) : d'abord le Syntoniseur, parmi ceux que des non-Syntoniseurs peuvent compléter, puis les non-Syntoniseurs, chacun seulement si la somme reste atteignable ; la sélection se termine d'elle-même quand la somme des niveaux est exacte ;
3. les matériaux vont au cimetière avant le choix de la case (msg 18) : les cases qu'ils libèrent sont proposées, ainsi que les zones extra ;
4. position ATK ou DEF recto (msg 19) ; puis les fenêtres de réponse d'une invocation, sans chaîne.

- **Pas de nouveau verbe** : V_SUMMON sur un monstre synchro de l'Extra Deck lance la procédure (une carte de l'Extra Deck ne s'invoque pas normalement). Choix des matériaux en D_TRIBUTE avec `ssel` (ni règle « toutes les cartes possibles d'office », ni nombre fixe), case et position en `ZC_PROC`, puis la même suite qu'une invocation normale (événement EV_SPSUMMON, `trig_point`, fenêtre rapide). `proc` est posé : Monster Reborn peut réanimer le monstre.
- **Banque** : `smat` [C, 2, FW] (filtres Syntoniseur / non-Syntoniseurs, champs MAT_FIELDS), `syn` [C]. Un seul Syntoniseur, un ou plusieurs non-Syntoniseurs (le cas courant).
- **Matériaux** (`syn_mats`) : monstres face recto du terrain du joueur, de niveau ≥ 1, ni Xyz ni Link. Les non-Syntoniseurs excluent TYPE_TUNER.
- **Existence** (`synchro_ok`, cache `sz_c`) : masque des niveaux de Syntoniseurs présents et masque de bits des sommes de non-Syntoniseurs (au moins un, `reach`, paquets binaires). Invocable s'il existe t avec un Syntoniseur de niveau t et la somme L − t atteignable. Les matériaux étant sur le terrain, une case se libère toujours. Calculé seulement si une partie du lot a un monstre synchro dans son Extra Deck.
- **Choix** (`syn_picks`) : Syntoniseur de niveau l si L − l est atteignable par les non-Syntoniseurs ; non-Syntoniseur de niveau l si le reste L − u − l est atteignable sans lui (une passe `reach` par niveau).
- Cartes : Gaia Knight, the Force of Earth (6), Scrap Archfiend (7), Naturia Leodrake (9, Syntoniseur TERRE + non-Syntoniseurs TERRE) ; Syntoniseurs normaux Tune Warrior, Water Spirit, Flamvell Guard, Genex Controller, Ally Mind. Mode `syn` de `compare.py` : 20 normaux de niveau ≤ 4, 12 Syntoniseurs, 2 Monster Reborn, 6 autres ; Extra Deck de 6 (2 de chaque synchro).
- Les decks de test (`cards.test_decks`) tirent aussi les Syntoniseurs parmi les matériaux, et les Extra Decks mêlent fusions et synchros.
- Comparaison avec l'oracle : syn 100/100 (graines 10000-10099 ; sur les 10 premières : 50 invocations synchro, 41 choix de matériaux à plusieurs options, Monster Reborn sur des synchros). Toute la non-régression reste verte ; `obs_check`, `vmap_check` et `sb_check` passent.
- CPU (B = 64, `--calls 20`, sans Extra Deck) : 364 décisions/s (322-345 au volet 5b) : pas de surcoût mesurable.

**Jalon 5, volet d (première partie) : Xyz (2026-10-31).** Ce que montrent l'oracle et ygopro-core (`Xyz.AddProcedure(c, f, lv, ct)`) :
1. idle : option « spsummon » ; matériaux un par un (msg 26) : exactement ct monstres face recto du terrain du joueur, de niveau égal au rang ; la sélection se termine d'elle-même à ct ;
2. `Duel.Overlay(c, g, true)` attache les matériaux au Xyz **encore dans l'Extra Deck** (l'instantané les montre là pendant le choix de la case) ; puis case (msg 18), position (msg 19), fenêtres d'invocation ;
3. ordre des matériaux : avec envoi au cimetière des matériaux des matériaux, `XyzOverlay` ne trie pas. L'ordre est celui du `card_set`, c'est-à-dire le `cardid` (ordre de création des cartes), et non l'ordre de sélection ni la case ;
4. quand un Xyz quitte le terrain, ses matériaux vont au cimetière de leur propriétaire dans un second `send_to` (REASON_LOST_TARGET), après le déplacement et les événements du groupe, triés par case du Xyz puis rang.

- **Zone Z_OVERLAY** et champ `ov_par` (index du Xyz porteur) ; `seq` = rang parmi ses matériaux ; `ctrl` = son contrôleur. Nouveau champ `uid` (ordre de création façon ygopro : deck 0, Extra 0, deck 1, Extra 1), permuté avec les slots quand `new_game` reçoit une clé : l'ordre des matériaux ne dépend jamais du slot.
- `attach` (O(N), rang par `uid`). `drop_overlays` (dans `refresh`, donc juste après l'instruction ou l'action qui a déplacé le Xyz) envoie au cimetière les matériaux d'un Xyz qui n'est plus sur le terrain et a bougé après l'attachement. Ce n'est donc pas le cas d'un Xyz encore dans l'Extra Deck pendant son invocation. L'événement EV_TO_GRAVE est levé, avec Z_OVERLAY comme zone d'origine : ce n'est pas « depuis le terrain ».
- **Procédure** : comme la synchro, V_SUMMON sur la carte de l'Extra Deck, D_TRIBUTE, `psel` = PS_SYN / PS_XYZ (remplace `ssel`). Banque : `xmat` [C, FW] (filtre MAT_FIELDS), `xct` [C]. Cache `xz_c` (au moins xct matériaux possibles), calculé seulement si une partie du lot a un Xyz dans son Extra Deck.
- **Pas de niveau** : un Xyz (ou un Link) n'est matériau ni rituel, ni synchro, ni Xyz, et ne passe un filtre de niveau que s'il est sans borne (`filter_mask`, `mat_mask`).
- **Observation** : les matériaux attachés sont publics. Token en zone Z_OVERLAY, avec seq = case du Xyz × 8 + rang. `obs_check` permute aussi `rz_c` (oublié au volet 5b), `sz_c`, `xz_c`, `uid` et renumérote `ov_par`.
- `driver.py` : l'instantané lit QUERY_OVERLAY_CARD (codes des matériaux, ajoutés à la carte seulement s'il y en a : les anciens modes ne changent pas).
- Cartes : Gem-Knight Pearl (2 niveau 4), Gagagigo the Risen (3 niveau 4). Mode `xyz` de `compare.py` : 24 normaux de niveau 4, 6 autres, 2 Monster Reborn, 2 Dark Hole, 6 autres cartes ; Extra Deck de 6.
- Mémoire : le retrait des matériaux est fait par `drop_overlays`, appelé par `refresh` (une seule copie dans le graphe). La première version le faisait dans `group_send`, recopié à chaque appel : un `compare.py` est monté à 3,7 Go et a été tué par l'OOM killer. Un `compare.py` atteint environ 3,6 Go dès le volet 5c ; `regress.sh` lance donc 3 processus par défaut au lieu de 4.
- `drop_overlays` ne touche à rien (horloge `clk`, `ev_p`) quand aucun matériau n'est à retirer : `refresh` n'est pas appelé le même nombre de fois dans `step` et dans `step_batch`, et `sb_check` l'a détecté.
- Comparaison avec l'oracle : xyz 100/100 (graines 11000-11099 ; sur les 10 premières : 92 invocations Xyz, 150 choix de matériaux, 42 Dark Hole). Toute la non-régression reste verte. Après la correction de `drop_overlays` : xyz et syn 100/100, et `sb_check`, `vmap_check` et `obs_check` passent.

**Jalon 5, volet d (fin) : détacher un matériau (2026-10-31).** Number 17: Leviathan Dragon (rang 3) a deux effets :
- une ignition, une fois par tour pour cette carte (`Cost.DetachFromSelf(1)`) : +500 ATK tant qu'elle reste sur le terrain (RESETS_STANDARD_DISABLE) ;
- un effet continu : pas d'attaque directe si elle n'a plus de matériau.

L'oracle présente le choix du matériau à détacher en msg 15. Sa localisation est 132 (OVERLAY | MZONE), avec seq = case du Xyz et position = rang du matériau ; `driver.py` le rend comme (« overlay », (case, rang)).
- **Pas de nouvelle brique pour détacher** : dans un filtre, la zone Z_OVERLAY ne voit que les matériaux de la carte source (`filter_mask`), d'où `FILTER(0, F(me=Z_OVERLAY)), CHOOSE(1, 0, 1), MOVE(1)`. La même règle sert aux conditions de compte des effets continus (`cont(..., cnt=(F(me=Z_OVERLAY), 0, 0))` : aucun matériau).
- `detach_fix` (dans `refresh`) : un matériau sorti de Z_OVERLAY perd `ov_par`, et les rangs des matériaux restants sont resserrés (`card::xyz_remove`). O(8N), seulement s'il y a eu un détachement.
- **Modificateurs permanents** : durée TD_PERM. À la fin du tour, `keep_perm` garde dans la table ceux qui portent encore sur une carte restée sur le terrain, et efface les autres. L'ordre d'application vient toujours de `tm_t`.
- **MC_NO_DIRECT / FL_NO_DIRECT** (EFFECT_CANNOT_DIRECT_ATTACK) : un monstre sans attaque directe ne peut attaquer que s'il existe un monstre adverse qui peut être attaqué.
- `ignition(..., opt=OPT_CARD)` (SetCountLimit(1) sans code). Leviathan est exclu des decks des anciens modes (J5D).
- Mode `xyz` : 16 normaux de niveau 4, 10 de niveau 3, 4 autres ; Extra Deck : 2 Gem-Knight Pearl, 2 Gagagigo, 2 Leviathan. xyz 100/100 (graines 11000-11099 ; sur les 10 premières : 87 invocations Xyz, 38 activations de Leviathan). Toute la non-régression reste verte ; `obs_check`, `vmap_check` et `sb_check` passent.

**Mémoire de compilation (2026-11-01).** Un `compare.py` montait à 3,6 Go, et 4 en parallèle dépassaient les 15 Go de la session. Mesures (taille du HLO, RSS après compilation) :
- `new_game` finissait par `lax.cond(forced >= 0, step, …)` : tout le moteur était compilé une seconde fois dans `new_game` (HLO 20 Mo, +2,2 Go). Dans les boucles de test (`vmap_check`, `obs_check`), deux copies de `step` se trouvaient dans le même programme. `new_game` s'arrête maintenant après `start_turn`. La première décision, éventuellement à option unique (fenêtre de la Draw Phase), est jouée par le `step` de l'appelant : action forcée dans `compare.py`, action légale tirée au hasard (alors la seule) dans les boucles.
- `step` commençait par un `settle` en plus de celui de sa boucle. Deux copies de `settle` (6,8 Mo chacune) représentaient 77 % du HLO. Les états d'entrée sont toujours à une décision (sortie de `step` ou de `new_game`), donc le `settle` initial est retiré. Même chose pour le `settle_b` initial de `step_batch`.
- Résultats :

  | | HLO avant | HLO après | Pic avant | Pic après |
  |---|---|---|---|---|
  | `step` | 17,6 Mo | 10,9 Mo | 3,3 Go (avec `new_game`) | 1,66 Go |
  | `step_batch` (B = 64) | 33,7 Mo | 22,4 Mo | 3,1 Go | 2,25 Go |

  `regress.sh` revient à 4 processus (pic total mesuré : 6,8 Go). Toute la non-régression reste verte ; `obs_check`, `vmap_check` et `sb_check` passent. Débit CPU inchangé (350 décisions/s à B = 64, contre 364 : dans le bruit). `new_game + legal` se compile en 23 s au lieu de plusieurs minutes.
- Gardé : les deux copies de `refresh` dans `refresh_b` (lot complet / sous-lot des parties modifiées) coûtent 4,4 Mo de HLO, mais c'est l'optimisation qui avait doublé le débit.
