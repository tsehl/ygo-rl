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

- Tokens = cartes visibles par le joueur (sa main, les deux terrains, les deux cimetières, les bannies recto, les matériels Xyz), regroupés par cumsum en O(N), max 48 + un token global (LP, phase, tour, chaîne en cours, décision demandée).
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
- `observe(bank, s, p)` produit 48 tokens au plus : sa main, tout le terrain (les cartes verso adverses sans identité), les cimetières et les bannies, plus un vecteur global (LP, tour, phase, décision, compteurs dont la main adverse et les decks). L'ordre des tokens ne dépend que d'informations visibles. `slot` sert seulement à reconstruire l'action et n'est pas une entrée du réseau.
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
- Pistes suivantes, à décider après une mesure sur GPU :
  - les exécutions d'effets (`exec_one`) calculent encore, sous vmap, toutes les branches de fin de chaîne et de déclencheurs ;
  - la boucle de `step` attend la partie la plus lente du lot. L'alternative : une micro-étape par appel, l'enveloppe RL n'interrogeant le réseau qu'aux vraies décisions.
