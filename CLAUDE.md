# Projet YGO RL — consignes pour Claude

Thomas écrit en français familier. Réponses en français, concises. Il veut être contredit quand il se trompe, pas de sycophantie.

## But
Moteur Yu-Gi-Oh complet en JAX, vectorisé (vmap) sur GPU. Les effets des cartes sont décomposés en briques atomiques. Le modèle s'entraîne en self-play sur des cartes et des decks procéduraux, donc il doit lire les arbres d'effets plutôt que des identifiants de carte. Les vraies cartes servent uniquement de benchmark et n'entrent jamais dans l'entraînement. On vise le jeu complet, sans version réduite.

- Conception, jalons et avancement : `ETAPE2.md`.
- Primitives tirées des scripts EDOPro : `ETAPE1.md`.
- Générateur procédural et données d'entraînement : `ETAPE3.md`.
- Benchmarks de faisabilité GPU : `bench/RESULTATS.md`.
- Historique complet de la session de conception (2026-10-28 → 30) : `notes/session-2026-10-28_30.md`. Les 14 sections « RÉSUMÉ (compaction) » en donnent l'essentiel ; grep pour retrouver une décision.

## État (2026-10-30)
- Jalons 1, 2, 3a et 3b faits : 20 vraies cartes, chaînes, déclencheurs, SEGOC, effets rapides. Le moteur rejoue à l'identique les parties de l'oracle ygopro-core dans tous les modes de `engine/compare.py` (normal, mixed, chain, trig, quick).
- Débit GPU (L4, `step_batch`, frac=0.05, `bench/engine_bench.py`) : 15,8k décisions/s avant le jalon 4, 13,5k après optimisation, **12,7k (B=1024) / 13,0k (B=4096) après le volet 5a sans Extra Deck, 10,7k avec un Extra Deck de 15 (`--extra 15`, N=110)**. Détails à la fin de `ETAPE2.md`. Mesurer le débit (CPU d'abord, `--b 64 --calls 20`) après chaque gros ajout au moteur : sous vmap, tout calcul ajouté dans `legal`, `refresh` ou une branche d'`exec_one` est payé partout.
- Colab depuis la VM cloud : `uv tool install google-colab-cli`, authentification par code (jeton dans ~/.config/colab-cli/token.json), domaine colab.research.google.com autorisé dans le réseau de l'environnement. `colab exec` ne répond plus quand la machine compile (CPU saturés) : prévoir des délais de 4-5 min.
- Jalon 4, volet 1 fait (2026-10-30) : setcode et filtres archétype / race / nom, effets continus de stats (K_CONT), modificateurs temporaires, cache ATK/DEF. Mode `cont` de `compare.py`.
- Jalon 4, volet 2 fait (2026-10-30) : fenêtres du damage step, couches ATK/DEF complètes (SET, SET_FINAL, SET_BASE, ordre des id d'effet), limites par carte. Modes `dmg` et `ds` de `compare.py`.
- Jalon 4, volet 3a fait (2026-10-30) : interdictions et protections (ne peut pas attaquer / être cible d'attaque / être ciblé, immunité aux pièges, indestructible au combat, pas d'invocation spéciale, pas d'activation de pièges). Mode `flag` de `compare.py`.
- Jalon 4, volet 3b fait (2026-10-30) : annulation d'effets (EFFECT_DISABLE, DISABLE_EFFECT, RESET_DISABLE, immunité dépendante de l'historique). Mode `neg` de `compare.py`.
- Jalon 5, volet a fait (2026-10-31) : Extra Deck, fusion (Polymerization, recettes par filtres, choix des matériaux, zones extra MR5). Mode `fus` de `compare.py`.
- Jalon 5, volet b fait (2026-10-31) : rituel (Greater / Equal, MSG_SELECT_SUM dans `driver.py`). Mode `rit` de `compare.py`.
- Jalon 5, volet c fait (2026-10-31) : synchro (procédure de l'Extra Deck par V_SUMMON, Syntoniseur puis non-Syntoniseurs). Mode `syn` de `compare.py`.
- Jalon 5, volet d fait (2026-11-01) : Xyz (zone des matériaux Z_OVERLAY, `ov_par`, `uid` ; matériaux au cimetière quand le Xyz part ; détacher en coût par un filtre Z_OVERLAY ; modificateurs permanents TD_PERM). Mode `xyz` de `compare.py`.
- Jalon 5, volet e fait (2026-11-01) : Link (matériaux par énumération des 128 sous-ensembles de cases, sommes de lien en masque de bits, cases désignées par les flèches, pas de DEF ni de niveau). Modes `lnk` et `xlk` de `compare.py`. Extra Link (composante de co-liens sur 12 cases physiques) testé avec trois Link fictifs sans effet (`TEST_CARDS` de `oracle/driver.py`, scripts dans `oracle/test_scripts/`) : cartes de test de règle seulement, jamais d'entraînement.
- Volet e, fin (2026-11-01) : Knightmare Phoenix / Cerberus (co-lien `cl_c` / F_ZX, statuts ST_SPSUM / ST_LINKSUM, indestructible par les effets et DESTROY, label de maillon, question oui / non D_YESNO + A_YES, matériaux de noms différents). Mode `kn` de `compare.py`.
- Jalon 5, volet f fait (2026-11-01) : pendule (zones pendule cases 0 / 4, invocation pendule depuis la main et l'Extra Deck, placement carte par carte, redirection vers l'Extra Deck, activation niée → cimetière via ST_ACTDIS). Mode `pen` de `compare.py`.
- Débit GPU de tout l'Extra Deck mesuré (L4, 2026-11-01) : 9,2k décisions/s sans Extra Deck, 5,8k avec un Extra Deck de 15 (B=1024), 3,1k à B=4096 (le débit baisse quand B monte).
- Optimisation (2026-11-01) : caches synchro / Xyz / Link recalculés au point de décision seulement (`refresh_ex`), arrêt anticipé dans `step_batch` (action « continuer » = n_actions(N)). CPU B=256 Extra Deck 15 : 152 → 222 décisions/s ; L4 (2026-11-02, Extra Deck 15) 5,5k → 9,4k (B=1024), 3,1k → 7,0k (B=4096). Décomposition par variante à la fin de `ETAPE2.md`.
- Optimisation, deuxième passe (2026-11-02) : double passe dans `settle_b`, tours des actions forcées en sous-lot de B/4. L4, Extra Deck 15 : 12,4k (B=1024) / 12,6k (B=4096) décisions/s. `step_batch` compile deux fois `rnd` : compilation et pic de mémoire ~×2 (5,8 Go sur CPU).
- Jalon 6, étape 1 faite (2026-11-02) : générateur procédural minimal `gen/cardgen.py` (spécification en dict → briques, budget grossier, `to_bank` à formes fixes : une compilation pour toutes les banques) et fuzz `gen/fuzz.py` (parties aléatoires, invariants d'état après chaque décision). Graines 0-9 propres après correction de la position d'un Link invoqué spécialement. Détails dans `ETAPE3.md`. Étape suivante : compilateur spécification → Lua, pour passer les cartes générées dans `compare.py`.
- Jalon 6, étape 2 (2026-11-02) : cartes générées compilées en Lua (`gen/tolua.py`, bibliothèque `gen/procgen.lua`) et comparées à l'oracle (`gen/compare_gen.py --banks 0-3 --games 10`). Quatre règles corrigées dans le moteur : interdiction d'invocation spéciale pour fusion / rituel / procédures, joueur d'événement PLAYER_NONE pour une invocation spéciale par effet, point d'événement après la défausse de fin de tour (R_HAND, sans fenêtre libre), modificateur d'une carte suspendu tant qu'elle est annulée (`tm_src`). Banques 0-9 : toutes les parties identiques. Restrictions du générateur (limites connues du moteur) : un seul déclencheur par carte, pas de temp.attack, « ne peut pas attaquer » compte aussi l'invocation par flip. Détails dans `ETAPE3.md`.
- Jalon 6, étape 3 (2026-11-02) : réseau (`train/model.py`, encodeur de cartes depuis les briques + transformer de politique) et PPO self-play (`train/ppo.py`). Premier run L4 (B=1024, d=128) : 2,8k décisions/s au total (mise à jour = moitié du temps), 87,7 % contre l'aléatoire sur une banque jamais vue à l'itération 49. La session Colab a disparu vers 75 min (log et paramètres perdus) : relever souvent, copier les paramètres hors de la VM. Fuzz graines 0-3 propre. Détails dans `ETAPE3.md`.
- Étape 3, suite (2026-11-03) : `--bf16` (mise à jour 11 → 7 s par itération), encodeur de cartes compacté (LC=48), `act_step` (politique + step_batch + remplacement en une fonction compilée), évaluation contre la politique d'il y a `--old-lag` itérations. Run 2 (L4, 200 it, 4,0k décisions/s) : 97,6 % contre l'aléatoire, 54-60 % contre l'itération −60 en fin de run. Paramètres dans `train/ckpt/`. Rapatrier les paramètres pendant un run : `colab exec` d'un script qui imprime le pickle f16 gzip en base64 (~3 Mo). Le rollout coûte ~165 ms par appel d'`act_step`, contre 88 ms pour le moteur seul et 13 ms pour `act` (L4) : la banque en argument et les parties neuves ne coûtent rien, l'écart vient de la position (`ppo.py --prof`) : `step_batch` coûte 133 ms en milieu de partie contre 53 ms au début, parce que les `while_loop` du lot font plus de tours. Le bench du moteur (débuts de partie) sous-estime le coût de l'entraînement (voir « Débit : où part le temps » dans `ETAPE3.md`).
- Optimisation, troisième passe (2026-11-04) : `auto_pass` exact (teste `act_ok` sur ≤ KA=2 candidats), recalculé en sous-lot pour les seules parties modifiées. `step_batch` L4 B=1024, jeu aléatoire : milieu de partie 129 → 113 ms, début 101 → 91 ms. Diagnostic par `bench/loop_stats.py` (tours de chaque boucle, `step_batch(stats=True)`). Il reste surtout la boucle d'exécution (~17 tours par appel, ~47 parties actives sur 1024). Voir « Milieu de partie : tours de boucle » dans `ETAPE3.md`.
- Colab, plusieurs moteurs dans une session : `YGO_JAX_CACHE` (cache persistant) + `bench/engine_bench.py --compile-only` lancés en parallèle, puis les mesures une par une. Compter ~6 min par vague de compilations parallèles.
- Colab : `colab exec -f` ne passe pas avec un fichier de 93 Ko. Envoyer le code en morceaux de 24 Ko (un fichier par morceau), puis lancer le travail en arrière-plan (Popen, sorties vers /dev/null) et interroger le log.
- Mémoire : ~1,7 Go par processus `compare.py` (pic de compilation ; 3,6 Go avant la correction du 2026-11-01, voir « Mémoire de compilation » dans `ETAPE2.md`). Surveiller la taille du HLO de `step` / `step_batch` : une fonction lourde appelée à deux endroits est compilée deux fois. Ne pas dérouler en Python de boucles sur les entrées de `stats` (l'OOM killer est passé à 3,8 Go par processus).
- Session cloud : 4 cœurs, 15 Go. Les tests CPU se font ici (`engine/regress.sh` lance toute la non-régression, ~45 min). Colab ne sert qu'au GPU.

## Règles
- **Vérifier contre l'oracle** avant tout commit qui touche le moteur. L'oracle est ygopro-core compilé localement par `setup.sh`, voir `oracle/BUILD.md`. Il faut lancer :
  - `compare.py` dans tous les modes, avec les graines de `ETAPE2.md` (`engine/regress.sh` fait tout) ;
  - `obs_check.py` ;
  - `vmap_check.py` ;
  - `sb_check.py` (step_batch == vmap(step)).
- Pas d'algorithme en O(N²) dans le moteur. Sous vmap, `switch` et `cond` calculent toutes leurs branches. Il faut donc masquer les bornes de boucle par l'opcode (voir le bug `draw_n` dans `ETAPE2.md`).
- `compare.py` doit appeler `new_game` via `jax.jit`. Sinon `step` est recompilé à chaque partie, ce qui fait une fuite mémoire.
- GPU :
  - l'entraînement tourne sur GPU, jamais sur CPU ni avec envpool ;
  - les tests courts se font sur L4, les vrais runs sur A100 ;
  - budget : 200 unités de calcul (CU) Colab par mois. Le L4 coûte ~1,94 CU/h, la session CPU standard ~0,08 CU/h ;
  - une session Colab sans progrès depuis ~3 min doit être arrêtée tout de suite ;
  - toujours arrêter les sessions à la fin.
- Le travail mécanique (comptages, scripts d'extraction, grep) peut être confié à des agents Haiku avec une spécification précise.
- Travailler sur `main` et pousser après chaque commit (Thomas veut voir le code).
