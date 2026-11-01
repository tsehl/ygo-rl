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
- Prochaine étape (décidée avec Thomas : finir le jeu avant le générateur procédural) : jalon 5, volets e-f (link, pendule), puis mesure du débit GPU. Débit GPU mesuré seulement à la fin de l'Extra Deck.
- Mémoire : ~3,6 Go par processus `compare.py` (pic de compilation, depuis le jalon 5) ; `regress.sh` en lance 3 en parallèle (JOBS=3), 4 dépassent les 15 Go. Ne pas dérouler en Python de boucles sur les entrées de `stats` (l'OOM killer est passé à 3,8 Go par processus).
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
