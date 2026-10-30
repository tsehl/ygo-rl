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
- Débit GPU : ~15k décisions/s sur L4 avec `step_batch` (frac=0.05). Le moteur est maintenant limité par le GPU. Détails à la fin de `ETAPE2.md`.
- Prochaine étape : jalon 4 (effets continus, champ setcode, filtre d'archétype), puis ETAPE3 (générateur procédural).

## Règles
- **Vérifier contre l'oracle** avant tout commit qui touche le moteur. L'oracle est ygopro-core compilé localement par `setup.sh`, voir `oracle/BUILD.md`. Il faut lancer :
  - `compare.py` dans tous les modes, avec les graines de `ETAPE2.md` ;
  - `obs_check.py` ;
  - `vmap_check.py`.
- Pas d'algorithme en O(N²) dans le moteur. Sous vmap, `switch` et `cond` calculent toutes leurs branches. Il faut donc masquer les bornes de boucle par l'opcode (voir le bug `draw_n` dans `ETAPE2.md`).
- `compare.py` doit appeler `new_game` via `jax.jit`. Sinon `step` est recompilé à chaque partie, ce qui fait une fuite mémoire.
- GPU :
  - l'entraînement tourne sur GPU, jamais sur CPU ni avec envpool ;
  - les tests courts se font sur L4, les vrais runs sur A100 ;
  - budget : 200 unités de calcul (CU) Colab par mois. Le L4 coûte ~1,94 CU/h, la session CPU standard ~0,08 CU/h ;
  - une session Colab sans progrès depuis ~3 min doit être arrêtée tout de suite ;
  - toujours arrêter les sessions à la fin.
- Le travail mécanique (comptages, scripts d'extraction, grep) peut être confié à des agents Haiku avec une spécification précise.
- Pas de push sans l'accord de Thomas.
