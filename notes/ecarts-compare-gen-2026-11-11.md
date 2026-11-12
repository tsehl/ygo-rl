# Écarts restants de compare_gen (2026-11-11, codex)

`gen/compare_gen.py --banks 0-9 --games 10` : 99 / 100 avec `--revive 0` comme avec `--revive 4`, avant et après
le correctif du verrou d'attaque (mêmes parties, mêmes messages). Pas encore analysés.

1. `--revive 0`, banque 9 partie 5, décision 16 (tour 3) : choix de case d'une invocation spéciale (msg 18).
   `actions légales : oracle [884, 885, 886, 887] / jax [880, 882]`, oracle `('place', 0, 'mzone', 1..4)`.
   Le moteur propose d'autres cases (0 et 2) que l'oracle : cases occupées, zones bloquées ou désignées mal calculées.
2. `--revive 4`, banque 5 partie 2, décision 303 (tour 59) : le moteur termine la partie (vainqueur 1) alors que
   l'oracle demande encore une sélection de cartes (msg 15) parmi des cartes de j1 sur le terrain : fin de partie
   détectée trop tôt (ou effet de fin de partie absent), à préciser.

Reproduction (la graine de partie est son numéro) : `uv run --with jax python gen/compare_gen.py --banks 9 --seed0 5
--games 1 --dump` et `--banks 5 --seed0 2 --games 1 --revive 4 --dump`.
