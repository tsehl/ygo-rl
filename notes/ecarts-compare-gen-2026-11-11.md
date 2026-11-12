# Écarts restants de compare_gen (2026-11-11 → 12, codex) — corrigés

`gen/compare_gen.py --banks 0-9 --games 10` donnait 99 / 100 avec `--revive 0` comme avec `--revive 4`, avant et
après le correctif du verrou d'attaque. Les deux écarts sont des règles manquantes du moteur, corrigées le 2026-11-12.

1. `--revive 0`, banque 9 partie 5, décision 16 (tour 3) : choix de case d'une invocation spéciale (msg 18),
   `oracle [884, 885, 886, 887] / jax [880, 882]`. Cause : la carte source quitte sa place en coût, puis se désigne
   comme cible. Dans ygopro, `Duel.SetTargetCard` recrée la relation (`IsRelateToEffect`) de chaque cible, source
   comprise, même après son déplacement. Le moteur la jugeait hors relation (CF_SRC_UNRELATED : horloge de
   déplacement postérieure au début d'activation), donc l'effet suivant partait d'un état différent.
   Correctif : `src_retarget` (core.py) recale `ch_src_mv` sur l'horloge actuelle de la source quand elle est
   désignée comme cible (fin de sélection ciblée et OP_LABEL).
2. `--revive 4`, banque 5 partie 2, décision 303 (tour 59) : le moteur terminait la partie alors que l'oracle
   demandait une sélection de cibles (msg 15). Cause : coût de 1500 LP payé avec exactement 1500 LP. Le moteur
   déclarait la défaite au paiement (`damage` → `lose`). ygopro ne contrôle la victoire qu'aux points
   d'ajustement (Processors::Adjust, point_event) : l'activation va jusqu'au choix des cibles, la défaite est
   constatée ensuite.
   Correctif : OP_PAY_LP d'une activation retire les LP sans `lose` ; `adjust` (appelé à la mise en chaîne,
   push_link, et aux changements de phase) constate LP ≤ 0 d'un joueur quand l'autre a encore des LP. Coût d'une
   procédure d'invocation (K_PROC, sans maillon) : défaite immédiate comme avant.
   Limite connue : des dégâts pendant la résolution font toujours perdre tout de suite, et « les deux joueurs à 0 »
   (match nul) n'est pas représenté.

Validation : regress complet vert (dont sbp), compare_gen banques 0-9, 10 parties : 100 / 100 avec revive 0 et
revive 4.

Reproduction (la graine de partie est son numéro) : `uv run --with jax python gen/compare_gen.py --banks 9 --seed0 5
--games 1 --dump` et `--banks 5 --seed0 2 --games 1 --revive 4 --dump`.
