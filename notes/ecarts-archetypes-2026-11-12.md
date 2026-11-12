# Écarts trouvés par les banques à archétypes (2026-11-12)

`compare_gen.py --arch 0.5 --structured`, banques 0-9, revive 0 et 4 : 195/200 parties identiques avant correction.
Les cinq écarts venaient de trois règles du moteur que les decks à thème (synergies, pendule, Xyz) rendent fréquentes.

## 1. Invocation pendule en cours : STATUS_SUMMONING

Banques 4 g8 r4, 6 g9 r0, 3 g8 r0 (ATK +400 / +1000 / +300 côté JAX).

Pendant une invocation pendule de plusieurs cartes, l'oracle demande la case de chaque carte l'une après l'autre (msg 18).
Les cartes déjà posées ont encore STATUS_SUMMONING jusqu'à l'étape 27 de la procédure (operations.cpp). Trois effets
dans ygopro-core :

- `card::get_attack` / `get_defense` renvoient l'ATK / DEF de base ;
- leurs effets de terrain ne sont pas disponibles : `effect::is_available` exige STATUS_EFFECT_ENABLED, posé par
  `enable_field_effect(true)` à l'étape 27, qui donne aussi de nouveaux id d'effet ;
- `field::filter_matching_card` / `filter_field_card` les excluent des groupes de la zone monstres (un compteur
  « par monstre » ne les compte pas).

Moteur (ST_PSTEP marque ces cartes) :

- `stats` : pas de source d'effet continu, ATK / DEF de base ;
- `filter_mask` : exclues de la zone monstres ;
- fin de l'invocation : `enable` sur le groupe (nouvel en_t).

## 2. Matériaux Xyz : EVENT_TO_GRAVE au même point d'événement

Banque 1 g9 r0 : 600000324 (déclencheur « envoyé au cimetière ») était un matériau d'un Xyz détruit au combat. L'oracle
le propose dans la même chaîne que le déclencheur du Xyz. Dans le moteur, `drop_overlays` n'était appelé que par
`refresh`, donc après la collecte des déclencheurs : l'événement restait dans `tev` et partait au point suivant.

Correction : `dmg_send` appelle `drop_overlays` avant la collecte (l'appel de `refresh` reste, sans effet si les
matériaux sont déjà partis). Ailleurs, chaque instruction finit par un refresh avant le point d'événement suivant.
Une première version l'appelait dans `trig_point` : une copie par site du graphe, −20 % de débit CPU.

Banque 9 g4 r4 (JAX proposait en trop le déclencheur de cimetière de 600000370) passe aussi après ces corrections.
C'était probablement le même événement tardif.

## Validation

- `engine/regress.sh` complet : tous les modes de `compare.py` à 100 %, obs / vmap / sb / sbp propres.
- `compare_gen.py --banks 0-9 --games 10` : 100/100 en revive 0 et en revive 4.
- Les mêmes avec `--arch 0.5 --structured` : 100/100 en revive 0 et en revive 4.

Diagnostic ajouté à `compare.py` : le message d'un écart d'état donne aussi la décision en attente de l'oracle (msg
et options). C'est ce qui a montré que l'oracle était au milieu de l'invocation pendule (msg 18).
