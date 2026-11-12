# Défaut moteur : « ne peut pas attaquer » filtré sur l'ATK, avec un bonus d'ATK limité à une phase (2026-11-11)

Trouvé par `gen/compare_gen.py --banks 0-3 --games 10 --revive 4` (banques renforcées en résurrections, voir
`Gen(revive)` dans `gen/cardgen.py`) : 39 / 40 parties identiques. L'écart ne vient pas des résurrections.

Reproduction : `uv run --with jax python gen/compare_gen.py --banks 1 --games 4 --revive 4 --dump` (partie 3).

Position (tour 28, joueur 1, phase de combat) :
- j0 contrôle le piège continu 600000367 : `cont code 9` (ne peut pas attaquer), cible `{'op': 4, 'atk': (1500, ∞)}`
  (monstres adverses de 1500 ATK ou plus).
- j1 contrôle 600000386 : `cont code 1` (+400 ATK à ses monstres), `tp=1, ph=8` (pendant son tour, en phase de
  combat seulement).
- j1 a invoqué normalement 600000394 (Syntoniseur normal, 1200 ATK de base), 1600 ATK en phase de combat.

L'oracle n'offre pas l'attaque de 600000394 (1600 ≥ 1500) ; le moteur l'offre :
`actions légales : oracle [392, 881, 882] / jax [390, 392, 881, 882]`, écart `('attack', 600000394, …)`.

Cause : `filter_mask` lit `s.atk_c` / `s.df_c`, le cache de stats calculé au `refresh` précédent. Les filtres d'un
effet continu (cible ou compte) qui bornent l'ATK ou la DEF voyaient donc l'ATK d'avant le changement de phase ;
ygopro-core appelle `get_attack()` à jour. Ici le +400 conditionné par la phase de combat n'était pas encore dans le
cache quand la cible du verrou était évaluée.

Corrigé sur `codex` (`engine/core.py`, `refresh2` / `_second_b`) : après un refresh, si la banque a un effet
K_CONT dont un filtre borne l'ATK ou la DEF (`cont_reads_stats`, prédicat de banque) et qu'une ATK / DEF en cache a
franchi l'un de ses seuils (`cont_bounds`, `stats_moved`), le refresh est refait une fois avec les stats à jour. Dans
`step_batch`, cette seconde passe ne porte que sur le sous-lot des parties concernées ; `step` applique la même
règle (`sb_check` reste exact, y compris avec `SB_PROC=X`, banque procédurale). Limite : une dépendance circulaire
(bonus d'ATK filtré sur l'ATK) n'est pas itérée jusqu'au point fixe.

Coût. Les 20 premières banques procédurales ont toutes un tel filtre (verrous et MINATK du générateur). Une seconde
passe de `stats` systématique coûtait −16 % de débit CPU (banque procédurale 1, B=64, minimum par appel sur
4 répétitions), une version qui ne réévaluait que les bornes −22 % (vectoriser `stats` le ralentit de 7 % à elle
seule). Avec le test de franchissement de seuil, 28 parties sur 1 392 appels de `refresh_b` (60 appels de
`step_batch`, B=64) passent en seconde passe. Le chronométrage CPU de la VM reste trop bruité (−4 % à −24 % pour le
même code) pour chiffrer mieux : à mesurer sur L4 au prochain passage GPU.

Vérification : reproduction 4 / 4 identiques ; `engine/regress.sh` complet vert ; `compare_gen --banks 0-9
--games 10` 99 / 100 avec `--revive 0` et `--revive 4`. Les deux écarts restants existent aussi sans le correctif
(voir `notes/ecarts-compare-gen-2026-11-11.md`).

## Défaut associé : caches de `new_game` périmés

`sb_check.py` sur une banque procédurale (`SB_PROC=2` et `5`, option ajoutée pour ce correctif) échouait à t=0 sur
`fz_c`, avec et sans le correctif ci-dessus. `new_game` appelait `refresh` avant de piocher les mains de départ :
`fz_c` (fusions réalisables) et `rz_c` (rituels) restaient périmés jusqu'au premier changement des entrées de
`stats_key`. `step` (refresh_if) les gardait ; `step_batch` les recalculait quand il rafraîchissait tout le lot.
Exemple : banque 2, partie 6, deux exemplaires de 600000462 invocables par fusion d'après `fusion_ok`, pas d'après
`step`. L'entraînement ne suivait donc pas exactement le chemin vérifié contre l'oracle en début de partie. La
banque de test de `sb_check` ne le montrait pas. Corrigé : `refresh` après la pioche (un seul appel, pas de
compilation en plus). `regress.sh` lance désormais aussi `SB_PROC=2 sb_check.py` (ligne `sbp`).
