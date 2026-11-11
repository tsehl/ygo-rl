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

Hypothèse : la cible du verrou (filtre sur l'ATK) est évaluée avant que le modificateur conditionné par la phase
soit appliqué, ou n'est pas réévaluée au changement de phase (cache `fl_c` / `atk_c` de `refresh`). À vérifier
dans l'ordre de calcul de `refresh` (couches ATK puis drapeaux de verrou).

Pas corrigé : le chantier moteur est sur `codex`. Les banques par défaut (`revive=0`) ne l'ont pas fait apparaître
dans `compare_gen --banks 0-9` ; la combinaison est rare mais générique.
