# Test de faisabilité GPU — interpréteur d'effets (2026-10-28)

Script : `interp_bench.py`. 80 cartes/partie, 4 effets × 8 opcodes par carte (cartes aléatoires), 32 opcodes via `lax.switch`, chaîne de 4 maillons, masque d'actions légales avec conditions, tours/OPT/pioche/reset. Pas les vraies règles, mais la même forme de calcul.

Politique NN : transformer 4 couches, d=256, 4 têtes, 80 tokens (1 par carte), encodeur d'effets léger (sac d'opcodes). Inférence seule, pas d'entraînement.

| GPU | Moteur seul (aléatoire) | Moteur + transformer |
|---|---|---|
| T4, fp32 | ~89 000 coups/s (plafond dès B=4096) | 4 600 coups/s (B=1024) ; B=4096 bloqué > 20 min, coupé |
| L4, NN en bf16 | ~184 000 coups/s (B=4096), 165 000 (B=16384) | ~22 000-23 600 coups/s (B=512 à 2048) |

Coût total des tests : 0,50 crédit.

## Lecture

- Le moteur vectorisé marche sur GPU : ~185 k coups/s sur L4 malgré les 32 branches calculées à chaque cycle.
- Avec le réseau, le transformer prend ~90 % du temps (45 µs/coup contre 5 µs pour le moteur). C'est le réseau qui limite, pas le jeu.
- Le vrai moteur sera plus lourd (plus d'opcodes, vraies règles de chaîne) : peut-être 3 à 10× plus cher, donc du même ordre que le réseau. Ça reste viable.
- Estimation d'entraînement (rétropropagation + plusieurs passes PPO ≈ 3-4× l'inférence) : ~6 000-8 000 coups/s sur L4, soit ~25 M coups/h, et **~2 à 3 milliards de coups par mois** avec 200 crédits en L4.
- Le bf16 est indispensable : le L4 en bf16 va 5× plus vite que le T4 en fp32.
## v2 optimisée (`interp_bench_v2.py`, L4) — états identiques à la v1 vérifiés sur 256 parties × 60 coups

| | v1 | v2 |
|---|---|---|
| Moteur seul B=4096 | 184 000 | 250 000 coups/s |
| Moteur seul B=16384 | 165 000 | 281 000 coups/s |
| Moteur + NN B=1024 | 21 900 | 22 300 coups/s |
| Moteur + NN B=4096 | — | 21 900 coups/s |

- Moteur : gain réel mais modeste (×1,35 à ×1,7), loin du ×5-20 prédit. L'hypothèse « limité par la bande passante à cause des branches » était au mieux partielle. Le plafond restant vient probablement des ~32 cycles séquentiels par coup (une boucle = beaucoup de petits kernels lancés l'un après l'autre) et du tri O(N²) fait à chaque cycle. À profiler avant d'aller plus loin.
- Réseau : aucun gain, débit plat à ~22 000 coups/s quel que soit B, soit ~11 TFLOPS effectifs (~10 % du L4 en bf16). Le levier le plus sûr n'est pas technique : ne donner au réseau que les cartes visibles (main, terrain, cimetière, bannies ≈ 20-40 au lieu de 80, le deck est caché de toute façon), soit ~2-3× moins de calcul.

Coût cumulé de tous les tests : 0,58 crédit.

- Anomalies : sur T4, le cas NN B=4096 s'est bloqué sans explication. Sur L4, la CLI a renvoyé `TimeoutError: Timeout waiting for reply` après la fin du script (résultats complets, session bien arrêtée).

## Ablation (`ablation.py` = v2 + `ablation_body.py`, construit par `build_ablation.sh` ; log `ablation_l4.log`, L4, 0,09 crédit)

Réseau seul (pas de moteur), B=2048 sauf mention :

| Variante | coups/s | TFLOPS effectifs |
|---|---|---|
| Politique réelle, états figés | 25 000 | 13,3 |
| Transformer synthétique d=256, 4 couches, 80 tokens | 25 400 | 13,4 |
| idem, sans attention (FFN seule) | 58 300 | 19,6 |
| idem, sans FFN (attention seule) | 44 400 | 8,6 |
| d=128 | 52 200 | 7,3 |
| d=512 | 10 600 | 21,8 |
| 32 tokens au lieu de 80 | 74 500 | 15,3 |
| B=8192 | 25 200 | 13,4 |

Moteur seul (politique aléatoire), B=16384 :

| Variante | coups/s | gain |
|---|---|---|
| base (32 cycles, tri O(N²), reset à chaque coup) | 276 000 | — |
| 8 cycles | 358 000 | ×1,3 |
| rang O(N) (cumsum) au lieu du tri O(N²) | 1 062 000 | ×3,9 |
| sans reset à chaque coup | 787 000 | ×2,9 |
| les trois | 4 989 000 | ×18 |

### Lecture

- **Réseau : limité par la bande passante mémoire, pas par le calcul ni par l'overhead.** Les TFLOPS montent avec la largeur (7 → 13 → 22 pour d=128 → 256 → 512), le batch n'y change rien (le GPU est déjà plein), et le moteur ne coûte que ~12 % (25 000 seul contre 22 000 avec). Le L4 n'a que 300 Go/s pour 121 TFLOPS bf16 : avec d=256 et des séquences de 80 tokens, les activations (layernorm, gelu, scores d'attention) font trop d'allers-retours mémoire pour le calcul effectué.
- L'attention prend ~57 % du temps pour ~1/3 des FLOPs (8,6 TFLOPS) : sous vmap avec N=80 et 4 têtes, XLA ne la fusionne pas bien. Piste : implémentation cuDNN/flash, ou regrouper batch × têtes.
- Le levier n°1 confirmé : **moins de tokens** (32 au lieu de 80 → ×2,9). Ne donner que les cartes visibles.
- Hypothèse à vérifier (non mesurée) : un A100 (~1,5-2 To/s, 5× le L4) pourrait être plus rentable par crédit pour la partie réseau malgré un prix 3,4× plus élevé.
- **Moteur : le vrai coût n'était pas les cycles mais le tri O(N²) et le reset systématique.** Le tri sert aussi à la pioche (dans le reset). Le reset fabriquait une partie neuve complète à chaque coup pour chaque environnement, utilisée seulement à la fin d'une partie.
- Corrections pour le vrai moteur : stocker la position dans le deck comme un indice explicite (piocher = prendre les positions < n, en O(N)) ; reset seulement des parties finies, par exemple via une réserve de parties neuves pré-générées. Attention : le cumsum de l'ablation n'a pas la même sémantique (ordre d'index et non ordre du deck), c'est une borne du gain, pas une implémentation.
- Conséquence : le moteur peut aller à plusieurs millions de coups/s, il devient négligeable devant le réseau. Tout le budget d'optimisation va au réseau (tokens, attention, choix du GPU).

Coût cumulé de tous les tests : 0,67 crédit (solde 199,33).
