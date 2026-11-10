# Substituts de fusion et identité des effets — 2026-11-06

Implémentation poussée sur `codex` : `370c852`. Ce bilan complète
[`codex-validation-2026-11-06.md`](codex-validation-2026-11-06.md).

## Comportement

Les recettes autorisent un seul remplacement par substitut, pour les emplacements
nommés passés comme arguments numériques à `Fusion.AddProcMix` / `MixN`. Un filtre
fonctionnel utilisant `Card.IsCode` n'accorde pas cette permission. Deux cartes
possédant la propriété de substitut peuvent participer si une seule remplace un
matériau ; l'autre doit satisfaire normalement son emplacement.

Le traducteur reconnaît les effets passifs simples `EFFECT_FUSION_SUBSTITUTE`,
y compris une condition de zone représentable. Les callbacks de valeur et les
conditions arbitraires restent refusés. Les tests oracle exercent aussi un
substitut posé face cachée. Les contraintes de contrôle, de zone, d'annulation,
de destruction au combat et de place pour la fusion sont prises en compte.

Le catalogue passe de 446 à **507 / 13 541** traductions utilisables : 426 cartes
du deck principal, 81 de l'Extra Deck (60 fusions, 6 synchros, 12 Xyz, 3 Links).
Une traduction reconnue n'est pas une preuve que tous ses effets ont été exercés.

## Identité indépendante des effets

`oracle/effect_inspector.cpp` est une extension en lecture seule qui consulte
le code d'événement enregistré sur l'effet proposé par ygopro-core dans
`select_chains`. Aucun changement des règles, scripts officiels ou sources du
cœur de l'oracle. L'indice exact de l'option est conservé lorsque plusieurs
options ont le même tuple. Le masque légal JAX ne départage pas les effets.

L'extension est construite contre les en-têtes et le binaire épinglés ; compilation
et cache sont verrouillés et publiés atomiquement. `setup.sh` la construit ; une
première ambiguïté peut également déclencher sa construction. Linux est testé,
le chemin Darwin est adapté mais n'a pas été exécuté ici. Un conflit persistant
entre type, description et événement reste **non concluant**.

## Optimisation et modèles

L'affectation exacte utilise Hall sur les sous-ensembles des quatre emplacements,
avec histogrammes et transformées zêta. Aucune matrice carte × carte ni tenseur
`[2**FK, 2**FK, N]`. Pour FK=4 fixé, le coût croît linéairement avec le nombre de
cartes. Les banques sans substitut conservent le chemin de calcul précédent.
Les nouvelles métadonnées sont partagées dans la banque, sans état par partie.

Les nouveaux modèles lisent les permissions de substitution par emplacement.
Les anciens checkpoints conservent leur calcul en ignorant ces bits ; le test
utilise notamment le checkpoint réel du run 3. Aucun entraînement CPU effectué.

## Vérifications

| Contrôle | Résultat |
| --- | --- |
| Tests unitaires | 27 réussis, dont comparaison à l'énumération exhaustive, choix partiels et vmap |
| Scénarios de fusion antérieurs, montage actualisé | 16/16 parties identiques |
| Scénarios de substituts | 16/16 parties identiques, 8 cas positifs/négatifs |
| Cinq anciens blocages ambigus | 5/5 parties complètes identiques, inspection native effectivement appelée |
| Campagne traduite finale, graines 106–135, k=8 | 30/30 identiques, aucune source modifiée pendant le run |
| Régression historique, 18 modes | 1640/1640 identiques |
| Observations, version finale | 9025 états, 800 invariances, 99 contrôles positifs |
| vmap, version finale | lot égal au séquentiel, 16/16 parties finies |
| step_batch, version finale | égalité décision par décision, dont 289 rendus en résolution et 114 attentes forcées |

Les cinq replays reprennent les decks complets des graines 90, 97, 98, 100 et 103 ;
ils dépassent l'ancien blocage et totalisent respectivement 260, 404, 383, 333 et
395 décisions. Ils couvrent Photon Caesar, Sniffer Dragon, XX-Saber Ragigura et
Hidden Soldiers. Les fixtures synthétiques servent aux tests de règles uniquement.

La régression historique a tourné dans la copie figée une archive locale non versionnée, avant
la dernière optimisation du noyau de substitution. Son chemin sans substitut est
identique à la version finale. L'exécution combinée a été interrompue après les
18 modes ; obs/vmap/sb ont ensuite été terminés séparément sur la version finale,
avec code de sortie 0. Le noyau optimisé est vérifié séparément par l'énumération,
les scénarios de substituts et la campagne finale. Ne pas attribuer un code 0 à
l'exécution combinée interrompue.

Campagne finale : `translate/cache/runs/20261106T084839-8aba31d72b86.json`, signature
`53956ce4ed81d3cd7ad1c7cab553a0294f3299993ffd905b21db6c065bd04bc8`.
Le manifeste enregistre le commit parent car le lancement précède le commit ;
les SHA-256 de ses **39 fichiers sources** correspondent au contenu de `370c852`.
Les binaires, données, versions et dépendances sont également identifiés. Sur 319
cartes présentes, 176 sont invoquées ou ont un effet résolu ; cela
ne prouve pas une couverture complète des effets continus ou de chaque opération.

Le dernier commit de finition ne change que la documentation du moteur (borne mémoire),
la documentation du projet et le montage du test historique pour conserver les
permissions de recette. Il ne modifie pas le comportement du moteur de `370c852`.

Les logs bruts, l'empreinte du snapshot historique, la campagne finale et la
référence intermédiaire du noyau sont conservés localement dans le répertoire
ignoré `translate/cache/runs/verification-substitutes-20261106/`.

## Mesures CPU

Mesures sérialisées sur la VM à 4 vCPU effectifs, JAX 0.12.0.1 ; pas de GPU.
Les six mesures de moteur sont alternées et sérialisées (trois avant, trois après),
avec `bench/engine_bench.py ENGINE --b 64 --calls 20 --extra 15 --bank-arg`.
La référence est le moteur de `c01f17a`, avant les substituts. Banque de 760 cartes,
72 à effets tirables, 63 décisions par appel ; banque sans substitut.

| Mesure | Avant | Après |
| --- | --- | --- |
| Débit médian | 373 décisions/s | 370 décisions/s |
| Temps médian par appel | 168,2 ms | 169,7 ms |
| Temps par appel, trois répétitions | 167,1 / 174,5 / 168,2 ms | 170,4 / 166,8 / 169,7 ms |

L'écart médian de +0,9 % en temps reste inférieur à la dispersion observée :
pas de régression nette mesurée sur ce chemin. Ce benchmark de débuts de partie
ne chiffre pas le coût d'un lot contenant des substituts ni celui du PPO.

Le microbenchmark `bench/fusion_match_bench.py --reference REFERENCE --b 32 --n 110
--calls 50` vérifie d'abord l'égalité des sorties, puis alterne cinq séries :
**8,10 → 4,17 ms** médianes, soit environ **1,94×** plus rapide que la première
implémentation intermédiaire du calcul de substitution (référence conservée dans
les preuves locales). Il mesure uniquement ce noyau, avec des candidats synthétiques,
pas le moteur entier. Logs, échantillons et JSON des mesures sont archivés avec
les vérifications.

## Limites et suite

La borne FK=4 et le sous-ensemble strict des recettes restent en place. Les
recettes variables, alternatives complexes et callbacks arbitraires demandent
encore une représentation. Les ambiguïtés impossibles à identifier restent
refusées. La campagne aléatoire complète les scénarios ciblés ; elle ne les
remplace pas. Le prochain contrôle de débit GPU devra aussi mesurer le milieu
de partie et le coût d'entraînement, que les mesures CPU ci-dessus n'établissent pas.
