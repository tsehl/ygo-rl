# Validation et fusions — reprise du 6 novembre 2026

Branche : `codex`. Commits fonctionnels : `ebc5e5e`, `616eb68`, `47e0fcf`, `58bafb9`.

Runtime : Python 3.12.14, JAX/jaxlib 0.12.0.1, NumPy 2.5.4, luaparser 4.2.0 ; CPU, 4 vCPU effectifs et 32 Gio.

## Changements

- `compare.py`, `validate.py`, `regress.sh` propagent les échecs. Le validateur distingue
  divergence, erreur du pilote et correspondance non concluante. Une campagne vide est refusée.
- Campagnes indépendantes, avec signature des sources/données/dépendances, versions du runtime,
  paramètres, graines, decks, traces et couverture. La clé du cache inclut les versions du runtime. Le JSON final et le cache sont publiés
  atomiquement ; un journal conserve les parties terminées en cas d'interruption.
- Présence, tentative d'invocation, invocation réussie, activation, annulation et résolution
  non annulée sont distinguées à partir des messages de l'oracle. La couverture par description
  d'effet est conservée dans chaque partie. Une invocation ne prouve pas que tous les effets
  d'une carte ont été testés ; une résolution peut ne produire aucun changement d'état.
- Les descriptions sont associées sans utiliser la légalité JAX. Les types statiques permettent
  d'écarter les candidats impossibles ; les véritables ambiguïtés font échouer la campagne.
- Bug confirmé et corrigé : `aux.Stringid` utilisait `code * 16 + index` au lieu du codage
  `(code << 20) | (index & 0xfffff)` de la bibliothèque Lua épinglée. Le repli sur l'effet 0
  masquait cette erreur. Les graines 76 à 79 passent après correction.
- Recettes `Fusion.AddProcMix` / `AddProcMixN`, avec deux à quatre matériaux fixes, par code
  ou filtre représentable, et effet standard `Fusion.RegisterSummonEff(c)`.

## Vérifications terminées

- 18 tests du banc : propagation des échecs, ambiguïtés, descriptions, protocole oracle,
  recettes et refus explicites, provenance, écritures atomiques et cache interrompu.
- 16/16 parties ciblées contre l'oracle : matériaux distincts, répétés, filtres qui se recouvrent,
  matériau absent, quantité insuffisante ; scripts officiels Allvain, Theseus, Zirconia.
  Dans chacun des six scénarios positifs, quatre fusions ont effectivement été invoquées.
  Dans les deux scénarios négatifs, aucune fusion ni résolution de Polymerization.
- Inventaire : 446/13 541 cartes traduites (421 du deck principal et 25 de l'Extra Deck).
  Gain : Polymerization et quatre fusions (Allvain, Theseus, Zirconia, D.3.S. Frog).
  Traduire une recette ne garantit pas la disponibilité de tous ses matériaux dans le sous-ensemble
  jouable : Des Frog, matériau de D.3.S. Frog, reste notamment hors de ce sous-ensemble.
- Campagne aléatoire : graines 74–103, `--games 30 --k 8 --bisect` : **25 identiques,
  5 non concluantes, aucune divergence d'état observée**, code de sortie **2**.
  129 cartes invoquées ou avec un effet résolu dans des parties identiques uniquement, sur
  293 cartes présentes. Ce nombre ne constitue pas une validation exhaustive de leurs effets.
- Une partie supplémentaire (graine 104, Monster Reborn) vérifie la sauvegarde atomique réelle :
  1/1 identique, code 0. Une nouvelle reconstruction du cache (clé incluant les versions du runtime)
  suivie de la graine 105 passe également à 1/1, code 0.

## Ambiguïtés encore ouvertes

| Graine | Carte | Limite du pilote |
|---|---|---|
| 90, 98 | Photon Caesar (13492423) | Deux déclencheurs optionnels ; description générique 221 |
| 97 | Sniffer Dragon (27877771) | Deux déclencheurs optionnels ; description générique 221 |
| 100 | XX-Saber Ragigura (87292536) | Deux déclencheurs optionnels ; description générique 221 |
| 103 | Hidden Soldiers (2047519) | Deux activations sans descriptions distinctes |

Les scripts distinguent notamment les événements d'invocation normale, flip ou spéciale, mais
le message de choix ne fournit pas une identité d'effet suffisante pour les départager directement.
Ne pas réintroduire un choix basé sur `legal()` ou `tg_e` du moteur testé. Pour poursuivre,
il faut enrichir l'identification indépendante côté oracle ou construire des scénarios dont le
contexte d'événement est établi indépendamment.

Pour reproduire un cas original avec le sous-ensemble actuel :

```sh
YGO_JAX_CACHE=/workspace/.cache/ygo-jax .venv/bin/python translate/validate.py --games 1 --k 8 --seed0 90 --nosave
```

Le JSON sauvegardé contient également les decks, choix et messages originaux. `--bisect` demeure
un rejeu isolé, pas une réduction préservant la divergence ; aucune absence d'échec en isolation
ne suffit à résoudre une interaction.

## Limites des fusions

Les substituts de matériaux nommés, quantités variables, tables d'alternatives et filtres hors
`MAT_FIELDS` restent refusés. Les quelque 494 cartes de la piste initiale représentent un potentiel,
pas un gain acquis. Les effets des autres cartes doivent aussi être traduits et leurs matériaux
accessibles. Les tests fictifs de `tests/fixtures/fusion` ne servent jamais à l'entraînement.

## Versions effectivement testées

Les empreintes de tous les fichiers Python des trois dossiers engine, oracle et translate ont été comparées aux commits :

| Contrôle | Sources |
|---|---|
| Suite historique | `ebc5e5e` |
| Campagne traduite 74–103 et fusions ciblées | `616eb68` |
| Persistance atomique, graine 104 | `47e0fcf` |
| Cache versionné, graine 105 | `58bafb9` |

## Non-régression historique

- **1 640/1 640 parties identiques**, dans les 18 modes historiques ; commande terminée avec le code 0.
- obs : états vivants : 9025 ; 1. ok : toutes les actions légales ont un token ; 2. ok : 800 observations invariantes à l'information cachée et aux slots ; 3. ok : 99 contrôles positifs détectés
- vmap : ok : lot == séquentiel ; 16 / 16 parties finies ; tours : [71 65 68 57 59 35 58 49 69 72 65 66 67 69 40 69]
- sb : ok : step_batch == vmap(step) sur 32 parties × 150 décisions ; 0 finies ; ok : step_batch(frac=0.25) suit vmap(step) décision par décision (32 parties × 60 décisions ; 289 parties rendues en cours de résolution, 114 avec une action forcée en attente)

Exécution avec trois puis quatre processus CPU ; interpréteur de `.venv`.


Le noyau `engine/core.py` et les cartes codées dans `engine/cards.py` n'ont pas été modifiés.
La suite historique tourne sur une copie figée dont les 15 fichiers Python des dossiers engine,
oracle et translate ont été vérifiés identiques à `ebc5e5e`. Les correctifs suivants concernent
la correspondance des métadonnées des cartes traduites (testée dans la campagne 74–103) et la
persistance (testée séparément) ; le parcours historique n'utilise pas `DESC`.

Les campagnes interrompues pendant le développement sont diagnostiques et ne sont pas comptées
comme réussites. Les logs et les signatures de référence sont conservés localement, dans le cache
ignoré du dépôt ; le présent bilan est versionné.

## Artefacts locaux

- `translate/cache/runs/20261106T042310-7a79137af5ec.json` : campagne 74–103 ; empreintes des 15 fichiers Python vérifiées identiques à `616eb68`.
- `translate/cache/runs/20261106T042648-9381bddbf4e7.json` : contrôle de persistance ; empreintes vérifiées identiques à `47e0fcf`.
- `translate/cache/runs/20261106T050849-371df02ac546.json` : reconstruction avec clé versionnée et partie 105 ; sources vérifiées identiques à `58bafb9`.
- `translate/cache/runs/verification-20261106/` : logs compressés de la régression, résumé et empreintes de référence, tests ciblés et tests du banc.

Ces fichiers sont ignorés par Git. Les graines, versions et commandes du présent bilan permettent de refaire les contrôles.
