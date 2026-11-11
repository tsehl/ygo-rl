# Coûts de procédures, groupes conservés et rituels

Branche `codex`, base `6a10deb2cf4f440b7c7a42f9742f1686f2508849`.
Travail des 6–7 novembre 2026. Le chantier du moteur et du traducteur continue.

## Changements

- Les procédures paient leurs coûts avant le placement, sans créer de chaîne.
  Le chemin direct sans paiement reste disponible. Aucun tableau d'état ajouté.
- Les groupes virtuels du compilateur sont alloués aux quatre registres physiques
  par analyse de durée de vie. Les requêtes pures identiques sont partagées,
  les requêtes mortes et les tests redondants supprimés. Les mutations et la
  résolution invalident le partage. Les instructions qui modifient leur registre
  conservent leur emplacement physique et leurs groupes indépendants.
- Les requêtes capturent leurs références au moment de l'appel. `Filter`,
  `FilterSelect`, `FilterCount`, `Clone` et les comptes à la résolution sont
  réellement exécutés. Seules les gardes qui déplacent uniquement leur groupe
  peuvent être effacées : déplacer un groupe vide est déjà sans effet.
- Générateur et Lua : paiements LP, défausse et bannissement du cimetière pour
  les procédures. Les limites d'activation des magies/pièges générés sont
  également respectées par leurs briques JAX.
- Rituels Greater/Equal usuels : réutilisation des instructions existantes.
  Un niveau littéral exige l'égalité avec le niveau de chaque monstre nommé
  dans la base épinglée. La limite de réanimation d'un rituel sans effet ne
  consomme aucun emplacement d'effet.
- `stats()` tient compte des protections temporaires au combat. La valeur
  native constante 1 est normalisée en seuil d'ATK 0, y compris contre ATK 0.
- La validation réserve 15 slots d'Extra Deck par joueur ; `extra_counts`
  masque les slots inutilisés avant l'initialisation. Les decks natifs ne
  changent pas et les tailles asymétriques ne recompilent plus le moteur.
- Les zones déclarées permettent de distinguer deux effets sans description
  lorsque leurs métadonnées natives suffisent. Les cas qui se recouvrent
  restent non concluants, sans consulter la légalité JAX.

## Catalogue et campagnes

**700 / 13 541**, soit **128 ajouts, aucune perte** par rapport aux 572 de
la base. Aucune exception interne. 611 cartes du deck principal, 89 de l'Extra
Deck (60 fusions, 8 synchros, 18 Xyz, 3 Link).

Deux campagnes finales sur le snapshot immuable une archive locale non versionnée :

| Périmètre | Graines | Parties identiques | Cartes exercées / présentes |
| --- | --- | --- | --- |
| Catalogue complet | 166–195 | 30/30 | 181 / 347 |
| 128 nouveaux ajouts | 196–225 | 30/30 | 52 / 125 |

« Exercée » signifie invoquée ou effet résolu selon les messages oracle,
uniquement dans les parties identiques. Cela ne valide pas tous les effets
des 700 cartes.

Artefacts dans `translate/cache/runs/` du snapshot :
`20261106T190510-f223a8ca6a24.json` et
`20261106T190725-a1477fa98174.json`.
Les manifestes conservent les empreintes des sources, dépendances, base et
oracle, ainsi que les decks, choix et événements de chaque partie.

Les diagnostics intermédiaires restent conservés :

- Draw Muscle, graine 182 : protection temporaire ignorée, corrigée. Le replay
  reprend les decks et les **98 premières décisions natives** de la divergence
  initiale ; il passe désormais.
- Sealing Ceremony of Mokuton : proposition de l'effet lors de l'activation
  du piège continu (question native 94). Ce lot refuse explicitement cette
  composition en attendant sa représentation dans le moteur. Les anciennes
  572 cartes restent acceptées.
- Lunalight Purple Butterfly : descriptions nulles mais zones disjointes.
  La correspondance statique corrige le cas sans masque légal JAX.
  Deux parties ciblées exercent **8 activations en main et 6 au cimetière**.

## Vérifications ciblées

- **59 tests unitaires**. L'allocateur est comparé à un interpréteur indépendant
  sur 200 états aléatoires, avec mutations et groupes conservés.
- Procédures officielles à coût : **8/8**, **16 procédures jouées**
  (12 Deep-Space Cruiser IX et 4 Threshold Borg).
- Groupes conservés : **6/6**, **52 résolutions** (14 snapshots déplacés,
  14 gardes sur groupe vide, 24 déplacements conditionnels suivis de pioche).
- Paiements générés ciblés : **6/6**, **4 LP, 4 défausses, 2 bannissements**.
- Lot mixte : **4 × 40 décisions**, tous les champs, masques légaux et tokens
  d'action identiques entre `step_batch` et `vmap(step)` ;
  **12 procédures payées et 28 invocations normales**.
- Extra Decks de tailles **0/0, 1/0 et 3/1** : trois parties identiques,
  **2 fusions** invoquées. Une compilation d'initialisation pour quatre
  paires de tailles. Les invariants détectent aussi une carte active perdue
  ou un slot de remplissage réactivé.
- Rituels officiels : **4/4**, **4 rituels invoqués** (Black Luster Soldier
  et Demise), **6 résolutions de Demise**.
- Draw Muscle ciblé : **2/2**, **38 résolutions**, **28 attaques** vers un
  monstre protégé. Protection générée : **2/2**, **40 résolutions**,
  **6 attaques** vers un monstre protégé.

Suite ciblée : une archive locale non versionnée,
`sources_changed=false`. Son passage unitaire initial a échoué parce que le
snapshot ne contenait pas `train/ckpt/run3_it120_f16.pkl.gz`. Le fichier
d'origine a été rendu accessible et les **59/59 tests** ont été rejoués sans
modifier les sources (une archive locale non versionnée).
Le résultat initial reste conservé.

La régression historique a été interrompue après 11 modes terminés.
`core.py` et `cards.py` de ce snapshot v2 sont identiques à ceux du v4.
Le comparateur v4 ajoute la correspondance de portée des cartes traduites,
chemin absent des modes historiques. Les sept modes restants et obs/vmap/sb
sont repris sur le v4 sans compter les anciennes parties incomplètes.
Résultat consolidé dans
`translate/cache/runs/verification-couts-rituels-20261107/results.json` :
**1640/1640, obs/vmap/sb propres**, y compris `step_batch(frac=0.25)`.
Les sources exécutables archivées correspondent à celles de ce commit ;
seule la documentation README a été complétée après le snapshot.
Les dix parties générées ont aussi été rejouées sur ce snapshot : **10/10**.

## Performance et limites

CPU seulement, 4 vCPU effectifs. B=64, 20 appels, Extra Deck 15, banque de
référence passée en argument ; compilation et échauffement exclus.
Les deux moteurs sont préparés avant la mesure, puis les validations
concurrentes suspendues pendant les appels chronométrés. Trois répétitions
repartent du même état et des mêmes choix.

Avant : **373, 361, 381** décisions/s ; après : **323, 349, 356**.
Médianes **373 → 349**, soit **−6,4 %**. Ce relevé ne démontre aucun gain.
Les banques de référence ont la même empreinte. Des variantes optimisées sont
mesurées et validées séparément. Aucun résultat GPU ni entraînement CPU.

Les paiements libérant une zone pleine, les contraintes complexes de groupes,
les équipements, les raisons d'événement, les labels scalaires et les événements
de phase restent des chantiers distincts. Les motifs non représentés sont
refusés. L'activation initiale des pièges continus est développée dans une
copie séparée.
