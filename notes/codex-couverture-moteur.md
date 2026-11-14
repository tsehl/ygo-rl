# Élargissement autonome du moteur — branche codex

Accord de Thomas, 14 novembre 2026 : conduire l'élargissement du moteur sur
`codex`, avec des changements faciles à fusionner avec les autres sessions.
Des sous-agents Luna peuvent traiter des tâches simples et précisément bornées.

## Organisation

- Actualiser les références distantes avant chaque lot et avant publication.
  Intégrer les changements concurrents sans écraser leur historique.
- Un lot porte une capacité précise et ses validations. Éviter les renommages,
  reformatages et refontes sans rapport, surtout dans `engine/core.py` et
  `translate/translate.py`, partagés avec les autres sessions.
- Commits cohérents, poussés sur `codex` après validation ; indiquer dans le
  bilan les fichiers touchés, les limites restantes et les mesures disponibles.
- Réutiliser les briques et représentations existantes quand leur sémantique
  convient. Une nouvelle brique doit justifier son utilité, son exposition au
  modèle et son coût ; ne pas cacher une nouvelle sémantique dans un paramètre
  ambigu dans le seul but de réduire le nombre d'opcodes.
- Éviter le coût global d'une capacité rare : chemins absents éliminables au
  traçage, travail borné par les opérations actives, aucune matrice carte × carte.
- Sous-agents : audits, inventaires et scénarios ciblés en lecture seule par
  défaut ; un seul intégrateur pour les fichiers centraux.

## Critères de livraison

Une capacité doit être représentable par le moteur, traduisible dans un périmètre
explicite, testée contre l'oracle et accessible à la génération procédurale et à
l'observation lorsque nécessaire à l'apprentissage. Les cartes officielles
restent des benchmarks, jamais des données d'entraînement.

Exiger des scénarios positifs, négatifs et perturbés par une réponse, puis les
contrôles historiques, observation, vmap et step_batch prescrits par le dépôt.
Mesurer les ajouts structurels sur GPU, avec une référence comparable et des
positions de milieu de partie ; ne pas conclure à partir du seul bruit CPU.
Distinguer traduction acceptée, effet exercé et résultat effectivement vérifié.

## Premier lot : diagnostic des groupes et relations

Point de départ : `38805d6`. Les notes historiques décrivent plusieurs prototypes
qui sont depuis partiellement intégrés ; vérifier le code et les tests actuels
avant de reprendre leurs conclusions.

1. Cartographier les refus de cibles multiples, le groupe original, les références
   filtrées à la résolution et l'ordre natif de `GetFirst`.
2. Définir le plus petit sous-ensemble cohérent, sans généraliser les formes Lua
   dont la sémantique n'est pas représentée.
3. Exiger au minimum : deux cibles conservées ; première cible déplacée ; seconde
   cible déplacée ; cible partie puis revenue ; opération échouée qui conserve
   le groupe Lua d'entrée. Vérifier explicitement l'ordre de `GetFirst`.
4. Examiner les interactions avec `OP_BLOCK` et les invocations groupées du dernier
   commit avant d'élargir la traduction.
5. Publier le périmètre retenu et ses preuves avant le lot suivant.

Priorités suivantes à confirmer par ce diagnostic : événements de phase et
effets différés, puis complétude d'une famille déjà entamée comme l'équipement.
Le classement du premier refus par carte aide au diagnostic mais ne suffit pas
à choisir une priorité : regarder aussi les blocages cumulés et la réutilisation.

## Premier sous-lot réalisé

Traduction de la première cible originale après `SelectTarget` multiple, par
réutilisation de `FIRST(original=True)` et `REQ`. Six scénarios oracle passent
(4 736 décisions identiques) et dix tests unitaires de relation passent. Détails
et limites dans `translate/README.md`. Aucun fichier du moteur d'exécution ni
du réseau modifié. La suite unitaire complète garde trois échecs préexistants,
reproduits avec le traducteur précédent.

Lots suivants : distribution procédurale de cette composition et observation
nécessaire au choix ; groupes multiples via `SetTargetCard` / `GetTargetCards` ;
scénarios de déplacement puis retour. Ne pas assimiler le sous-lot livré à la
couverture complète des groupes et relations.

## Distribution procédurale : première cible originale

Les clauses de déplacement sélectionnant deux cibles peuvent désormais opérer
sur la première originale (`first_original`). Les autres sélections multiples
gardent leur comportement. Le coût estimé compte une seule carte affectée.
Compilation par `FIRST(original=True)`, `REQ(related=True)` et filtre existants :
aucun opcode, registre ou champ d'état ajouté, formes de banque conservées.
L'audit des briques distingue désormais MINATK des variantes FIRST par ses bits.

Validations du 14 novembre 2026 :

- `tests/first_target_generation_oracle_check.py` : trois scénarios, 2 146
  décisions identiques (sans réponse, première ou seconde cible détruite).
  La seconde clause de récupération ne passe que si la première opération réussit.
- `gen/compare_gen.py --banks 0-3 --games 5 --seed0 6100` : 20/20 parties
  identiques à l'oracle sur quatre banques procédurales.
- `test_procedure_generation.py` : trois tests passent, dont présence effective
  de la composition tirée par le générateur, budgets et exposition de l'instruction
  à l'encodeur de cartes. Cela ne valide pas l'observation dynamique des cibles.

Campagne officielle sur une copie figée de `67e0359` : 30/30 parties identiques,
10 573 décisions, graines 6000–6029, `--k 8`, limitées aux 17 codes gagnés par
le traducteur. Signature du manifeste :
`6414baf7331bc91ed00de0ad895d0dda5323767dc8226c150db8ebb823f115ba`.
Sept cartes ont résolu un effet : 4929256 (8), 28357177 (13), 36280194 (40),
77334267 (30), 90928333 (30), 93920745 (5), 96947648 (17).
97518132 a seulement été invoquée. Les neuf autres n'ont exercé aucun effet :
cette campagne ne prouve donc pas la validité de chacune des 17 traductions,
ni de chaque branche d'effet des sept cartes exercées.

Prochaine étape nécessaire : l'observation montre les sources de chaîne mais
pas leurs cibles. Ne pas annoncer une couverture d'apprentissage complète.
Ne pas reconstruire les cibles par union des registres : ils contiennent aussi
des candidats et peuvent être réutilisés. La première originale est ordonnée
par l'oracle natif ; exposer l'UID interne comme information au réseau demande
une justification de visibilité, pas une simple copie de ce champ.
