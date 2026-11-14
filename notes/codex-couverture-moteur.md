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
