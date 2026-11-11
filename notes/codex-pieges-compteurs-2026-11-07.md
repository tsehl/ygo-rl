# Activation initiale des pièges continus et compteurs

Suite de `714d94d`, développée et vérifiée dans un snapshot indépendant.
Le chantier complet du moteur et du traducteur se poursuit.

## Règles représentées

L'oracle peut proposer l'effet d'un piège continu pendant l'activation de sa
carte (question 94). Accepter remplace l'effet du même maillon ; cela ne crée
pas un second maillon. Refuser laisse l'activation vide et ne consomme pas le
compteur de l'autre effet. La nature « activation de carte » est conservée
indépendamment du genre de l'effet choisi, pour les conditions de chaîne et
les annulations. Un effet de carte continue cesse de résoudre si sa source
n'est plus liée, recto et en zone magie/piège au chargement du maillon.

Le sous-ensemble traduit comporte une activation sans callback coût/cible/
opération et un seul effet rapide libre depuis la zone magie/piège.
Les compositions avec plusieurs choix, événements déclenchés ou propriétés de
damage step supplémentaires restent explicitement refusées.

Le cœur épinglé rend les compteurs `EFFECT_COUNT_CODE_OATH` si l'activation
est niée. Il ne les rend pas pour une simple annulation d'effet. Un ancien
commentaire du compilateur inversait cette signification. La remise du compte
est désormais effectuée au chargement du maillon nié, au même stade que le
cœur natif, et respecte les compteurs par nom indexés.

## Représentation et coût

- Deux métadonnées partagées par banque : effet initial optionnel et compteur
  remboursable. Les banques sans ces possibilités ont des tableaux de largeur
  zéro, éliminant leurs chemins au traçage ; les banques procédurales gardent
  des formes fixes entre graines.
- Un entier pour l'effet courant et douze pour les maillons : **52 octets
  supplémentaires par partie**. Aucun nouveau registre de groupes ni opcode.
- Le réseau reçoit les trois bits de métadonnées et la nature publique du
  maillon. Les anciens checkpoints sont compatibles : les tests comparent leur
  résultat à une extension explicite des poids par des zéros, encodeur et
  politique compris. Aucun entraînement CPU.
- Les programmes de test dérivés regroupent les bornes de compte avec les
  filtres. Le bytecode d'exécution et ses tokens restent intacts. Les requêtes
  des candidats inactifs ont une borne de boucle nulle, y compris sous `vmap`.
  Les registres restent au nombre de quatre ; aucun calcul carte × carte.

## Résultats acquis

- **701 / 13 541** cartes traduites, sans exception interne : 612 du deck
  principal et 89 de l'Extra Deck. Sealing Ceremony of Mokuton est ajoutée aux
  700 du lot précédent ; aucune perte.
- **73 tests unitaires** passent, dont deux contrôles supplémentaires de la
  question 94 : une description identique ne suffit pas sans son contexte, et
  la source doit correspondre à l'activation en attente.
- Activation officielle : **4/4**, avec quatre acceptations et six refus
  effectivement choisis, puis des activations ultérieures de l'effet.
- Réponses adverses : **4/4**, avec quatre annulations par Seven Tools of the
  Bandit et quatre retraits de la source par Mystical Space Typhoon.
- Activation procédurale : **4/4**, douze questions proposées, 58 activations
  ultérieures effectivement payées.
- Compteurs : **4/4**, 24 réponses adverses par variante ; maximum d'une
  activation par tour sans OATH et de trois avec remise du compteur.
- Les cinq scripts ciblés du snapshot figé ont terminé avec code 0 et
  `sources_changed=false`, lot mixte de procédures compris.
- Nouveau lot de règles : **8 parties × 12 décisions**, tous les champs et
  masques identiques entre `step_batch` et `vmap(step)`, avec oui/non et les
  deux modes de compteurs exercés.
- Nouvelle campagne du catalogue, graines **226–255** : **30/30 parties
  identiques**, 174 cartes invoquées ou effets résolus parmi 350 cartes
  présentes. Artefact : `20261107T150551-565a60b9d8e8.json`.

Le snapshot est une archive locale non versionnée. La régression complète termine avec
code 0 : **1640/1640 dans les 18 modes**, obs/vmap/sb propres, retours en cours de
résolution compris. Campagne, résultats ciblés, sources et empreintes sont
archivés dans `translate/cache/runs/verification-pieges-compteurs-20261107/`.
Les sources d'exécution correspondent au snapshot ; les deux nouveaux tests du
comparateur et le script du lot mixte ont été vérifiés séparément sur `codex`.

## Mesure CPU

Référence identique, B=64, Extra Deck 15, 20 appels, banque en argument.
Compilation exclue, un passage supplémentaire d'échauffement après la mise en
attente des deux moteurs ; trois répétitions depuis les mêmes états et choix.
Les autres validations sont suspendues pendant les appels chronométrés.

Avant : **306, 308, 308** décisions/s ; après : **300, 310, 310**.
Médianes **308 → 310** : écart insuffisant pour annoncer un gain ou une
régression mesurable sur ce benchmark. Les champs communs de la banque sont
inchangés, à l'exception du programme de test dérivé ; les nouvelles métadonnées
sont vides dans cette banque de référence. Pas d'extrapolation GPU.

## Suite

Le registre des noms de quatre places peut oublier un compteur ou une
interdiction lorsque plus de quatre noms distincts sont utilisés dans le même
tour. Son correctif est développé séparément pour conserver les sources de la
régression actuelle. Les scénarios de référence restent archivés, y compris
leurs échecs intermédiaires de mise au point.
