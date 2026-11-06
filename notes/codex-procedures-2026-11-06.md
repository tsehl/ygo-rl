# Procédures de main et poursuite du moteur — 2026-11-06

Objectif : étendre le moteur et le traducteur jusqu'au jeu complet, avec validation
contre ygopro-core et suivi du coût sous JAX/vmap. Cette livraison poursuit le
chantier ; elle n'établit pas encore la complétude du traducteur.

## Implémentation

- Nouveau genre `K_PROC` dans l'emplacement 7 déjà réservé par l'encodeur :
  procédure inhérente depuis la main, réutilisant les programmes de conditions,
  les emplacements d'action et les placements du moteur. Elle ne construit pas
  de chaîne et ne consomme pas l'invocation normale.
- Conditions de zone, nombre de cartes et filtres représentables ; limites par
  nom via les compteurs existants. Une interdiction d'activation par nom ne
  s'applique pas à une procédure. Les interdictions d'invocation spéciale et
  l'absence de case principale libre s'appliquent.
- `TYPE_SPSUMMON` interdit correctement l'invocation normale et la pose. La
  métadonnée `erl` distingue la réanimation après invocation correcte et
  l'interdiction permanente d'invocation par effet (`aux.FALSE`). Le cas est
  également pris en compte dans l'invocation pendule.
- `REQ` exprime aussi une borne supérieure, sans nouvel opcode. Un nombre de
  cases libres stocké dans une variable locale reste symbolique et se compile
  vers le même `REQ_FREE` que l'appel direct.
- Les clones SINGLE identiques de déclencheurs d'invocation normale, flip et
  spéciale sont regroupés par union des événements, lorsqu'ils ont les mêmes
  programmes, limite partagée par nom (ou aucune limite), description et événements
  disjoints. Les déclencheurs à compteurs par effet restent refusés et ne sont
  pas regroupés. Aucun regroupement général de déclencheurs FIELD ou d'événements arbitraires.
- Le générateur produit ces procédures sur des monstres procéduraux ; le
  compilateur Lua, le comparateur et l'audit des briques les connaissent aussi.
  L'encodeur voit le genre, les conditions, filtres et limites ; aucun identifiant
  de vraie carte n'entre dans l'entraînement. Aucun entraînement CPU effectué.
- L'empreinte des campagnes inclut désormais aussi les sources Python et Lua du
  générateur.

Le catalogue passe de **507 à 572 / 13 541** traductions utilisables : 490 cartes
du deck principal et 82 de l’Extra Deck (60 fusions, 6 synchros, 13 Xyz, 3 Links).
**33 cartes** ont une procédure de main. Ce total décrit l’acceptation par le
compilateur, pas une preuve que tous leurs effets ont été exercés.

Les procédures à coût, opération, conditions non représentables, plusieurs
procédures pour la même carte ou limites non représentables restent refusées.
Les drapeaux inconnus de `SetCountLimit` sont explicitement refusés.

Cinq filtres de matériaux Fossil attendaient un contrôleur fourni par une
procédure non représentée. Un contexte nil dépilé devenait un accès hors borne :
ils sont maintenant refusés par `Unsupported`, avec un test sur les cinq scripts.
Le passage complet des 13 541 scripts ne produit plus d’exception interne. Les
572 traductions acceptées sont identiques avant/après ce durcissement (comparaison
intégrale des résultats du cache, pas seulement du total).

## Défaut révélé par les nouvelles campagnes

Banque générée 0, partie 0, 80 cartes principales et 10 Extra, focus activé : un
monstre FLIP marqué détruit au combat est encore présent sur le terrain lorsque
son effet est proposé par l'oracle. Son prédicat sur lui-même était rejeté par
le moteur, qui l'envoyait déjà au cimetière. Le défaut préexistait aux procédures.

Les requêtes de groupe et les prédicats (`Group:Filter` / `SELF`) permettent
maintenant ces cartes, comme `field::filter_matching_card` dans le cœur épinglé.
Le ciblage continue d'exclure les monstres marqués détruits au combat. La fixture
minimale est conservée dans
`tests/fixtures/damage_flip_self.json`, avec un contrôle de parties complètes et
une assertion d'activation réelle de l'effet. Un témoin séparé exécute cette
fixture avec le moteur de `e9131a0` et reproduit effectivement l'écart attendu ;
la version corrigée passe. Les sources du témoin ont été comparées au commit
de référence, pas seulement identifiées par le nom du répertoire.

## Optimisation

Aucun nouveau tableau de banque, champ d'état par partie, verbe d'action ou
dimension de modèle. Les conditions réutilisent le VM existant et son nombre
borné de candidats ; aucun calcul carte × carte n'est ajouté. Les formes des
banques procédurales restent fixes. Le contrôle des limites est partagé avec
les activations plutôt que de passer par la construction d'une chaîne.

Mesures CPU sur les 4 vCPU effectifs de la VM, JAX 0.12.0.1 : trois répétitions
alternées avant/après, exécutées une à une après tous les tests, compilation
exclue. Référence : moteur `e9131a0`, dont les fichiers ont été comparés au témoin.
Commande : `bench/engine_bench.py ENGINE --b 64 --calls 20 --extra 15 --bank-arg`.
Banque de référence de 760 cartes, 72 à effet tirables, sans les nouvelles cartes
à procédure de main, 63 décisions par appel.

| Mesure | Avant | Après |
| --- | --- | --- |
| Débit médian | 377 décisions/s | 383 décisions/s |
| Temps médian par appel | 166,6 ms | 163,9 ms |
| Trois temps par appel | 166,6 / 165,2 / 169,0 ms | 163,9 / 176,8 / 161,7 ms |

Le petit écart médian (−1,6 % en temps) reste inférieur à la dispersion observée.
Il ne permet pas d'affirmer un gain ; aucune régression nette n'est mesurée sur
ce scénario. Les compilations préalables tournant en parallèle des tests ne font
pas partie des mesures.

Sur la banque procédurale actuelle de graine 0, 400 cartes principales et 80 Extra,
`--proc 0 --b 64 --calls 20 --bank-arg` donne **303 décisions/s, 208,5 ms/appel**
(Extra Deck de 15, N=110, 63 décisions/appel). Ce scénario inclut les nouvelles
procédures ; c'est une mesure de la version actuelle, sans témoin équivalent avant
l'ajout du générateur. Le compteur imprimé « 0 à effet tirables » est celui des
indices de la banque historique et ne décrit pas les effets de cette banque
procédurale.

Ces benchmarks de débuts de partie ne mesurent ni le GPU ni le coût PPO de bout
en bout. Aucun entraînement n'a été lancé.

## Vérifications et provenance

| Contrôle | Résultat |
| --- | --- |
| Tests unitaires | 39 réussis, code de sortie 0 |
| Procédures de main, scénarios réels | 16/16 parties identiques, dont des cas négatifs |
| Lot mixte, procédures et invocations normales | 4 × 40 décisions : mêmes états/masques que vmap(step), actions tokenisées ; 5 choix de procédures, 34 normaux |
| Cartes générées, banques 0–1, parties 0–4, focus | 10/10 identiques, 44 procédures de main réellement exercées |
| FLIP sur soi-même après destruction au combat | 2/2 identiques, 8 activations réelles |
| Anciens scénarios ambigus, decks complets | 5/5 identiques, 7 identités lues côté oracle |
| Campagne traduite finale, graines 136–165, k=8 | 30/30 identiques ; 180/326 cartes présentes invoquées ou avec un effet résolu |
| Régression historique, moteur final | 1640/1640 identiques, 18 modes |
| Observations standard | 9025 états, 800 invariances, 99 contrôles positifs |
| vmap standard | lot égal au séquentiel, 16/16 parties finies |
| step_batch standard | égalité avec vmap(step), dont 289 pauses en résolution et 114 attentes forcées |

Le test d’identité déplie volontairement les clones des trois monstres concernés
pour continuer à exercer l’inspecteur natif. Les campagnes ordinaires et Moon
Gardna vérifient la représentation regroupée. Une limite par effet n’est pas
remplacée par une limite partagée.

Campagne finale : `translate/cache/runs/20261106T102301-313045afecef.json`,
signature `ac86cdbf17aabdcf94b9f9fcf20851317920d70a61a6cb048e73455cc24abb26`.
Aucune source modifiée pendant son exécution ; les empreintes de tous les fichiers
correspondent aux fichiers finaux. Le manifeste enregistre le commit parent
`e9131a0`, car la campagne précède le commit de livraison. Le contrôle additionnel
de lot mixte a été ajouté après cette campagne ; le code
du moteur et les traductions restent inchangés. La couverture de
résolution ne prouve pas qu'une opération a modifié l'état, ni que tous les effets
de chaque carte ont été exercés.

La régression historique utilise la copie figée `/tmp/ygo-hand-final-v3`. Les
fichiers du moteur sont identiques à ceux livrés ; les dernières modifications
du traducteur ne changent pas ce moteur. Les premières campagnes ont été arrêtées
pour corriger la divergence découverte ; seule la campagne finale est comptée ici.
La commande complète des 18 modes puis observations/vmap/step_batch termine avec
le code de sortie **0**. Les versions intermédiaires et leurs campagnes
interrompues ne valent pas validation de la version finale.

## Suite vers la complétude

Les refus restants doivent être traités par capacité sémantique, puis vérifiés
avec des scénarios positifs et négatifs qui exercent effectivement la capacité.
L'inventaire ne donne que le premier refus d'une carte : supprimer un refus peut
en révéler un autre sans rendre la carte utilisable.

Priorités issues de l'inventaire : procédures avec sélection et paiement de
matériaux/coûts (193 premiers refus : 166 combinent `SetTarget` et `SetOperation`,
27 utilisent une opération sans cible) ; équipements et leurs relations persistantes ; provenance des
déplacements (`IsReason`) et événements supplémentaires ; déclencheurs de phase,
effets temporaires enregistrés par une résolution et resets ; recettes Extra Deck
variables/alternatives ; compteurs, unicité et limitations d'invocation.

La cible « traducteur complet » demande aussi de dépasser proprement les bornes
actuelles (deux effets, un déclencheur en attente par carte, quatre registres,
longueur des programmes et budgets de candidats / compteurs nommés), sans augmenter globalement tous les tenseurs et toutes les branches.
Conserver des classes de capacité statiques et mesurer les banques représentatives
avant de modifier ces bornes. Une traduction syntaxiquement acceptée n'est pas
une preuve de conformité aux règles ou de couverture complète des effets.


Les logs bruts, les empreintes du snapshot, l'inventaire final, le témoin de
régression et ses sources, les échantillons des mesures et leur script sont
conservés dans le répertoire local ignoré
`translate/cache/runs/verification-procedures-20261106/`. `sha256.json` identifie
les fichiers de preuve ; le JSON de campagne reste également dans `cache/runs/`.
