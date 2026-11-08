# Historique d’invocation et invocation générique depuis l’Extra Deck

Travail en cours, développé séparément du lot de capacités. Sources de travail : `/tmp/ygo-summon-history-v2/`. Le moteur du dépôt reste celui soumis à la régression historique du lot de capacités.

## Règles distinguées

- Un monstre Fusion face cachée dans l’Extra Deck peut être invoqué par un effet générique tel que Cyber-Stein, sous réserve de ses restrictions.
- Cette invocation n’accomplit pas sa procédure. Une fois au cimetière, un monstre à réanimation limitée ainsi invoqué ne peut pas être réanimé.
- Fusion et rituel accomplissent explicitement leur procédure. Venir de l’Extra Deck ne suffit plus à le déduire.
- Le dernier type d’invocation est distinct des statuts « ce tour » et « tant que sur le terrain ». Vampire Dragon doit encore savoir au cimetière qu’il avait été invoqué par sacrifice. Une réanimation remplace cet historique.

Le moteur de travail ajoute un seul entier par carte : type, zone d’origine et joueur sont empaquetés. Les filtres réutilisent un sélecteur signé du champ de statut ; aucun nouvel opcode ni nouvelle largeur de filtre. Les invocations Fusion sont identifiées explicitement dans les programmes manuels, le traducteur et le générateur. Les méthodes `IsNormalSummoned`, `IsTributeSummoned`, `IsSpecialSummoned`, `IsFusionSummoned`, `IsRitualSummoned`, `IsSynchroSummoned`, `IsXyzSummoned`, `IsPendulumSummoned`, `IsLinkSummoned` et `IsSummonType` utilisent cet historique.

La première prise en charge de `GetLocationCountFromEx(tp,tp,nil,c)>0` est limitée aux filtres Fusion / Synchro / Xyz. Les variantes Link, matériaux retirés, zones imposées, autre joueur et autres seuils restent refusées.

Les observations masquent l’historique des cartes cachées. Le modèle reçoit ses bits ; les deux générations antérieures d’entrées dynamiques restent compatibles par préfixe, vérifiées contre une extension explicite à poids nuls.

## État des vérifications

- Catalogue statique : 717 / 13 541 (701 dans le dépôt actuel). La conservation exacte de l’ancien ensemble et une campagne native large restent à vérifier.
- 83 tests unitaires passent sur le premier état figé. Un quatrième test d’historique a ensuite été ajouté : les quatre tests de ce module passent, y compris les deux préfixes de checkpoint avec un historique non nul.
- Cyber-Stein : 2/2 parties ciblées identiques, Extra Deck → terrain → cimetière sans procédure complète.
- Vampire Dragon : scénario sacrifice → cimetière → réanimation → cimetière en cours.
- Génération ciblée : scénarios pour les cinq invocations de procédure, avec assertion du nombre d’effets réellement résolus, préparés pour l’oracle.

Ce lot n’est pas encore validé pour intégration. La mesure CPU complète et la régression historique de ses sources finales restent nécessaires. Aucune mesure GPU ni entraînement CPU.


## Version finale en vérification

La copie `/tmp/ygo-summon-history-v4/` contient aussi le contrôle d’invocabilité à la résolution, les zones des Pendules recto de l’Extra Deck, l’historique d’une tentative échouée et l’envoi au cimetière par règle après un manque de place. Six tests ciblés de résolution passent ; la série complète passe **90 tests**. Les scénarios générés du premier historique passent **10/10**, avec des conditions effectivement résolues pour Fusion, Rituel, Synchro, Xyz et Lien. Vampire Dragon passe deux parcours avec **7 effets résolus par partie**. Deux réponses invoquant une interdiction avant la résolution de Reborn passent également. Les scénarios sont relancés sur v4, dont les deux suites historiques complètes et la campagne large restent en cours.

Un garde-fou de compilation est développé séparément dans `/tmp/ygo-summon-type-guard/` : les appels bruts à `SpecialSummon` / `IsCanBeSpecialSummoned` acceptent seulement un type générique, jusqu’à la représentation des autres types indépendamment de `CompleteProcedure`. Les helpers Fusion / Rituel gardent leur représentation explicite. Trois tests de ce garde-fou passent ; aucun type explicite n’apparaît parmi les 701 scripts du lot précédent. Le contrôle du catalogue complet de cette variante est en cours. Ce changement n’altère pas le moteur v4 figé.


## Mesure intermédiaire de v5

Sur CPU, banque identique entre versions, B=64 et Extra Deck 15, les trois répétitions donnent 304 / 316 / 310 décisions/s avant et 294 / 343 / 352 après. Les empreintes des champs d'état communs sont identiques ; le nouveau champ d'historique est exclu de cette comparaison et vérifié séparément. La dispersion est trop forte pour conclure à un gain. La banque de référence avant utilise le même encodage de marqueur, avec le test de procédure ajusté pour ne pas considérer un marqueur négatif comme une procédure complète. Pas de mesure GPU.


Le cache reconstruit à partir des sources v5 donne **717 scripts** (624 du deck principal, 93 de l’Extra Deck). Comparaison des ensembles avec le cache précédent : **16 gagnés, aucune perte, aucune exception interne**. La campagne oracle sur les graines 256–285 est en cours.


Version v5 : les marqueurs génériques de type d'invocation (dont celui de Monster Reborn) sont conservés sans déclarer une procédure accomplie. Les 94 tests unitaires et tous les scénarios ciblés passent, dont deux marqueurs distincts, les réponses bloquant une invocation, les lots mixtes et dix parties générées. Campagne sur les graines 256–285 : **30/30** identiques ; **195 / 353** cartes présentes ont été invoquées ou ont résolu un effet. Les sources de la campagne n'ont pas changé pendant son exécution. Les 18 modes historiques et obs/vmap/sb restent à terminer sur cette version.

## Correction du cache des effets continus

Une tentative d'invocation peut changer l'historique de la carte sans la déplacer. Si un effet continu dépend du type d'invocation, la clé de rafraîchissement précédente ne voyait pas ce changement : un test reproductible conservait 600 ATK au lieu des 500 attendus. La copie `/tmp/ygo-summon-history-refresh` ajoute l'historique à cette clé ; elle passe ce scénario comparé à un rafraîchissement complet et ses **95 tests unitaires**. Ses sources figées et empreintes sont archivées dans `translate/cache/runs/verification-historique-20261107/refresh/`. Les scénarios ciblés sont relancés sur cette copie ; la régression historique complète reste nécessaire avant intégration. Les résultats de v5 ci-dessus précèdent cette correction et ne sont pas présentés comme validation de ses sources finales.

Sur la copie corrigée du cache des effets continus, **95 tests unitaires**, tous les scénarios ciblés et **10 / 10 parties générées** passent. La campagne des graines 256–285 est à nouveau **30 / 30**, avec 717 scripts traduits ; l'empreinte des sources n'a pas changé pendant cette campagne. Les sources ont été reportées dans la branche locale et leur régression historique complète est en cours avant commit. Un problème distinct de partage des compteurs OATH a été découvert dans une autre copie : il n'a pas encore été intégré à ce lot.

Mesure CPU de la version corrigée du cache : **385 / 364 / 383 décisions/s avant** et **424 / 427 / 425 après**, B=64, Extra Deck 15, banque passée en argument, 20 appels par répétition hors compilation et échauffement. Les empreintes des états communs sont identiques, hors nouveau champ d'historique. La banque est identique ; le moteur de référence inclut seulement l'ajustement du test de procédure nécessaire à l'encodage des marqueurs, comme pour la mesure v5. Ces mesures séquentielles ne suffisent pas à conclure à une accélération ; elles ne montrent pas de ralentissement net sur cet essai. Journaux : `translate/cache/runs/verification-historique-20261107/refresh/mesure-cpu/`. Pas de mesure GPU.

La régression historique est terminée : **1640/1640 parties identiques**, vmap et step_batch conformes. Le premier `obs_check` a échoué parce que son test de permutation ne déplaçait pas le nouveau champ `suminfo` avec les cartes. Après correction de ce test, les **800 observations** vérifiées sont invariantes et les autres contrôles d'observation passent. Les sources moteur n'ont pas changé pendant la régression ; journaux archivés sous `translate/cache/runs/verification-historique-20261107/refresh/regression-complete/`.
