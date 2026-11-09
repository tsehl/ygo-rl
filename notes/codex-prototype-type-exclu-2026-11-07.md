# Prototype : exclusion de types

Une copie de travail séparée étend le filtre existant avec un champ `not_typ` : le masque positif conserve le sens « au moins un de ces types », tandis que le nouveau masque exclut tous les types indiqués. Cette représentation couvre par exemple « monstre et pas monstre à effet », sans ajouter d'opcode. Le filtre général, le filtre allégé de matériaux et le générateur Lua utilisent la même règle.

Cinq tests ciblés passent, dont l'inclusion et l'exclusion combinées pour les matériaux, le refus d'une disjonction non représentable et la compatibilité d'un modèle entraîné avec la version précédente. Les nouvelles informations sont ajoutées à la fin des caractéristiques afin de conserver le préfixe des anciens modèles. Un scénario natif avec les deux types de monstres est en cours.

Une première variante utilisait un signe négatif dans le champ existant, sans pouvoir représenter simultanément inclusion et exclusion ; elle a été remplacée. Sur l'échantillon initial de 80 refus, cette première variante traduisait une carte (Arnis, déjà gagnée grâce à la raison de combat), sans exception interne. Ce résultat ne permet aucune estimation de couverture globale pour la nouvelle représentation.

Ce prototype inclut la copie de travail des raisons et n'est pas validé pour intégration. Sources locales : `translate/cache/runs/prototype-type-exclu/sources/` ; mettre à jour les empreintes avant la validation finale. La mesure CPU du prototype des raisons ne comprend pas ce champ supplémentaire.

Après ajout de la correction OATH, les cinq tests ciblés et les dix tests d'activation combinée passent ; ces derniers incluent le chargement d'un ancien checkpoint réel. La nouvelle version traduit également une carte sur l'échantillon de 80, toujours Arnis, sans exception interne. La comparaison native à deux types a passé deux parties sur la version précédente ; elle est relancée sur les sources exactes intégrant OATH.

Sur les sources intégrant OATH, le scénario natif est passé **2 / 2** : un monstre normal et un monstre à effet sont présents au cimetière, puis dix recherches par partie sélectionnent conformément au filtre combiné. La comparaison contrôle les actions légales et l'état du duel. Journal : `translate/cache/runs/prototype-type-exclu/oracle-v3.log`.

Le générateur utilise désormais parfois « monstre et pas monstre à effet ». Dans deux banques de test de 90 cartes, respectivement deux et trois cartes comportent ce filtre. La série unitaire est relancée sur les sources finales du générateur ; la couverture réelle des filtres doit encore être mesurée dans les parties.

L'audit des recettes a révélé que `PG.mat` ignorait les contraintes de type des matériaux. Le générateur Lua les conserve maintenant, y compris lorsqu'elles sont combinées à un nom. Un scénario de fusion avec deux matériaux répondant à « monstre sans effet » et un cas négatif avec des monstres à effet est en cours ; la série unitaire est relancée sur cette version.

Les recettes générées conservent aussi la différence entre un nom brut, qui peut autoriser un substitut de fusion, et un filtre de nom combiné à d'autres contraintes, qui ne l'autorise pas. Un test ciblé du masque de substitution passe ; les scénarios natifs et la série unitaire sont relancés sur cette version.

La version corrigée passe **112 tests unitaires**. Le scénario de fusion passe **4 / 4 parties** contre l'oracle : une polymérisation et une fusion par partie avec les matériaux normaux, aucune avec les matériaux à effet exclus. Journaux : `translate/cache/runs/prototype-type-exclu/`.

Une recette Lua `Fusion.AddProcMix` avec deux fois « monstre et pas monstre à effet » a également été traduite : les deux filtres produits contiennent bien les deux contraintes. Test reproductible : `tests/negative_type_translation_check.py` dans la copie du prototype.

La campagne générée passe **10 / 10 parties** sur deux banques. Cinq cartes générées comportent un filtre d'exclusion ; quatre ont été activées ou invoquées, et deux ont eu au moins un effet résolu. Cela ne prouve pas à lui seul que chaque filtre a servi : les scénarios ciblés de recherche et de fusion vérifient séparément cette logique.

Mesure CPU B=64, Extra Deck 15, table OATH allouée dans les deux versions, 20 appels par répétition hors compilation/échauffement : **419 / 419 / 416 décisions/s avant**, **344 / 374 / 392 après**, médianes 419 → 374 (−10,7 %). Les champs d'état communs ont des empreintes identiques. Le filtre d'exclusion est nul dans cette banque : la mesure porte sur le coût de sa représentation et de son traitement, pas sur une fréquence de filtres réellement utilisés. La série après remonte de 344 à 392, ce qui révèle une variabilité temporelle ; une mesure supplémentaire et une recherche d'optimisation sont nécessaires avant d'interpréter précisément l'écart. Aucun résultat GPU.

Mesure répétée dans l'ordre inverse : **416 / 417 / 418 décisions/s après**, puis **384 / 398 / 413 avant**, mêmes paramètres et mêmes empreintes d'état. L'écart change de signe lorsque l'ordre change : il n'est pas possible d'attribuer précisément la baisse de la première mesure au champ supplémentaire. Une variante évitant le traitement si l'exclusion est vide est évaluée séparément.

Variante conditionnelle isolée (boucle de borne zéro pour un filtre d'exclusion vide) : **407 / 400 / 403 décisions/s avant**, **348 / 354 / 339 après**, mêmes banques et états communs. Elle ralentit cet essai et a été écartée ; les mesures, bruitées et CPU uniquement, ne démontrent pas un résultat GPU.
