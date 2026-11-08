# Audit de huit scripts utilisant les labels

Échantillon reproductible : huit scripts tirés avec la graine 20261108 parmi les 161 dont le premier refus du traducteur combiné était `méthode :SetLabel sur Eff`. Ce nombre décrit uniquement le premier refus, pas le gain qu'apporterait une nouvelle représentation.

Les usages observés dépassent le label booléen temporaire actuellement porté par le maillon : valeur 100 puis remise à zéro dans une condition (50619462), valeur d'ATK d'une carte sélectionnée (74711057), choix 0/1 réutilisé plus tard (35011819), nombre de cartes et label multiple contenant un code (59131526), joueur responsable (87340664), et indicateurs remis à zéro à des moments différents (2295440, 87210505). Le script 43711255 utilise aussi un objet associé à un label.

La représentation doit distinguer la durée de vie propre à l'effet, les valeurs associées à un maillon et, lorsque nécessaire, les valeurs multiples ou objets référencés. Un seul booléen de chaîne ne suffit pas. Les effets de bord dans une condition doivent être comparés à l'oracle ; les variantes non représentées doivent continuer à être refusées. Cet audit ne prétend pas que ces huit cartes deviendront traduisibles avec le seul label.
