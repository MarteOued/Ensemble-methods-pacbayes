# Vérification de mes implémentations contre scikit-learn
Moyennes sur 6 jeux (wdbc, pima, australian, iono, german, yeast3) x 10 découpages train/test, soit 60 cas par méthode. Mêmes découpages, mêmes graines, mêmes réglages pour ma version et celle de scikit-learn ; 200 arbres / tours.

## Tableau 1 : erreur de test et accord des prédictions
« Accord » = part des exemples de test où les deux versions prédisent la même classe. « Réf. » = même mesure entre scikit-learn et scikit-learn relancé avec une autre graine (niveau d'écart normal entre deux exécutions).

| Méthode | Erreur moi | Erreur sklearn | Accord | Accord réf. | Verdict |
|---|---|---|---|---|---|
| Bagging (arbres complets) | 13,30 % | 13,17 % | 98,02 % | 98,10 % | proche, écart du même ordre que sklearn contre lui-même |
| AdaBoost, stumps | 13,25 % | 13,25 % | 100,00 % | 100,00 % | identique |
| AdaBoost, profondeur 3 | 13,65 % | 13,71 % | 99,16 % | 99,49 % | proche, écart du même ordre que sklearn contre lui-même |
| Stacking | 12,59 % | 12,59 % | 100,00 % | – | identique |

## Tableau 2 : contrôles internes
Écart maximal sur les 10 premiers tours d'AdaBoost (alpha de sklearn = 2 fois le mien, j'ai donc comparé 2 x mon alpha) et sur les coefficients du méta-modèle du stacking. « Cas divergents » = nombre de cas sur 60 où alpha s'écarte de plus de 0.01 ; entre parenthèses, le même nombre pour sklearn contre lui-même.

| Méthode | Écart max alpha | Écart max epsilon | Écart max coefficients | Cas divergents |
|---|---|---|---|---|
| AdaBoost, stumps | 0,000 | 0,000 | – | 0 (0) |
| AdaBoost, profondeur 3 | 1,290 | 0,110 | – | 11 (9) |
| Stacking | – | – | 0,000 | – |

## Tableau 3 : erreur out-of-bag du bagging
L'erreur OOB est calculée sur le train, sans jeu de validation.

| Jeu | OOB moi | OOB sklearn | Erreur de test (moi) |
|---|---|---|---|
| australian | 14,37 % | 14,37 % | 13,43 % |
| german | 24,13 % | 23,99 % | 23,83 % |
| iono | 8,00 % | 7,76 % | 8,77 % |
| pima | 25,55 % | 25,61 % | 23,33 % |
| wdbc | 4,55 % | 4,42 % | 5,38 % |
| yeast3 | 5,66 % | 5,44 % | 5,02 % |
| **moyenne** | 13,71 % | 13,60 % | 13,30 % |

## Tableau 4 : détail par jeu

| Jeu | Méthode | Erreur moi | Erreur sklearn | Accord |
|---|---|---|---|---|
| australian | AdaBoost, stumps | 15,12 % | 15,12 % | 100,00 % |
| australian | AdaBoost, profondeur 3 | 15,12 % | 15,12 % | 99,61 % |
| australian | Bagging (arbres complets) | 13,43 % | 12,95 % | 98,26 % |
| australian | Stacking | 12,61 % | 12,61 % | 100,00 % |
| german | AdaBoost, stumps | 24,20 % | 24,20 % | 100,00 % |
| german | AdaBoost, profondeur 3 | 25,80 % | 25,80 % | 99,93 % |
| german | Bagging (arbres complets) | 23,83 % | 24,10 % | 96,00 % |
| german | Stacking | 23,90 % | 23,90 % | 100,00 % |
| iono | AdaBoost, stumps | 8,58 % | 8,58 % | 100,00 % |
| iono | AdaBoost, profondeur 3 | 7,26 % | 7,45 % | 99,43 % |
| iono | Bagging (arbres complets) | 8,77 % | 8,11 % | 98,96 % |
| iono | Stacking | 8,30 % | 8,30 % | 100,00 % |
| pima | AdaBoost, stumps | 23,03 % | 23,03 % | 100,00 % |
| pima | AdaBoost, profondeur 3 | 24,89 % | 24,94 % | 96,93 % |
| pima | Bagging (arbres complets) | 23,33 % | 23,42 % | 96,28 % |
| pima | Stacking | 22,60 % | 22,60 % | 100,00 % |
| wdbc | AdaBoost, stumps | 3,92 % | 3,92 % | 100,00 % |
| wdbc | AdaBoost, profondeur 3 | 3,51 % | 3,63 % | 99,06 % |
| wdbc | Bagging (arbres complets) | 5,38 % | 5,38 % | 99,06 % |
| wdbc | Stacking | 3,39 % | 3,39 % | 100,00 % |
| yeast3 | AdaBoost, stumps | 4,66 % | 4,66 % | 100,00 % |
| yeast3 | AdaBoost, profondeur 3 | 5,31 % | 5,31 % | 100,00 % |
| yeast3 | Bagging (arbres complets) | 5,02 % | 5,07 % | 99,55 % |
| yeast3 | Stacking | 4,75 % | 4,75 % | 100,00 % |
