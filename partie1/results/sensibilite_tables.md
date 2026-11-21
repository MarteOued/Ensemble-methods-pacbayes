# Étude de sensibilité : pourquoi le bagging ou le boosting l'emporte-t-il ?
6 jeux réels équilibrés (australian, german, iono, pima, vehicle, wdbc), 10 découpages 70/30 chacun ; erreur de test mesurée sur des étiquettes propres, en %. Les étiquettes d'ENTRAÎNEMENT sont inversées avec la probabilité indiquée.

## Tableau S1 : effet du bruit sur les étiquettes (erreur de test moyenne, %)
| Méthode | bruit 0 % | bruit 5 % | bruit 10 % | bruit 20 % | Hausse 0 → 20 % (pts) |
|---|---|---|---|---|---|
| Arbre seul | 18,33 | 21,27 | 25,25 | 31,13 | +12,80 |
| Bagging (le mien) | 13,26 | 13,90 | 14,68 | 17,43 | +4,17 |
| Forêt aléatoire | 12,44 | 12,80 | 13,48 | 15,35 | +2,91 |
| AdaBoost stumps (le mien) | 13,10 | 14,67 | 15,66 | 18,36 | +5,27 |
| AdaBoost prof. 3 (le mien) | 13,13 | 15,01 | 17,12 | 23,01 | +9,88 |
| Hist. gradient boosting | 13,09 | 14,57 | 16,46 | 21,67 | +8,58 |

### Hausse de l'erreur entre 0 % et 20 % de bruit, par jeu (points)
| Jeu | Arbre seul | Bagging (le mien) | Forêt aléatoire | AdaBoost stumps (le mien) | AdaBoost prof. 3 (le mien) | Hist. gradient boosting |
|---|---|---|---|---|---|---|
| australian | +12,0 | +3,1 | +3,0 | +1,9 | +9,3 | +9,6 |
| german | +7,3 | +4,7 | +1,6 | +2,7 | +7,4 | +7,2 |
| iono | +11,8 | +4,8 | +4,9 | +10,1 | +13,2 | +12,0 |
| pima | +9,5 | +3,4 | +1,2 | +3,8 | +7,2 | +5,8 |
| vehicle | +18,3 | +4,0 | +3,7 | +5,4 | +11,7 | +9,1 |
| wdbc | +18,0 | +5,0 | +3,0 | +7,7 | +10,4 | +7,8 |
| **moyenne** | +12,8 | +4,2 | +2,9 | +5,3 | +9,9 | +8,6 |

## Tableau S2 : profondeur des arbres de base (erreur de test moyenne, %)
200 arbres ou tours dans tous les cas. « complet » = profondeur illimitée (pas défini pour AdaBoost).

| Profondeur | AdaBoost, 0 % bruit | Bagging, 0 % bruit | AdaBoost, 20 % bruit | Bagging, 20 % bruit |
|---|---|---|---|---|
| 1 | 13,10 | 20,02 | 18,36 | 20,04 |
| 2 | 13,08 | 16,12 | 21,07 | 16,87 |
| 3 | 13,13 | 14,72 | 23,01 | 15,49 |
| 5 | 13,66 | 13,39 | 20,98 | 15,12 |
| complet | – | 13,26 | – | 17,43 |

## Tableau S3 : erreur de test selon le nombre d'arbres (bagging) ou de tours (AdaBoost), %
| Bruit | Méthode | 1 | 2 | 5 | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|---|---|
| 0 % | Bagging (le mien) | 19,36 | 18,00 | 15,34 | 14,15 | 13,76 | 13,46 | 13,19 | 13,26 |
| 0 % | AdaBoost stumps (le mien) | 20,67 | 22,07 | 16,86 | 15,20 | 14,24 | 13,31 | 13,01 | 13,10 |
| 0 % | AdaBoost prof. 3 (le mien) | 16,36 | 17,74 | 14,87 | 14,39 | 13,48 | 13,02 | 12,98 | 13,13 |
| 20 % | Bagging (le mien) | 32,03 | 27,17 | 24,61 | 20,94 | 19,49 | 18,15 | 17,56 | 17,43 |
| 20 % | AdaBoost stumps (le mien) | 21,13 | 21,15 | 18,89 | 17,79 | 17,81 | 17,45 | 17,72 | 18,36 |
| 20 % | AdaBoost prof. 3 (le mien) | 19,73 | 19,73 | 21,46 | 21,10 | 21,46 | 21,93 | 22,71 | 23,01 |

## Tableau S4 : biais et variance sur jeux simulés (%)
Pour chaque ligne : 30 modèles entraînés sur 30 échantillons indépendants de 500 exemples, évalués sur 5000 points communs. Erreur = biais + variance nette. Le bruit est ajouté aux étiquettes d'entraînement seulement ; le test est propre.

| Jeu simulé | Bruit | Méthode | Erreur | Biais | Variance nette |
|---|---|---|---|---|---|
| additive | 0 % | Arbre seul | 12,91 | 7,22 | 5,69 |
| additive | 0 % | Bagging (le mien) | 9,76 | 8,20 | 1,56 |
| additive | 0 % | Forêt aléatoire | 9,06 | 7,78 | 1,28 |
| additive | 0 % | AdaBoost stumps (le mien) | 5,77 | 2,94 | 2,83 |
| additive | 0 % | AdaBoost prof. 3 (le mien) | 7,67 | 4,56 | 3,11 |
| additive | 0 % | Hist. gradient boosting | 7,64 | 4,74 | 2,90 |
| additive | 20 % | Arbre seul | 29,19 | 9,12 | 20,07 |
| additive | 20 % | Bagging (le mien) | 15,59 | 8,58 | 7,01 |
| additive | 20 % | Forêt aléatoire | 14,30 | 8,20 | 6,10 |
| additive | 20 % | AdaBoost stumps (le mien) | 17,99 | 7,12 | 10,87 |
| additive | 20 % | AdaBoost prof. 3 (le mien) | 23,43 | 7,88 | 15,55 |
| additive | 20 % | Hist. gradient boosting | 19,99 | 7,08 | 12,91 |
| checkerboard | 0 % | Arbre seul | 4,86 | 0,64 | 4,22 |
| checkerboard | 0 % | Bagging (le mien) | 3,20 | 0,56 | 2,64 |
| checkerboard | 0 % | Forêt aléatoire | 5,73 | 0,92 | 4,81 |
| checkerboard | 0 % | AdaBoost stumps (le mien) | 49,78 | 48,02 | 1,76 |
| checkerboard | 0 % | AdaBoost prof. 3 (le mien) | 4,05 | 0,78 | 3,27 |
| checkerboard | 0 % | Hist. gradient boosting | 3,71 | 1,48 | 2,23 |
| checkerboard | 20 % | Arbre seul | 26,67 | 2,80 | 23,87 |
| checkerboard | 20 % | Bagging (le mien) | 18,69 | 2,20 | 16,49 |
| checkerboard | 20 % | Forêt aléatoire | 20,97 | 2,78 | 18,19 |
| checkerboard | 20 % | AdaBoost stumps (le mien) | 49,56 | 46,46 | 3,10 |
| checkerboard | 20 % | AdaBoost prof. 3 (le mien) | 30,56 | 10,88 | 19,68 |
| checkerboard | 20 % | Hist. gradient boosting | 19,58 | 3,24 | 16,34 |
