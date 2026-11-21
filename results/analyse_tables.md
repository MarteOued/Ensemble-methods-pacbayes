# Analyse de la partie 1
Valeur par jeu et par méthode = moyenne sur 10 découpages. 25 jeux (15 équilibrés, 10 déséquilibrés). Rangs : 1 = meilleur. Tests au risque de 5 %. Les 10 méthodes principales sont comparées ; l'arbre de profondeur 5 et Naive Bayes ne servent qu'à la section 6.

## 1. Résultats par jeu (métrique mise en avant, en %)
Accuracy pour les jeux équilibrés, balanced accuracy pour les jeux déséquilibrés. En gras : la meilleure méthode du jeu. Résultats complets (4 métriques, écarts-types) dans `analyse_annexe.md`.

### Jeux équilibrés : accuracy
| Jeu | m | Arbre seul | Régression logistique | kNN (k=7) | Bagging (le mien) | Forêt aléatoire | AdaBoost stumps (le mien) | AdaBoost prof. 3 (le mien) | Gradient boosting | Hist. gradient boosting | Stacking (le mien) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| splice | 3175 | 93,0 | 84,6 | 76,3 | 96,1 | **97,1** | 94,1 | 96,8 | 97,1 | 96,9 | 94,8 |
| sonar | 208 | 73,3 | 74,9 | 77,8 | 78,4 | 82,5 | 83,0 | **85,4** | 81,9 | 84,6 | 78,7 |
| balance | 625 | 85,2 | 95,2 | 90,4 | 86,6 | 87,9 | **98,2** | 92,8 | 89,6 | 88,2 | 94,9 |
| australian | 690 | 81,2 | 86,3 | 84,9 | 86,6 | **87,6** | 84,9 | 84,9 | 86,7 | 85,8 | 87,4 |
| heart | 270 | 71,2 | 83,3 | 81,9 | 78,9 | 82,6 | 79,3 | 77,2 | 79,1 | 79,3 | **83,7** |
| bupa | 345 | 63,3 | 69,7 | 61,7 | 72,0 | **73,8** | 72,4 | 69,5 | 71,4 | 68,8 | 69,2 |
| spambase | 4597 | 90,9 | 92,4 | 90,3 | 94,2 | 95,1 | 93,9 | 94,8 | 94,8 | **95,6** | 93,5 |
| wdbc | 569 | 92,4 | **97,3** | 95,3 | 94,6 | 95,5 | 96,1 | 96,5 | 95,1 | 96,2 | 96,6 |
| iono | 351 | 85,8 | 86,4 | 83,5 | 91,2 | **93,4** | 91,4 | 92,7 | 92,8 | 92,6 | 91,7 |
| pima | 768 | 70,8 | 77,0 | 73,3 | 76,7 | 76,5 | 77,0 | 75,1 | 76,1 | 74,6 | **77,4** |
| wine | 178 | 92,2 | **99,6** | 98,5 | 97,0 | 98,7 | 98,0 | 94,6 | 95,0 | 98,1 | 98,9 |
| glass | 214 | 78,5 | 75,2 | 76,5 | 84,0 | 86,0 | 85,1 | **87,4** | 85,2 | 81,8 | 78,9 |
| newthyroid | 215 | 92,2 | 88,6 | 92,0 | 95,1 | **95,8** | 95,2 | 94,8 | 94,6 | 94,9 | 94,0 |
| german | 1000 | 66,7 | **77,2** | 71,4 | 76,2 | 76,1 | 75,8 | 74,2 | 76,2 | 75,8 | 76,1 |
| vehicle | 846 | 93,0 | 96,6 | 93,0 | 95,2 | 96,3 | 96,3 | **97,8** | 96,1 | 96,4 | 96,6 |
| **rang moyen** |  | 9,33 | 5,07 | 8,07 | 5,77 | 3,20 | 4,73 | 4,90 | 4,83 | 4,93 | 4,17 |

### Jeux déséquilibrés : balanced accuracy
| Jeu | m | Arbre seul | Régression logistique | kNN (k=7) | Bagging (le mien) | Forêt aléatoire | AdaBoost stumps (le mien) | AdaBoost prof. 3 (le mien) | Gradient boosting | Hist. gradient boosting | Stacking (le mien) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| abalone17 | 4177 | **53,4** | 50,0 | 50,0 | 50,0 | 50,0 | 49,9 | 50,5 | 49,8 | 51,1 | 50,0 |
| yeast6 | 1484 | 72,0 | 64,4 | **75,7** | 73,4 | 66,7 | 75,6 | 67,8 | 66,9 | 73,4 | 64,0 |
| wine4 | 1599 | **55,6** | 49,9 | 49,9 | 50,5 | 50,0 | 51,1 | 50,8 | 51,3 | 50,1 | 50,0 |
| libras | 360 | 73,8 | **87,9** | 71,3 | 68,4 | 68,6 | 82,9 | 79,2 | 76,9 | 85,6 | 84,3 |
| satimage | 6435 | 74,5 | 50,7 | **80,8** | 74,7 | 74,5 | 74,3 | 79,2 | 76,6 | 79,8 | 77,8 |
| pageblocks | 5473 | 89,7 | 81,4 | 87,6 | 92,4 | 92,5 | 88,4 | 92,5 | 92,1 | **92,8** | 89,0 |
| yeast3 | 1484 | 83,2 | 83,0 | 84,4 | 87,0 | 84,2 | **88,8** | 85,7 | 86,3 | 85,4 | 84,3 |
| bankmarketing | 45211 | 70,3 | 66,0 | 63,4 | 72,2 | 68,7 | 66,4 | 70,5 | 70,4 | **73,0** | 67,1 |
| abalone8 | 4177 | **56,4** | 50,1 | 54,2 | 53,6 | 53,6 | 50,0 | 52,7 | 53,9 | 55,5 | 51,1 |
| segmentation | 2310 | 92,4 | 76,9 | 89,3 | 93,8 | 94,9 | 91,5 | **96,9** | 95,6 | 96,4 | 91,2 |
| **rang moyen** |  | 4,90 | 8,60 | 6,10 | 5,00 | 6,20 | 6,10 | 3,90 | 4,90 | 2,70 | 6,60 |

## 2. Rangs moyens, test de Friedman et différence critique de Nemenyi
En gras : le meilleur rang moyen. Deux méthodes dont les rangs moyens diffèrent de moins que CD ne sont pas significativement différentes. Les diagrammes de différences critiques sont dans `figures/cd/`.

### Tous les jeux (N = 25, CD = 2,71)
| Méthode | Accuracy | Balanced accuracy | F1 (minorité) | AUC |
|---|---|---|---|---|
| Arbre seul | 9,36 | 7,44 | 7,64 | 9,88 |
| Régression logistique | 5,64 | 6,72 | 6,62 | 5,32 |
| kNN (k=7) | 6,90 | 7,32 | 7,26 | 7,80 |
| Bagging (le mien) | 5,50 | 5,24 | 5,40 | 5,72 |
| Forêt aléatoire | **3,48** | 4,52 | 4,50 | **3,08** |
| AdaBoost stumps (le mien) | 5,04 | 5,12 | 5,12 | 5,12 |
| AdaBoost prof. 3 (le mien) | 4,98 | 4,36 | 4,56 | 4,36 |
| Gradient boosting | 5,38 | 4,76 | 4,64 | 5,00 |
| Hist. gradient boosting | 4,70 | **3,92** | **3,76** | 4,22 |
| Stacking (le mien) | 4,02 | 5,60 | 5,50 | 4,50 |
| *p de Friedman* | < 0,001 | < 0,001 | < 0,001 | < 0,001 |

### Jeux équilibrés (N = 15, CD = 3,50)
| Méthode | Accuracy | Balanced accuracy | F1 (minorité) | AUC |
|---|---|---|---|---|
| Arbre seul | 9,33 | 9,13 | 9,07 | 9,80 |
| Régression logistique | 5,07 | 5,47 | 5,53 | 5,27 |
| kNN (k=7) | 8,07 | 8,13 | 8,20 | 7,60 |
| Bagging (le mien) | 5,77 | 5,40 | 5,67 | 5,80 |
| Forêt aléatoire | **3,20** | **3,40** | **3,47** | **3,27** |
| AdaBoost stumps (le mien) | 4,73 | 4,47 | 4,53 | 4,80 |
| AdaBoost prof. 3 (le mien) | 4,90 | 4,67 | 4,67 | 4,67 |
| Gradient boosting | 4,83 | 4,67 | 4,53 | 5,07 |
| Hist. gradient boosting | 4,93 | 4,73 | 4,47 | 4,23 |
| Stacking (le mien) | 4,17 | 4,93 | 4,87 | 4,50 |
| *p de Friedman* | < 0,001 | < 0,001 | < 0,001 | < 0,001 |

### Jeux déséquilibrés (N = 10, CD = 4,28)
| Méthode | Accuracy | Balanced accuracy | F1 (minorité) | AUC |
|---|---|---|---|---|
| Arbre seul | 9,40 | 4,90 | 5,50 | 10,00 |
| Régression logistique | 6,50 | 8,60 | 8,25 | 5,40 |
| kNN (k=7) | 5,15 | 6,10 | 5,85 | 8,10 |
| Bagging (le mien) | 5,10 | 5,00 | 5,00 | 5,60 |
| Forêt aléatoire | 3,90 | 6,20 | 6,05 | **2,80** |
| AdaBoost stumps (le mien) | 5,50 | 6,10 | 6,00 | 5,60 |
| AdaBoost prof. 3 (le mien) | 5,10 | 3,90 | 4,40 | 3,90 |
| Gradient boosting | 6,20 | 4,90 | 4,80 | 4,90 |
| Hist. gradient boosting | 4,35 | **2,70** | **2,70** | 4,20 |
| Stacking (le mien) | **3,80** | 6,60 | 6,45 | 4,50 |
| *p de Friedman* | 0,002 | 0,002 | 0,011 | < 0,001 |

## 3. Jeux déséquilibrés : le classement change avec la métrique
Rang moyen sur les jeux déséquilibrés pour chaque métrique, avec la position entre parenthèses. La dernière colonne donne le nombre de places gagnées (+) ou perdues (-) quand on passe de l'accuracy à la balanced accuracy. Figure : `figures/rank_by_metric_imbalanced.png`.

| Méthode | Accuracy | Balanced accuracy | F1 (minorité) | AUC | Places gagnées (acc. → bal. acc.) |
|---|---|---|---|---|---|
| Arbre seul | 9,40 (10) | 4,90 (3) | 5,50 (5) | 10,00 (10) | +7 |
| Régression logistique | 6,50 (9) | 8,60 (10) | 8,25 (10) | 5,40 (6) | -1 |
| kNN (k=7) | 5,15 (6) | 6,10 (6) | 5,85 (6) | 8,10 (9) | +0 |
| Bagging (le mien) | 5,10 (4) | 5,00 (5) | 5,00 (4) | 5,60 (7) | -1 |
| Forêt aléatoire | 3,90 (2) | 6,20 (8) | 6,05 (8) | 2,80 (1) | -6 |
| AdaBoost stumps (le mien) | 5,50 (7) | 6,10 (6) | 6,00 (7) | 5,60 (7) | +1 |
| AdaBoost prof. 3 (le mien) | 5,10 (4) | 3,90 (2) | 4,40 (2) | 3,90 (2) | +2 |
| Gradient boosting | 6,20 (8) | 4,90 (3) | 4,80 (3) | 4,90 (5) | +5 |
| Hist. gradient boosting | 4,35 (3) | 2,70 (1) | 2,70 (1) | 4,20 (3) | +2 |
| Stacking (le mien) | 3,80 (1) | 6,60 (9) | 6,45 (9) | 4,50 (4) | -8 |

Corrélation de Kendall entre les classements de méthodes selon chaque métrique (1 = même classement) :

|  | Accuracy | Balanced accuracy | F1 (minorité) | AUC |
|---|---|---|---|---|
| Accuracy | 1,00 | -0,05 | 0,04 | 0,48 |
| Balanced accuracy | -0,05 | 1,00 | 0,93 | 0,05 |
| F1 (minorité) | 0,04 | 0,93 | 1,00 | 0,09 |
| AUC | 0,48 | 0,05 | 0,09 | 1,00 |

### L'accuracy est trompeuse : comparaison avec la réponse « toujours la classe majoritaire »
Accuracy d'un classifieur qui répond toujours la classe majoritaire, contre la meilleure accuracy et la meilleure balanced accuracy obtenues par les 10 méthodes.

| Jeu | Minorité (%) | Acc. toujours-majoritaire (%) | Meilleure accuracy (%) | Meilleure bal. acc. (%) |
|---|---|---|---|---|
| abalone17 | 1,4 | 98,6 | 98,6 (kNN (k=7)) | 53,4 (Arbre seul) |
| yeast6 | 2,4 | 97,6 | 98,4 (kNN (k=7)) | 75,7 (kNN (k=7)) |
| wine4 | 3,3 | 96,7 | 96,7 (Stacking (le mien)) | 55,6 (Arbre seul) |
| libras | 6,7 | 93,3 | 98,4 (Régression logistique) | 87,9 (Régression logistique) |
| satimage | 9,7 | 90,3 | 94,8 (Hist. gradient boosting) | 80,8 (kNN (k=7)) |
| pageblocks | 10,2 | 89,8 | 97,6 (Hist. gradient boosting) | 92,8 (Hist. gradient boosting) |
| yeast3 | 11,0 | 89,0 | 95,3 (kNN (k=7)) | 88,8 (AdaBoost stumps (le mien)) |
| bankmarketing | 11,7 | 88,3 | 90,8 (Gradient boosting) | 73,0 (Hist. gradient boosting) |
| abalone8 | 13,6 | 86,4 | 86,4 (AdaBoost stumps (le mien)) | 56,4 (Arbre seul) |
| segmentation | 14,3 | 85,7 | 98,7 (AdaBoost prof. 3 (le mien)) | 96,9 (AdaBoost prof. 3 (le mien)) |

## 4. Gain des ensembles par rapport à un arbre seul et à une autre famille de modèles
Pour chaque ensemble : différence moyenne de la métrique en points (ensemble moins référence), victoires/égalités/défaites sur les jeux (G/N/P), et p de Wilcoxon par paires corrigé par Holm. Trois situations : accuracy sur les 25 jeux, balanced accuracy sur les 10 jeux déséquilibrés, AUC sur les 25 jeux.

### Référence : arbre seul
| Méthode | Δ acc. | G/N/P acc. | p acc. | Δ bal. acc. | G/N/P bal. acc. | p bal. acc. | Δ AUC | G/N/P AUC | p AUC |
|---|---|---|---|---|---|---|---|---|---|
| Bagging (le mien) | +3,85 | 25/0/0 | < 0,001 | -0,52 | 6/0/4 | 1,000 | +13,95 | 25/0/0 | < 0,001 |
| Forêt aléatoire | +4,82 | 25/0/0 | < 0,001 | -1,75 | 4/0/6 | 1,000 | +15,13 | 25/0/0 | < 0,001 |
| AdaBoost stumps (le mien) | +4,50 | 23/0/2 | < 0,001 | -0,23 | 3/0/7 | 1,000 | +13,92 | 25/0/0 | < 0,001 |
| AdaBoost prof. 3 (le mien) | +4,42 | 25/0/0 | < 0,001 | +0,46 | 6/0/4 | 1,000 | +13,89 | 25/0/0 | < 0,001 |
| Gradient boosting | +4,25 | 25/0/0 | < 0,001 | -0,15 | 6/0/4 | 1,000 | +13,79 | 25/0/0 | < 0,001 |
| Hist. gradient boosting | +4,26 | 25/0/0 | < 0,001 | +2,18 | 7/0/3 | 1,000 | +13,85 | 25/0/0 | < 0,001 |
| Stacking (le mien) | +4,32 | 25/0/0 | < 0,001 | -1,24 | 3/0/7 | 1,000 | +13,93 | 25/0/0 | < 0,001 |

### Référence : régression logistique
| Méthode | Δ acc. | G/N/P acc. | p acc. | Δ bal. acc. | G/N/P bal. acc. | p bal. acc. | Δ AUC | G/N/P AUC | p AUC |
|---|---|---|---|---|---|---|---|---|---|
| Bagging (le mien) | +1,13 | 15/0/10 | 1,000 | +5,58 | 9/0/1 | 0,258 | +2,07 | 12/0/13 | 0,763 |
| Forêt aléatoire | +2,09 | 16/0/9 | 0,426 | +4,35 | 9/0/1 | 0,258 | +3,24 | 14/0/11 | 0,220 |
| AdaBoost stumps (le mien) | +1,78 | 14/1/10 | 0,356 | +5,88 | 7/0/3 | 0,244 | +2,04 | 14/0/11 | 0,537 |
| AdaBoost prof. 3 (le mien) | +1,70 | 11/1/13 | 1,000 | +6,57 | 9/0/1 | 0,223 | +2,01 | 16/0/9 | 0,763 |
| Gradient boosting | +1,52 | 13/0/12 | 1,000 | +5,95 | 8/0/2 | 0,258 | +1,91 | 12/0/13 | 0,763 |
| Hist. gradient boosting | +1,54 | 12/0/13 | 1,000 | +8,28 | 9/0/1 | 0,068 | +1,97 | 13/0/12 | 0,763 |
| Stacking (le mien) | +1,60 | 18/1/6 | 0,106 | +4,86 | 8/0/2 | 0,258 | +2,05 | 18/0/7 | 0,062 |

### Référence : kNN
| Méthode | Δ acc. | G/N/P acc. | p acc. | Δ bal. acc. | G/N/P bal. acc. | p bal. acc. | Δ AUC | G/N/P AUC | p AUC |
|---|---|---|---|---|---|---|---|---|---|
| Bagging (le mien) | +2,36 | 15/0/10 | 0,024 | +0,95 | 5/0/5 | 1,000 | +4,06 | 20/0/5 | < 0,001 |
| Forêt aléatoire | +3,33 | 20/1/4 | 0,003 | -0,27 | 4/1/5 | 1,000 | +5,23 | 22/0/3 | < 0,001 |
| AdaBoost stumps (le mien) | +3,01 | 15/1/9 | 0,015 | +1,25 | 6/0/4 | 1,000 | +4,03 | 21/0/4 | < 0,001 |
| AdaBoost prof. 3 (le mien) | +2,93 | 18/0/7 | 0,010 | +1,94 | 7/0/3 | 1,000 | +4,00 | 23/0/2 | < 0,001 |
| Gradient boosting | +2,76 | 16/0/9 | 0,015 | +1,33 | 6/0/4 | 1,000 | +3,89 | 21/0/4 | < 0,001 |
| Hist. gradient boosting | +2,77 | 17/0/8 | 0,008 | +3,65 | 8/0/2 | 0,342 | +3,95 | 21/0/4 | < 0,001 |
| Stacking (le mien) | +2,83 | 22/1/2 | < 0,001 | +0,24 | 5/1/4 | 1,000 | +4,03 | 23/0/2 | < 0,001 |

## 5. Bagging ou boosting ?
### 5.1 Meilleur du type bagging (bagging, forêt aléatoire) contre meilleur du type boosting (AdaBoost stumps, AdaBoost prof. 3, gradient boosting, Hist. gradient boosting), par jeu, sur la métrique mise en avant. Attention : le boosting a 4 candidats contre 2, ce qui le favorise légèrement. Jeux triés par taille.

| Jeu | m | d | Minorité (%) | Meilleur bagging (%) | Meilleur boosting (%) | Boosting - bagging (pts) | Vainqueur |
|---|---|---|---|---|---|---|---|
| wine | 178 | 13 | 33,1 | 98,7 (Forêt aléatoire) | 98,1 (Hist. gradient boosting) | -0,56 | bagging |
| sonar | 208 | 60 | 46,6 | 82,5 (Forêt aléatoire) | 85,4 (AdaBoost prof. 3 (le mien)) | +2,86 | boosting |
| glass | 214 | 9 | 32,7 | 86,0 (Forêt aléatoire) | 87,4 (AdaBoost prof. 3 (le mien)) | +1,38 | boosting |
| newthyroid | 215 | 5 | 30,2 | 95,8 (Forêt aléatoire) | 95,2 (AdaBoost stumps (le mien)) | -0,62 | bagging |
| heart | 270 | 13 | 44,4 | 82,6 (Forêt aléatoire) | 79,3 (AdaBoost stumps (le mien)) | -3,33 | bagging |
| bupa | 345 | 6 | 42,0 | 73,8 (Forêt aléatoire) | 72,4 (AdaBoost stumps (le mien)) | -1,35 | bagging |
| iono | 351 | 34 | 35,9 | 93,4 (Forêt aléatoire) | 92,8 (Gradient boosting) | -0,57 | bagging |
| libras | 360 | 90 | 6,7 | 68,6 (Forêt aléatoire) | 85,6 (Hist. gradient boosting) | +17,04 | boosting |
| wdbc | 569 | 30 | 37,3 | 95,5 (Forêt aléatoire) | 96,5 (AdaBoost prof. 3 (le mien)) | +0,99 | boosting |
| balance | 625 | 4 | 46,1 | 87,9 (Forêt aléatoire) | 98,2 (AdaBoost stumps (le mien)) | +10,32 | boosting |
| australian | 690 | 14 | 44,5 | 87,6 (Forêt aléatoire) | 86,7 (Gradient boosting) | -0,87 | bagging |
| pima | 768 | 8 | 34,9 | 76,7 (Bagging (le mien)) | 77,0 (AdaBoost stumps (le mien)) | +0,30 | boosting |
| vehicle | 846 | 18 | 23,5 | 96,3 (Forêt aléatoire) | 97,8 (AdaBoost prof. 3 (le mien)) | +1,46 | boosting |
| german | 1000 | 24 | 30,0 | 76,2 (Bagging (le mien)) | 76,2 (Gradient boosting) | +0,00 | égalité |
| yeast6 | 1484 | 8 | 2,4 | 73,4 (Bagging (le mien)) | 75,6 (AdaBoost stumps (le mien)) | +2,18 | boosting |
| yeast3 | 1484 | 8 | 11,0 | 87,0 (Bagging (le mien)) | 88,8 (AdaBoost stumps (le mien)) | +1,81 | boosting |
| wine4 | 1599 | 11 | 3,3 | 50,5 (Bagging (le mien)) | 51,3 (Gradient boosting) | +0,77 | boosting |
| segmentation | 2310 | 19 | 14,3 | 94,9 (Forêt aléatoire) | 96,9 (AdaBoost prof. 3 (le mien)) | +2,00 | boosting |
| splice | 3175 | 60 | 48,1 | 97,1 (Forêt aléatoire) | 97,1 (Gradient boosting) | -0,03 | bagging |
| abalone17 | 4177 | 10 | 1,4 | 50,0 (Forêt aléatoire) | 51,1 (Hist. gradient boosting) | +1,05 | boosting |
| abalone8 | 4177 | 10 | 13,6 | 53,6 (Forêt aléatoire) | 55,5 (Hist. gradient boosting) | +1,86 | boosting |
| spambase | 4597 | 57 | 39,4 | 95,1 (Forêt aléatoire) | 95,6 (Hist. gradient boosting) | +0,46 | boosting |
| pageblocks | 5473 | 10 | 10,2 | 92,5 (Forêt aléatoire) | 92,8 (Hist. gradient boosting) | +0,24 | boosting |
| satimage | 6435 | 36 | 9,7 | 74,7 (Bagging (le mien)) | 79,8 (Hist. gradient boosting) | +5,06 | boosting |
| bankmarketing | 45211 | 51 | 11,7 | 72,2 (Bagging (le mien)) | 73,0 (Hist. gradient boosting) | +0,73 | boosting |

### 5.2 Qui l'emporte selon la taille et le déséquilibre
| Groupe | Nb jeux | Boosting gagne | Bagging gagne | Égalité |
|---|---|---|---|---|
| m < 1000 | 13 | 7 | 6 | 0 |
| 1000 ≤ m < 5000 | 9 | 7 | 1 | 1 |
| m ≥ 5000 | 3 | 3 | 0 | 0 |
| jeux équilibrés | 15 | 7 | 7 | 1 |
| jeux déséquilibrés | 10 | 10 | 0 | 0 |

### 5.3 Comparaisons directes
Différence (première méthode moins seconde), G/N/P = victoires/égalités/défaites de la première, p de Wilcoxon corrigé par Holm. Accuracy sur 25 jeux ; balanced accuracy sur 10 jeux déséquilibrés.

| Comparaison | Δ acc. | G/N/P acc. | p acc. | Δ bal. acc. | G/N/P bal. acc. | p bal. acc. |
|---|---|---|---|---|---|---|
| Bagging (le mien) - AdaBoost stumps (le mien) | -0,65 | 10/0/15 | 0,937 | -0,29 | 6/0/4 | 1,000 |
| Bagging (le mien) - AdaBoost prof. 3 (le mien) | -0,57 | 12/0/13 | 0,937 | -0,99 | 4/0/6 | 1,000 |
| Forêt aléatoire - Gradient boosting | +0,57 | 20/0/5 | 0,014 | -1,60 | 2/0/8 | 0,082 |
| Forêt aléatoire - Hist. gradient boosting | +0,55 | 14/0/11 | 0,921 | -3,93 | 0/0/10 | 0,010 |
| Forêt aléatoire - AdaBoost prof. 3 (le mien) | +0,39 | 16/0/9 | 0,937 | -2,22 | 2/0/8 | 0,078 |

## 6. Le stacking bat-il le meilleur de ses membres ? À quel coût ?
### 6.1 Résultats par jeu (métrique mise en avant, en %)

| Jeu | Arbre prof. 5 (membre) | kNN (k=7) | Régression logistique | Naive Bayes (membre) | Stacking (le mien) | Meilleur membre | Stacking - meilleur membre (pts) |
|---|---|---|---|---|---|---|---|
| splice | 94,1 | 76,3 | 84,6 | 87,3 | **94,8** | Arbre prof. 5 (membre) | +0,71 |
| sonar | 72,4 | 77,8 | 74,9 | 66,5 | **78,7** | kNN (k=7) | +0,95 |
| balance | 82,1 | 90,4 | **95,2** | 95,1 | 94,9 | Régression logistique | -0,32 |
| australian | 84,4 | 84,9 | 86,3 | 81,1 | **87,4** | Régression logistique | +1,11 |
| heart | 74,0 | 81,9 | 83,3 | **83,8** | 83,7 | Naive Bayes (membre) | -0,12 |
| bupa | 65,4 | 61,7 | **69,7** | 59,0 | 69,2 | Régression logistique | -0,48 |
| spambase | 90,5 | 90,3 | 92,4 | 81,9 | **93,5** | Régression logistique | +1,07 |
| wdbc | 92,2 | 95,3 | **97,3** | 93,0 | 96,6 | Régression logistique | -0,64 |
| iono | 87,0 | 83,5 | 86,4 | 88,4 | **91,7** | Naive Bayes (membre) | +3,30 |
| pima | 73,5 | 73,3 | 77,0 | 75,7 | **77,4** | Régression logistique | +0,43 |
| wine | 92,2 | 98,5 | **99,6** | 98,5 | 98,9 | Régression logistique | -0,74 |
| glass | 77,7 | 76,5 | 75,2 | 61,2 | **78,9** | Arbre prof. 5 (membre) | +1,23 |
| newthyroid | 92,6 | 92,0 | 88,6 | 92,3 | **94,0** | Arbre prof. 5 (membre) | +1,38 |
| german | 70,2 | 71,4 | **77,2** | 71,2 | 76,1 | Régression logistique | -1,07 |
| vehicle | 91,9 | 93,0 | **96,6** | 65,7 | **96,6** | Régression logistique | +0,00 |
| abalone17 | 50,5 | 50,0 | 50,0 | **59,9** | 50,0 | Naive Bayes (membre) | -9,87 |
| yeast6 | 72,1 | **75,7** | 64,4 | 64,3 | 64,0 | kNN (k=7) | -11,68 |
| wine4 | 51,1 | 49,9 | 49,9 | **56,7** | 50,0 | Naive Bayes (membre) | -6,72 |
| libras | 73,4 | 71,3 | **87,9** | 85,9 | 84,3 | Régression logistique | -3,57 |
| satimage | 74,4 | 80,8 | 50,7 | **84,2** | 77,8 | Naive Bayes (membre) | -6,45 |
| pageblocks | 88,9 | 87,6 | 81,4 | 69,0 | **89,0** | Arbre prof. 5 (membre) | +0,15 |
| yeast3 | **88,2** | 84,4 | 83,0 | 58,1 | 84,3 | Arbre prof. 5 (membre) | -3,93 |
| bankmarketing | 65,5 | 63,4 | 66,0 | **71,7** | 67,1 | Naive Bayes (membre) | -4,54 |
| abalone8 | 50,2 | 54,2 | 50,1 | **65,7** | 51,1 | Naive Bayes (membre) | -14,54 |
| segmentation | 90,0 | 89,3 | 76,9 | 80,0 | **91,2** | Arbre prof. 5 (membre) | +1,18 |

### 6.2 Stacking contre ses membres (25 jeux)
| Référence | Stacking - référence (pts) | G/N/P du stacking | p de Wilcoxon |
|---|---|---|---|
| meilleur membre de chaque jeu (choisi après coup) | -2,13 | 10/1/14 | 0,170 |
| Régression logistique, membre au meilleur rang moyen | +3,07 | 17/1/7 | 0,010 |

### 6.3 Coût du stacking
Le stacking entraîne chaque membre 6 fois (5 plis + un réentraînement final) et doit faire prédire chaque pli pour obtenir les prédictions out-of-fold. Un rapport de 6 par rapport à la somme des entraînements n'est donc qu'un minimum pour des modèles dont le coût est à l'entraînement : un modèle « paresseux » comme le kNN ne coûte presque rien à entraîner, son coût est dans la prédiction (voir 6.4). C'est pourquoi je donne aussi le rapport à la somme entraînement + prédiction.

| Mesure | Valeur |
|---|---|
| Somme des temps d'entraînement des 4 membres (s), médiane des jeux | 0,0050 |
| Temps d'entraînement du stacking (s), médiane des jeux | 0,0345 |
| Rapport stacking / somme des entraînements des membres, médiane des jeux | 8,0 |
| Rapport stacking / somme des entraînements des membres, bankmarketing | 40,9 |
| Rapport stacking / somme (entraînement + prédiction) des membres, médiane des jeux | 3,7 |
| Rapport stacking / somme (entraînement + prédiction) des membres, bankmarketing | 2,0 |

### 6.4 Où va le temps du stacking (bankmarketing, un découpage, un seul cœur)
Temps passé, pour chaque membre, à s'entraîner sur les 5 plis et à prédire les plis écartés (les prédictions out-of-fold). Cela exclut le réentraînement final et le méta-modèle.

| Membre | Entraînement des 5 plis (s) | Prédictions out-of-fold (s) | Total (s) |
|---|---|---|---|
| Arbre prof. 5 (membre) | 0,192 | 0,004 | 0,196 |
| kNN (k=7) | 0,028 | 2,296 | 2,323 |
| Régression logistique | 0,049 | 0,006 | 0,054 |
| Naive Bayes (membre) | 0,027 | 0,006 | 0,034 |
| **Total des 4 membres** | 0,296 | 2,312 | 2,607 |

## 7. Coût de calcul (un seul cœur)
Médiane sur les 25 jeux du temps moyen par découpage, et temps sur le plus gros jeu (bankmarketing, 31 647 exemples d'entraînement).

| Méthode | Entraînement, médiane (s) | Rapport à l'arbre seul | Entraînement, bankmarketing (s) | Prédiction, médiane (s) |
|---|---|---|---|---|
| Arbre seul | 0,002 | 1,0 | 0,17 | 0,0001 |
| Régression logistique | 0,002 | 0,7 | 0,01 | 0,0002 |
| kNN (k=7) | 0,001 | 0,2 | 0,01 | 0,0035 |
| Bagging (le mien) | 0,407 | 168,4 | 31,57 | 0,0160 |
| Forêt aléatoire | 0,155 | 64,1 | 3,37 | 0,0074 |
| AdaBoost stumps (le mien) | 0,178 | 73,6 | 2,39 | 0,0143 |
| AdaBoost prof. 3 (le mien) | 0,303 | 125,3 | 6,15 | 0,0153 |
| Gradient boosting | 0,284 | 117,4 | 5,77 | 0,0007 |
| Hist. gradient boosting | 0,107 | 44,3 | 1,76 | 0,0034 |
| Stacking (le mien) | 0,035 | 14,3 | 2,79 | 0,0017 |

