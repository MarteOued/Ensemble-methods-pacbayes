# Jeux de données utilisés
25 jeux : 15 équilibrés et 10 déséquilibrés. Un jeu est dit **déséquilibré** si sa classe minoritaire représente moins de 15 % des exemples. m = nombre d'exemples, d = dimension (nombre de variables après encodage), « Minoritaires » = nombre d'exemples de la classe minoritaire, « Dans le test » = nombre attendu dans un test de 30 %.

Métrique mise en avant : accuracy (erreur) et AUC pour les jeux équilibrés ; balanced accuracy, F1 de la classe minoritaire et AUC pour les jeux déséquilibrés.

## Jeux déséquilibrés (10)
| Jeu | m | d | Minorité | Minoritaires | Dans le test | Groupe |
|---|---|---|---|---|---|---|
| abalone17 | 4177 | 10 | 1,4 % | 58 | 17 | déséquilibré |
| yeast6 | 1484 | 8 | 2,4 % | 35 | 10 | déséquilibré |
| wine4 | 1599 | 11 | 3,3 % | 53 | 16 | déséquilibré |
| libras | 360 | 90 | 6,7 % | 24 | 7 | déséquilibré |
| satimage | 6435 | 36 | 9,7 % | 626 | 188 | déséquilibré |
| pageblocks | 5473 | 10 | 10,2 % | 560 | 168 | déséquilibré |
| yeast3 | 1484 | 8 | 11,0 % | 163 | 49 | déséquilibré |
| bankmarketing | 45211 | 51 | 11,7 % | 5289 | 1587 | déséquilibré |
| abalone8 | 4177 | 10 | 13,6 % | 568 | 170 | déséquilibré |
| segmentation | 2310 | 19 | 14,3 % | 330 | 99 | déséquilibré |

## Jeux équilibrés (15)
| Jeu | m | d | Minorité | Minoritaires | Dans le test | Groupe |
|---|---|---|---|---|---|---|
| vehicle | 846 | 18 | 23,5 % | 199 | 60 | équilibré |
| german | 1000 | 24 | 30,0 % | 300 | 90 | équilibré |
| newthyroid | 215 | 5 | 30,2 % | 65 | 20 | équilibré |
| glass | 214 | 9 | 32,7 % | 70 | 21 | équilibré |
| wine | 178 | 13 | 33,1 % | 59 | 18 | équilibré |
| pima | 768 | 8 | 34,9 % | 268 | 80 | équilibré |
| iono | 351 | 34 | 35,9 % | 126 | 38 | équilibré |
| wdbc | 569 | 30 | 37,3 % | 212 | 64 | équilibré |
| spambase | 4597 | 57 | 39,4 % | 1812 | 544 | équilibré |
| bupa | 345 | 6 | 42,0 % | 145 | 44 | équilibré |
| heart | 270 | 13 | 44,4 % | 120 | 36 | équilibré |
| australian | 690 | 14 | 44,5 % | 307 | 92 | équilibré |
| balance | 625 | 4 | 46,1 % | 288 | 86 | équilibré |
| sonar | 208 | 60 | 46,6 % | 97 | 29 | équilibré |
| splice | 3175 | 60 | 48,1 % | 1527 | 458 | équilibré |
