# Question 5 : apprentissage classique contre minimisation de borne
6 jeux, 10 découpages. Méthodes classiques entraînées sur prior + S (70 % des données), PAC-Bayes avec le prior appris. Erreur en test : vote pour PAC-Bayes. Moyenne ± écart-type sur les découpages.

Classement des méthodes de la partie 1 sur ces six jeux (rang moyen en accuracy, 1 = meilleure) : Forêt aléatoire 3,33, Stacking (le mien) 3,50, Gradient boosting 4,08, Régression logistique 4,25, Hist. gradient boosting 4,92, Bagging (le mien) 5,25, AdaBoost prof. 3 (le mien) 5,58, AdaBoost stumps (le mien) 5,75, kNN (k=7) 8,67, Arbre seul 9,67.

## Tableau 1 : erreur en test de toutes les méthodes
| Jeu | AdaBoost stumps (le mien) | Forêt aléatoire | Régression logistique | Stacking (le mien) | PAC-Bayes stumps, A | PAC-Bayes stumps, B | PAC-Bayes linéaire, A | PAC-Bayes linéaire, B |
|---|---|---|---|---|---|---|---|---|
| wdbc | 0,039 ± 0,013 | 0,050 ± 0,016 | 0,030 ± 0,009 | 0,037 ± 0,015 | 0,092 ± 0,019 | 0,092 ± 0,019 | 0,035 ± 0,014 | 0,034 ± 0,015 |
| pima | 0,251 ± 0,036 | 0,239 ± 0,027 | 0,239 ± 0,024 | 0,240 ± 0,024 | 0,274 ± 0,017 | 0,274 ± 0,017 | 0,245 ± 0,018 | 0,243 ± 0,016 |
| australian | 0,146 ± 0,020 | 0,133 ± 0,010 | 0,139 ± 0,014 | 0,138 ± 0,023 | 0,147 ± 0,022 | 0,147 ± 0,022 | 0,139 ± 0,021 | 0,139 ± 0,024 |
| iono | 0,079 ± 0,024 | 0,068 ± 0,015 | 0,117 ± 0,014 | 0,083 ± 0,019 | 0,184 ± 0,027 | 0,184 ± 0,027 | 0,126 ± 0,017 | 0,126 ± 0,017 |
| german | 0,246 ± 0,014 | 0,244 ± 0,011 | 0,239 ± 0,017 | 0,248 ± 0,013 | 0,298 ± 0,003 | 0,295 ± 0,008 | 0,248 ± 0,022 | 0,246 ± 0,022 |
| spambase | 0,060 ± 0,006 | 0,050 ± 0,006 | 0,079 ± 0,006 | 0,065 ± 0,006 | 0,203 ± 0,008 | 0,207 ± 0,009 | 0,082 ± 0,006 | 0,082 ± 0,006 |
| **moyenne des 6 jeux** | 0,137 | 0,131 | 0,141 | 0,135 | 0,200 | 0,200 | 0,146 | 0,145 |
| **rang moyen** | 4,00 | 2,00 | 2,67 | 3,33 | 7,58 | 7,42 | 4,75 | 4,25 |

## Tableau 2 : les classifieurs PAC-bayésiens et leurs certificats
Le certificat de Seeger majore le risque de GIBBS (avec probabilité 95 %) ; 2 x certificat majore le risque du VOTE.

| Jeu | Famille | Borne | Certificat de Seeger (Gibbs) | 2 x certificat (vote) | Risque de Gibbs (test) | Risque du vote (test) | KL |
|---|---|---|---|---|---|---|---|
| wdbc | stumps | A | 0,156 ± 0,010 | 0,312 ± 0,020 | 0,100 ± 0,017 | 0,092 ± 0,019 | 1,64 ± 0,36 |
| wdbc | stumps | B | 0,156 ± 0,010 | 0,311 ± 0,021 | 0,097 ± 0,017 | 0,092 ± 0,019 | 2,24 ± 0,58 |
| wdbc | linéaire | A | 0,130 ± 0,009 | 0,259 ± 0,018 | 0,071 ± 0,011 | 0,035 ± 0,014 | 1,80 ± 0,34 |
| wdbc | linéaire | B | 0,128 ± 0,008 | 0,256 ± 0,017 | 0,069 ± 0,010 | 0,034 ± 0,015 | 2,12 ± 0,46 |
| pima | stumps | A | 0,337 ± 0,009 | 0,674 ± 0,017 | 0,273 ± 0,015 | 0,274 ± 0,017 | 1,86 ± 1,09 |
| pima | stumps | B | 0,337 ± 0,009 | 0,674 ± 0,017 | 0,274 ± 0,017 | 0,274 ± 0,017 | 2,17 ± 1,18 |
| pima | linéaire | A | 0,354 ± 0,014 | 0,707 ± 0,027 | 0,275 ± 0,018 | 0,245 ± 0,018 | 2,21 ± 1,50 |
| pima | linéaire | B | 0,353 ± 0,013 | 0,705 ± 0,027 | 0,272 ± 0,019 | 0,243 ± 0,016 | 2,81 ± 1,00 |
| australian | stumps | A | 0,219 ± 0,017 | 0,438 ± 0,033 | 0,147 ± 0,022 | 0,147 ± 0,022 | 0,37 ± 0,18 |
| australian | stumps | B | 0,219 ± 0,017 | 0,439 ± 0,033 | 0,147 ± 0,022 | 0,147 ± 0,022 | 0,37 ± 0,18 |
| australian | linéaire | A | 0,249 ± 0,008 | 0,498 ± 0,015 | 0,170 ± 0,013 | 0,139 ± 0,021 | 2,09 ± 0,28 |
| australian | linéaire | B | 0,247 ± 0,008 | 0,494 ± 0,016 | 0,167 ± 0,013 | 0,139 ± 0,024 | 2,80 ± 0,40 |
| iono | stumps | A | 0,298 ± 0,013 | 0,596 ± 0,026 | 0,189 ± 0,022 | 0,184 ± 0,027 | 3,35 ± 0,36 |
| iono | stumps | B | 0,299 ± 0,013 | 0,598 ± 0,025 | 0,188 ± 0,023 | 0,184 ± 0,027 | 3,52 ± 0,60 |
| iono | linéaire | A | 0,305 ± 0,016 | 0,611 ± 0,031 | 0,181 ± 0,008 | 0,126 ± 0,017 | 3,82 ± 0,48 |
| iono | linéaire | B | 0,306 ± 0,016 | 0,611 ± 0,031 | 0,182 ± 0,008 | 0,126 ± 0,017 | 3,76 ± 0,59 |
| german | stumps | A | 0,370 ± 0,006 | 0,740 ± 0,012 | 0,298 ± 0,004 | 0,298 ± 0,003 | 1,40 ± 1,27 |
| german | stumps | B | 0,370 ± 0,006 | 0,740 ± 0,011 | 0,299 ± 0,006 | 0,295 ± 0,008 | 1,67 ± 1,02 |
| german | linéaire | A | 0,370 ± 0,007 | 0,740 ± 0,013 | 0,286 ± 0,009 | 0,248 ± 0,022 | 5,24 ± 0,36 |
| german | linéaire | B | 0,370 ± 0,006 | 0,741 ± 0,013 | 0,288 ± 0,009 | 0,246 ± 0,022 | 4,60 ± 0,42 |
| spambase | stumps | A | 0,246 ± 0,003 | 0,492 ± 0,007 | 0,210 ± 0,008 | 0,203 ± 0,008 | 1,30 ± 1,05 |
| spambase | stumps | B | 0,246 ± 0,003 | 0,492 ± 0,007 | 0,210 ± 0,008 | 0,207 ± 0,009 | 1,46 ± 0,88 |
| spambase | linéaire | A | 0,149 ± 0,005 | 0,297 ± 0,009 | 0,116 ± 0,006 | 0,082 ± 0,006 | 6,37 ± 0,29 |
| spambase | linéaire | B | 0,148 ± 0,005 | 0,297 ± 0,009 | 0,115 ± 0,006 | 0,082 ± 0,006 | 6,84 ± 0,44 |

## Tableau 3 : différence d'erreur en test, classique - PAC-Bayes (en points ; négatif = le classique est meilleur)
Dernière colonne : nombre d'exécutions (sur 60) où le classique est strictement meilleur.

| Comparaison | wdbc | pima | australian | iono | german | spambase | Moyenne | Classique meilleur |
|---|---|---|---|---|---|---|---|---|
| AdaBoost sur stumps contre stumps PAC-bayésiens (A) | -5,3 | -2,3 | -0,1 | -10,5 | -5,2 | -14,4 | -6,3 | 53/60 |
| Régression logistique contre linéaire PAC-bayésien (A) | -0,5 | -0,5 | 0,0 | -0,9 | -1,0 | -0,3 | -0,5 | 38/60 |
| Forêt aléatoire contre linéaire PAC-bayésien (A) | 1,5 | -0,6 | -0,5 | -5,8 | -0,5 | -3,2 | -1,5 | 36/60 |
| Forêt aléatoire contre stumps PAC-bayésiens (A) | -4,2 | -3,5 | -1,4 | -11,6 | -5,4 | -15,3 | -6,9 | 57/60 |

## Tableau 4 : ce que garantit chaque approche
PAC-Bayes : certificat de Seeger (sur Gibbs), 2 x certificat (sur le vote) et erreur de test du vote. Forêt aléatoire : erreur de test et borne « de test » (95 %) calculée avec le jeu de test, qui n'existe pas au moment de l'apprentissage.

| Jeu | Linéaire : certificat | Linéaire : 2 x certificat | Linéaire : erreur du vote | Stumps : certificat | Stumps : 2 x certificat | Stumps : erreur du vote | Forêt : erreur | Forêt : borne de test |
|---|---|---|---|---|---|---|---|---|
| wdbc | 0,130 | 0,259 | 0,035 | 0,156 | 0,312 | 0,092 | 0,050 | 0,100 |
| pima | 0,354 | 0,707 | 0,245 | 0,337 | 0,674 | 0,274 | 0,239 | 0,312 |
| australian | 0,249 | 0,498 | 0,139 | 0,219 | 0,438 | 0,147 | 0,133 | 0,198 |
| iono | 0,305 | 0,611 | 0,126 | 0,298 | 0,596 | 0,184 | 0,068 | 0,144 |
| german | 0,370 | 0,740 | 0,248 | 0,370 | 0,740 | 0,298 | 0,244 | 0,307 |
| spambase | 0,149 | 0,297 | 0,082 | 0,246 | 0,492 | 0,203 | 0,050 | 0,066 |

## Tableau 5 : le prix du découpage en trois parties (entraîner sur S seul, 56 %, au lieu de 70 %)
Hausse de l'erreur de test (en points) quand la méthode classique n'utilise que S.

| Méthode | wdbc | pima | australian | iono | german | spambase | Moyenne |
|---|---|---|---|---|---|---|---|
| AdaBoost stumps (le mien) | 0,4 | -0,2 | 0,7 | 0,4 | 0,8 | -0,0 | 0,4 |
| Forêt aléatoire | 0,5 | -0,2 | 0,1 | -0,6 | -0,4 | 0,1 | -0,1 |
| Régression logistique | 0,1 | 0,1 | 0,0 | -0,2 | -0,0 | 0,2 | 0,0 |
| Stacking (le mien) | 0,2 | 0,0 | 0,5 | -0,2 | 0,4 | 0,3 | 0,2 |

## Tableau 6 : meilleure méthode de chaque approche, par jeu
Attention : le meilleur de chaque approche est choisi en regardant le test, ce qui avantage légèrement les deux.

| Jeu | Meilleur classique | Erreur | Meilleur PAC-Bayes | Erreur | Écart (PAC-Bayes - classique, points) |
|---|---|---|---|---|---|
| wdbc | Régression logistique | 0,030 | PAC-Bayes linéaire, B | 0,034 | 0,4 |
| pima | Forêt aléatoire | 0,239 | PAC-Bayes linéaire, B | 0,243 | 0,4 |
| australian | Forêt aléatoire | 0,133 | PAC-Bayes linéaire, A | 0,139 | 0,5 |
| iono | Forêt aléatoire | 0,068 | PAC-Bayes linéaire, A | 0,126 | 5,8 |
| german | Régression logistique | 0,239 | PAC-Bayes linéaire, B | 0,246 | 0,8 |
| spambase | Forêt aléatoire | 0,050 | PAC-Bayes linéaire, B | 0,082 | 3,2 |

