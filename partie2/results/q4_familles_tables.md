# Question 4 : stumps contre classifieurs linéaires, et autres posteriors
6 jeux, 10 découpages, prior appris pour les deux familles. Moyenne ± écart-type sur les découpages. Les posteriors de stumps avec les bornes A et B ont été vérifiés identiques à ceux de `run_categorical` (prior appris) sur 4 cas.

## Tableau 1 : les deux familles, avec une borne minimisée (A ou B)
| Jeu | Posterior | Certificat de Seeger | KL | Risque de Gibbs (test) | Risque du vote (test) | Gibbs - vote |
|---|---|---|---|---|---|---|
| wdbc | Stumps, borne A | 0,156 ± 0,010 | 1,64 ± 0,36 | 0,100 ± 0,017 | 0,092 ± 0,019 | 0,008 |
| wdbc | Stumps, borne B | 0,156 ± 0,010 | 2,24 ± 0,58 | 0,097 ± 0,017 | 0,092 ± 0,019 | 0,005 |
| wdbc | Linéaire, borne A | 0,130 ± 0,009 | 1,80 ± 0,34 | 0,071 ± 0,011 | 0,035 ± 0,014 | 0,036 |
| wdbc | Linéaire, borne B | 0,128 ± 0,008 | 2,12 ± 0,46 | 0,069 ± 0,010 | 0,034 ± 0,015 | 0,035 |
| pima | Stumps, borne A | 0,337 ± 0,009 | 1,86 ± 1,09 | 0,273 ± 0,015 | 0,274 ± 0,017 | -0,001 |
| pima | Stumps, borne B | 0,337 ± 0,009 | 2,17 ± 1,18 | 0,274 ± 0,017 | 0,274 ± 0,017 | 0,000 |
| pima | Linéaire, borne A | 0,354 ± 0,014 | 2,21 ± 1,50 | 0,275 ± 0,018 | 0,245 ± 0,018 | 0,030 |
| pima | Linéaire, borne B | 0,353 ± 0,013 | 2,81 ± 1,00 | 0,272 ± 0,019 | 0,243 ± 0,016 | 0,028 |
| australian | Stumps, borne A | 0,219 ± 0,017 | 0,37 ± 0,18 | 0,147 ± 0,022 | 0,147 ± 0,022 | 0,000 |
| australian | Stumps, borne B | 0,219 ± 0,017 | 0,37 ± 0,18 | 0,147 ± 0,022 | 0,147 ± 0,022 | 0,000 |
| australian | Linéaire, borne A | 0,249 ± 0,008 | 2,09 ± 0,28 | 0,170 ± 0,013 | 0,139 ± 0,021 | 0,031 |
| australian | Linéaire, borne B | 0,247 ± 0,008 | 2,80 ± 0,40 | 0,167 ± 0,013 | 0,139 ± 0,024 | 0,028 |
| iono | Stumps, borne A | 0,298 ± 0,013 | 3,35 ± 0,36 | 0,189 ± 0,022 | 0,184 ± 0,027 | 0,005 |
| iono | Stumps, borne B | 0,299 ± 0,013 | 3,52 ± 0,60 | 0,188 ± 0,023 | 0,184 ± 0,027 | 0,004 |
| iono | Linéaire, borne A | 0,305 ± 0,016 | 3,82 ± 0,48 | 0,181 ± 0,008 | 0,126 ± 0,017 | 0,055 |
| iono | Linéaire, borne B | 0,306 ± 0,016 | 3,76 ± 0,59 | 0,182 ± 0,008 | 0,126 ± 0,017 | 0,055 |
| german | Stumps, borne A | 0,370 ± 0,006 | 1,40 ± 1,27 | 0,298 ± 0,004 | 0,298 ± 0,003 | 0,001 |
| german | Stumps, borne B | 0,370 ± 0,006 | 1,67 ± 1,02 | 0,299 ± 0,006 | 0,295 ± 0,008 | 0,004 |
| german | Linéaire, borne A | 0,370 ± 0,007 | 5,24 ± 0,36 | 0,286 ± 0,009 | 0,248 ± 0,022 | 0,037 |
| german | Linéaire, borne B | 0,370 ± 0,006 | 4,60 ± 0,42 | 0,288 ± 0,009 | 0,246 ± 0,022 | 0,041 |
| spambase | Stumps, borne A | 0,246 ± 0,003 | 1,30 ± 1,05 | 0,210 ± 0,008 | 0,203 ± 0,008 | 0,006 |
| spambase | Stumps, borne B | 0,246 ± 0,003 | 1,46 ± 0,88 | 0,210 ± 0,008 | 0,207 ± 0,009 | 0,002 |
| spambase | Linéaire, borne A | 0,149 ± 0,005 | 6,37 ± 0,29 | 0,116 ± 0,006 | 0,082 ± 0,006 | 0,034 |
| spambase | Linéaire, borne B | 0,148 ± 0,005 | 6,84 ± 0,44 | 0,115 ± 0,006 | 0,082 ± 0,006 | 0,033 |

## Tableau 2 : autres posteriors de stumps (Dirac sur le meilleur, uniforme sur les 5 et les 20 meilleurs)
Pour comparaison, le posterior de la borne A. Un Dirac a un vote égal à son stump : « Gibbs - vote » y vaut 0.

| Jeu | Posterior | Certificat de Seeger | KL | Risque de Gibbs (test) | Risque du vote (test) | Gibbs - vote |
|---|---|---|---|---|---|---|
| wdbc | Dirac sur le meilleur stump | 0,157 ± 0,010 | 3,42 ± 0,68 | 0,099 ± 0,016 | 0,099 ± 0,016 | 0,000 |
| wdbc | Uniforme sur les 5 meilleurs | 0,160 ± 0,010 | 1,82 ± 0,32 | 0,103 ± 0,023 | 0,092 ± 0,021 | 0,011 |
| wdbc | Uniforme sur les 20 meilleurs | 0,186 ± 0,012 | 0,85 ± 0,24 | 0,121 ± 0,018 | 0,104 ± 0,023 | 0,018 |
| wdbc | Stumps, borne A | 0,156 ± 0,010 | 1,64 ± 0,36 | 0,100 ± 0,017 | 0,092 ± 0,019 | 0,008 |
| pima | Dirac sur le meilleur stump | 0,338 ± 0,009 | 2,54 ± 0,90 | 0,274 ± 0,017 | 0,274 ± 0,017 | 0,000 |
| pima | Uniforme sur les 5 meilleurs | 0,361 ± 0,008 | 1,27 ± 0,70 | 0,289 ± 0,019 | 0,265 ± 0,017 | 0,024 |
| pima | Uniforme sur les 20 meilleurs | 0,415 ± 0,006 | 1,09 ± 0,16 | 0,333 ± 0,009 | 0,328 ± 0,018 | 0,005 |
| pima | Stumps, borne A | 0,337 ± 0,009 | 1,86 ± 1,09 | 0,273 ± 0,015 | 0,274 ± 0,017 | -0,001 |
| australian | Dirac sur le meilleur stump | 0,220 ± 0,017 | 0,44 ± 0,19 | 0,147 ± 0,022 | 0,147 ± 0,022 | 0,000 |
| australian | Uniforme sur les 5 meilleurs | 0,328 ± 0,010 | 1,20 ± 0,32 | 0,241 ± 0,018 | 0,220 ± 0,026 | 0,021 |
| australian | Uniforme sur les 20 meilleurs | 0,428 ± 0,007 | 2,10 ± 0,53 | 0,328 ± 0,011 | 0,243 ± 0,021 | 0,085 |
| australian | Stumps, borne A | 0,219 ± 0,017 | 0,37 ± 0,18 | 0,147 ± 0,022 | 0,147 ± 0,022 | 0,000 |
| iono | Dirac sur le meilleur stump | 0,301 ± 0,014 | 4,22 ± 0,58 | 0,182 ± 0,026 | 0,182 ± 0,026 | 0,000 |
| iono | Uniforme sur les 5 meilleurs | 0,319 ± 0,013 | 2,69 ± 0,19 | 0,200 ± 0,019 | 0,156 ± 0,018 | 0,044 |
| iono | Uniforme sur les 20 meilleurs | 0,370 ± 0,014 | 1,88 ± 0,27 | 0,257 ± 0,012 | 0,194 ± 0,036 | 0,063 |
| iono | Stumps, borne A | 0,298 ± 0,013 | 3,35 ± 0,36 | 0,189 ± 0,022 | 0,184 ± 0,027 | 0,005 |
| german | Dirac sur le meilleur stump | 0,377 ± 0,011 | 4,58 ± 0,95 | 0,316 ± 0,024 | 0,316 ± 0,024 | 0,000 |
| german | Uniforme sur les 5 meilleurs | 0,377 ± 0,003 | 2,26 ± 0,26 | 0,304 ± 0,009 | 0,299 ± 0,006 | 0,005 |
| german | Uniforme sur les 20 meilleurs | 0,376 ± 0,001 | 0,74 ± 0,17 | 0,301 ± 0,002 | 0,300 ± 0,000 | 0,001 |
| german | Stumps, borne A | 0,370 ± 0,006 | 1,40 ± 1,27 | 0,298 ± 0,004 | 0,298 ± 0,003 | 0,001 |
| spambase | Dirac sur le meilleur stump | 0,247 ± 0,005 | 2,78 ± 1,64 | 0,211 ± 0,007 | 0,211 ± 0,007 | 0,000 |
| spambase | Uniforme sur les 5 meilleurs | 0,256 ± 0,004 | 1,98 ± 1,22 | 0,216 ± 0,005 | 0,178 ± 0,010 | 0,038 |
| spambase | Uniforme sur les 20 meilleurs | 0,292 ± 0,003 | 5,36 ± 1,44 | 0,247 ± 0,003 | 0,179 ± 0,013 | 0,068 |
| spambase | Stumps, borne A | 0,246 ± 0,003 | 1,30 ± 1,05 | 0,210 ± 0,008 | 0,203 ± 0,008 | 0,006 |

## Tableau 3 : de combien le vote est-il meilleur que le classifieur de Gibbs ?
Écart Gibbs - vote (positif : le vote est meilleur) et rapport vote / Gibbs (1 = aucun avantage ; 0,5 = vote deux fois meilleur), avec la borne A.

| Jeu | Stumps : Gibbs - vote | Stumps : vote / Gibbs | Linéaire : Gibbs - vote | Linéaire : vote / Gibbs |
|---|---|---|---|---|
| wdbc | 0,008 | 0,92 | 0,036 | 0,48 |
| pima | -0,001 | 1,01 | 0,030 | 0,89 |
| australian | 0,000 | 1,00 | 0,031 | 0,81 |
| iono | 0,005 | 0,97 | 0,055 | 0,70 |
| german | 0,001 | 1,00 | 0,037 | 0,87 |
| spambase | 0,006 | 0,97 | 0,034 | 0,71 |
| **moyenne des 6 jeux** | 0,003 | 0,98 | 0,037 | 0,74 |

## Tableau 4 : le meilleur k pour le vote (uniforme sur les k meilleurs stumps)
k de la grille qui minimise le risque du vote moyen en test ; ce risque, comparé à celui du Dirac (k = 1), du posterior de la borne A et du classifieur linéaire (borne A) ; risque de Gibbs à ce k et au Dirac. « tous » = uniforme sur tous les stumps. C'est un diagnostic (le k est choisi sur le test), pas une méthode.

| Jeu | Meilleur k pour le vote | Vote à ce k | Vote du Dirac (k = 1) | Vote du posterior A | Vote du linéaire A | Gibbs à ce k | Gibbs du Dirac |
|---|---|---|---|---|---|---|---|
| wdbc | 100 | 0,073 | 0,099 | 0,092 | 0,035 | 0,212 | 0,099 |
| pima | 5 | 0,265 | 0,274 | 0,274 | 0,245 | 0,289 | 0,274 |
| australian | 1 | 0,147 | 0,147 | 0,147 | 0,139 | 0,147 | 0,147 |
| iono | 5 | 0,156 | 0,182 | 0,184 | 0,126 | 0,200 | 0,182 |
| german | 3 | 0,294 | 0,316 | 0,298 | 0,248 | 0,304 | 0,316 |
| spambase | 10 | 0,170 | 0,211 | 0,203 | 0,082 | 0,227 | 0,211 |

## Tableau 5 : quelle famille a le meilleur certificat et le meilleur vote ? (borne A)
| Jeu | Certificat stumps | Certificat linéaire | Linéaire meilleur certificat | Vote stumps | Vote linéaire |
|---|---|---|---|---|---|
| wdbc | 0,156 | 0,130 | 10/10 | 0,092 | 0,035 |
| pima | 0,337 | 0,354 | 0/10 | 0,274 | 0,245 |
| australian | 0,219 | 0,249 | 0/10 | 0,147 | 0,139 |
| iono | 0,298 | 0,305 | 2/10 | 0,184 | 0,126 |
| german | 0,370 | 0,370 | 4/10 | 0,298 | 0,248 |
| spambase | 0,246 | 0,149 | 10/10 | 0,203 | 0,082 |

Le classifieur linéaire a un meilleur certificat de Seeger que les stumps dans 26/60 exécutions.

