# Question 2 : comparaison des bornes A et B sur des stumps (prior uniforme)
6 jeux, 10 découpages chacun. Moyenne ± écart-type sur les découpages. Mes résultats ont été vérifiés identiques à ceux de `run_categorical` sur 6 exécutions.

## Tableau 1 : les jeux utilisés
Tailles de la partie prior (14 %), de l'échantillon de borne S (56 %) et du test (30 %) ; « Stumps » = nombre de votants (moyenne sur les découpages).

| Jeu | m | d | Prior | S | Test | Stumps |
|---|---|---|---|---|---|---|
| wdbc | 569 | 30 | 79 | 319 | 171 | 587 |
| pima | 768 | 8 | 107 | 430 | 231 | 146 |
| australian | 690 | 14 | 96 | 387 | 207 | 151 |
| iono | 351 | 34 | 49 | 196 | 106 | 614 |
| german | 1000 | 24 | 140 | 560 | 300 | 168 |
| spambase | 4597 | 57 | 643 | 2574 | 1380 | 371 |

## Tableau 2 : wdbc
| Borne | Valeur de la borne | Certificat de Seeger | KL | Votants effectifs | Risque de Gibbs (test) | Risque du vote (test) |
|---|---|---|---|---|---|---|
| A (linéaire) | 0,211 ± 0,008 | 0,171 ± 0,010 | 4,67 ± 0,26 | 4,1 ± 0,9 | 0,102 ± 0,018 | 0,093 ± 0,015 |
| B (McAllester) | 0,213 ± 0,008 | 0,170 ± 0,010 | 5,37 ± 0,70 | 2,6 ± 1,2 | 0,099 ± 0,018 | 0,095 ± 0,015 |

## Tableau 3 : les six jeux
| Jeu | Borne | Valeur de la borne | Certificat de Seeger | KL | Votants effectifs | Risque de Gibbs (test) | Risque du vote (test) |
|---|---|---|---|---|---|---|---|
| wdbc | A (linéaire) | 0,211 ± 0,008 | 0,171 ± 0,010 | 4,67 ± 0,26 | 4,1 ± 0,9 | 0,102 ± 0,018 | 0,093 ± 0,015 |
| wdbc | B (McAllester) | 0,213 ± 0,008 | 0,170 ± 0,010 | 5,37 ± 0,70 | 2,6 ± 1,2 | 0,099 ± 0,018 | 0,095 ± 0,015 |
| pima | A (linéaire) | 0,358 ± 0,010 | 0,350 ± 0,011 | 4,59 ± 0,29 | 1,4 ± 0,4 | 0,273 ± 0,016 | 0,274 ± 0,017 |
| pima | B (McAllester) | 0,358 ± 0,010 | 0,350 ± 0,011 | 4,56 ± 0,38 | 1,5 ± 0,5 | 0,273 ± 0,015 | 0,274 ± 0,017 |
| australian | A (linéaire) | 0,268 ± 0,014 | 0,243 ± 0,017 | 4,95 ± 0,22 | 1,1 ± 0,3 | 0,147 ± 0,022 | 0,147 ± 0,022 |
| australian | B (McAllester) | 0,267 ± 0,014 | 0,243 ± 0,017 | 4,94 ± 0,22 | 1,1 ± 0,3 | 0,147 ± 0,022 | 0,147 ± 0,022 |
| iono | A (linéaire) | 0,333 ± 0,011 | 0,315 ± 0,014 | 5,73 ± 0,32 | 1,7 ± 0,6 | 0,187 ± 0,022 | 0,175 ± 0,024 |
| iono | B (McAllester) | 0,337 ± 0,011 | 0,315 ± 0,014 | 5,78 ± 0,32 | 1,7 ± 0,6 | 0,186 ± 0,023 | 0,175 ± 0,024 |
| german | A (linéaire) | 0,378 ± 0,005 | 0,376 ± 0,005 | 3,79 ± 0,82 | 2,9 ± 1,7 | 0,307 ± 0,014 | 0,301 ± 0,014 |
| german | B (McAllester) | 0,381 ± 0,005 | 0,377 ± 0,005 | 3,73 ± 1,22 | 3,9 ± 2,9 | 0,306 ± 0,013 | 0,300 ± 0,014 |
| spambase | A (linéaire) | 0,257 ± 0,003 | 0,253 ± 0,003 | 5,47 ± 0,32 | 1,4 ± 0,5 | 0,211 ± 0,007 | 0,211 ± 0,007 |
| spambase | B (McAllester) | 0,260 ± 0,003 | 0,253 ± 0,003 | 5,70 ± 0,36 | 1,3 ± 0,5 | 0,211 ± 0,007 | 0,211 ± 0,007 |

## Tableau 4 : les deux bornes conduisent-elles au même posterior ?
Distance en variation totale entre Q_A et Q_B (0 = identiques, 1 = supports disjoints) ; recouvrement Σ min(Q_A, Q_B) ; nombre de découpages où le stump de poids maximal est le même, ou porte sur la même variable ; poids du stump le plus pondéré de chaque posterior ; nombre effectif de votants.

| Jeu | Distance (var. totale) | Recouvrement | Même stump principal | Même variable principale | Poids max A | Poids max B | Votants eff. A | Votants eff. B |
|---|---|---|---|---|---|---|---|---|
| wdbc | 0,23 | 0,77 | 10/10 | 10/10 | 0,40 | 0,61 | 4,1 | 2,6 |
| pima | 0,05 | 0,95 | 10/10 | 10/10 | 0,85 | 0,82 | 1,4 | 1,5 |
| australian | 0,00 | 1,00 | 10/10 | 10/10 | 0,95 | 0,95 | 1,1 | 1,1 |
| iono | 0,02 | 0,98 | 10/10 | 10/10 | 0,71 | 0,73 | 1,7 | 1,7 |
| german | 0,11 | 0,89 | 10/10 | 10/10 | 0,62 | 0,60 | 2,9 | 3,9 |
| spambase | 0,05 | 0,95 | 10/10 | 10/10 | 0,84 | 0,89 | 1,4 | 1,3 |

## Tableau 5 : quel certificat reporter ?
« Seeger ≥ test » : nombre de découpages où le certificat de Seeger est supérieur au risque de Gibbs mesuré sur le test (il doit l'être avec probabilité 95 %). « Écart » = certificat - risque de test (moyenne). « Borne - Seeger » : de combien la borne minimisée dépasse le certificat de Seeger du même posterior. « Seeger A < Seeger B » : nombre de découpages où le posterior de A a le meilleur certificat, et écart moyen Seeger(B) - Seeger(A). Dernière colonne : nombre d'exécutions où R(vote) ≤ 2 R(Gibbs) sur le test.

| Jeu | Seeger ≥ test (A) | Seeger ≥ test (B) | Écart (A) | Écart (B) | Borne - Seeger (A) | Borne - Seeger (B) | Seeger A < Seeger B | Écart moyen B - A | Vote ≤ 2 x Gibbs |
|---|---|---|---|---|---|---|---|---|---|
| wdbc | 10/10 | 10/10 | 0,069 | 0,071 | 0,040 | 0,043 | 0/10 | -0,0014 | 20/20 |
| pima | 10/10 | 10/10 | 0,077 | 0,077 | 0,008 | 0,008 | 7/10 | 0,0001 | 20/20 |
| australian | 10/10 | 10/10 | 0,097 | 0,096 | 0,025 | 0,023 | 10/10 | 0,0000 | 20/20 |
| iono | 10/10 | 10/10 | 0,128 | 0,129 | 0,018 | 0,022 | 0/10 | -0,0001 | 20/20 |
| german | 10/10 | 10/10 | 0,070 | 0,070 | 0,002 | 0,005 | 5/10 | 0,0002 | 20/20 |
| spambase | 10/10 | 10/10 | 0,042 | 0,042 | 0,004 | 0,008 | 7/10 | 0,0000 | 20/20 |

## Tableau 6 : λ choisi par l'algorithme de la borne A
| Jeu | λ médian choisi |
|---|---|
| wdbc | 128 |
| pima | 256 |
| australian | 256 |
| iono | 128 |
| german | 256 |
| spambase | 512 |

## Résumé sur les 60 exécutions
- Certificat de Seeger ≥ risque de Gibbs en test : A 60/60, B 60/60.
- Le posterior de A a un certificat de Seeger plus petit que celui de B dans 29/60 cas ; le risque de Gibbs en test de A est plus petit dans 29/60 cas ; le risque du vote de A est plus petit ou égal dans 59/60 cas.
- Votants effectifs : A 2,1 en moyenne, B 2,0 ; B concentre tout son poids sur un seul stump (votants effectifs < 1,5) dans 34/60 cas, A dans 31/60 cas.

