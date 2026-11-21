# Question 3 : impact du prior
6 jeux, 10 découpages chacun, moyenne ± écart-type sur les découpages. « Départ » = prior uniforme (stumps) ou N(0, I) (linéaire) ; « appris » = prior appris sur la partie prior. Mes résultats ont été vérifiés identiques à ceux de `run_categorical` et `run_gaussian` sur 8 cas.

## Tableau 1 : Stumps (catégoriel)
| Jeu | Prior | Borne | Valeur de la borne | Certificat de Seeger | KL | Risque de Gibbs (test) | Risque du vote (test) |
|---|---|---|---|---|---|---|---|
| wdbc | départ | A | 0,211 ± 0,008 | 0,171 ± 0,010 | 4,67 ± 0,26 | 0,102 ± 0,018 | 0,093 ± 0,015 |
| wdbc | départ | B | 0,213 ± 0,008 | 0,170 ± 0,010 | 5,37 ± 0,70 | 0,099 ± 0,018 | 0,095 ± 0,015 |
| wdbc | appris | A | 0,188 ± 0,008 | 0,156 ± 0,010 | 1,64 ± 0,36 | 0,100 ± 0,017 | 0,092 ± 0,019 |
| wdbc | appris | B | 0,195 ± 0,008 | 0,156 ± 0,010 | 2,24 ± 0,58 | 0,097 ± 0,017 | 0,092 ± 0,019 |
| pima | départ | A | 0,358 ± 0,010 | 0,350 ± 0,011 | 4,59 ± 0,29 | 0,273 ± 0,016 | 0,274 ± 0,017 |
| pima | départ | B | 0,358 ± 0,010 | 0,350 ± 0,011 | 4,56 ± 0,38 | 0,273 ± 0,015 | 0,274 ± 0,017 |
| pima | appris | A | 0,342 ± 0,008 | 0,337 ± 0,009 | 1,86 ± 1,09 | 0,273 ± 0,015 | 0,274 ± 0,017 |
| pima | appris | B | 0,345 ± 0,008 | 0,337 ± 0,009 | 2,17 ± 1,18 | 0,274 ± 0,017 | 0,274 ± 0,017 |
| australian | départ | A | 0,268 ± 0,014 | 0,243 ± 0,017 | 4,95 ± 0,22 | 0,147 ± 0,022 | 0,147 ± 0,022 |
| australian | départ | B | 0,267 ± 0,014 | 0,243 ± 0,017 | 4,94 ± 0,22 | 0,147 ± 0,022 | 0,147 ± 0,022 |
| australian | appris | A | 0,233 ± 0,014 | 0,219 ± 0,017 | 0,37 ± 0,18 | 0,147 ± 0,022 | 0,147 ± 0,022 |
| australian | appris | B | 0,240 ± 0,014 | 0,219 ± 0,017 | 0,37 ± 0,18 | 0,147 ± 0,022 | 0,147 ± 0,022 |
| iono | départ | A | 0,333 ± 0,011 | 0,315 ± 0,014 | 5,73 ± 0,32 | 0,187 ± 0,022 | 0,175 ± 0,024 |
| iono | départ | B | 0,337 ± 0,011 | 0,315 ± 0,014 | 5,78 ± 0,32 | 0,186 ± 0,023 | 0,175 ± 0,024 |
| iono | appris | A | 0,315 ± 0,010 | 0,298 ± 0,013 | 3,35 ± 0,36 | 0,189 ± 0,022 | 0,184 ± 0,027 |
| iono | appris | B | 0,321 ± 0,010 | 0,299 ± 0,013 | 3,52 ± 0,60 | 0,188 ± 0,023 | 0,184 ± 0,027 |
| german | départ | A | 0,378 ± 0,005 | 0,376 ± 0,005 | 3,79 ± 0,82 | 0,307 ± 0,014 | 0,301 ± 0,014 |
| german | départ | B | 0,381 ± 0,005 | 0,377 ± 0,005 | 3,73 ± 1,22 | 0,306 ± 0,013 | 0,300 ± 0,014 |
| german | appris | A | 0,371 ± 0,005 | 0,370 ± 0,006 | 1,40 ± 1,27 | 0,298 ± 0,004 | 0,298 ± 0,003 |
| german | appris | B | 0,374 ± 0,005 | 0,370 ± 0,006 | 1,67 ± 1,02 | 0,299 ± 0,006 | 0,295 ± 0,008 |
| spambase | départ | A | 0,257 ± 0,003 | 0,253 ± 0,003 | 5,47 ± 0,32 | 0,211 ± 0,007 | 0,211 ± 0,007 |
| spambase | départ | B | 0,260 ± 0,003 | 0,253 ± 0,003 | 5,70 ± 0,36 | 0,211 ± 0,007 | 0,211 ± 0,007 |
| spambase | appris | A | 0,249 ± 0,003 | 0,246 ± 0,003 | 1,30 ± 1,05 | 0,210 ± 0,008 | 0,203 ± 0,008 |
| spambase | appris | B | 0,253 ± 0,003 | 0,246 ± 0,003 | 1,46 ± 0,88 | 0,210 ± 0,008 | 0,207 ± 0,009 |

## Tableau 2 : Linéaire (gaussien)
| Jeu | Prior | Borne | Valeur de la borne | Certificat de Seeger | KL | Risque de Gibbs (test) | Risque du vote (test) |
|---|---|---|---|---|---|---|---|
| wdbc | départ | A | 0,226 ± 0,007 | 0,173 ± 0,010 | 10,15 ± 0,67 | 0,077 ± 0,009 | 0,035 ± 0,011 |
| wdbc | départ | B | 0,225 ± 0,008 | 0,176 ± 0,010 | 8,67 ± 0,64 | 0,083 ± 0,009 | 0,038 ± 0,011 |
| wdbc | appris | A | 0,169 ± 0,008 | 0,130 ± 0,009 | 1,80 ± 0,34 | 0,071 ± 0,011 | 0,035 ± 0,014 |
| wdbc | appris | B | 0,174 ± 0,007 | 0,128 ± 0,008 | 2,12 ± 0,46 | 0,069 ± 0,010 | 0,034 ± 0,015 |
| pima | départ | A | 0,395 ± 0,011 | 0,391 ± 0,012 | 8,96 ± 0,46 | 0,280 ± 0,018 | 0,244 ± 0,020 |
| pima | départ | B | 0,398 ± 0,011 | 0,391 ± 0,012 | 8,22 ± 0,51 | 0,283 ± 0,018 | 0,244 ± 0,018 |
| pima | appris | A | 0,357 ± 0,013 | 0,354 ± 0,014 | 2,21 ± 1,50 | 0,275 ± 0,018 | 0,245 ± 0,018 |
| pima | appris | B | 0,360 ± 0,013 | 0,353 ± 0,013 | 2,81 ± 1,00 | 0,272 ± 0,019 | 0,243 ± 0,016 |
| australian | départ | A | 0,317 ± 0,009 | 0,299 ± 0,011 | 11,74 ± 0,45 | 0,180 ± 0,012 | 0,140 ± 0,026 |
| australian | départ | B | 0,320 ± 0,009 | 0,300 ± 0,010 | 11,17 ± 0,52 | 0,182 ± 0,012 | 0,141 ± 0,025 |
| australian | appris | A | 0,263 ± 0,006 | 0,249 ± 0,008 | 2,09 ± 0,28 | 0,170 ± 0,013 | 0,139 ± 0,021 |
| australian | appris | B | 0,267 ± 0,006 | 0,247 ± 0,008 | 2,80 ± 0,40 | 0,167 ± 0,013 | 0,139 ± 0,024 |
| iono | départ | A | 0,394 ± 0,012 | 0,381 ± 0,014 | 8,91 ± 0,28 | 0,216 ± 0,011 | 0,137 ± 0,020 |
| iono | départ | B | 0,394 ± 0,012 | 0,377 ± 0,014 | 11,03 ± 0,47 | 0,202 ± 0,011 | 0,132 ± 0,022 |
| iono | appris | A | 0,322 ± 0,013 | 0,305 ± 0,016 | 3,82 ± 0,48 | 0,181 ± 0,008 | 0,126 ± 0,017 |
| iono | appris | B | 0,326 ± 0,013 | 0,306 ± 0,016 | 3,76 ± 0,59 | 0,182 ± 0,008 | 0,126 ± 0,017 |
| german | départ | A | 0,419 ± 0,006 | 0,418 ± 0,006 | 11,84 ± 0,25 | 0,308 ± 0,010 | 0,249 ± 0,021 |
| german | départ | B | 0,422 ± 0,006 | 0,417 ± 0,006 | 13,48 ± 0,39 | 0,303 ± 0,010 | 0,247 ± 0,020 |
| german | appris | A | 0,372 ± 0,006 | 0,370 ± 0,007 | 5,24 ± 0,36 | 0,286 ± 0,009 | 0,248 ± 0,022 |
| german | appris | B | 0,376 ± 0,006 | 0,370 ± 0,006 | 4,60 ± 0,42 | 0,288 ± 0,009 | 0,246 ± 0,022 |
| spambase | départ | A | 0,214 ± 0,004 | 0,191 ± 0,005 | 40,88 ± 0,85 | 0,123 ± 0,006 | 0,083 ± 0,006 |
| spambase | départ | B | 0,216 ± 0,004 | 0,191 ± 0,005 | 40,11 ± 1,10 | 0,124 ± 0,006 | 0,083 ± 0,006 |
| spambase | appris | A | 0,161 ± 0,004 | 0,149 ± 0,005 | 6,37 ± 0,29 | 0,116 ± 0,006 | 0,082 ± 0,006 |
| spambase | appris | B | 0,165 ± 0,004 | 0,148 ± 0,005 | 6,84 ± 0,44 | 0,115 ± 0,006 | 0,082 ± 0,006 |

## Tableau 3 : ce qui change avec le prior appris (appris - départ, moyenne sur les découpages)
Une différence négative de KL ou de certificat est une amélioration. « Certificat meilleur » : nombre de découpages où le certificat de Seeger est plus petit avec le prior appris.

| Famille | Borne | Jeu | Δ KL | Δ certificat de Seeger | Certificat meilleur | Δ risque de Gibbs | Δ risque du vote |
|---|---|---|---|---|---|---|---|
| Stumps (catégoriel) | A | wdbc | -3,03 | -0,015 | 10/10 | -0,002 | -0,001 |
| Stumps (catégoriel) | A | pima | -2,73 | -0,013 | 10/10 | 0,000 | 0,001 |
| Stumps (catégoriel) | A | australian | -4,58 | -0,024 | 10/10 | 0,000 | 0,000 |
| Stumps (catégoriel) | A | iono | -2,38 | -0,017 | 10/10 | 0,002 | 0,008 |
| Stumps (catégoriel) | A | german | -2,39 | -0,006 | 10/10 | -0,008 | -0,003 |
| Stumps (catégoriel) | A | spambase | -4,17 | -0,007 | 10/10 | -0,001 | -0,008 |
| Stumps (catégoriel) | B | wdbc | -3,13 | -0,014 | 10/10 | -0,002 | -0,004 |
| Stumps (catégoriel) | B | pima | -2,40 | -0,013 | 10/10 | 0,001 | 0,000 |
| Stumps (catégoriel) | B | australian | -4,58 | -0,024 | 10/10 | 0,000 | 0,000 |
| Stumps (catégoriel) | B | iono | -2,26 | -0,016 | 10/10 | 0,002 | 0,008 |
| Stumps (catégoriel) | B | german | -2,05 | -0,007 | 10/10 | -0,008 | -0,005 |
| Stumps (catégoriel) | B | spambase | -4,23 | -0,007 | 10/10 | -0,001 | -0,004 |
| Linéaire (gaussien) | A | wdbc | -8,36 | -0,044 | 10/10 | -0,006 | 0,000 |
| Linéaire (gaussien) | A | pima | -6,75 | -0,038 | 10/10 | -0,005 | 0,001 |
| Linéaire (gaussien) | A | australian | -9,65 | -0,050 | 10/10 | -0,010 | -0,001 |
| Linéaire (gaussien) | A | iono | -5,08 | -0,075 | 10/10 | -0,035 | -0,010 |
| Linéaire (gaussien) | A | german | -6,60 | -0,048 | 10/10 | -0,022 | -0,001 |
| Linéaire (gaussien) | A | spambase | -34,51 | -0,042 | 10/10 | -0,008 | -0,001 |
| Linéaire (gaussien) | B | wdbc | -6,55 | -0,048 | 10/10 | -0,014 | -0,004 |
| Linéaire (gaussien) | B | pima | -5,41 | -0,039 | 10/10 | -0,011 | -0,000 |
| Linéaire (gaussien) | B | australian | -8,37 | -0,052 | 10/10 | -0,016 | -0,001 |
| Linéaire (gaussien) | B | iono | -7,28 | -0,071 | 10/10 | -0,021 | -0,006 |
| Linéaire (gaussien) | B | german | -8,88 | -0,047 | 10/10 | -0,015 | -0,000 |
| Linéaire (gaussien) | B | spambase | -33,27 | -0,043 | 10/10 | -0,009 | -0,001 |

## Tableau 4 : résumé global (appris - départ, sur les 60 exécutions par famille et par borne)
| Famille | Borne | Δ KL | Δ certificat de Seeger | Certificat meilleur | Δ risque de Gibbs | Δ risque du vote |
|---|---|---|---|---|---|---|
| Stumps (catégoriel) | A | -3,21 | -0,014 | 60/60 | -0,002 | -0,000 |
| Stumps (catégoriel) | B | -3,11 | -0,013 | 60/60 | -0,001 | -0,001 |
| Linéaire (gaussien) | A | -11,82 | -0,049 | 60/60 | -0,014 | -0,002 |
| Linéaire (gaussien) | B | -11,63 | -0,050 | 60/60 | -0,014 | -0,002 |

## Tableau 5 : le prior appris lui-même, évalué sur le test
Taille de la partie prior et de l'échantillon S ; pour les stumps, nombre effectif de votants du prior appris et risques de Gibbs et du vote de P lui-même ; pour le linéaire, norme de w0 et risques du prior N(w0, I) (Gibbs) et du classifieur sign⟨w0, x⟩ (vote).

| Jeu | Partie prior | S | Stumps : votants eff. du prior | Gibbs de P | Vote de P | Linéaire : ‖w0‖ | Gibbs de N(w0, I) | Vote de w0 |
|---|---|---|---|---|---|---|---|---|
| wdbc | 79 | 319 | 33,1 | 0,156 | 0,082 | 3,25 | 0,118 | 0,053 |
| pima | 107 | 430 | 16,1 | 0,330 | 0,285 | 3,07 | 0,310 | 0,266 |
| australian | 96 | 387 | 2,3 | 0,197 | 0,147 | 3,61 | 0,214 | 0,139 |
| iono | 49 | 196 | 149,6 | 0,344 | 0,332 | 3,31 | 0,269 | 0,158 |
| german | 140 | 560 | 28,9 | 0,311 | 0,300 | 3,83 | 0,342 | 0,257 |
| spambase | 643 | 2574 | 2,4 | 0,216 | 0,213 | 6,75 | 0,154 | 0,091 |

