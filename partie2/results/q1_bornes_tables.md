# Question 1 : les trois bornes PAC-bayésiennes
δ = 0,05, grille de λ : 1, 2, ..., 16384 (15 valeurs, ln(|grille|/δ) = 5,70). Risque observé = 0,1 sauf mention contraire. Les relations théoriques utilisées pour interpréter les résultats (Seeger ≤ McAllester ; formule du λ optimal) ont été vérifiées par assertions sur 180 combinaisons (m, KL, risque observé).

## Tableau 1 : valeur des trois bornes (risque observé 0,1)
| m | KL | A (grille) | λ choisi | B (McAllester) | Seeger | La plus serrée |
|---|---|---|---|---|---|---|
| 100 | 1 | 0,285 | 64 | 0,287 | 0,247 | Seeger |
| 100 | 5 | 0,344 | 128 | 0,334 | 0,293 | Seeger |
| 100 | 20 | 0,461 | 128 | 0,460 | 0,426 | Seeger |
| 100 | 50 | 0,638 | 256 | 0,629 | 0,604 | Seeger |
| 1000 | 1 | 0,158 | 256 | 0,164 | 0,143 | Seeger |
| 1000 | 5 | 0,174 | 256 | 0,178 | 0,153 | Seeger |
| 1000 | 20 | 0,214 | 512 | 0,216 | 0,184 | Seeger |
| 1000 | 50 | 0,273 | 512 | 0,269 | 0,230 | Seeger |
| 10000 | 1 | 0,119 | 1024 | 0,122 | 0,113 | Seeger |
| 10000 | 5 | 0,123 | 1024 | 0,126 | 0,116 | Seeger |
| 10000 | 20 | 0,138 | 1024 | 0,138 | 0,124 | Seeger |
| 10000 | 50 | 0,153 | 2048 | 0,154 | 0,135 | Seeger |

## Tableau 2 : le prix de la grille
Trois versions de la borne A : (a) λ optimal continu, SANS borne de l'union (valeur de référence, mais non valable car λ dépend des données) ; (b) λ optimal continu AVEC la borne de l'union sur la grille (ln|grille|) ; (c) meilleur λ de la grille (borne réellement utilisée). Coût de l'union = (b) - (a) ; coût de la discrétisation = (c) - (b) ; prix total de la grille = (c) - (a).

| m | KL | λ* théorique | (a) sans grille | (b) avec l'union | (c) grille réelle | Coût de l'union | Coût de la discrétisation | Prix total |
|---|---|---|---|---|---|---|---|---|
| 100 | 1 | 57 | 0,2413 | 0,2831 | 0,2847 | 0,0417 | 0,0017 | 0,0434 |
| 100 | 5 | 80 | 0,2999 | 0,3313 | 0,3436 | 0,0314 | 0,0123 | 0,0437 |
| 100 | 20 | 136 | 0,4391 | 0,4585 | 0,4608 | 0,0194 | 0,0023 | 0,0217 |
| 100 | 50 | 206 | 0,6148 | 0,6277 | 0,6376 | 0,0130 | 0,0098 | 0,0228 |
| 1000 | 1 | 179 | 0,1447 | 0,1579 | 0,1582 | 0,0132 | 0,0003 | 0,0135 |
| 1000 | 5 | 253 | 0,1632 | 0,1732 | 0,1738 | 0,0099 | 0,0007 | 0,0106 |
| 1000 | 20 | 429 | 0,2072 | 0,2134 | 0,2142 | 0,0061 | 0,0008 | 0,0070 |
| 1000 | 50 | 651 | 0,2628 | 0,2669 | 0,2728 | 0,0041 | 0,0059 | 0,0100 |
| 10000 | 1 | 565 | 0,1141 | 0,1183 | 0,1193 | 0,0042 | 0,0010 | 0,0052 |
| 10000 | 5 | 800 | 0,1200 | 0,1231 | 0,1233 | 0,0031 | 0,0001 | 0,0033 |
| 10000 | 20 | 1356 | 0,1339 | 0,1358 | 0,1379 | 0,0019 | 0,0021 | 0,0040 |
| 10000 | 50 | 2059 | 0,1515 | 0,1528 | 0,1528 | 0,0013 | 0,0000 | 0,0013 |

## Tableau 3 : la constante de la pénalité, ln(|grille|/δ) pour A et ln(2√m/δ) pour B
Les deux bornes ont la forme « risque + racine((KL + constante) / (2m)) » (A au λ optimal). La constante de B croît avec m ; celle de A est fixe. Elles sont égales quand 2√m = |grille|, c'est-à-dire m = 56,25. Au-delà, la constante de A est plus petite.

| m | A : ln(|grille|/δ) | B : ln(2√m/δ) | Constante la plus petite |
|---|---|---|---|
| 50 | 5,70 | 5,64 | B |
| 100 | 5,70 | 5,99 | A |
| 1000 | 5,70 | 7,14 | A |
| 10000 | 5,70 | 8,29 | A |
| 100000 | 5,70 | 9,45 | A |

## Tableau 4 : effet du risque observé (m = 1000, KL = 5)
| Risque observé | A (grille) | B (McAllester) | Seeger | La plus serrée |
|---|---|---|---|---|
| 0,00 | 0,074 | 0,078 | 0,012 | Seeger |
| 0,01 | 0,084 | 0,088 | 0,034 | Seeger |
| 0,05 | 0,124 | 0,128 | 0,091 | Seeger |
| 0,10 | 0,174 | 0,178 | 0,153 | Seeger |
| 0,20 | 0,274 | 0,278 | 0,267 | Seeger |
| 0,30 | 0,374 | 0,378 | 0,374 | A |
| 0,50 | 0,574 | 0,578 | 0,577 | A |

## Tableau 5 : la borne A selon λ (m = 1000, KL = 10, risque observé 0,1)
Meilleur λ de la grille : 256 (borne 0,1933). λ* théorique = 354. Pour comparaison, B = 0,1926 et Seeger = 0,1645.

| λ | Borne A(λ) |
|---|---|
| 1 | 15,8039 |
| 2 | 7,9521 |
| 4 | 4,0264 |
| 8 | 2,0640 |
| 16 | 1,0835 |
| 32 | 0,5947 |
| 64 | 0,3534 |
| 128 | 0,2387 |
| 256 | 0,1933 |
| 512 | 0,1947 |
| 1024 | 0,2433 |
| 2048 | 0,3637 |
| 4096 | 0,6158 |
| 8192 | 1,1259 |
| 16384 | 2,1490 |

## Balayage global (risque observé 0,1)
Sur 160 combinaisons (m de 50 à 100 000, KL de 0 à 100) : la borne la plus serrée est Seeger dans 160 cas, A dans 0 cas, B dans 0 cas ; A est plus petite que B dans 119 cas.

