<!--
NOTES DE RÉDACTION (à supprimer avant le rendu)
- Brouillon de la partie 1 du rapport, rédigé à la première personne. Format Markdown : se copie dans Word ou LaTeX.
- Budget : le rapport entier fait 15 pages au maximum ; la partie 1 doit tenir en 8 à 9 pages environ (figures et tableaux compris).
- Tous les chiffres cités viennent des fichiers de results/ (la source est indiquée sous chaque tableau). Les tableaux eux-mêmes
  ne sont pas recopiés ici : les coller depuis results/*.md, pour ne jamais retaper un chiffre à la main.
- La section 3.2 (bruit, profondeur, biais et variance) s'appuie sur l'étude de sensibilité (code/sensibilite.py). Plus aucun passage [À COMPLÉTER].
- La partie 1 a grossi avec la section 3.2 : elle approche 10 pages avec figures et tableaux. Si le total dépasse 15 pages, déplacer les tableaux détaillés (Tableaux 4, 5, 6) en annexe, ce que le sujet autorise.
- Les passages marqués [HYPOTHÈSE] sont des explications plausibles mais non testées : les garder comme telles.
- Relire chaque phrase de la section 3 en la comparant au tableau indiqué.
-->

# Partie 1 : méthodes d'ensemble, implémentation et comparaison

## 1. Rappel des algorithmes

Je note $m$ le nombre d'exemples d'entraînement, $(x_i, y_i)$ les exemples avec $y_i \in \{-1,+1\}$. Tous les ensembles ci-dessous utilisent des arbres de décision comme briques de base.

**Arbre de décision.** Un arbre complet (profondeur illimitée) a un biais faible mais une variance forte : il change beaucoup si l'on modifie un peu l'échantillon. C'est cette variance que le bagging réduit. Un arbre de profondeur 1 (un *stump*) a au contraire un biais fort et une variance faible : c'est un classifieur faible typique du boosting.

**Bagging et erreur out-of-bag (implémenté).** Pour $b = 1,\dots,B$, je tire un bootstrap $\mathcal{B}_b$ de $m$ exemples avec remise et j'entraîne un arbre $h_b$ dessus. La prédiction est le vote majoritaire $H(x) = \mathrm{sign}\big(\sum_b h_b(x)\big)$. Un bootstrap contient en moyenne une fraction $1 - 1/e \approx 63{,}2\,\%$ d'exemples distincts ; les autres sont dits *hors-sac*. L'erreur out-of-bag est
$$\widehat{R}_{\mathrm{OOB}} = \frac{1}{m}\sum_{i=1}^{m} \mathbb{1}\Big[\mathrm{sign}\Big(\sum_{b:\, i \notin \mathcal{B}_b} h_b(x_i)\Big) \neq y_i\Big],$$
c'est-à-dire que chaque exemple n'est prédit que par les arbres qui ne l'ont pas vu. C'est une estimation de l'erreur de test qui ne demande pas de jeu de validation. **Forêt aléatoire** : bagging où, à chaque coupure, seul un sous-ensemble aléatoire de $\sqrt{d}$ variables est examiné, ce qui décorrèle les arbres (scikit-learn).

**AdaBoost discret (implémenté).** Les poids initiaux sont $w_i = 1/m$. À chaque tour $t$, j'entraîne un arbre $h_t$ avec les poids $w$, puis
$$\varepsilon_t = \sum_{i:\,h_t(x_i)\neq y_i} w_i, \qquad \alpha_t = \tfrac12 \ln\frac{1-\varepsilon_t}{\varepsilon_t}, \qquad w_i \leftarrow \frac{w_i\, e^{-\alpha_t y_i h_t(x_i)}}{Z_t},$$
où $Z_t$ renormalise les poids. Le classifieur final est $H(x) = \mathrm{sign}\big(\sum_t \alpha_t h_t(x)\big)$. Les exemples mal classés voient leur poids augmenter, donc l'arbre suivant se concentre dessus : le boosting réduit surtout le biais. En contrepartie, la perte exponentielle donne un poids croissant aux exemples mal étiquetés, ce qui rend la méthode sensible au bruit. Je m'arrête si $\varepsilon_t \ge 1/2$ et je borne $\varepsilon_t$ à $10^{-10}$ s'il est nul.

**Gradient boosting (scikit-learn).** Modèle additif construit par étapes : chaque arbre ajusté est celui qui suit la direction de plus forte décroissance de la perte logistique (le gradient) du modèle courant. `HistGradientBoostingClassifier` est la même idée, mais les variables sont d'abord discrétisées en 255 classes, ce qui rend les coupures beaucoup plus rapides.

**Stacking (implémenté).** Des modèles de base de familles différentes (arbre de profondeur 5, kNN, régression logistique, Naive Bayes) sont combinés par un méta-modèle (régression logistique). Pour éviter la fuite d'information (un modèle de base prédit parfaitement ses propres données d'entraînement), le méta-modèle est entraîné sur des prédictions *out-of-fold* : je découpe le train en 5 plis ; pour chaque pli, chaque modèle de base est entraîné sur les 4 autres et prédit le pli écarté. Chaque exemple reçoit donc la prédiction d'un modèle qui ne l'a jamais vu. Les modèles de base sont ensuite réentraînés sur tout le train. Chaque modèle de base est ainsi entraîné $5+1 = 6$ fois.

**Vérification contre scikit-learn.** J'ai comparé chacune de mes trois implémentations à son équivalent (`BaggingClassifier`, `AdaBoostClassifier`, `StackingClassifier`), avec les mêmes réglages et exactement les mêmes découpages, sur 6 jeux (wdbc, pima, australian, iono, german, yeast3) et 10 découpages chacun, soit 60 cas par méthode (Tableau 2).

> **Tableau 2.** Vérification de mes implémentations. Source : `results/verify_tables.md` (tableaux 1 et 2).

- **AdaBoost sur stumps et stacking** : prédictions identiques à celles de scikit-learn sur les 60 cas (accord de 100 %), avec des poids $\alpha_t$ et des erreurs $\varepsilon_t$ identiques sur les 10 premiers tours, et des coefficients du méta-modèle identiques.
- **Bagging** : l'erreur de test est de 13,30 % pour la mienne contre 13,17 % pour scikit-learn ; l'accord sur les prédictions est de 98,02 %, au niveau de l'accord de scikit-learn avec lui-même quand on change la graine (98,10 %). Les bootstraps dépendent du hasard : on ne peut pas attendre l'identité. L'erreur OOB moyenne est de 13,71 % pour la mienne contre 13,60 % (Figure 2).
- **AdaBoost sur arbres de profondeur 3** : accord de 99,16 % (99,49 % pour scikit-learn contre lui-même). Mes $\alpha_t$ s'écartent de ceux de scikit-learn de plus de 0,01 dans 11 cas sur 60 ; scikit-learn s'écarte de lui-même dans 9 cas sur 60 quand on ne change que sa graine. L'explication est que, lorsque deux coupures sont à égalité parfaite, la graine décide laquelle est choisie, et un seul choix différent au tour 2 modifie tous les poids suivants. J'ai vérifié que ma formule de mise à jour des poids redonne exactement les $\alpha_t$ de scikit-learn lorsque ce hasard est écarté.

> **Figure 2.** `figures/verify_oob.png` : mon erreur OOB contre celle de scikit-learn (gauche) ; l'erreur OOB contre l'erreur de test (droite).

## 2. Protocole expérimental

### 2.1 Jeux de données

J'utilise 25 jeux : les 22 de `bk.REAL_SMALL` et `bk.REAL_LARGE`, auxquels j'ajoute `yeast6`, `wine4` et `abalone17`, que le chargeur fourni sait lire mais qui ne figurent pas dans ces listes. Je les ai retenus parce qu'ils sont beaucoup plus déséquilibrés (1,4 % à 3,3 % de minorité) et donnent donc plus de matière à l'étude du déséquilibre. J'ai écarté `abalone20` (26 exemples positifs) et `hayes` (132 exemples), trop petits pour donner des résultats fiables. Le sujet demande au moins 15 jeux ; en prendre 25 rend les rangs moyens et les tests plus fiables. Aucun jeu n'est sous-échantillonné, y compris `bankmarketing` (45 211 exemples).

> **Tableau 1.** Les 25 jeux : $m$, $d$, proportion de la classe minoritaire. Source : `results/datasets_table.md`. **Figure 1** : `figures/datasets_minority.png`.

**Seuil de déséquilibre.** Je dis qu'un jeu est *déséquilibré* si sa classe minoritaire représente moins de 15 % des exemples. Ce seuil est conventionnel. Je l'ai retenu parce qu'il classe comme déséquilibrés les cinq jeux cités par le sujet (yeast3, libras, satimage, pageblocks, abalone8) et parce qu'aucun jeu n'a une proportion de minorité entre 14,3 % (segmentation) et 23,5 % (vehicle) : tout seuil dans cet intervalle donne la même partition. On obtient 15 jeux équilibrés et 10 jeux déséquilibrés. Sur les jeux les plus déséquilibrés, le test ne contient que très peu d'exemples minoritaires (7 pour libras, 10 pour yeast6, 16 pour wine4, 17 pour abalone17) : les métriques centrées sur cette classe y sont très bruitées.

### 2.2 Découpages et prétraitement

Pour chaque jeu, je répète 10 fois un découpage train/test **stratifié** 70/30 (graine 0 pour la génération des découpages, puis un modèle par découpage avec la graine de son numéro de découpage de 0 à 9). La stratification conserve la proportion de chaque classe dans le train et le test, ce qui est indispensable quand il y a peu d'exemples minoritaires. Les 10 méthodes reçoivent exactement les mêmes découpages. Les variables ne sont pas standardisées d'avance : la standardisation est faite à l'intérieur des méthodes qui en ont besoin (kNN, régression logistique, modèles de base du stacking), sur le train uniquement, pour que le test n'influence jamais l'entraînement.

### 2.3 Méthodes et hyperparamètres

Les hyperparamètres sont **fixes et identiques sur tous les jeux** ; aucun réglage jeu par jeu n'est fait. Tous les ensembles ont 200 arbres ou 200 tours, pour comparer les méthodes à effort égal. Le sujet demande de comprendre quand et pourquoi les méthodes aident, pas de les optimiser, et un réglage par jeu aurait multiplié le temps de calcul. L'influence de ces choix est étudiée à part (section 3.2).

| Méthode | Réglage |
|---|---|
| Arbre seul | profondeur illimitée |
| Régression logistique | variables standardisées, $C = 1$ |
| kNN | variables standardisées, $k = 7$ (choix arbitraire) |
| Bagging (le mien) | 200 arbres complets |
| Forêt aléatoire | 200 arbres, $\sqrt d$ variables par coupure |
| AdaBoost stumps (le mien) | 200 tours, profondeur 1 |
| AdaBoost profondeur 3 (le mien) | 200 tours, profondeur 3 |
| Gradient boosting | 200 arbres, profondeur 3, pas d'apprentissage 0,1 |
| Hist. gradient boosting | 200 itérations, **sans** arrêt précoce |
| Stacking (le mien) | arbre (prof. 5), kNN ($k=7$), régression logistique, Naive Bayes ; méta-modèle logistique ; 5 plis |

Pour `HistGradientBoostingClassifier`, j'ai désactivé l'arrêt précoce : par défaut, scikit-learn ne l'active qu'au-delà de 10 000 exemples, ce qui ferait tourner la méthode différemment selon la taille du jeu. Je n'applique **aucune pondération des classes** : le sujet demande quelles méthodes souffrent du déséquilibre, donc je les laisse non corrigées pour que cet effet soit visible. Pour répondre à la question sur le stacking, j'ai aussi évalué séparément ses deux membres qui ne sont pas déjà des baselines (arbre de profondeur 5 et Naive Bayes) ; ils ne comptent pas dans les classements.

### 2.4 Métriques

Je calcule les quatre métriques sur tous les jeux, et je mets en avant celles qui conviennent à la nature du jeu :

- **accuracy** (taux de bonnes prédictions) et **AUC** pour les jeux équilibrés ;
- **balanced accuracy** (moyenne des rappels des deux classes), **F1 de la classe minoritaire** et **AUC** pour les jeux déséquilibrés, où l'accuracy est trompeuse : sur un jeu à 10 % de positifs, répondre toujours « négatif » donne 90 % d'accuracy.

La classe minoritaire n'est pas toujours celle étiquetée $+1$ : je la détermine sur le train de chaque jeu, pour que le F1 porte toujours sur la classe rare. L'AUC est calculée à partir du score continu de chaque méthode. Le seuil de décision est celui par défaut de chaque méthode ; je ne l'ajuste pas. Je mesure aussi les temps d'entraînement et de prédiction. Toutes les méthodes tournent sur un seul cœur, pour que les temps soient comparables entre méthodes.

### 2.5 Analyse statistique

Pour chaque jeu et chaque méthode, je prends la moyenne de la métrique sur les 10 découpages ; chaque jeu compte donc pour un seul point. Comparer directement des moyennes entre jeux n'aurait pas de sens (une erreur de 5 % ne signifie pas la même chose sur un jeu facile et sur un jeu difficile). Sur chaque jeu, je classe donc les méthodes (rang 1 = la meilleure, rang moyen en cas d'égalité), puis je moyenne les rangs sur les jeux (Demšar, 2006). Le test de **Friedman** teste l'hypothèse que toutes les méthodes sont équivalentes ; s'il rejette, le test post-hoc de **Nemenyi** dit quelles paires diffèrent : deux méthodes sont significativement différentes au risque de 5 % si leurs rangs moyens s'écartent de plus de
$$CD = q_{0{,}05}\sqrt{\frac{k(k+1)}{6N}}$$
avec $k$ méthodes et $N$ jeux ($CD = 2{,}71$ pour $N = 25$ jeux et $k = 10$ méthodes ; $CD = 4{,}28$ pour les $N = 10$ jeux déséquilibrés). Pour les comparaisons ciblées (une méthode contre l'arbre seul, bagging contre boosting), j'utilise le test de **Wilcoxon par paires** sur les jeux, avec la correction de **Holm** pour tenir compte du nombre de comparaisons.

## 3. Résultats

Les valeurs par jeu, avec écarts-types, sont dans `results/analyse_annexe.md` ; les rangs moyens sont dans le Tableau 3.

> **Tableau 3.** Rangs moyens et test de Friedman, par métrique (tous les jeux, jeux équilibrés, jeux déséquilibrés). Source : `results/analyse_tables.md`, section 2. **Figure 3** : diagramme de différences critiques, accuracy, tous les jeux (`figures/cd/cd_tous_accuracy.png`).

Le test de Friedman rejette l'équivalence des 10 méthodes pour les quatre métriques (p < 0,001 sur les 25 jeux ; p = 0,002 en accuracy et en balanced accuracy, 0,011 en F1, < 0,001 en AUC sur les 10 jeux déséquilibrés). En accuracy sur les 25 jeux, la forêt aléatoire a le meilleur rang moyen (3,48), l'arbre seul le plus mauvais (9,36).

### 3.1 Réponses aux questions du sujet

**(1) Quel gain par rapport à un arbre seul, et à un modèle d'une autre famille ?**

> **Tableau 4.** Gain de chaque ensemble par rapport à l'arbre seul, à la régression logistique et au kNN. Source : `results/analyse_tables.md`, section 4.

- **Par rapport à l'arbre seul, le gain est net.** Les sept ensembles gagnent entre +3,85 points (bagging) et +4,82 points (forêt aléatoire) d'accuracy, sur 25 jeux sur 25 pour presque tous (23 sur 25 pour AdaBoost sur stumps), avec p < 0,001 ; en AUC le gain est d'environ 14 à 15 points. Au sens de Nemenyi (CD = 2,71), tous les modèles sauf le kNN sont significativement meilleurs que l'arbre seul en accuracy.
- **Par rapport à la régression logistique, je ne mets pas en évidence de gain** : en accuracy, les écarts vont de +1,13 à +2,09 points mais aucun n'est significatif (p corrigés entre 0,106 et 1,000), et la régression logistique gagne sur beaucoup de jeux (12 sur 25 contre le gradient boosting, par exemple). Un modèle linéaire simple est très compétitif.
- **Par rapport au kNN, les ensembles gagnent** : +2,36 à +3,33 points d'accuracy, p ≤ 0,024 pour les sept ensembles ; au sens de Nemenyi, seuls la forêt aléatoire et le stacking sont significativement meilleurs que le kNN.
- **Exception notable sur les jeux déséquilibrés.** En balanced accuracy sur les 10 jeux déséquilibrés, aucun ensemble ne bat significativement l'arbre seul (écarts de −1,75 à +2,18 points, p corrigés de 1,000). [HYPOTHÈSE, non testée : les ensembles moyennent beaucoup d'arbres et rendent des scores plus lissés, donc dépassent moins souvent le seuil par défaut pour la classe rare, alors qu'un arbre complet la prédit plus facilement. Voir aussi le point (4).]

**(2) Bagging ou boosting ?**

> **Tableau 5.** Meilleur du type bagging contre meilleur du type boosting, par jeu ; comparaisons directes. Source : `results/analyse_tables.md`, section 5.

- **Mon bagging et mes deux AdaBoost ne se distinguent pas** sur les 25 jeux : −0,65 point (contre les stumps) et −0,57 point (contre les arbres de profondeur 3), avec p corrigé de 0,937 dans les deux cas.
- **Le résultat dépend de la métrique et de la méthode de référence.** En accuracy, la forêt aléatoire bat le gradient boosting (+0,57 point, 20 victoires sur 25, p = 0,014) mais pas le Hist. gradient boosting (+0,55 point, p = 0,921). Sur les 10 jeux déséquilibrés en balanced accuracy, c'est l'inverse : le Hist. gradient boosting bat la forêt aléatoire sur **les 10 jeux** (écart de 3,93 points, p = 0,010).
- **Par jeu, en prenant le meilleur de chaque famille**, le boosting l'emporte sur les 10 jeux déséquilibrés ; sur les 15 jeux équilibrés, c'est 7 victoires pour le boosting, 7 pour le bagging et une égalité. Cette comparaison favorise légèrement le boosting, qui compte 4 candidats contre 2. Selon la taille : 7 victoires du boosting sur 13 jeux avec $m<1000$, 7 sur 9 (et 1 égalité) pour $1000 \le m < 5000$, 3 sur 3 pour $m \ge 5000$ : le boosting semble avantagé sur les grands jeux, mais il n'y en a que 3.
- Les écarts les plus grands sont sur libras (+17,04 points pour le boosting), balance (+10,32), et satimage (+5,06). Sur les trois jeux où tout est proche de 50 % de balanced accuracy (abalone17, wine4, abalone8), les écarts d'un ou deux points ne sont pas interprétables.

→ L'explication de ces constats (bruit, profondeur, nombre d'arbres, biais et variance) fait l'objet de la section 3.2. Sur les jeux réels, je ne contrôle ni le bruit ni la structure du problème, donc je n'y fais pas d'hypothèse : je l'étudie par des expériences où je ne change qu'une chose à la fois.

**(3) Le stacking bat-il le meilleur de ses membres ? À quel coût ?**

> **Tableau 6.** Stacking et ses quatre membres, par jeu ; comparaisons ; coût. Source : `results/analyse_tables.md`, section 6.

- **Non, il ne bat pas le meilleur membre.** Face au meilleur membre de chaque jeu (choisi après coup, donc dans un sens favorable aux membres), le stacking perd en moyenne 2,13 points (10 victoires, 1 égalité, 14 défaites ; p = 0,170, non significatif).
- **Mais il évite d'avoir à choisir.** Face au membre qui a le meilleur rang moyen (la régression logistique), il gagne +3,07 points (17 victoires, 1 égalité, 7 défaites ; p = 0,010). Sur les 15 jeux équilibrés, il ne perd jamais plus de 1,1 point par rapport au meilleur membre (et le dépasse sur 8 d'entre eux, avec une égalité sur vehicle).
- **Il s'effondre sur certains jeux déséquilibrés** : −11,68 points sur yeast6, −14,54 sur abalone8, −9,87 sur abalone17 par rapport au meilleur membre ; sur abalone17 et wine4, sa balanced accuracy est de 50,0 % et son F1 sur la classe minoritaire est de 0 sur les 10 découpages : il prédit toujours la classe majoritaire. [HYPOTHÈSE : le méta-modèle logistique, entraîné sur des probabilités de membres très déséquilibrées, dépasse rarement le seuil de 0,5 pour la classe rare.]
- **Coût.** Chaque membre est entraîné 6 fois. Le temps d'entraînement du stacking est de 8,0 fois la somme des temps d'entraînement de ses membres (médiane sur les jeux), et de 40,9 fois sur `bankmarketing`. Ce rapport dépasse largement 6 parce que le kNN, qui ne coûte presque rien à entraîner, coûte à la prédiction, et que le stacking doit prédire chaque pli : sur `bankmarketing`, les prédictions out-of-fold du kNN représentent 2,30 s sur 2,61 s de temps cumulé pour les membres sur les 5 plis (88 %). Rapporté à la somme (entraînement + prédiction) des membres, le coût du stacking est de 3,7 fois en médiane (2,0 fois sur `bankmarketing`).

**(4) Jeux déséquilibrés : quelles méthodes souffrent, et que change la métrique ?**

> **Tableau 7.** Classement des méthodes sur les jeux déséquilibrés selon la métrique ; accuracy contre classifieur « toujours majoritaire ». Source : `results/analyse_tables.md`, section 3. **Figure 4** : `figures/rank_by_metric_imbalanced.png`. **Figure 5** : `figures/cd/cd_desequilibres_balanced_accuracy.png`.

- **L'accuracy est trompeuse.** Sur trois jeux (abalone17, wine4, abalone8), la meilleure accuracy obtenue par les 10 méthodes (98,6 %, 96,7 %, 86,4 %) est exactement celle d'un classifieur qui répond toujours la classe majoritaire ; or leurs meilleures balanced accuracy ne sont que de 53,4 %, 55,6 % et 56,4 %, c'est-à-dire proches du hasard.
- **Le classement change fortement avec la métrique.** Entre l'accuracy et la balanced accuracy, le stacking passe du rang 1 au rang 9, la forêt aléatoire du rang 2 au rang 8, alors que l'arbre seul passe du rang 10 au rang 3 et que le gradient boosting passe du rang 8 au rang 3. La corrélation de Kendall entre les classements selon l'accuracy et la balanced accuracy est de −0,05 : ce sont deux classements quasiment indépendants. Les classements selon la balanced accuracy et le F1 sont, eux, très proches (0,93).
- **Les méthodes qui souffrent** (mauvais rang en balanced accuracy et en F1) : la régression logistique (dernière, rang moyen 8,60 en balanced accuracy, avec 50,7 % sur satimage), le stacking et la forêt aléatoire. **Les méthodes qui résistent** : le Hist. gradient boosting (1er en balanced accuracy et en F1, rang moyen 2,70), AdaBoost sur arbres de profondeur 3 et l'arbre seul. Au sens de Nemenyi (CD = 4,28), seules deux différences sont significatives en balanced accuracy : Hist. gradient boosting et AdaBoost profondeur 3 contre la régression logistique.
- **La forêt aléatoire est 1re en AUC (rang moyen 2,80) mais 8e en balanced accuracy.** [HYPOTHÈSE, non testée : ses scores ordonnent bien les exemples, mais le seuil de décision par défaut (0,5) fait rarement prédire la classe rare. Pour la tester, il faudrait ajuster le seuil sur un jeu de validation.]

### 3.2 Pourquoi ? Bruit, profondeur, nombre d'arbres, biais et variance

**Protocole.** Les expériences utilisent les mêmes méthodes et réglages qu'en 2.3 (200 arbres ou tours). (a) *Bruit* : sur 6 jeux réels équilibrés (australian, german, iono, pima, vehicle, wdbc, 10 découpages 70/30 chacun), j'inverse au hasard 0, 5, 10 ou 20 % des étiquettes d'**entraînement** ; l'erreur est mesurée sur un test aux étiquettes propres. Je me limite aux jeux équilibrés pour que l'effet du bruit ne se mélange pas à celui du déséquilibre. (b) *Profondeur* : bagging et AdaBoost avec des arbres de profondeur 1, 2, 3, 5 (et complets pour le bagging). (c) *Nombre d'arbres* : j'évalue les $k$ premiers arbres d'un même modèle à 200 arbres. (d) *Biais et variance* : sur deux problèmes simulés dont je connais la vraie classe, `additive` (somme de fonctions d'une seule variable, qu'une somme de stumps peut représenter) et `checkerboard` (damier, où la classe dépend d'une *combinaison* de deux variables, qu'aucun stump ne peut capter), j'entraîne 30 modèles sur 30 échantillons indépendants de 500 exemples et je les compare à la vraie classe sur 5 000 points communs. Pour la perte 0-1, le *biais* d'un point vaut 1 si la prédiction majoritaire des 30 modèles est fausse ; la *variance* est la fréquence à laquelle un modèle s'écarte de cette prédiction majoritaire ; l'erreur est égale au biais plus la variance nette (Domingos, 2000).

> **Tableaux 9 à 12** et **Figures 6 à 8**. Source : `results/sensibilite_tables.md` (tableaux S1 à S4) ; `figures/sens_bruit.png`, `figures/sens_courbes.png`, `figures/sens_biais_variance.png`.

- **Sans bruit ajouté, il n'y a pas de gagnant sur ces six jeux** : les ensembles ont tous une erreur comprise entre 12,4 % (forêt aléatoire) et 13,3 % (bagging), contre 18,3 % pour l'arbre seul (Tableau 9). Cela concorde avec l'égalité 7-7 de la section 3.1 (2) sur les jeux équilibrés.
- **Le bruit pénalise davantage le boosting que le bagging.** Quand on passe de 0 % à 20 % de bruit, l'erreur augmente de +2,91 points pour la forêt aléatoire, +4,17 pour mon bagging, +5,27 pour AdaBoost sur stumps, +8,58 pour le Hist. gradient boosting, +9,88 pour AdaBoost sur profondeur 3 et +12,80 pour l'arbre seul. Sur chacun des six jeux, la hausse de mon bagging est inférieure à celle d'AdaBoost sur profondeur 3 (par exemple +4,8 contre +13,2 points sur iono). Le boosting sur stumps est nettement moins touché que sur profondeur 3 : la sensibilité au bruit augmente avec la capacité des arbres de base.
- **Le nombre d'arbres n'a pas le même effet** (Tableau 11, Figure 7). Avec 20 % de bruit, l'erreur du bagging diminue de façon régulière quand on ajoute des arbres (32,03 % avec 1 arbre, 17,43 % avec 200) et se stabilise : le bagging ne sur-apprend pas avec le nombre d'arbres. L'erreur d'AdaBoost sur profondeur 3, elle, *augmente* avec le nombre de tours (21,46 % après 5 tours, 23,01 % après 200) ; celle d'AdaBoost sur stumps passe par un minimum (17,45 % à 50 tours) puis remonte (18,36 % à 200). Sans bruit, tout converge vers 13 % à partir de 50 à 100 arbres. [HYPOTHÈSE, mécanisme non testé directement : c'est l'explication classique de la sensibilité d'AdaBoost au bruit, à savoir que sa perte exponentielle donne un poids croissant aux exemples mal étiquetés, que des arbres assez profonds finissent par ajuster. Mes résultats sont compatibles avec cette explication sans la démontrer.]
- **Le bagging a besoin d'arbres profonds, le boosting non** (Tableau 10). Sans bruit, l'erreur du bagging baisse quand la profondeur augmente (20,02 % à profondeur 1, 16,12 % à 2, 14,72 % à 3, 13,39 % à 5, 13,26 % pour des arbres complets), alors qu'AdaBoost est quasiment insensible à la profondeur (13,08 % à 13,66 %) et atteint 13,10 % avec de simples stumps. Un bagging de stumps reste donc bloqué à 20 %. Avec 20 % de bruit, c'est AdaBoost sur stumps qui est le meilleur de ses variantes (18,36 %, contre 23,01 % à profondeur 3), et la profondeur 5 est préférable aux arbres complets pour le bagging (15,12 % contre 17,43 %) : limiter la profondeur aide aussi le bagging quand les étiquettes sont bruitées.
- **Biais et variance** (Tableau 12, Figure 8).
  - *Le bagging réduit la variance mais pas le biais.* Sur `additive` sans bruit, la variance nette tombe de 5,69 % (arbre seul) à 1,56 % (bagging), mais le biais ne baisse pas (7,22 % pour l'arbre, 8,20 % pour le bagging). Avec 20 % de bruit, la variance de l'arbre atteint 20,07 % sur `additive` et 23,87 % sur `checkerboard` ; le bagging la ramène à 7,01 % et 16,49 %, alors qu'AdaBoost sur profondeur 3 la laisse élevée (15,55 % et 19,68 %).
  - *AdaBoost sur stumps réduit le biais quand la structure est additive.* Sur `additive` sans bruit, c'est la meilleure méthode (erreur 5,77 %, biais 2,94 %), loin devant le bagging (9,76 %, biais 8,20 %) et la forêt aléatoire (9,06 %).
  - *Mais il ne peut pas représenter une interaction.* Sur `checkerboard`, AdaBoost sur stumps fait 49,78 % d'erreur, soit le niveau du hasard, avec un biais de 48,02 % : une somme de fonctions d'une seule variable ne peut pas capter un damier. Les autres méthodes y font entre 3,2 % (bagging) et 5,7 % (forêt aléatoire).

**Synthèse : pourquoi bagging ou boosting ?** Les quatre facteurs cités par le sujet se lisent ainsi. *Biais et variance* : le bagging est une méthode de réduction de variance, qui fonctionne donc avec des arbres profonds (faible biais, forte variance) ; le boosting réduit le biais et fonctionne avec des arbres simples. *Profondeur* : le boosting n'a pas besoin d'arbres profonds, le bagging si. *Bruit* : le bruit fait surtout monter la variance, que le bagging réduit mais que le boosting laisse monter, d'où l'avantage du bagging quand les étiquettes sont bruitées. *Structure du problème* : le boosting de stumps excelle sur un problème additif mais échoue sur un problème d'interactions. *Taille du jeu* : je ne trouve pas d'effet net (section 3.1 (2)) ; le boosting l'emporte sur les trois plus grands jeux, mais trois jeux ne suffisent pas à conclure.

**Ce que cette étude n'explique pas.** Elle ne porte que sur des jeux équilibrés : elle n'explique donc pas l'avantage du boosting sur les 10 jeux déséquilibrés observé en 3.1 (2). Je ne sais pas non plus quel niveau de bruit contiennent mes jeux réels : sans bruit ajouté, bagging et boosting y sont à égalité (Tableau 9).

**Limites.** Seulement 6 jeux réels : je décris des tendances et je ne fais pas de test statistique sur 6 points. Les deux jeux simulés sont des cas extrêmes (parfaitement additif, parfaitement interactif) qui illustrent un mécanisme sans représenter des jeux réalistes. Je ne teste que du bruit symétrique sur les étiquettes. Sur `checkerboard`, la forêt aléatoire est pénalisée : avec seulement deux variables, scikit-learn n'en examine qu'une par coupure ($\lfloor\sqrt{2}\rfloor = 1$), ce qui la gêne pour une interaction ; son résultat (5,7 %) n'est donc pas représentatif de la méthode.

### 3.3 Coût de calcul

> **Tableau 8.** Temps d'entraînement et de prédiction (un seul cœur). Source : `results/analyse_tables.md`, section 7.

La médiane des temps d'entraînement va de 0,001 s (kNN) à 0,41 s (mon bagging) ; sur `bankmarketing`, mon bagging demande 31,6 s, contre 3,4 s pour la forêt aléatoire et 1,8 s pour le Hist. gradient boosting. Ma version du bagging est écrite en Python pur (une boucle sur les arbres) et n'est pas optimisée ; ces temps ne comparent donc pas équitablement les algorithmes, seulement ces implémentations.

### 3.4 Limites

- **Les jeux ne sont pas tous indépendants** : abalone8 et abalone17 viennent du même fichier, comme yeast3 et yeast6, alors que le test de Friedman suppose des jeux indépendants.
- **Peu de jeux déséquilibrés** (10) : la différence critique est grande (4,28), donc peu de différences ressortent comme significatives, même quand elles existent.
- **Bruit sur les très petites classes** : 7 à 17 exemples minoritaires dans le test pour libras, yeast6, wine4 et abalone17.
- **Pas de réglage par jeu** et **seuil de décision par défaut** : les résultats sont valables pour ces réglages et peuvent changer si l'on règle les hyperparamètres ou le seuil, en particulier sur les jeux déséquilibrés.
- **Les temps** dépendent de ma machine et de mes implémentations ; ils sont indicatifs.

## 4. Reproductibilité (à reprendre dans le README)

| Élément | Commande | Fichiers produits |
|---|---|---|
| Tableau 1, Figure 1 | `python code/datasets.py` | `results/datasets_table.md`, `figures/datasets_minority.png` |
| Tableau 2, Figure 2 | `python code/verify.py` | `results/verify_tables.md`, `figures/verify_oob.png` |
| Expériences (résultats bruts) | `python code/run_experiments.py` | `results/exp_raw.csv` |
| Tableaux 3 à 8, Figures 3 à 5 | `python code/analyse.py` | `results/analyse_tables.md`, `results/analyse_annexe.md`, `figures/cd/`, `figures/rank_by_metric_imbalanced.png` |
| Tableaux 9 à 12, Figures 6 à 8 | `python code/sensibilite.py` | `results/sensibilite_tables.md`, `figures/sens_bruit.png`, `figures/sens_courbes.png`, `figures/sens_biais_variance.png` |

Les fichiers à lire pour comprendre le code : `code/ensembles.py` (mes trois implémentations), `code/datasets.py`, `code/run_experiments.py` (le protocole), `code/analyse.py` (l'analyse statistique).
