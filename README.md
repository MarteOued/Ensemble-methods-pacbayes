# Méthodes d'ensemble et apprentissage PAC-bayésien

Projet du cours de méthodes d'ensemble (Master 2 MALIA, Lyon 2). Ce dépôt contient tout ce qu'il faut pour relire mon
travail et refaire chaque tableau et chaque figure du rapport : le code, les résultats bruts, les tableaux générés et les
figures.

Le sujet a deux parties :

- **Partie 1, méthodes d'ensemble.** J'ai écrit moi-même le bagging (avec son erreur out-of-bag), AdaBoost et le stacking, je les
  ai vérifiés contre scikit-learn, puis je les ai comparés à des forêts aléatoires, à du gradient boosting et à des baselines sur
  25 jeux de données, avec une métrique adaptée à chaque jeu. Je me suis aussi demandé *pourquoi* le bagging ou le boosting gagne.
- **Partie 2, bornes PAC-bayésiennes.** Le code des bornes était fourni. J'ai exécuté, comparé et expliqué : les trois bornes
  (question 1), les deux algorithmes qui les minimisent (question 2), l'effet du prior (question 3), le passage des stumps aux
  classifieurs linéaires (question 4), et la comparaison avec l'apprentissage classique (question 5).

## Sommaire

1. [Organisation du dépôt](#organisation-du-dépôt)
2. [Pour démarrer](#pour-démarrer)
3. [Comment lire le code](#comment-lire-le-code)
4. [Partie 1 : ce qu'il y a dedans](#partie-1--méthodes-densemble)
5. [Partie 2 : ce qu'il y a dedans](#partie-2--bornes-pac-bayésiennes)
6. [Les résultats en bref](#les-résultats-en-bref)
7. [Conventions et reproductibilité](#conventions-et-reproductibilité)
8. [Le rapport (LaTeX)](#le-rapport-latex)
9. [Si ça ne marche pas](#si-ça-ne-marche-pas)
10. [Déclaration d'usage de l'IA](#déclaration-dusage-de-lia-et-vérification)
11. [Limites](#limites)
12. [Références](#références)

---

## Organisation du dépôt

```
.
├── README.md                    ce fichier
├── requirements.txt             les paquets Python nécessaires
│
├── code/   common/   data/      fournis avec le sujet, je n'y ai pas touché
│   ├── code/benchkit.py             chargement des jeux de données, simulateurs
│   ├── code/pacbayes_tools.py       le code PAC-bayésien du prof (partie 2)
│   ├── common/tptools.py            fonctions utilitaires (dont le chargeur de jeux)
│   └── data/                        les fichiers de données et prepdata.py
│
├── rapport/                     le rapport final (LaTeX) : rapport.tex, rapport.pdf, build_tables.py, build_mapping.py,
│                                correspondance.md (numéro de tableau/figure -> commande), generated/ (tableaux LaTeX)
│
├── partie1/                     méthodes d'ensemble
│   ├── code/                        mes scripts (7 fichiers)
│   ├── results/                     tableaux (.md) et résultats bruts (.csv)
│   └── figures/                     figures (.png)
│
└── partie2/                     bornes PAC-bayésiennes
    ├── code/                        mes scripts (5 fichiers, un par question)
    ├── results/                     tableaux (.md) et résultats bruts (.csv)
    └── figures/                     figures (.png)
```

Les trois dossiers `code/`, `common/` et `data/` à la racine sont ceux du sujet. **Mon code à moi est dans `partie1/code/`
et `partie2/code/`.** J'ai gardé les dossiers fournis intacts pour qu'on puisse toujours les comparer à l'original.

---

## Pour démarrer

**1. Installer les paquets** (Python 3 ; testé avec Python 3.9.6) :

```bash
python3 -m pip install -r requirements.txt
```

Sous macOS, la commande s'appelle `python3` (pas `python`). Toutes les commandes de ce README se lancent depuis la racine du dépôt.

Les paquets sont `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pandas` (et `joblib`, `threadpoolctl`, que scikit-learn installe de
toute façon).

**2. Vérifier que tout fonctionne**, en quelques secondes, depuis la racine du dépôt :

```bash
python3 partie1/code/datasets.py       # décrit les 25 jeux de données
python3 partie2/code/q1_bornes.py      # trace les trois bornes PAC-bayésiennes
```

Chaque script retrouve seul ses chemins : on peut le lancer depuis n'importe quel dossier. Il écrit ses résultats dans
`results/` et `figures/` de sa partie, et indique à la fin ce qu'il a produit.

**3. Tout refaire** (environ 15 minutes en tout, dont 10 pour `run_experiments.py`) :

```bash
python3 partie1/code/datasets.py
python3 partie1/code/verify.py
python3 partie1/code/run_experiments.py      # le plus long : environ 10 min
python3 partie1/code/analyse.py
python3 partie1/code/sensibilite.py          # environ 3 min
python3 partie2/code/q1_bornes.py
python3 partie2/code/q2_comparaison.py
python3 partie2/code/q3_prior.py
python3 partie2/code/q4_familles.py
python3 partie2/code/q5_classique.py         # environ 45 s
```

Si je ne veux pas attendre 10 minutes, je n'ai pas besoin de relancer `run_experiments.py` : son résultat,
`partie1/results/exp_raw.csv`, est déjà dans le dépôt, et `analyse.py` ainsi que `q5_classique.py` le lisent directement.

---

## Comment lire le code

Chaque script commence par un court en-tête (une dizaine de lignes) : ce que fait le fichier, ce que demande le sujet quand c'est
le cas, et la commande pour le lancer (sauf `ensembles.py` et `tables.py`, qui sont importés par les autres scripts et ne se lancent pas). C'est le moyen le plus rapide de comprendre un fichier avant d'en lire le code.

Ordre de lecture que je conseille :

1. `partie1/code/tables.py` : le plus court, il écrit les tableaux Markdown (utilisé partout).
2. `partie1/code/ensembles.py` : mes trois algorithmes (bagging, AdaBoost, stacking).
3. `partie1/code/datasets.py` puis `run_experiments.py` : les jeux de données et le protocole d'expérience.
4. `partie1/code/analyse.py` : les rangs, les tests statistiques, les tableaux.
5. `code/pacbayes_tools.py` (fourni) : les bornes et les algorithmes de la partie 2.
6. `partie2/code/q1_bornes.py` à `q5_classique.py`, dans l'ordre des questions.

---

## Partie 1 : méthodes d'ensemble

### Ce que j'y ai fait

- **Implémentations** (`ensembles.py`) : `MyBagging` (200 arbres complets, erreur out-of-bag), `MyAdaBoost` (AdaBoost discret, arbres
  de profondeur choisie) et `MyStacking` (arbre, kNN, régression logistique, Naive Bayes, avec un méta-modèle entraîné sur des
  prédictions out-of-fold). Elles suivent l'interface de scikit-learn (`fit`, `predict`, `decision_function`).
- **Vérification** (`verify.py`) : chacune est comparée à son équivalent scikit-learn, avec les mêmes réglages et les mêmes
  découpages.
- **Jeux de données** (`datasets.py`) : 25 jeux (les 22 des listes du cours, plus `yeast6`, `wine4` et `abalone17`, qui sont encore
  plus déséquilibrés). Un jeu est dit *déséquilibré* si sa classe minoritaire fait moins de 15 % : cela donne 15 jeux équilibrés et
  10 déséquilibrés.
- **Expériences** (`run_experiments.py`) : 12 méthodes × 25 jeux × 10 découpages stratifiés 70/30, avec quatre métriques
  (accuracy, balanced accuracy, F1 de la classe minoritaire, AUC) et les temps de calcul.
- **Analyse** (`analyse.py`) : rangs moyens, test de Friedman, différence critique de Nemenyi, tests de Wilcoxon avec correction de Holm.
- **Le « pourquoi »** (`sensibilite.py`) : effet du bruit sur les étiquettes, de la profondeur des arbres, du nombre d'arbres, et
  décomposition biais-variance sur deux problèmes simulés.

### Les scripts

| Script | Ce qu'il fait | Ce qu'il lit | Ce qu'il écrit (dans `partie1/`) | Durée |
|---|---|---|---|---|
| `tables.py` | fonctions pour écrire des tableaux Markdown | rien (bibliothèque) | rien | – |
| `ensembles.py` | mes trois algorithmes | rien (bibliothèque) | rien | – |
| `verify.py` | compare mes algorithmes à ceux de scikit-learn sur 6 jeux × 10 découpages | `data/` | `results/verify_tables.md`, `verify_raw.csv`, `verify_summary.csv`, `verify_per_dataset.csv`, `figures/verify_oob.png` | 1 à 2 min |
| `datasets.py` | décrit les 25 jeux et les classe équilibré / déséquilibré | `data/` | `results/datasets_table.md`, `datasets_table.csv`, `figures/datasets_minority.png` | quelques s |
| `run_experiments.py` | lance les 12 méthodes sur les 25 jeux, 10 découpages chacun | `data/` | `results/exp_raw.csv` (et `exp_raw_parts/`) | environ 10 min |
| `analyse.py` | rangs, tests, tableaux et diagrammes de différences critiques | `results/exp_raw.csv` | `results/analyse_tables.md`, `analyse_annexe.md`, `figures/cd/` (12 figures), `figures/rank_by_metric_imbalanced.png` | quelques s |
| `sensibilite.py` | bruit, profondeur, nombre d'arbres, biais-variance | `data/` | `results/sensibilite_tables.md`, `sens_reel.csv`, `sens_biaisvariance.csv`, `figures/sens_bruit.png`, `sens_courbes.png`, `sens_biais_variance.png` | environ 3 min |

Comment ils s'enchaînent : `datasets.py` fixe la liste des jeux et le seuil de déséquilibre ; `run_experiments.py` les utilise avec
`ensembles.py` pour produire `exp_raw.csv` ; `analyse.py` lit ce fichier. `verify.py` et `sensibilite.py` sont indépendants de
`exp_raw.csv`.

### Options utiles

`run_experiments.py` :

| Option | Effet |
|---|---|
| `--datasets wdbc yeast3` | ne traite que ces jeux |
| `--n-rep 2` | 2 découpages au lieu de 10 (pour un test rapide) |
| `--jobs 4` | nombre de jeux traités en parallèle (6 par défaut) |
| `--out chemin.csv` | fichier de sortie |
| `--force` | refait tout, même les jeux déjà sauvegardés |

Chaque jeu terminé est sauvegardé dans `results/exp_raw_parts/`, donc un calcul interrompu reprend où il s'était arrêté.
**Si je change la liste des méthodes, il faut relancer avec `--force`**, sinon il réutilise d'anciens fichiers.

Un test rapide :

```bash
python3 partie1/code/run_experiments.py --datasets wdbc yeast3 --n-rep 2 --out partie1/results/exp_test.csv
```

`sensibilite.py --skip-run` refait seulement les tableaux et les figures à partir des CSV déjà calculés.

### Les méthodes de `run_experiments.py` (les noms de la colonne `method`)

| Nom | Méthode |
|---|---|
| `tree` | arbre de décision complet |
| `logreg` | régression logistique (variables standardisées) |
| `knn` | 7 plus proches voisins (variables standardisées) |
| `bagging` | **mon** bagging, 200 arbres |
| `rf` | forêt aléatoire (scikit-learn), 200 arbres |
| `adaboost_d1` | **mon** AdaBoost, 200 tours, stumps (profondeur 1) |
| `adaboost_d3` | **mon** AdaBoost, 200 tours, profondeur 3 |
| `gb` | `GradientBoostingClassifier`, 200 arbres |
| `hgb` | `HistGradientBoostingClassifier`, 200 itérations |
| `stacking` | **mon** stacking, 4 modèles de base + méta-modèle logistique |
| `tree_d5`, `nb` | deux membres du stacking (arbre de profondeur 5, Naive Bayes), utilisés seulement pour la question « le stacking bat-il ses membres ? » |

### Le fichier de résultats bruts `exp_raw.csv`

Une ligne par (jeu, découpage, méthode), soit 3 000 lignes. Colonnes : `dataset`, `split` (0 à 9), `method`, `accuracy`, `error`,
`balanced_accuracy`, `f1_minority` (F1 de la classe rare), `auc`, `t_fit` et `t_pred` (secondes, sur un seul cœur),
`oob_error` (rempli seulement pour mon bagging).

---

## Partie 2 : bornes PAC-bayésiennes

### L'idée, en quelques lignes

On dispose d'un ensemble de classifieurs (les *votants*) : soit des **stumps** (arbres à une seule coupure), soit tous les
**classifieurs linéaires**. On choisit :

- un **prior P**, une distribution sur les votants fixée *avant* de voir l'échantillon d'apprentissage S ;
- un **posterior Q**, appris sur S.

Le **classifieur de Gibbs** tire un votant au hasard selon Q ; le **vote majoritaire** prend le signe du vote pondéré par Q. La
**KL(Q‖P)** mesure de combien Q s'est éloigné de P. Une **borne PAC-bayésienne** dit : avec probabilité 95 %, le vrai risque du classifieur
de Gibbs est au plus « risque observé sur S + pénalité qui dépend de la KL et de la taille de S ». Cette valeur, calculée sans jeu
de test, est un **certificat**. Les trois bornes étudiées sont A (linéaire, avec un paramètre λ pris dans une grille), B (McAllester)
et Seeger (la plus serrée).

Le protocole du sujet découpe chaque jeu en trois : 14 % pour le prior, 56 % pour l'échantillon S (où on apprend Q et on calcule le
certificat), 30 % pour le test (seulement pour mesurer les vrais risques à la fin).

### Ce qui est fourni et ce qui est de moi

`code/pacbayes_tools.py` contient les trois bornes, les algorithmes qui minimisent A et B, et deux fonctions `run_categorical` /
`run_gaussian` qui font tout pour un jeu et un découpage. **Je ne l'ai pas modifié.** Mes scripts l'importent.

Comme `run_categorical` ne renvoie que des résumés (pas la distribution Q elle-même), j'en ai refait les étapes une par une avec
les fonctions fournies, pour pouvoir comparer les posteriors. **Chacun de mes scripts vérifie par une assertion que ma version donne
exactement les mêmes résultats que la fonction fournie** : si je me trompais, il s'arrêterait.

### Les scripts

Un script par question du sujet. Tous écrivent dans `partie2/results/` et `partie2/figures/`.

| Script | Question du sujet | Ce qu'il fait | Ce qu'il écrit | Durée |
|---|---|---|---|---|
| `q1_bornes.py` | 1. Calculer les bornes | trace A, B et Seeger en fonction de m, de la KL et de λ ; mesure le « prix de la grille » ; vérifie par le calcul que Seeger ≤ McAllester et la formule du λ optimal | `results/q1_bornes_tables.md`, `figures/q1_bornes_m.png`, `q1_bornes_kl.png`, `q1_bornes_lambda.png` | 2 s |
| `q2_comparaison.py` | 2. Comparer les deux bornes | sur 6 jeux × 10 découpages : valeur de la borne, certificat de Seeger, KL, nombre effectif de votants, risques de Gibbs et du vote, distance entre les deux posteriors, courbes en λ | `results/q2_comparaison_tables.md`, `q2_runs.csv`, `q2_posteriors.csv`, `q2_lambda.csv`, `figures/q2_certificats.png`, `q2_lambda_wdbc.png`, `q2_lambda_tous.png`, `q2_kl_neff_lambda.png` | 2 s |
| `q3_prior.py` | 3. Impact du prior | prior de départ (uniforme, ou N(0, I)) contre prior appris sur la partie « prior », pour les stumps et les classifieurs linéaires | `results/q3_prior_tables.md`, `q3_runs.csv`, `q3_priors.csv`, `figures/q3_certificats.png`, `q3_kl.png`, `q3_risques.png` | 4 s |
| `q4_familles.py` | 4. Changer de famille | stumps contre linéaire ; ajoute un Dirac sur le meilleur stump et l'uniforme sur les k meilleurs ; courbe en k pour tester pourquoi le vote de stumps est faible | `results/q4_familles_tables.md`, `q4_runs.csv`, `q4_k_curve.csv`, `figures/q4_familles.png`, `q4_courbe_k.png` | 3 s |
| `q5_classique.py` | 5. Classique contre minimisation de borne | AdaBoost sur stumps, forêt aléatoire, régression logistique et mon stacking, contre les classifieurs PAC-bayésiens, avec les certificats | `results/q5_classique_tables.md`, `q5_classique.csv`, `q5_pacbayes.csv`, `figures/q5_erreurs.png`, `q5_nuage.png`, `q5_garanties.png` | 45 s |

### Qui dépend de qui

- `q1_bornes.py` n'utilise aucune donnée (seulement les trois fonctions `bound_*` du fichier fourni).
- `q2_comparaison.py` et `q3_prior.py` sont indépendants l'un de l'autre.
- `q4_familles.py` réutilise une fonction de `q3_prior.py` (`gau_run`, les classifieurs linéaires).
- `q5_classique.py` réutilise `q3_prior.py` et `q4_familles.py`, ainsi que la partie 1 (`ensembles.py`, `run_experiments.py`), et il lit
  `partie1/results/exp_raw.csv` pour trouver ma meilleure méthode de la partie 1.

### Réglages communs à toute la partie 2

δ = 0,05 (donc des certificats à 95 %) ; grille de λ : 1, 2, 4, …, 16 384 ; σ = 1 pour les gaussiennes ; six jeux
(`wdbc`, `pima`, `australian`, `iono`, `german`, `spambase`), choisis équilibrés et de tailles variées ; dix découpages (graines
0 à 9 de la fonction `split` du code fourni). J'ai écarté les jeux déséquilibrés parce qu'une borne sur le risque 0-1 y favorise le
classifieur « toujours majoritaire ».

### Les fichiers CSV de la partie 2

| Fichier | Une ligne = | Colonnes principales |
|---|---|---|
| `q2_runs.csv` | un jeu, un découpage, une borne (A ou B) | `bound` (valeur de la borne minimisée), `seeger` (certificat), `KL`, `emp` (risque empirique), `neff` (votants effectifs), `test_gibbs`, `test_vote`, `lam` (λ choisi pour A) |
| `q2_posteriors.csv` | un jeu, un découpage | `tv` (distance entre Q_A et Q_B), `overlap`, `same_top`, `same_top_feature`, `wmax_A`, `wmax_B` |
| `q2_lambda.csv` | un jeu, un découpage, un λ de la grille | `bound`, `seeger`, `KL`, `neff`, `test_gibbs`, `test_vote` |
| `q3_runs.csv` | un jeu, un découpage, un prior, une borne | `family` (`cat` = stumps, `gau` = linéaire), `prior` (`base` ou `learned`), plus les colonnes de `q2_runs.csv` |
| `q3_priors.csv` | un jeu, un découpage, un prior | propriétés du prior lui-même (`prior_neff`, `w0_norm`, risques du prior) |
| `q4_runs.csv` | un jeu, un découpage, un posterior | `post` (`stumps_A`, `stumps_B`, `dirac`, `top5`, `top20`, `lin_A`, `lin_B`), `seeger`, `KL`, `test_gibbs`, `test_vote` |
| `q4_k_curve.csv` | un jeu, un découpage, un k | posterior uniforme sur les k meilleurs stumps (`all` = tous les stumps) |
| `q5_classique.csv` | un jeu, un découpage, une méthode classique | `error`, `error_S_only` (entraîné sur S seul), `holdout_bound` |
| `q5_pacbayes.csv` | un jeu, un découpage, un posterior | `seeger`, `test_gibbs`, `test_vote`, `cert_vote` (= 2 × certificat, majorant du risque du vote) |

---

## Les résultats en bref

Détails, chiffres et limites dans le rapport (`rapport/rapport.pdf`).

**Partie 1**

- Les ensembles gagnent environ 4 points d'accuracy sur un arbre seul (significatif, p < 0,001). Face à la régression logistique, je ne
  mets pas de gain en évidence.
- Sur les jeux déséquilibrés, le classement des méthodes change fortement avec la métrique : l'accuracy y est trompeuse.
- Le stacking ne bat pas le meilleur de ses membres, et son coût vient surtout des prédictions out-of-fold du kNN.
- Le bruit sur les étiquettes pénalise plus le boosting que le bagging ; le bagging a besoin d'arbres profonds, le boosting non.

**Partie 2**

- La borne de Seeger est la plus serrée ; A et B conduisent presque au même posterior (le même stump principal dans 60 cas sur 60).
- Un prior appris sur la partie « prior » réduit fortement la KL et resserre le certificat dans tous les cas, sans changer beaucoup
  les risques en test.
- Minimiser une borne sur le risque de Gibbs concentre le posterior sur un seul stump : le vote de stumps n'a plus d'intérêt. Pour les
  classifieurs linéaires, le vote est nettement meilleur que Gibbs.
- Le classifieur linéaire PAC-bayésien est proche des méthodes classiques, et il fournit un certificat sans jeu de test, mais ce
  certificat est pessimiste.

---

## Conventions et reproductibilité

- **Les fichiers `.md` des dossiers `results/` sont écrits par le code**, jamais à la main : ce sont les tableaux du rapport. Les `.csv`
  sont les résultats bruts dont ils sont tirés.
- **Toutes les graines sont fixées** : les métriques se reproduisent à l'identique. Seuls les **temps de calcul** varient un peu d'une
  exécution à l'autre.
- Les découpages sont **stratifiés** (même proportion de chaque classe dans le train et le test), et la standardisation n'est jamais
  faite avant le découpage : elle utilise seulement le train.
- Quelques fichiers locaux ne sont pas dans le dépôt (voir `.gitignore`) : `exp_raw_parts/` (sauvegardes intermédiaires), les `.log` et
  les fichiers de test `*_test*`.
- Les chemins sont tous relatifs au script : le dépôt marche une fois cloné, où qu'on le mette.

---

## Le rapport (LaTeX)

Le rapport final est `rapport/rapport.tex` (et `rapport/rapport.pdf` une fois compilé). Il rassemble les deux parties, avec les tableaux
détaillés en annexe.

**Les tableaux ne sont jamais écrits à la main.** Mes scripts écrivent leurs tableaux en Markdown dans `results/`. Le script
`rapport/build_tables.py` les lit, choisit ceux du rapport par leur titre et les convertit en LaTeX dans `rapport/generated/` (un fichier
par tableau, avec légende et étiquette). `rapport.tex` les inclut avec `\input`. Si je relance une expérience, je relance ce script et le
rapport est à jour.

Pour recompiler le rapport depuis la racine du dépôt :

```bash
cd rapport
python3 build_tables.py        # régénère les tableaux LaTeX à partir des résultats
latexmk -pdf rapport.tex      # avec une distribution LaTeX (MacTeX, TeX Live...)
# ou, sans installer de distribution LaTeX : tectonic -X compile rapport.tex
```

Je l'ai compilé avec [tectonic](https://tectonic-typesetting.github.io/) 0.17 (un compilateur LaTeX autonome, un seul exécutable) : le corps du rapport fait 15 pages, plus 10 pages d'annexes (7 de tableaux et 3 de figures complémentaires), soit 25 pages au total. Le PDF compilé est `rapport/rapport.pdf`.

**Pour retrouver comment reproduire un tableau ou une figure précis du PDF**, il y a une table de correspondance générée automatiquement : [`rapport/correspondance.md`](rapport/correspondance.md). Elle donne, pour chaque numéro de tableau et de figure du rapport, le fichier source et la commande qui le produit. Elle est produite par `rapport/build_mapping.py` à partir des numéros réellement attribués par LaTeX :

```bash
cd rapport
tectonic -X compile --keep-intermediates rapport.tex    # ou latexmk -pdf rapport.tex, qui garde rapport.aux
python3 build_mapping.py                                       # écrit correspondance.md
```

Le rapport s'appuie sur les figures de `partie1/figures/` et `partie2/figures/` (chemins relatifs : il faut le compiler depuis le dossier
`rapport/` du dépôt). Il utilise `babel` en français, `booktabs` et `graphicx`, présents dans toute distribution LaTeX courante.

---

## Si ça ne marche pas

| Problème | Cause probable et solution |
|---|---|
| `ModuleNotFoundError` (numpy, sklearn, pandas…) | `python3 -m pip install -r requirements.txt` |
| `run_experiments.py` est long | normal (environ 10 min) : lancer avec `--jobs` plus grand, ou ne pas le relancer et utiliser `exp_raw.csv` fourni |
| `KeyError` dans `run_experiments.py` après avoir changé les méthodes | d'anciens fichiers de `exp_raw_parts/` sont réutilisés : ajouter `--force` |
| Mes temps de calcul ne sont pas les mêmes que dans les tableaux | normal : ils dépendent de la machine |
| Un script de la partie 2 plante sur `exp_raw.csv` | lancer d'abord `partie1/code/run_experiments.py` (seul `q5_classique.py` en a besoin) |

---

## Déclaration d'usage de l'IA et vérification

**Usage.** J'ai réalisé ce projet seule, avec l'aide de l'assistant d'IA Claude. J'ai avancé étape par étape : je commençais par demander à Claude de m'expliquer les notions que je ne comprenais pas, puis je collais la consigne ; Claude proposait un plan et le code, et je lisais, je demandais des changements et je validais avant de passer à l'étape suivante. Claude a écrit la majorité du code et m'a aidée à comprendre mes résultats. Les choix de protocole ont été discutés avec lui, et je les ai validés ou modifiés (par exemple, j'ai demandé de garder tous les jeux sans sous-échantillonnage, d'ajouter l'étude de sensibilité, de séparer le dépôt en deux dossiers, un par partie, de produire les tableaux par le code plutôt que de les recopier, de rédiger le rapport en LaTeX, de choisir les jeux de données utilisés (25 jeux, dont trois jeux très déséquilibrés ajoutés à ceux du cours), de remettre en question le seuil de déséquilibre de 15 % avant de le valider, etc.). J'ai rédigé le texte du rapport à l'aide des résultats du code, puis Claude l'a corrigé et mis en forme en LaTeX.

**Vérification.** Aucun tableau n'est recopié à la main : tous sont générés par le code. J'ai fait relancer toute la chaîne depuis un dépôt vierge : les résultats et les figures sont identiques, sauf les temps de calcul, qui varient d'environ 30 %. Les chiffres du texte ont été recalculés par des scripts de contrôle, et les fichiers fournis par le sujet n'ont pas été modifiés.

## Limites

- Je n'ai pas réglé les hyperparamètres jeu par jeu (nombre d'arbres, k du kNN, σ, etc.) : les résultats valent pour ces réglages.
- Pour les méthodes d'ensemble, le seuil de décision est celui par défaut et je n'ai pas corrigé le déséquilibre des classes, pour que
  son effet reste visible.
- Les jeux ne sont pas tous indépendants (`abalone8` et `abalone17` viennent du même fichier, comme `yeast3` et `yeast6`), alors que
  le test de Friedman le suppose.
- La partie 2 ne porte que sur six jeux équilibrés et dix découpages : je décris des tendances, sans test statistique.
- Plusieurs explications sont marquées `[hypothèse]` dans le rapport : elles sont plausibles mais je ne les ai pas testées.

---

## Références

- J. Demšar. *Statistical comparisons of classifiers over multiple data sets.* JMLR, 7, 2006.
- P. Germain, A. Lacasse, F. Laviolette, M. Marchand. *PAC-Bayesian learning of linear classifiers.* ICML, 2009.
- H. He, E. A. Garcia. *Learning from imbalanced data.* IEEE TKDE, 21, 2009.
- J. Langford, J. Shawe-Taylor. *PAC-Bayes & margins.* NIPS, 2002.
- D. McAllester. *PAC-Bayesian model averaging.* COLT, 1999.
- P. Domingos. *A unified bias-variance decomposition for zero-one and squared loss.* AAAI, 2000.
