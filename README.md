# Projet : méthodes d'ensemble et apprentissage PAC-bayésien (M2 MALIA)

Ce dépôt contient le code, les résultats, les figures et le rapport des deux parties du projet.
Chaque tableau et chaque figure du rapport se reproduit avec une commande indiquée ci-dessous.

## Organisation du dépôt

```
.
├── README.md
├── requirements.txt
├── code/   common/   data/     fournis avec le sujet, non modifiés (benchkit.py, pacbayes_tools.py, jeux de données)
├── partie1/                    méthodes d'ensemble : implémentation et comparaison
│   ├── code/                   scripts de la partie 1
│   ├── results/                tableaux (.md) et résultats bruts (.csv)
│   ├── figures/                figures (.png)
│   └── rapport/                texte de la partie 1
└── partie2/                    bornes PAC-bayésiennes et leur minimisation
    ├── code/   results/   figures/   rapport/
```

Les dossiers `code/`, `common/` et `data/` de la racine sont ceux du sujet. Le code que j'ai écrit est dans
`partie1/code/` et `partie2/code/`. Chaque script retrouve seul les chemins dont il a besoin : on peut le lancer
depuis n'importe quel dossier.

## Installation

Python 3 (testé avec 3.9.6) et les paquets de `requirements.txt` :

```bash
pip install -r requirements.txt
```

## Partie 1 : reproduire les tableaux et les figures

Les commandes se lancent depuis la racine du dépôt, dans cet ordre (chaque script en lit un précédent) :

| Contenu du rapport | Commande | Fichiers produits (dans `partie1/`) | Durée indicative |
|---|---|---|---|
| Tableau 1 (jeux de données), Figure 1 | `python partie1/code/datasets.py` | `results/datasets_table.md`, `figures/datasets_minority.png` | quelques secondes |
| Tableau 2 (vérification contre scikit-learn), Figure 2 | `python partie1/code/verify.py` | `results/verify_tables.md`, `figures/verify_oob.png` | 1 à 2 min |
| Expériences : 25 jeux, 10 découpages, 12 méthodes | `python partie1/code/run_experiments.py` | `results/exp_raw.csv` | environ 10 min (6 cœurs) |
| Tableaux 3 à 8, Figures 3 à 5 | `python partie1/code/analyse.py` | `results/analyse_tables.md`, `results/analyse_annexe.md`, `figures/cd/`, `figures/rank_by_metric_imbalanced.png` | quelques secondes |
| Tableaux 9 à 12, Figures 6 à 8 (étude de sensibilité) | `python partie1/code/sensibilite.py` | `results/sensibilite_tables.md`, `figures/sens_bruit.png`, `figures/sens_courbes.png`, `figures/sens_biais_variance.png` | environ 3 min |

Remarques :

- `results/exp_raw.csv` (résultats bruts des expériences) est fourni : on peut donc relancer `analyse.py` sans
  refaire les 10 minutes de calcul.
- Les métriques sont reproductibles à l'identique (toutes les graines sont fixées). Les **temps de calcul**
  varient légèrement d'une exécution à l'autre.
- `run_experiments.py --jobs N` règle le nombre de jeux traités en parallèle (6 par défaut). Chaque jeu terminé est
  sauvegardé dans `partie1/results/exp_raw_parts/`, donc un calcul interrompu reprend où il s'est arrêté.
- Un test rapide (2 jeux, 2 découpages) : `python partie1/code/run_experiments.py --datasets wdbc yeast3 --n-rep 2 --out partie1/results/exp_test.csv`.

Fichiers de code de la partie 1 :

| Fichier | Rôle |
|---|---|
| `ensembles.py` | mes implémentations du bagging (avec erreur out-of-bag), d'AdaBoost discret et du stacking |
| `verify.py` | vérification de ces implémentations contre scikit-learn |
| `datasets.py` | liste des 25 jeux, seuil de déséquilibre (15 %), tableau de description |
| `run_experiments.py` | protocole expérimental : méthodes, hyperparamètres, métriques |
| `analyse.py` | rangs moyens, tests de Friedman, Nemenyi, Wilcoxon, tableaux et figures |
| `sensibilite.py` | bruit, profondeur, nombre d'arbres, biais et variance |
| `tables.py` | écriture des tableaux en Markdown (utilisé par les autres scripts) |

Le texte de la partie 1 du rapport est dans `partie1/rapport/partie1.md`.

## Partie 2 : bornes PAC-bayésiennes

À venir. Le code fourni par le sujet est `code/pacbayes_tools.py` ; mes scripts, résultats, figures et le texte du
rapport iront dans `partie2/`.
