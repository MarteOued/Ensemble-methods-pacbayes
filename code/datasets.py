"""datasets.py : les jeux de données que j'utilise, leur description et leur classement (équilibré / déséquilibré)

CE QUE DEMANDE LE SUJET
    Partie 1, « Jeux de données » : utiliser au moins 15 jeux parmi ceux du cours, les décrire dans un
    tableau (nombre d'exemples m, dimension d, proportion de la classe minoritaire), et indiquer le
    seuil à partir duquel je qualifie un jeu de déséquilibré. Le sujet précise aussi que le choix de
    la métrique dépend de la nature du jeu (équilibré ou non), et cite yeast3, libras, satimage,
    pageblocks et abalone8 comme fortement déséquilibrés.

CE QUE J'AI FAIT DANS CE FICHIER
    - J'ai fixé la liste de mes 25 jeux : les 22 de bk.REAL_SMALL et bk.REAL_LARGE, plus trois jeux que
      prepdata.py sait charger mais qui ne figurent pas dans ces listes (yeast6, wine4, abalone17).
      Je les ai ajoutés parce qu'ils sont beaucoup plus déséquilibrés (entre 1,4 % et 3,3 % de
      minorité), donc plus parlants pour étudier l'effet du déséquilibre. J'ai écarté abalone20
      (26 positifs seulement) et hayes (132 exemples) : trop petits pour des résultats fiables.
    - J'ai fixé le seuil : un jeu est déséquilibré si sa classe minoritaire représente moins de 15 %
      des exemples. Ce seuil est conventionnel. Je l'ai choisi parce qu'il classe comme déséquilibrés
      les cinq jeux cités par le sujet, et parce qu'aucun jeu ne se trouve entre 14,3 % (segmentation)
      et 23,5 % (vehicle) : tout seuil dans cet intervalle donne exactement la même partition.
    - Je calcule m, d, la proportion de minorité, et aussi le nombre absolu d'exemples minoritaires
      et celui attendu dans un test à 30 %. Ce dernier chiffre me sert à savoir quels jeux donneront
      des résultats très bruités (par exemple abalone17 : seulement 17 positifs dans le test).
    - Je génère le tableau du rapport (en Markdown) et une figure des proportions de minorité.

À QUOI ÇA ME SERT ENSUITE
    run_experiments.py et analyse.py importent d'ici la liste des jeux (DATASETS), le seuil et la
    fonction is_imbalanced : je ne recopie jamais ces choix ailleurs. Le groupe d'un jeu décide aussi
    de la métrique que je mets en avant : accuracy et AUC pour un jeu équilibré ; balanced accuracy,
    F1 de la classe minoritaire et AUC pour un jeu déséquilibré.

Usage : python datasets.py   (écrit results/datasets_table.md, results/datasets_table.csv et
                              figures/datasets_minority.png)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import benchkit as bk
from tables import md_table, pct

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIGURES = os.path.join(HERE, "..", "figures")

# Les 22 jeux des listes du cours + 3 jeux plus fortement déséquilibrés que prepdata.py sait charger.
EXTRA = ["yeast6", "wine4", "abalone17"]
DATASETS = bk.REAL_SMALL + bk.REAL_LARGE + EXTRA

IMBALANCE_THRESHOLD = 0.15      # minorité < 15 % des exemples => jeu déséquilibré
TEST_SIZE = 0.3                 # même valeur que dans les expériences (hold-out 70/30)

# Jeux que le sujet cite comme fortement déséquilibrés (je vérifie que mon seuil les classe bien ainsi).
SUJET_IMBALANCED = ["yeast3", "libras", "satimage", "pageblocks", "abalone8"]


def minority_rate(y):
    """Proportion de la classe minoritaire (étiquettes dans {-1,+1})."""
    return float(min(np.mean(y == 1), np.mean(y == -1)))


def is_imbalanced(rate):
    return rate < IMBALANCE_THRESHOLD


def load(name):
    """Charge un jeu : X (m x d), y dans {-1,+1}. Les variables ne sont PAS standardisées."""
    return bk.real_dataset(name)


def describe():
    """Tableau (DataFrame) décrivant chacun de mes jeux."""
    rows = []
    for name in DATASETS:
        X, y = load(name)
        rate = minority_rate(y)
        n_min = int(round(rate * len(y)))
        rows.append(dict(dataset=name, m=X.shape[0], d=X.shape[1], minority_rate=rate, n_minority=n_min,
                         n_minority_test=int(round(n_min * TEST_SIZE)), imbalanced=is_imbalanced(rate)))
    return pd.DataFrame(rows)


def write_markdown(df, path):
    n_imb = int(df.imbalanced.sum())
    out = ["# Jeux de données utilisés",
           f"{len(df)} jeux : {len(df) - n_imb} équilibrés et {n_imb} déséquilibrés. Un jeu est dit "
           f"**déséquilibré** si sa classe minoritaire représente moins de {pct(IMBALANCE_THRESHOLD, 0)} "
           "des exemples. m = nombre d'exemples, d = dimension (nombre de variables après encodage), "
           f"« Minoritaires » = nombre d'exemples de la classe minoritaire, « Dans le test » = nombre "
           f"attendu dans un test de {int(TEST_SIZE * 100)} %.", "",
           "Métrique mise en avant : accuracy (erreur) et AUC pour les jeux équilibrés ; balanced "
           "accuracy, F1 de la classe minoritaire et AUC pour les jeux déséquilibrés.\n"]
    header = ["Jeu", "m", "d", "Minorité", "Minoritaires", "Dans le test", "Groupe"]
    for title, flag in [("Jeux déséquilibrés", True), ("Jeux équilibrés", False)]:
        sub = df[df.imbalanced == flag].sort_values("minority_rate")
        rows = [[r.dataset, r.m, r.d, pct(r.minority_rate, 1), r.n_minority, r.n_minority_test,
                 "déséquilibré" if flag else "équilibré"] for r in sub.itertuples()]
        out += [f"## {title} ({len(sub)})", md_table(header, rows), ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def plot_minority(df, path):
    d = df.sort_values("minority_rate")
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = ["tab:red" if f else "tab:blue" for f in d.imbalanced]
    ax.barh(d.dataset, 100 * d.minority_rate, color=colors)
    ax.axvline(100 * IMBALANCE_THRESHOLD, color="k", ls="--", lw=1)
    ax.text(100 * IMBALANCE_THRESHOLD + 0.5, 0.2, f"seuil {pct(IMBALANCE_THRESHOLD, 0)}", fontsize=9)
    ax.set(xlabel="proportion de la classe minoritaire (%)",
           title="Déséquilibre des jeux de données (rouge : déséquilibré)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    df = describe()
    # Contrôle : mon seuil doit classer comme déséquilibrés les jeux cités par le sujet.
    flagged = set(df[df.imbalanced].dataset)
    missing = [n for n in SUJET_IMBALANCED if n not in flagged]
    assert not missing, f"jeux cités par le sujet mais classés équilibrés avec mon seuil : {missing}"
    df.to_csv(os.path.join(RESULTS, "datasets_table.csv"), index=False)
    write_markdown(df, os.path.join(RESULTS, "datasets_table.md"))
    plot_minority(df, os.path.join(FIGURES, "datasets_minority.png"))
    print(f"{len(df)} jeux ; déséquilibrés (< {IMBALANCE_THRESHOLD:.0%}) : {int(df.imbalanced.sum())}")
    print("Contrôle : les 5 jeux cités par le sujet sont bien classés déséquilibrés.")
    print("Écrit : results/datasets_table.md, results/datasets_table.csv, figures/datasets_minority.png")


if __name__ == "__main__":
    main()
