"""datasets.py : les jeux de données utilisés, leur description et le seuil de déséquilibre

Sujet : au moins 15 jeux, décrits dans un tableau (m, d, proportion de la classe minoritaire), avec un seuil
de déséquilibre.
Ce que fait le fichier : je fixe ici mes 25 jeux (les 22 de bk.REAL_SMALL / REAL_LARGE + yeast6, wine4, abalone17)
et mon seuil (déséquilibré = classe minoritaire < 15 %), puis je calcule m, d et la part de minorité, et j'écris
le tableau (Markdown, CSV) et une figure. Les autres scripts importent DATASETS et is_imbalanced d'ici, pour que
ces choix ne soient écrits qu'à un seul endroit. La justification du seuil et des jeux écartés est dans le rapport.

Usage : python3 datasets.py   (écrit results/datasets_table.md/.csv et figures/datasets_minority.png)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))   # code fourni par le prof (benchkit.py)
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
