"""run_experiments.py : les expériences de la partie 1 (toutes les méthodes sur tous les jeux)

Sujet : comparer mes bagging, AdaBoost et stacking avec RF, gradient boosting, HistGB et des baselines sur au
moins 15 jeux, avec une métrique adaptée, et décrire le protocole.
Ce que fait le fichier : pour chacun de mes 25 jeux et chacun de 10 découpages stratifiés 70/30 (graines 0 à 9),
j'entraîne 10 méthodes (tree, logreg, knn, bagging, rf, adaboost_d1, adaboost_d3, gb, hgb, stacking ; plus
tree_d5 et nb, membres du stacking) avec les mêmes réglages pour tous les jeux (200 arbres), puis je mesure
accuracy, balanced accuracy, F1 de la minorité, AUC et temps. Une ligne par (jeu, découpage, méthode) dans
results/exp_raw.csv. Tout tourne sur un seul cœur (temps comparables) ; chaque jeu est sauvegardé dès qu'il est
fini, donc une interruption ne fait refaire que les jeux manquants. Le protocole complet et la raison de chaque
choix sont décrits dans le rapport (section 2).

Utilisé par : analyse.py (lit exp_raw.csv ; importe METHODS et LABELS).

Usage : python3 run_experiments.py                       (25 jeux, 10 découpages)
        python3 run_experiments.py --datasets wdbc yeast3 --n-rep 2 --out ../results/exp_test.csv   (test rapide)
"""
import os
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import datasets as ds
from ensembles import MyBagging, MyAdaBoost, MyStacking

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

N_REP = 10          # découpages train/test par jeu
TEST_SIZE = 0.3
N_TREES = 200       # arbres / tours de tous les ensembles

# nom -> fonction (graine) -> modèle neuf. Les graines varient d'un découpage à l'autre.
METHODS = {
    "tree": lambda s: DecisionTreeClassifier(random_state=s),
    "logreg": lambda s: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "knn": lambda s: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    "bagging": lambda s: MyBagging(N_TREES, random_state=s),
    "rf": lambda s: RandomForestClassifier(N_TREES, n_jobs=1, random_state=s),
    "adaboost_d1": lambda s: MyAdaBoost(N_TREES, max_depth=1, random_state=s),
    "adaboost_d3": lambda s: MyAdaBoost(N_TREES, max_depth=3, random_state=s),
    "gb": lambda s: GradientBoostingClassifier(n_estimators=N_TREES, max_depth=3, learning_rate=0.1, random_state=s),
    "hgb": lambda s: HistGradientBoostingClassifier(max_iter=N_TREES, early_stopping=False, random_state=s),
    "stacking": lambda s: MyStacking(random_state=s),
    # Les deux membres du stacking qui ne sont pas déjà des baselines (même réglage que dans
    # ensembles.default_base_models). Ils servent UNIQUEMENT à la question « le stacking bat-il le meilleur
    # de ses membres ? » : ils ne participent pas aux classements des 10 méthodes principales.
    "tree_d5": lambda s: DecisionTreeClassifier(max_depth=5, random_state=s),
    "nb": lambda s: GaussianNB(),
}
MAIN_METHODS = ["tree", "logreg", "knn", "bagging", "rf", "adaboost_d1", "adaboost_d3", "gb", "hgb", "stacking"]
MEMBER_METHODS = ["tree_d5", "nb"]                     # + knn et logreg, déjà dans les méthodes principales
LABELS = {"tree": "Arbre seul", "logreg": "Régression logistique", "knn": "kNN (k=7)", "bagging": "Bagging (le mien)",
          "rf": "Forêt aléatoire", "adaboost_d1": "AdaBoost stumps (le mien)", "adaboost_d3": "AdaBoost prof. 3 (le mien)",
          "gb": "Gradient boosting", "hgb": "Hist. gradient boosting", "stacking": "Stacking (le mien)",
          "tree_d5": "Arbre prof. 5 (membre)", "nb": "Naive Bayes (membre)"}

# Les gros jeux d'abord, pour qu'ils démarrent tôt quand on parallélise.
BIG_FIRST = ["bankmarketing", "spambase", "satimage", "pageblocks", "abalone8", "abalone17", "splice"]


def score_of(model, X):
    """Score continu d'un modèle (pour l'AUC) : decision_function si elle existe, sinon P(y=+1)."""
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def run_dataset(name, n_rep, parts_dir, force=False):
    """Toutes les méthodes sur tous les découpages d'un jeu ; sauvegarde results/exp_raw_parts/<jeu>.csv."""
    part = os.path.join(parts_dir, f"{name}.csv")
    if os.path.exists(part) and not force:
        print(f"[{name}] déjà fait, ignoré", flush=True)
        return
    warnings.simplefilter("ignore")
    t0 = time.perf_counter()
    X, y = ds.load(name)
    rows = []
    for r, (tr, te) in enumerate(StratifiedShuffleSplit(n_rep, test_size=TEST_SIZE, random_state=0).split(X, y)):
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        minority = 1 if np.mean(ytr == 1) < 0.5 else -1          # classe rare, déterminée sur le train
        for method, make in METHODS.items():
            model = make(r)
            t = time.perf_counter(); model.fit(Xtr, ytr); t_fit = time.perf_counter() - t
            t = time.perf_counter(); pred = model.predict(Xte); t_pred = time.perf_counter() - t
            acc = float(np.mean(pred == yte))
            rows.append(dict(dataset=name, split=r, method=method, accuracy=acc, error=1 - acc,
                             balanced_accuracy=balanced_accuracy_score(yte, pred),
                             f1_minority=f1_score(yte, pred, pos_label=minority, zero_division=0),
                             auc=roc_auc_score(yte, score_of(model, Xte)),
                             t_fit=t_fit, t_pred=t_pred,
                             oob_error=getattr(model, "oob_error_", np.nan)))
    os.makedirs(parts_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(part, index=False)
    print(f"[{name}] fini en {time.perf_counter() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=ds.DATASETS)
    ap.add_argument("--n-rep", type=int, default=N_REP)
    ap.add_argument("--jobs", type=int, default=6, help="nombre de jeux traités en parallèle")
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp_raw.csv"))
    ap.add_argument("--force", action="store_true", help="refaire même les jeux déjà sauvegardés")
    a = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
    parts_dir = os.path.splitext(a.out)[0] + "_parts"
    names = sorted(a.datasets, key=lambda n: (BIG_FIRST.index(n) if n in BIG_FIRST else len(BIG_FIRST)))
    from joblib import Parallel, delayed
    t0 = time.perf_counter()
    Parallel(n_jobs=a.jobs)(delayed(run_dataset)(n, a.n_rep, parts_dir, a.force) for n in names)

    raw = pd.concat([pd.read_csv(os.path.join(parts_dir, f"{n}.csv")) for n in names], ignore_index=True)
    raw.to_csv(a.out, index=False)
    print(f"\n{len(raw)} lignes écrites dans {os.path.relpath(a.out)} ({time.perf_counter() - t0:.0f} s)")
    g = raw.groupby("method")[["accuracy", "balanced_accuracy", "f1_minority", "auc", "t_fit"]].mean()
    print(g.loc[list(METHODS)].round(4).to_string())


if __name__ == "__main__":
    main()
