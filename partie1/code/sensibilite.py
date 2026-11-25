"""sensibilite.py : pourquoi le bagging ou le boosting l'emporte-t-il ? (question 2 de la partie 1)

Sujet : « Bagging ou boosting : sur quels jeux chacun l'emporte-t-il, et pourquoi (taille, bruit, profondeur,
biais et variance) ? » Sur des jeux réels je ne contrôle pas ces facteurs ; ici je change une seule chose à la fois.
Ce que fait le fichier : quatre petites expériences, avec les méthodes de run_experiments.py :
  1. bruit : j'inverse 0, 5, 10, 20 % des étiquettes d'entraînement sur 6 jeux équilibrés ;
  2. profondeur : AdaBoost et bagging avec des arbres de profondeur croissante, avec et sans bruit ;
  3. nombre d'arbres : courbes d'erreur de test en fonction du nombre d'arbres ou de tours ;
  4. biais et variance (décomposition de Domingos, perte 0-1) sur deux jeux simulés dont je connais la vraie
     réponse : additif (bk.additive) et damier (bk.checkerboard).
Il écrit results/sensibilite_tables.md, les CSV et les figures sens_*.png.
Limites assumées : 6 jeux réels (tendances, pas de test), deux jeux simulés extrêmes, bruit symétrique seulement.

Usage : python3 sensibilite.py            (calcule puis écrit tableaux et figures)
        python3 sensibilite.py --skip-run (refait seulement tableaux et figures à partir des CSV)
        python3 sensibilite.py --n-rep 2 --R 4 --out-tag test   (test rapide)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))   # code fourni par le prof (benchkit.py)
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

import benchkit as bk
import datasets as ds
from ensembles import MyBagging, MyAdaBoost
from run_experiments import METHODS, LABELS
from tables import md_table, num

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIGURES = os.path.join(HERE, "..", "figures")

REAL = ["wdbc", "pima", "australian", "iono", "german", "vehicle"]    # jeux réels équilibrés, de tailles variées
NOISE = [0.0, 0.05, 0.10, 0.20]            # part des étiquettes d'entraînement inversées
NOISE_METHODS = ["tree", "bagging", "rf", "adaboost_d1", "adaboost_d3", "hgb"]
CURVE_NOISE = [0.0, 0.20]
CURVE_METHODS = ["bagging", "adaboost_d1", "adaboost_d3"]
CHECK = [1, 2, 5, 10, 20, 50, 100, 200]    # nombres d'arbres / de tours auxquels je lis l'erreur
DEPTH_NOISE = [0.0, 0.20]
DEPTHS = [1, 2, 3, 5]
N_TREES = 200
N_REP = 10

GENS = {"additive": lambda m, seed: bk.additive(m, d=10, n_inf=5, seed=seed),
        "checkerboard": lambda m, seed: bk.checkerboard(m, d=2, k=4, seed=seed)}
BV_METHODS = ["tree", "bagging", "rf", "adaboost_d1", "adaboost_d3", "hgb"]
BV_NOISE = [0.0, 0.20]
BV_R, BV_M, BV_MTEST = 30, 500, 5000


def sgn(s):
    return np.where(s > 0, 1, -1)


def curve(model, Xte, yte):
    """Erreur de test après 1, 2, 5, ... arbres (bagging) ou tours (AdaBoost), pour un même modèle."""
    if hasattr(model, "staged_decision_function"):
        errs = [float(np.mean(sgn(s) != yte)) for s in model.staged_decision_function(Xte)]
    else:
        votes = np.cumsum([MyBagging._vote(t, Xte) for t in model.estimators_], axis=0)
        errs = [float(np.mean(sgn(v) != yte)) for v in votes]
    return {T: errs[min(T, len(errs)) - 1] for T in CHECK}


# ----------------------------------------------------------------------------------------------------
# Expériences 1, 2, 3 : jeux réels avec bruit
# ----------------------------------------------------------------------------------------------------
def run_real(name, n_rep):
    warnings.simplefilter("ignore")
    X, y = ds.load(name)
    rows = []
    for r, (tr, te) in enumerate(StratifiedShuffleSplit(n_rep, test_size=0.3, random_state=0).split(X, y)):
        Xtr, Xte, yte = X[tr], X[te], y[te]
        for noise in NOISE:
            ytr = bk.flip_labels(y[tr], noise, seed=r) if noise > 0 else y[tr]
            base = dict(dataset=name, split=r, noise=noise)
            for meth in NOISE_METHODS:
                model = METHODS[meth](r).fit(Xtr, ytr)
                rows.append(dict(base, kind="final", method=meth, depth="", n_iter=N_TREES,
                                 error=float(np.mean(model.predict(Xte) != yte))))
                if noise in CURVE_NOISE and meth in CURVE_METHODS:
                    for T, e in curve(model, Xte, yte).items():
                        rows.append(dict(base, kind="curve", method=meth, depth="", n_iter=T, error=e))
            if noise in DEPTH_NOISE:
                for depth in DEPTHS + ["complet"]:
                    d = None if depth == "complet" else depth
                    models = {"bagging": MyBagging(N_TREES, max_depth=d, random_state=r)}
                    if d is not None:
                        models["adaboost"] = MyAdaBoost(N_TREES, max_depth=d, random_state=r)
                    for fam, model in models.items():
                        model.fit(Xtr, ytr)
                        rows.append(dict(base, kind="depth", method=fam, depth=str(depth), n_iter=N_TREES,
                                         error=float(np.mean(model.predict(Xte) != yte))))
    print(f"[{name}] fini", flush=True)
    return rows


# ----------------------------------------------------------------------------------------------------
# Expérience 4 : biais et variance sur jeux simulés
# ----------------------------------------------------------------------------------------------------
def run_biasvar(gen_name, noise, R):
    warnings.simplefilter("ignore")
    gen = GENS[gen_name]
    Xte, yte = gen(BV_MTEST, seed=10_000)
    preds = {m: np.empty((R, BV_MTEST)) for m in BV_METHODS}
    for r in range(R):
        X, y = gen(BV_M, seed=r)
        if noise > 0:
            y = bk.flip_labels(y, noise, seed=r)
        for m in BV_METHODS:
            preds[m][r] = METHODS[m](r).fit(X, y).predict(Xte)
    rows = []
    for m, P in preds.items():
        main = sgn(P.mean(axis=0))                         # prédiction majoritaire des R modèles, en chaque point
        bias_x = (main != yte).astype(float)
        var_x = (P != main).mean(axis=0)
        err = float((P != yte).mean())
        net = float(((1 - 2 * bias_x) * var_x).mean())
        assert abs(err - (bias_x.mean() + net)) < 1e-9, "identité biais + variance nette = erreur non vérifiée"
        rows.append(dict(generator=gen_name, noise=noise, method=m, error=err, bias=float(bias_x.mean()),
                         variance=float(var_x.mean()), variance_net=net))
    print(f"[{gen_name}, bruit {noise:.0%}] fini", flush=True)
    return rows


# ----------------------------------------------------------------------------------------------------
# Tableaux et figures
# ----------------------------------------------------------------------------------------------------
def pt(x, nd=2):
    return num(100 * x, nd)


def build_outputs(real, bv, tag):
    sfx = f"_{tag}" if tag else ""
    out = ["# Étude de sensibilité : pourquoi le bagging ou le boosting l'emporte-t-il ?",
           f"{real.dataset.nunique()} jeux réels équilibrés ({', '.join(sorted(real.dataset.unique()))}), "
           f"{real.split.nunique()} découpages 70/30 chacun ; erreur de test mesurée sur des étiquettes propres, en %. "
           "Les étiquettes d'ENTRAÎNEMENT sont inversées avec la probabilité indiquée.\n"]

    # --- Tableau S1 : bruit ----------------------------------------------------------------------
    fin = real[real.kind == "final"]
    per = fin.groupby(["dataset", "noise", "method"]).error.mean().reset_index()      # moyenne sur les découpages
    mean = per.groupby(["noise", "method"]).error.mean().unstack("noise")             # puis sur les jeux
    rows = []
    for m in NOISE_METHODS:
        rows.append([LABELS[m]] + [pt(mean.loc[m, n]) for n in NOISE] + [f"{100 * (mean.loc[m, 0.20] - mean.loc[m, 0.0]):+.2f}".replace(".", ",")])
    out += ["## Tableau S1 : effet du bruit sur les étiquettes (erreur de test moyenne, %)",
            md_table(["Méthode"] + [f"bruit {int(n * 100)} %" for n in NOISE] + ["Hausse 0 → 20 % (pts)"], rows), ""]
    delta = per.pivot_table(index=["dataset", "method"], columns="noise", values="error")
    delta = (delta[0.20] - delta[0.0]).unstack("method")[NOISE_METHODS]
    rows = [[d] + [f"{100 * delta.loc[d, m]:+.1f}".replace(".", ",") for m in NOISE_METHODS] for d in delta.index]
    rows.append(["**moyenne**"] + [f"{100 * delta[m].mean():+.1f}".replace(".", ",") for m in NOISE_METHODS])
    out += ["### Hausse de l'erreur entre 0 % et 20 % de bruit, par jeu (points)",
            md_table(["Jeu"] + [LABELS[m] for m in NOISE_METHODS], rows), ""]

    # --- Tableau S2 : profondeur -----------------------------------------------------------------
    dep = real[real.kind == "depth"].groupby(["dataset", "noise", "method", "depth"]).error.mean().groupby(
        ["noise", "method", "depth"]).mean()
    rows = []
    for d in [str(x) for x in DEPTHS] + ["complet"]:
        row = [d]
        for n in DEPTH_NOISE:
            for fam in ("adaboost", "bagging"):
                row.append(pt(dep[(n, fam, d)]) if (n, fam, d) in dep.index else "–")
        rows.append(row)
    out += ["## Tableau S2 : profondeur des arbres de base (erreur de test moyenne, %)",
            "200 arbres ou tours dans tous les cas. « complet » = profondeur illimitée (pas défini pour AdaBoost).\n",
            md_table(["Profondeur", "AdaBoost, 0 % bruit", "Bagging, 0 % bruit", "AdaBoost, 20 % bruit", "Bagging, 20 % bruit"], rows), ""]

    # --- Tableau S3 : courbes ----------------------------------------------------------------------
    cur = real[real.kind == "curve"].groupby(["dataset", "noise", "method", "n_iter"]).error.mean().groupby(
        ["noise", "method", "n_iter"]).mean()
    rows = []
    for n in CURVE_NOISE:
        for m in CURVE_METHODS:
            rows.append([f"{int(n * 100)} %", LABELS[m]] + [pt(cur[(n, m, T)]) for T in CHECK])
    out += ["## Tableau S3 : erreur de test selon le nombre d'arbres (bagging) ou de tours (AdaBoost), %",
            md_table(["Bruit", "Méthode"] + [str(T) for T in CHECK], rows), ""]

    # --- Tableau S4 : biais et variance ---------------------------------------------------------------
    rows = []
    for g in GENS:
        for n in BV_NOISE:
            for m in BV_METHODS:
                r = bv[(bv.generator == g) & (bv.noise == n) & (bv.method == m)].iloc[0]
                rows.append([g, f"{int(n * 100)} %", LABELS[m], pt(r.error), pt(r.bias), pt(r.variance_net)])
    out += ["## Tableau S4 : biais et variance sur jeux simulés (%)",
            f"Pour chaque ligne : {BV_R} modèles entraînés sur {BV_R} échantillons indépendants de {BV_M} exemples, "
            f"évalués sur {BV_MTEST} points communs. Erreur = biais + variance nette. Le bruit est ajouté aux "
            "étiquettes d'entraînement seulement ; le test est propre.\n",
            md_table(["Jeu simulé", "Bruit", "Méthode", "Erreur", "Biais", "Variance nette"], rows), ""]
    with open(os.path.join(RESULTS, f"sensibilite_tables{sfx}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    # --- Figure S1 : bruit ----------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for m in NOISE_METHODS:
        y = [100 * mean.loc[m, n] for n in NOISE]
        ax[0].plot([100 * n for n in NOISE], y, "o-", label=LABELS[m])
        ax[1].plot([100 * n for n in NOISE], [v - y[0] for v in y], "o-", label=LABELS[m])
    ax[0].set(xlabel="étiquettes inversées dans le train (%)", ylabel="erreur de test (%)", title="Erreur de test")
    ax[1].set(xlabel="étiquettes inversées dans le train (%)", ylabel="hausse par rapport à 0 % (points)", title="Dégradation due au bruit")
    ax[0].legend(fontsize=7)
    fig.suptitle(f"Effet du bruit sur les étiquettes (moyenne de {real.dataset.nunique()} jeux réels équilibrés)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, f"sens_bruit{sfx}.png"), dpi=150); plt.close(fig)

    # --- Figure S2 : courbes ---------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, n in enumerate(CURVE_NOISE):
        for m in CURVE_METHODS:
            ax[i].plot(CHECK, [100 * cur[(n, m, T)] for T in CHECK], "o-", label=LABELS[m])
        ax[i].set(xscale="log", xlabel="nombre d'arbres / de tours", title=f"{int(n * 100)} % de bruit")
    ax[0].set_ylabel("erreur de test (%)"); ax[0].legend(fontsize=8)
    fig.suptitle("Erreur de test selon le nombre d'arbres (bagging) ou de tours (AdaBoost)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, f"sens_courbes{sfx}.png"), dpi=150); plt.close(fig)

    # --- Figure S3 : biais et variance -----------------------------------------------------------------------
    fig, axes = plt.subplots(len(GENS), len(BV_NOISE), figsize=(11, 7), sharey=True)
    for i, g in enumerate(GENS):
        for j, n in enumerate(BV_NOISE):
            ax = axes[i, j]
            sub = bv[(bv.generator == g) & (bv.noise == n)].set_index("method").loc[BV_METHODS]
            x = np.arange(len(BV_METHODS))
            ax.bar(x, 100 * sub.bias, color="tab:red", label="biais")
            ax.bar(x, 100 * sub.variance_net, bottom=100 * sub.bias, color="tab:blue", label="variance nette")
            ax.set_xticks(x); ax.set_xticklabels([LABELS[m].replace(" (le mien)", "") for m in BV_METHODS], rotation=30, ha="right", fontsize=7)
            ax.set_title(f"{g}, {int(n * 100)} % de bruit", fontsize=10)
            for xi, e in zip(x, 100 * sub.error):
                ax.text(xi, e + 0.5, f"{e:.1f}", ha="center", fontsize=7)
    axes[0, 0].legend(fontsize=8); axes[0, 0].set_ylabel("erreur = biais + variance nette (%)")
    axes[1, 0].set_ylabel("erreur = biais + variance nette (%)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, f"sens_biais_variance{sfx}.png"), dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rep", type=int, default=N_REP)
    ap.add_argument("--R", type=int, default=BV_R, help="nombre d'échantillons pour la décomposition biais-variance")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--skip-run", action="store_true")
    ap.add_argument("--out-tag", default="")
    a = ap.parse_args()
    sfx = f"_{a.out_tag}" if a.out_tag else ""
    os.makedirs(RESULTS, exist_ok=True); os.makedirs(FIGURES, exist_ok=True)
    p_real, p_bv = os.path.join(RESULTS, f"sens_reel{sfx}.csv"), os.path.join(RESULTS, f"sens_biaisvariance{sfx}.csv")
    if not a.skip_run:
        os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
        from joblib import Parallel, delayed
        t0 = time.perf_counter()
        res = Parallel(n_jobs=a.jobs)(delayed(run_real)(n, a.n_rep) for n in REAL)
        pd.DataFrame([r for rows in res for r in rows]).to_csv(p_real, index=False)
        res = Parallel(n_jobs=a.jobs)(delayed(run_biasvar)(g, n, a.R) for g in GENS for n in BV_NOISE)
        pd.DataFrame([r for rows in res for r in rows]).to_csv(p_bv, index=False)
        print(f"calcul terminé en {time.perf_counter() - t0:.0f} s")
    build_outputs(pd.read_csv(p_real), pd.read_csv(p_bv), a.out_tag)
    print(f"Écrit : results/sensibilite_tables{sfx}.md et figures/sens_*{sfx}.png")


if __name__ == "__main__":
    main()
