"""verify.py : je vérifie que mes implémentations donnent les mêmes résultats que scikit-learn

Sujet : « Vérifiez chaque implémentation par rapport à son équivalent scikit-learn ».
Ce que fait le fichier : sur 6 jeux et 10 découpages, j'entraîne ma version et celle de scikit-learn (mêmes
réglages, mêmes découpages) et je compare l'erreur de test, l'erreur out-of-bag (bagging), les poids alpha
(AdaBoost) et les coefficients du méta-modèle (stacking), ainsi que la part de prédictions identiques.
Note : sklearn utilise alpha = ln((1-eps)/eps), soit 2 fois le mien, ce qui ne change pas le vote ; les écarts
restants viennent des égalités entre coupures d'arbres (voir colonnes « _ref » : sklearn diverge de lui-même).

Usage : python3 verify.py   (écrit results/verify_tables.md, results/verify_*.csv, figures/verify_oob.png)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))   # code fourni par le prof (benchkit.py)
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

import benchkit as bk
from ensembles import MyBagging, MyAdaBoost, MyStacking, default_base_models
from tables import md_table, pct, num

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIGURES = os.path.join(HERE, "..", "figures")

DATASETS = ["wdbc", "pima", "australian", "iono", "german", "yeast3"]   # 5 équilibrés + 1 déséquilibré
N_REP = 10          # nombre de découpages train/test par jeu
TEST_SIZE = 0.3
N_TREES = 200       # arbres du bagging et tours d'AdaBoost
N_ROUNDS_CHECK = 10  # nombre de premiers tours d'AdaBoost dont je compare alpha et epsilon
SEED_REF = 1000      # décalage de graine pour la référence « sklearn contre sklearn »
ALPHA_TOL = 0.01     # au-delà, je dis que les alpha des premiers tours ont « divergé »


def adaboost_sklearn(depth, seed):
    return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=depth), n_estimators=N_TREES,
                              algorithm="SAMME", random_state=seed)


def run_one(name, X, y, r, tr, te):
    """Les 4 comparaisons (bagging, adaboost d=1, adaboost d=3, stacking) pour un découpage."""
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    rows = []

    def row(method, mine, sk, ref=None, **extra):
        """ref = scikit-learn relancé avec une AUTRE graine : il me donne le niveau d'écart « normal »
        entre deux exécutions de la même méthode (voir le commentaire de main)."""
        pm, ps = mine.predict(Xte), sk.predict(Xte)
        if ref is not None:
            pr = ref.predict(Xte)
            extra.update(err_ref=float(np.mean(pr != yte)), agreement_ref=float(np.mean(ps == pr)))
        rows.append(dict(dataset=name, split=r, method=method,
                         err_mine=float(np.mean(pm != yte)), err_sk=float(np.mean(ps != yte)),
                         agreement=float(np.mean(pm == ps)), **extra))

    # --- bagging (arbres complets) ---
    mine = MyBagging(N_TREES, random_state=r).fit(Xtr, ytr)
    sk = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=N_TREES, oob_score=True,
                           random_state=r).fit(Xtr, ytr)
    ref = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=N_TREES,
                            random_state=r + SEED_REF).fit(Xtr, ytr)
    row("bagging", mine, sk, ref, oob_mine=mine.oob_error_, oob_sk=1.0 - sk.oob_score_)

    # --- AdaBoost : stumps puis arbres de profondeur 3 ---
    for d in (1, 3):
        mine = MyAdaBoost(N_TREES, max_depth=d, random_state=r).fit(Xtr, ytr)
        sk = adaboost_sklearn(d, r).fit(Xtr, ytr)
        ref = adaboost_sklearn(d, r + SEED_REF).fit(Xtr, ytr)
        n = min(N_ROUNDS_CHECK, len(mine.alphas_), len(sk.estimator_weights_), len(ref.estimator_weights_))
        row(f"adaboost_d{d}", mine, sk, ref,
            alpha_diff=float(np.max(np.abs(2 * mine.alphas_[:n] - sk.estimator_weights_[:n]))),
            alpha_diff_ref=float(np.max(np.abs(ref.estimator_weights_[:n] - sk.estimator_weights_[:n]))),
            eps_diff=float(np.max(np.abs(np.array(mine.errors_[:n]) - sk.estimator_errors_[:n]))))

    # --- stacking : mêmes modèles de base, mêmes plis, même méta-modèle ---
    mine = MyStacking(random_state=r).fit(Xtr, ytr)
    sk = StackingClassifier(estimators=default_base_models(r), final_estimator=LogisticRegression(max_iter=1000),
                            cv=StratifiedKFold(5, shuffle=True, random_state=r),
                            stack_method="predict_proba").fit(Xtr, ytr)
    row("stacking", mine, sk,
        coef_diff=float(np.max(np.abs(mine.meta_.coef_ - sk.final_estimator_.coef_))))
    return rows


METHODS = [("bagging", "Bagging (arbres complets)"), ("adaboost_d1", "AdaBoost, stumps"),
           ("adaboost_d3", "AdaBoost, profondeur 3"), ("stacking", "Stacking")]


def verdict(s):
    """Verdict automatique : « identique » si les prédictions coïncident partout ; sinon « proche » si
    mon accord avec sklearn est au niveau de l'accord de sklearn avec lui-même (autre graine)."""
    if s["accord_moy"] >= 0.9999:
        return "identique"
    if s["accord_moy"] >= s["accord_ref_moy"] - 0.01:
        return "proche, écart du même ordre que sklearn contre lui-même"
    return "à examiner"


def write_markdown(raw, summary, per_ds, path):
    """Les tableaux du rapport pour cette étape, écrits en Markdown depuis les résultats."""
    n_cas = int(summary["n_cas"].iloc[0])
    out = ["# Vérification de mes implémentations contre scikit-learn",
           f"Moyennes sur {raw.dataset.nunique()} jeux ({', '.join(DATASETS)}) x {N_REP} découpages "
           f"train/test, soit {n_cas} cas par méthode. Mêmes découpages, mêmes graines, mêmes réglages "
           f"pour ma version et celle de scikit-learn ; {N_TREES} arbres / tours.\n",
           "## Tableau 1 : erreur de test et accord des prédictions",
           "« Accord » = part des exemples de test où les deux versions prédisent la même classe. "
           "« Réf. » = même mesure entre scikit-learn et scikit-learn relancé avec une autre graine "
           "(niveau d'écart normal entre deux exécutions).\n"]
    rows = [[lab, pct(summary.loc[k, "err_mine"]), pct(summary.loc[k, "err_sk"]),
             pct(summary.loc[k, "accord_moy"]), pct(summary.loc[k, "accord_ref_moy"]),
             verdict(summary.loc[k])] for k, lab in METHODS]
    out.append(md_table(["Méthode", "Erreur moi", "Erreur sklearn", "Accord", "Accord réf.", "Verdict"], rows))

    out += ["\n## Tableau 2 : contrôles internes",
            "Écart maximal sur les 10 premiers tours d'AdaBoost (alpha de sklearn = 2 fois le mien, "
            "j'ai donc comparé 2 x mon alpha) et sur les coefficients du méta-modèle du stacking. "
            f"« Cas divergents » = nombre de cas sur {n_cas} où alpha s'écarte de plus de {ALPHA_TOL} ; "
            "entre parenthèses, le même nombre pour sklearn contre lui-même.\n"]
    g = raw.groupby("method")
    rows = []
    for k, lab in METHODS:
        if k.startswith("adaboost"):
            rows.append([lab, num(g["alpha_diff"].max()[k]), num(g["eps_diff"].max()[k]), "–",
                         f"{int(summary.loc[k, 'alpha_diverge'])} ({int(summary.loc[k, 'alpha_diverge_ref'])})"])
        elif k == "stacking":
            rows.append([lab, "–", "–", num(g["coef_diff"].max()[k]), "–"])
    out.append(md_table(["Méthode", "Écart max alpha", "Écart max epsilon", "Écart max coefficients",
                         "Cas divergents"], rows))

    out += ["\n## Tableau 3 : erreur out-of-bag du bagging",
            "L'erreur OOB est calculée sur le train, sans jeu de validation.\n"]
    b = raw[raw.method == "bagging"].groupby("dataset")[["oob_mine", "oob_sk", "err_mine"]].mean()
    rows = [[d, pct(r.oob_mine), pct(r.oob_sk), pct(r.err_mine)] for d, r in b.iterrows()]
    rows.append(["**moyenne**", pct(b.oob_mine.mean()), pct(b.oob_sk.mean()), pct(b.err_mine.mean())])
    out.append(md_table(["Jeu", "OOB moi", "OOB sklearn", "Erreur de test (moi)"], rows))

    out += ["\n## Tableau 4 : détail par jeu", ""]
    names = dict(METHODS)
    rows = [[d, names[m], pct(r.err_mine), pct(r.err_sk), pct(r.agreement)] for (d, m), r in per_ds.iterrows()]
    out.append(md_table(["Jeu", "Méthode", "Erreur moi", "Erreur sklearn", "Accord"], rows))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    warnings.simplefilter("ignore")
    rows = []
    for name in DATASETS:
        X, y = bk.real_dataset(name)
        sss = StratifiedShuffleSplit(n_splits=N_REP, test_size=TEST_SIZE, random_state=0)
        for r, (tr, te) in enumerate(sss.split(X, y)):
            rows += run_one(name, X, y, r, tr, te)
        print(f"{name:11s} fait", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(RESULTS, "verify_raw.csv"), index=False)

    # Résumé par méthode : écart d'erreur apparié (moi - sklearn), accord, détails propres à chaque méthode.
    raw["diff"] = raw["err_mine"] - raw["err_sk"]
    g = raw.groupby("method")
    summary = pd.DataFrame({
        "err_mine": g["err_mine"].mean(), "err_sk": g["err_sk"].mean(),
        "diff_moy": g["diff"].mean(), "diff_std": g["diff"].std(),
        "accord_moy": g["agreement"].mean(), "accord_min": g["agreement"].min(),
        "accord_ref_moy": g["agreement_ref"].mean(),          # sklearn(graine s) contre sklearn(graine s')
        "oob_mine": g["oob_mine"].mean(), "oob_sk": g["oob_sk"].mean(),
        "alpha_diverge": g["alpha_diff"].apply(lambda v: int((v > ALPHA_TOL).sum()) if v.notna().any() else np.nan),         # moi contre sklearn
        "alpha_diverge_ref": g["alpha_diff_ref"].apply(lambda v: int((v > ALPHA_TOL).sum()) if v.notna().any() else np.nan),  # sklearn contre sklearn
        "n_cas": g["split"].count(),
        "eps_diff_max": g["eps_diff"].max(), "coef_diff_max": g["coef_diff"].max()})
    summary.to_csv(os.path.join(RESULTS, "verify_summary.csv"))
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n=== Résumé (moyennes sur", len(DATASETS), "jeux x", N_REP, "découpages) ===")
    print(summary.round(4).to_string())

    per_ds = raw.groupby(["dataset", "method"])[["err_mine", "err_sk", "agreement"]].mean().round(4)
    per_ds.to_csv(os.path.join(RESULTS, "verify_per_dataset.csv"))
    print("\n=== Par jeu ===")
    print(per_ds.to_string())
    write_markdown(raw, summary, raw.groupby(["dataset", "method"])[["err_mine", "err_sk", "agreement"]].mean(),
                   os.path.join(RESULTS, "verify_tables.md"))
    print("\nTableaux lisibles : results/verify_tables.md")

    # Figure : mon erreur OOB contre celle de sklearn, et l'OOB contre l'erreur de test (bagging).
    b = raw[raw.method == "bagging"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    lim = [0, max(b.oob_mine.max(), b.oob_sk.max()) * 1.05]
    ax[0].scatter(b.oob_sk, b.oob_mine, s=14)
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set(xlabel="erreur OOB, scikit-learn", ylabel="erreur OOB, mon bagging",
              title="Mon OOB contre celui de sklearn")
    lim2 = [0, max(b.oob_mine.max(), b.err_mine.max()) * 1.05]
    ax[1].scatter(b.oob_mine, b.err_mine, s=14)
    ax[1].plot(lim2, lim2, "k--", lw=1)
    ax[1].set(xlabel="erreur OOB (train)", ylabel="erreur de test", title="L'OOB prédit l'erreur de test")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "verify_oob.png"), dpi=150)
    print("\nFigure : figures/verify_oob.png")


if __name__ == "__main__":
    main()
