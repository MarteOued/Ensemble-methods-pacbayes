"""q5_classique.py : question 5 de la partie 2, apprentissage classique contre minimisation de borne

Sujet : sur au moins 6 jeux, comparer les risques en test d'ensembles appris classiquement (AdaBoost sur stumps, forêt
aléatoire, meilleure méthode de la partie 1, régression logistique) avec ceux des classifieurs obtenus en minimisant les
bornes ; donner les certificats ; dire ce que PAC-Bayes apporte et à quel prix.
Ce que fait le fichier : mêmes 6 jeux et 10 découpages. Les méthodes classiques sont entraînées sur prior + S (70 %) ; ma
« meilleure méthode de la partie 1 » est recalculée depuis results/exp_raw.csv (forêt aléatoire, plus mon stacking). Les
méthodes PAC-bayésiennes (stumps et linéaire, bornes A et B, prior appris) sont comparées par leur risque de vote, avec
certificat de Seeger sur le risque de Gibbs et borne 2 x certificat sur le vote. Trois ajouts pour le « à quel prix » :
une borne de test pour les méthodes classiques, les mêmes méthodes entraînées sur S seul (56 %), et des comparaisons à
famille égale (AdaBoost contre stumps PAC-bayésiens, logreg contre linéaire PAC-bayésien).
Limites assumées : 6 jeux (tendances, pas de test) ; aucune des deux familles n'est réglée jeu par jeu.

Usage : python3 q5_classique.py   (écrit partie2/results/q5_*.csv, q5_classique_tables.md et partie2/figures/q5_*.png)
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))              # code fourni par le prof (pacbayes_tools.py)
sys.path.insert(0, os.path.join(HERE, "..", "..", "partie1", "code"))   # tables.py, run_experiments.py (partie 1)
import pacbayes_tools as pb
from tables import md_table, num
from run_experiments import METHODS, LABELS
from q3_prior import gau_run
from q4_familles import cat_full

RESULTS = os.path.join(HERE, "..", "results")
FIGURES = os.path.join(HERE, "..", "figures")
PART1 = os.path.join(HERE, "..", "..", "partie1", "results", "exp_raw.csv")

DATASETS = ["wdbc", "pima", "australian", "iono", "german", "spambase"]
N_SPLITS = 10
PB = {"stumps_A": "PAC-Bayes stumps, A", "stumps_B": "PAC-Bayes stumps, B", "lin_A": "PAC-Bayes linéaire, A", "lin_B": "PAC-Bayes linéaire, B"}


def classement_partie1():
    """Rang moyen en accuracy des 10 méthodes de la partie 1 sur les six jeux (1 = meilleure)."""
    r = pd.read_csv(PART1)
    M = ["tree", "logreg", "knn", "bagging", "rf", "adaboost_d1", "adaboost_d3", "gb", "hgb", "stacking"]
    m = r[r.dataset.isin(DATASETS)].groupby(["dataset", "method"]).accuracy.mean().unstack()[M].loc[DATASETS]
    return pd.Series(np.array([stats.rankdata(-x) for x in m.values]).mean(0), index=M).sort_values()


def choisir_methodes(rang):
    noms = ["adaboost_d1", "rf", "logreg"]
    noms += [m for m in rang.index[:4] if m not in noms and m not in ("tree", "knn")][:1]      # la meilleure de la partie 1 non déjà listée
    return noms


# ----------------------------------------------------------------------------------------------------
# Expériences
# ----------------------------------------------------------------------------------------------------
def collecter(noms):
    cls, pbr = [], []
    for name in DATASETS:
        X, y = pb.real_dataset(name)
        for seed in range(N_SPLITS):
            Xp, yp, Xs, ys, Xte, yte = pb.split(X, y, seed)
            Xtr, ytr = np.vstack([Xp, Xs]), np.concatenate([yp, ys])
            for m in noms:
                err = float(np.mean(METHODS[m](seed).fit(Xtr, ytr).predict(Xte) != yte))
                err_s = float(np.mean(METHODS[m](seed).fit(Xs, ys).predict(Xte) != yte))                # contrôle : S seul
                cls.append(dict(dataset=name, split=seed, method=m, error=err, error_S_only=err_s,
                                holdout_bound=pb.kl_inv_upper(err, np.log(1 / pb.DELTA) / len(yte))))
            posts, _, _ = cat_full(X, y, seed)
            out, _ = gau_run(X, y, seed, True)
            for key, r in (("stumps_A", posts["stumps_A"]), ("stumps_B", posts["stumps_B"]), ("lin_A", out["A"]), ("lin_B", out["B"])):
                pbr.append(dict(dataset=name, split=seed, post=key, seeger=r["seeger"], KL=r["KL"], test_gibbs=r["test_gibbs"],
                                test_vote=r["test_vote"], cert_vote=min(1.0, 2 * r["seeger"])))
        print(f"[{name}] fini", flush=True)
    return pd.DataFrame(cls), pd.DataFrame(pbr)


# ----------------------------------------------------------------------------------------------------
# Tableaux
# ----------------------------------------------------------------------------------------------------
def ms(x, nd=3):
    x = np.asarray(x, float)
    return f"{num(x.mean(), nd)} ± {num(x.std(ddof=1), nd)}"


def sel(df, **kw):
    for k, v in kw.items():
        df = df[df[k] == v]
    return df.sort_values("split").reset_index(drop=True)


def erreurs(cls, pbr, noms):
    """Tableau jeux x méthodes des erreurs moyennes en test (vote pour PAC-Bayes)."""
    cols = {LABELS[m]: [sel(cls, dataset=d, method=m).error.mean() for d in DATASETS] for m in noms}
    cols.update({PB[p]: [sel(pbr, dataset=d, post=p).test_vote.mean() for d in DATASETS] for p in PB})
    return pd.DataFrame(cols, index=DATASETS)


def tableaux(cls, pbr, noms, rang):
    E = erreurs(cls, pbr, noms)
    ranks = pd.Series(np.array([stats.rankdata(r) for r in E.values]).mean(0), index=E.columns)
    out = ["# Question 5 : apprentissage classique contre minimisation de borne",
           f"{len(DATASETS)} jeux, {N_SPLITS} découpages. Méthodes classiques entraînées sur prior + S (70 % des données), PAC-Bayes avec le prior appris. "
           "Erreur en test : vote pour PAC-Bayes. Moyenne ± écart-type sur les découpages.\n",
           "Classement des méthodes de la partie 1 sur ces six jeux (rang moyen en accuracy, 1 = meilleure) : "
           + ", ".join(f"{LABELS[m]} {num(v, 2)}" for m, v in rang.items()) + ".\n"]

    head = ["Jeu"] + list(E.columns)
    rows = [[d] + [ms(sel(cls, dataset=d, method=m).error) for m in noms] + [ms(sel(pbr, dataset=d, post=p).test_vote) for p in PB] for d in DATASETS]
    rows.append(["**moyenne des 6 jeux**"] + [num(E[c].mean(), 3) for c in E.columns])
    rows.append(["**rang moyen**"] + [num(ranks[c], 2) for c in E.columns])
    out += ["## Tableau 1 : erreur en test de toutes les méthodes", md_table(head, rows), ""]

    rows = []
    for d in DATASETS:
        for fam, fam_lab in (("stumps", "stumps"), ("lin", "linéaire")):
            for b in ("A", "B"):
                s = sel(pbr, dataset=d, post=f"{fam}_{b}")
                rows.append([d, fam_lab, b, ms(s.seeger), ms(s.cert_vote), ms(s.test_gibbs), ms(s.test_vote), ms(s.KL, 2)])
    out += ["## Tableau 2 : les classifieurs PAC-bayésiens et leurs certificats",
            "Le certificat de Seeger majore le risque de GIBBS (avec probabilité 95 %) ; 2 x certificat majore le risque du VOTE.\n",
            md_table(["Jeu", "Famille", "Borne", "Certificat de Seeger (Gibbs)", "2 x certificat (vote)", "Risque de Gibbs (test)", "Risque du vote (test)", "KL"], rows), ""]

    pairs = [("adaboost_d1", "stumps_A", "AdaBoost sur stumps contre stumps PAC-bayésiens (A)"),
             ("logreg", "lin_A", "Régression logistique contre linéaire PAC-bayésien (A)"),
             ("rf", "lin_A", "Forêt aléatoire contre linéaire PAC-bayésien (A)"),
             ("rf", "stumps_A", "Forêt aléatoire contre stumps PAC-bayésiens (A)")]
    rows = []
    for m, p, lab in pairs:
        diffs = []
        for d in DATASETS:
            diffs.append(sel(cls, dataset=d, method=m).error.values - sel(pbr, dataset=d, post=p).test_vote.values)
        allv = np.concatenate(diffs)
        rows.append([lab] + [num(100 * v.mean(), 1) for v in diffs] + [num(100 * allv.mean(), 1), f"{int((allv < -1e-12).sum())}/{len(allv)}"])
    out += ["## Tableau 3 : différence d'erreur en test, classique - PAC-Bayes (en points ; négatif = le classique est meilleur)",
            "Dernière colonne : nombre d'exécutions (sur 60) où le classique est strictement meilleur.\n",
            md_table(["Comparaison"] + DATASETS + ["Moyenne", "Classique meilleur"], rows), ""]

    rows = []
    for d in DATASETS:
        la, sa = sel(pbr, dataset=d, post="lin_A"), sel(pbr, dataset=d, post="stumps_A")
        rf = sel(cls, dataset=d, method="rf")
        rows.append([d, num(la.seeger.mean(), 3), num(la.cert_vote.mean(), 3), num(la.test_vote.mean(), 3),
                     num(sa.seeger.mean(), 3), num(sa.cert_vote.mean(), 3), num(sa.test_vote.mean(), 3),
                     num(rf.error.mean(), 3), num(rf.holdout_bound.mean(), 3)])
    out += ["## Tableau 4 : ce que garantit chaque approche",
            "PAC-Bayes : certificat de Seeger (sur Gibbs), 2 x certificat (sur le vote) et erreur de test du vote. Forêt aléatoire : erreur de test et borne "
            "« de test » (95 %) calculée avec le jeu de test, qui n'existe pas au moment de l'apprentissage.\n",
            md_table(["Jeu", "Linéaire : certificat", "Linéaire : 2 x certificat", "Linéaire : erreur du vote", "Stumps : certificat", "Stumps : 2 x certificat",
                      "Stumps : erreur du vote", "Forêt : erreur", "Forêt : borne de test"], rows), ""]

    rows = []
    for m in noms:
        row = [LABELS[m]]
        for d in DATASETS:
            s = sel(cls, dataset=d, method=m)
            row.append(num(100 * (s.error_S_only - s.error).mean(), 1))
        allm = cls[cls.method == m]
        row.append(num(100 * (allm.error_S_only - allm.error).mean(), 1))
        rows.append(row)
    out += ["## Tableau 5 : le prix du découpage en trois parties (entraîner sur S seul, 56 %, au lieu de 70 %)",
            "Hausse de l'erreur de test (en points) quand la méthode classique n'utilise que S.\n",
            md_table(["Méthode"] + DATASETS + ["Moyenne"], rows), ""]

    gap = []
    for d in DATASETS:
        best_cls = min(noms, key=lambda m: sel(cls, dataset=d, method=m).error.mean())
        best_pb = min(PB, key=lambda p: sel(pbr, dataset=d, post=p).test_vote.mean())
        ec, ep = sel(cls, dataset=d, method=best_cls).error.mean(), sel(pbr, dataset=d, post=best_pb).test_vote.mean()
        gap.append([d, LABELS[best_cls], num(ec, 3), PB[best_pb], num(ep, 3), num(100 * (ep - ec), 1)])
    out += ["## Tableau 6 : meilleure méthode de chaque approche, par jeu",
            "Attention : le meilleur de chaque approche est choisi en regardant le test, ce qui avantage légèrement les deux.\n",
            md_table(["Jeu", "Meilleur classique", "Erreur", "Meilleur PAC-Bayes", "Erreur", "Écart (PAC-Bayes - classique, points)"], gap), ""]
    return out, E, ranks


# ----------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------
def fig_erreurs(E, path):
    fig, ax = plt.subplots(figsize=(14, 4.6))
    n = len(E.columns); x = np.arange(len(E)); w = 0.8 / n
    cmap = ["#7f7f7f", "#4c4c4c", "#a6a6a6", "#1f77b4", "#ff9896", "#d62728", "#ffbb78", "#ff7f0e"]
    for j, c in enumerate(E.columns):
        ax.bar(x + (j - (n - 1) / 2) * w, E[c], w * 0.95, color=cmap[j % len(cmap)], label=c)
    ax.set_xticks(x); ax.set_xticklabels(E.index); ax.set_ylabel("erreur en test (vote pour PAC-Bayes)")
    ax.legend(fontsize=7, ncol=2); ax.set_title("Apprentissage classique (gris, bleu) contre minimisation de borne (rouge, orange)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_nuage(cls, pbr, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (m, p, titre) in zip(axes, (("rf", "lin_A", "Forêt aléatoire contre linéaire PAC-bayésien"),
                                         ("adaboost_d1", "stumps_A", "AdaBoost sur stumps contre stumps PAC-bayésiens"))):
        for d in DATASETS:
            ax.scatter(sel(cls, dataset=d, method=m).error, sel(pbr, dataset=d, post=p).test_vote, s=18, label=d)
        lim = [0, 0.4]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set(xlabel=f"erreur de test : {LABELS[m]}", ylabel="erreur de test du vote PAC-Bayes (borne A)", title=titre, xlim=lim, ylim=lim)
    axes[0].legend(fontsize=7)
    fig.suptitle("Chaque point = un découpage (60) ; au-dessus de la diagonale, le classique est meilleur")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_garanties(cls, pbr, path):
    fig, ax = plt.subplots(figsize=(13, 4.6))
    x = np.arange(len(DATASETS)); w = 0.15
    items = [("Forêt aléatoire : erreur de test", lambda d: sel(cls, dataset=d, method="rf").error.mean(), "#4c4c4c"),
             ("Linéaire PAC-Bayes : erreur du vote", lambda d: sel(pbr, dataset=d, post="lin_A").test_vote.mean(), "#d62728"),
             ("Linéaire PAC-Bayes : certificat (Gibbs)", lambda d: sel(pbr, dataset=d, post="lin_A").seeger.mean(), "#ff9896"),
             ("Linéaire PAC-Bayes : 2 x certificat (vote)", lambda d: sel(pbr, dataset=d, post="lin_A").cert_vote.mean(), "#ffbb78"),
             ("Forêt aléatoire : borne de test (95 %)", lambda d: sel(cls, dataset=d, method="rf").holdout_bound.mean(), "#a6a6a6")]
    for j, (lab, f, c) in enumerate(items):
        ax.bar(x + (j - 2) * w, [f(d) for d in DATASETS], w * 0.95, color=c, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(DATASETS); ax.set_ylabel("erreur / garantie"); ax.legend(fontsize=7)
    ax.set_title("Erreur de test et garanties : PAC-Bayes (certificat calculé sur l'échantillon d'entraînement) contre borne de test")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    os.makedirs(RESULTS, exist_ok=True); os.makedirs(FIGURES, exist_ok=True)
    t0 = time.perf_counter()
    rang = classement_partie1()
    noms = choisir_methodes(rang)
    print("classement de la partie 1 sur ces jeux :", {k: round(v, 2) for k, v in rang.items()})
    print("méthodes classiques retenues :", noms, flush=True)
    cls, pbr = collecter(noms)
    cls.to_csv(os.path.join(RESULTS, "q5_classique.csv"), index=False)
    pbr.to_csv(os.path.join(RESULTS, "q5_pacbayes.csv"), index=False)
    md, E, _ = tableaux(cls, pbr, noms, rang)
    with open(os.path.join(RESULTS, "q5_classique_tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    fig_erreurs(E, os.path.join(FIGURES, "q5_erreurs.png"))
    fig_nuage(cls, pbr, os.path.join(FIGURES, "q5_nuage.png"))
    fig_garanties(cls, pbr, os.path.join(FIGURES, "q5_garanties.png"))
    print(f"terminé en {time.perf_counter() - t0:.0f} s : partie2/results/q5_classique_tables.md et partie2/figures/q5_*.png")


if __name__ == "__main__":
    main()
