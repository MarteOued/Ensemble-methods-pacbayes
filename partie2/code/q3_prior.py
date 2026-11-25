"""q3_prior.py : question 3 de la partie 2, l'impact du prior

Sujet : comparer un prior uniforme (catégoriel) ou N(0, I) (gaussien) à un prior appris sur la partie prior ; que
change-t-il pour la KL, les certificats et les risques ? Pourquoi un prior appris est-il légitime ?
Ce que fait le fichier : mêmes 6 jeux et mêmes découpages que la question 2. Pour chaque exécution et chaque borne (A, B),
je lance l'algorithme avec le prior de départ puis avec le prior appris (stumps : posterior de Gibbs sur la partie prior ;
linéaire : w0 minimisant la borne B sur la partie prior, prior N(w0, I), sigma = 1). Je refais pas à pas run_categorical
et run_gaussian (vérifiés identiques par assertion) pour lire aussi le prior lui-même, et je compare KL, certificat de
Seeger, risques de Gibbs et de vote, et le nombre de découpages où le certificat s'améliore.
Légitimité : le prior est appris sur la partie prior, disjointe de S ; il est donc indépendant de S et les bornes
restent valables. Le prix : S ne contient que 56 % des données au lieu de 70 %.

Usage : python3 q3_prior.py   (écrit partie2/results/q3_*.csv, q3_prior_tables.md et partie2/figures/q3_*.png)
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))              # code fourni par le prof (pacbayes_tools.py)
sys.path.insert(0, os.path.join(HERE, "..", "..", "partie1", "code"))   # tables.py
import pacbayes_tools as pb
from tables import md_table, num

RESULTS = os.path.join(HERE, "..", "results")
FIGURES = os.path.join(HERE, "..", "figures")

DATASETS = ["wdbc", "pima", "australian", "iono", "german", "spambase"]
N_SPLITS = 10
SIGMA = 1.0
FAMILY = {"cat": "Stumps (catégoriel)", "gau": "Linéaire (gaussien)"}
PRIOR = {"base": "départ", "learned": "appris"}      # départ = uniforme (stumps) ou N(0, I) (linéaire)
BOUND = {"A": "A (linéaire)", "B": "B (McAllester)"}


# ----------------------------------------------------------------------------------------------------
# Les deux pipelines, refaits pas à pas pour garder le prior
# ----------------------------------------------------------------------------------------------------
def cat_run(X, y, seed, learned):
    Xp, yp, Xs, ys, Xte, yte = pb.split(X, y, seed)
    H = pb.make_stumps(Xp)
    V, Vte = pb.stump_votes(H, Xs), pb.stump_votes(H, Xte)
    P = np.full(len(H), 1 / len(H))
    if learned:
        P = np.maximum(pb.gibbs_posterior(P, pb.voter_risks(pb.stump_votes(H, Xp), yp), len(yp) / 4), 1e-300)
        P /= P.sum()
    L, m = pb.voter_risks(V, ys), len(ys)
    A, _ = pb.learn_linear_categorical(P, L, m)
    B = pb.learn_mcallester_categorical(P, L, m)
    out = {k: dict(bound=r["bound"], seeger=r["seeger"], KL=r["KL"], emp=r["emp"], neff=1 / np.sum(r["Q"] ** 2),
                   test_gibbs=pb.gibbs_risk(r["Q"], Vte, yte), test_vote=pb.vote_risk(r["Q"], Vte, yte))
           for k, r in (("A", A), ("B", B))}
    prior = dict(prior_neff=1 / float(np.sum(P ** 2)), prior_gibbs=pb.gibbs_risk(P, Vte, yte), prior_vote=pb.vote_risk(P, Vte, yte),
                 w0_norm=np.nan, n_prior=len(yp), m=m)
    return out, prior


def gau_run(X, y, seed, learned):
    Xp, yp, Xs, ys, Xte, yte = pb.split(X, y, seed)
    Zp, Zs, Zte = [pb.add_bias(Z) for Z in pb.standardize(Xp, Xs, Xte)]
    w0 = np.zeros(Zs.shape[1])
    if learned:
        w0 = pb.learn_mcallester_gaussian(Zp, yp, w0, SIGMA)["w"]
    A, _ = pb.learn_linear_gaussian(Zs, ys, w0, SIGMA)
    B = pb.learn_mcallester_gaussian(Zs, ys, w0, SIGMA)
    out = {k: dict(bound=r["bound"], seeger=r["seeger"], KL=r["KL"], emp=r["emp"], neff=np.nan,
                   test_gibbs=pb.probit_risk(r["w"], Zte, yte, SIGMA), test_vote=pb.linear_vote_risk(r["w"], Zte, yte))
           for k, r in (("A", A), ("B", B))}
    prior = dict(prior_neff=np.nan, w0_norm=float(np.linalg.norm(w0)), n_prior=len(yp), m=len(ys),
                 prior_gibbs=pb.probit_risk(w0, Zte, yte, SIGMA) if learned else np.nan,
                 prior_vote=pb.linear_vote_risk(w0, Zte, yte) if learned else np.nan)
    return out, prior


def verifier_contre_les_fonctions_fournies():
    n = 0
    for name, seeds in [("wdbc", (0, 1)), ("pima", (2,)), ("iono", (3,))]:
        X, y = pb.real_dataset(name)
        for seed in seeds:
            for learned in (False, True):
                mine, _ = cat_run(X, y, seed, learned)
                ref = pb.run_categorical(X, y, seed=seed, prior="learned" if learned else "uniform")
                mine2, _ = gau_run(X, y, seed, learned)
                ref2 = pb.run_gaussian(X, y, seed=seed, prior="learned" if learned else "zero", sigma=SIGMA)
                for got, r in ((mine, ref), (mine2, ref2)):
                    for key, label in (("A", "A (linear)"), ("B", "B (McAllester)")):
                        for k, v in r[label].items():
                            assert np.isclose(got[key][k], v, rtol=1e-8, atol=1e-10), (name, seed, learned, key, k, got[key][k], v)
                n += 1
    return n


def collecter():
    rows, priors = [], []
    for name in DATASETS:
        X, y = pb.real_dataset(name)
        for seed in range(N_SPLITS):
            for fam, fn in (("cat", cat_run), ("gau", gau_run)):
                for learned in (False, True):
                    out, prior = fn(X, y, seed, learned)
                    for k in ("A", "B"):
                        rows.append(dict(family=fam, dataset=name, split=seed, prior="learned" if learned else "base", bound_name=k, **out[k]))
                    priors.append(dict(family=fam, dataset=name, split=seed, prior="learned" if learned else "base", **prior))
        print(f"[{name}] fini", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(priors)


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


def tableaux(rows, priors, n_verifs):
    out = ["# Question 3 : impact du prior",
           f"{len(DATASETS)} jeux, {N_SPLITS} découpages chacun, moyenne ± écart-type sur les découpages. « Départ » = prior uniforme "
           f"(stumps) ou N(0, I) (linéaire) ; « appris » = prior appris sur la partie prior. Mes résultats ont été vérifiés identiques à "
           f"ceux de `run_categorical` et `run_gaussian` sur {n_verifs} cas.\n"]
    head = ["Jeu", "Prior", "Borne", "Valeur de la borne", "Certificat de Seeger", "KL", "Risque de Gibbs (test)", "Risque du vote (test)"]
    for fam in ("cat", "gau"):
        table = []
        for d in DATASETS:
            for pr in ("base", "learned"):
                for k in ("A", "B"):
                    s = sel(rows, family=fam, dataset=d, prior=pr, bound_name=k)
                    table.append([d, PRIOR[pr], k, ms(s.bound), ms(s.seeger), ms(s.KL, 2), ms(s.test_gibbs), ms(s.test_vote)])
        out += [f"## Tableau {1 if fam == 'cat' else 2} : {FAMILY[fam]}", md_table(head, table), ""]

    # --- Tableau 3 : ce qui change (appris - départ), par découpage ----------------------------------
    table, glob = [], []
    for fam in ("cat", "gau"):
        for k in ("A", "B"):
            allD = {c: [] for c in ("KL", "seeger", "test_gibbs", "test_vote")}
            n_better = n_tot = 0
            for d in DATASETS:
                b = sel(rows, family=fam, dataset=d, prior="base", bound_name=k)
                l = sel(rows, family=fam, dataset=d, prior="learned", bound_name=k)
                diff = {c: (l[c] - b[c]).values for c in allD}
                better = int((diff["seeger"] < -1e-9).sum())
                table.append([FAMILY[fam], k, d, num(diff["KL"].mean(), 2), num(diff["seeger"].mean(), 3), f"{better}/{N_SPLITS}",
                              num(diff["test_gibbs"].mean(), 3), num(diff["test_vote"].mean(), 3)])
                for c in allD:
                    allD[c].append(diff[c])
                n_better += better; n_tot += N_SPLITS
            cc = {c: np.concatenate(v) for c, v in allD.items()}
            glob.append([FAMILY[fam], k, num(cc["KL"].mean(), 2), num(cc["seeger"].mean(), 3), f"{n_better}/{n_tot}",
                         num(cc["test_gibbs"].mean(), 3), num(cc["test_vote"].mean(), 3)])
    out += ["## Tableau 3 : ce qui change avec le prior appris (appris - départ, moyenne sur les découpages)",
            "Une différence négative de KL ou de certificat est une amélioration. « Certificat meilleur » : nombre de découpages où "
            "le certificat de Seeger est plus petit avec le prior appris.\n",
            md_table(["Famille", "Borne", "Jeu", "Δ KL", "Δ certificat de Seeger", "Certificat meilleur", "Δ risque de Gibbs", "Δ risque du vote"], table), "",
            "## Tableau 4 : résumé global (appris - départ, sur les 60 exécutions par famille et par borne)",
            md_table(["Famille", "Borne", "Δ KL", "Δ certificat de Seeger", "Certificat meilleur", "Δ risque de Gibbs", "Δ risque du vote"], glob), ""]

    # --- Tableau 5 : le prior appris lui-même -----------------------------------------------------
    table = []
    for d in DATASETS:
        pc = sel(priors, family="cat", dataset=d, prior="learned"); pg = sel(priors, family="gau", dataset=d, prior="learned")
        table.append([d, int(pc.n_prior.iloc[0]), int(pc.m.iloc[0]), num(pc.prior_neff.mean(), 1), num(pc.prior_gibbs.mean(), 3), num(pc.prior_vote.mean(), 3),
                      num(pg.w0_norm.mean(), 2), num(pg.prior_gibbs.mean(), 3), num(pg.prior_vote.mean(), 3)])
    out += ["## Tableau 5 : le prior appris lui-même, évalué sur le test",
            "Taille de la partie prior et de l'échantillon S ; pour les stumps, nombre effectif de votants du prior appris et risques de Gibbs et du vote de P "
            "lui-même ; pour le linéaire, norme de w0 et risques du prior N(w0, I) (Gibbs) et du classifieur sign⟨w0, x⟩ (vote).\n",
            md_table(["Jeu", "Partie prior", "S", "Stumps : votants eff. du prior", "Gibbs de P", "Vote de P", "Linéaire : ‖w0‖", "Gibbs de N(w0, I)",
                      "Vote de w0"], table), ""]
    return out


# ----------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------
def fig_certificats(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)
    x = np.arange(len(DATASETS)); w = 0.38
    for ax, fam in zip(axes, ("cat", "gau")):
        for j, (pr, col) in enumerate((("base", "lightsteelblue"), ("learned", "tab:blue"))):
            cert = [sel(rows, family=fam, dataset=d, prior=pr, bound_name="A").seeger.mean() for d in DATASETS]
            gib = [sel(rows, family=fam, dataset=d, prior=pr, bound_name="A").test_gibbs.mean() for d in DATASETS]
            ax.bar(x + (j - 0.5) * w, cert, w * 0.95, color=col, label=f"certificat de Seeger, prior {PRIOR[pr]}")
            ax.scatter(x + (j - 0.5) * w, gib, color="k", marker="_", s=200, zorder=3, label="risque de Gibbs (test)" if j == 0 else None)
        ax.set_xticks(x); ax.set_xticklabels(DATASETS, rotation=20); ax.set_title(FAMILY[fam] + ", borne A")
    axes[0].set_ylabel("certificat / risque"); axes[0].legend(fontsize=7)
    fig.suptitle("Certificat de Seeger avec le prior de départ et avec le prior appris (moyenne sur 10 découpages)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_kl(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    x = np.arange(len(DATASETS)); w = 0.38
    for ax, fam in zip(axes, ("cat", "gau")):
        for j, (pr, col) in enumerate((("base", "lightsalmon"), ("learned", "tab:red"))):
            kl = [sel(rows, family=fam, dataset=d, prior=pr, bound_name="A").KL.mean() for d in DATASETS]
            ax.bar(x + (j - 0.5) * w, kl, w * 0.95, color=col, label=f"prior {PRIOR[pr]}")
        ax.set_xticks(x); ax.set_xticklabels(DATASETS, rotation=20); ax.set_title(FAMILY[fam] + ", borne A"); ax.set_ylabel("KL(Q‖P)")
        ax.legend(fontsize=8)
    fig.suptitle("KL du posterior avec le prior de départ et avec le prior appris (moyenne sur 10 découpages)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_risques(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)
    x = np.arange(len(DATASETS)); w = 0.2
    for ax, fam in zip(axes, ("cat", "gau")):
        for j, (pr, met, col) in enumerate((("base", "test_gibbs", "lightcoral"), ("learned", "test_gibbs", "tab:red"),
                                           ("base", "test_vote", "plum"), ("learned", "test_vote", "tab:purple"))):
            v = [sel(rows, family=fam, dataset=d, prior=pr, bound_name="A")[met].mean() for d in DATASETS]
            ax.bar(x + (j - 1.5) * w, v, w * 0.95, color=col, label=f"{'Gibbs' if met == 'test_gibbs' else 'vote'}, prior {PRIOR[pr]}")
        ax.set_xticks(x); ax.set_xticklabels(DATASETS, rotation=20); ax.set_title(FAMILY[fam] + ", borne A")
    axes[0].set_ylabel("risque en test"); axes[0].legend(fontsize=7)
    fig.suptitle("Risques en test avec le prior de départ et avec le prior appris (moyenne sur 10 découpages)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    os.makedirs(RESULTS, exist_ok=True); os.makedirs(FIGURES, exist_ok=True)
    t0 = time.perf_counter()
    n = verifier_contre_les_fonctions_fournies()
    print(f"{n} cas identiques aux fonctions fournies", flush=True)
    rows, priors = collecter()
    rows.to_csv(os.path.join(RESULTS, "q3_runs.csv"), index=False)
    priors.to_csv(os.path.join(RESULTS, "q3_priors.csv"), index=False)
    with open(os.path.join(RESULTS, "q3_prior_tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(tableaux(rows, priors, n)) + "\n")
    fig_certificats(rows, os.path.join(FIGURES, "q3_certificats.png"))
    fig_kl(rows, os.path.join(FIGURES, "q3_kl.png"))
    fig_risques(rows, os.path.join(FIGURES, "q3_risques.png"))
    print(f"terminé en {time.perf_counter() - t0:.0f} s : partie2/results/q3_prior_tables.md et partie2/figures/q3_*.png")


if __name__ == "__main__":
    main()
