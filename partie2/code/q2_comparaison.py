"""q2_comparaison.py : question 2 de la partie 2, comparer les bornes A et B sur des stumps

Sujet : sur wdbc puis au moins 5 autres jeux, comparer les deux algorithmes (bornes minimisées, certificat de Seeger,
KL, nombre effectif de votants, risques en test), dire s'ils donnent le même posterior et quel certificat reporter ;
tracer, pour A, la borne, la KL et les risques en fonction de lambda.
Ce que fait le fichier : sur 6 jeux équilibrés (wdbc, pima, australian, iono, german, spambase) et 10 découpages
(14 % prior, 56 % échantillon S, 30 % test), prior uniforme, je refais pas à pas run_categorical avec les fonctions
fournies pour récupérer les posteriors Q_A et Q_B (mon script vérifie par assertion que c'est identique à
run_categorical). Je les compare (distance en variation totale, recouvrement, votants effectifs), je calcule les
certificats de Seeger et je les confronte au risque de Gibbs mesuré sur le test.
Limites assumées : 10 découpages et 6 jeux (tendances) ; la partie prior ne fait que 14 % des données.

Usage : python3 q2_comparaison.py   (écrit partie2/results/q2_*.csv, q2_comparaison_tables.md, partie2/figures/q2_*.png)
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
NAMES = {"A": "A (linéaire)", "B": "B (McAllester)"}


# ----------------------------------------------------------------------------------------------------
# Une exécution (un jeu, un découpage) : refait run_categorical en gardant les posteriors
# ----------------------------------------------------------------------------------------------------
def run_one(X, y, seed):
    Xp, yp, Xs, ys, Xte, yte = pb.split(X, y, seed)
    H = pb.make_stumps(Xp)
    V, Vte = pb.stump_votes(H, Xs), pb.stump_votes(H, Xte)
    P = np.full(len(H), 1 / len(H))
    L, m = pb.voter_risks(V, ys), len(ys)
    A, rows = pb.learn_linear_categorical(P, L, m)
    B = pb.learn_mcallester_categorical(P, L, m)

    def summary(r):
        return dict(bound=r["bound"], seeger=r["seeger"], KL=r["KL"], emp=r["emp"], neff=1 / np.sum(r["Q"] ** 2),
                    test_gibbs=pb.gibbs_risk(r["Q"], Vte, yte), test_vote=pb.vote_risk(r["Q"], Vte, yte))

    return dict(H=H, A=A, B=B, rows=rows, summary=summary, sizes=(len(yp), m, len(yte)), Vte=Vte, yte=yte)


def verifier_contre_run_categorical():
    """Mes résultats doivent être IDENTIQUES à ceux de pb.run_categorical (sinon le script s'arrête)."""
    n = 0
    for name, seeds in [("wdbc", (0, 1, 2)), ("pima", (0, 3)), ("iono", (4,))]:
        X, y = pb.real_dataset(name)
        for seed in seeds:
            mine, ref = run_one(X, y, seed), pb.run_categorical(X, y, seed=seed)
            for key, label in (("A", "A (linear)"), ("B", "B (McAllester)")):
                got = mine["summary"](mine[key])
                for k, v in ref[label].items():
                    assert np.isclose(got[k], v, rtol=1e-9, atol=1e-12), (name, seed, key, k, got[k], v)
            n += 1
    return n


def collecter():
    runs, pairs, lams, info = [], [], [], []
    for name in DATASETS:
        X, y = pb.real_dataset(name)
        n_stumps = []
        for seed in range(N_SPLITS):
            res = run_one(X, y, seed)
            n_stumps.append(len(res["H"]))
            for key in ("A", "B"):
                r = res[key]
                runs.append(dict(dataset=name, split=seed, bound_name=key, lam=r.get("lam", np.nan), **res["summary"](r)))
            QA, QB = res["A"]["Q"], res["B"]["Q"]
            ia, ib = int(np.argmax(QA)), int(np.argmax(QB))
            pairs.append(dict(dataset=name, split=seed, tv=0.5 * float(np.abs(QA - QB).sum()),
                              overlap=float(np.minimum(QA, QB).sum()), same_top=ia == ib,
                              same_top_feature=res["H"][ia][0] == res["H"][ib][0],
                              wmax_A=float(QA.max()), wmax_B=float(QB.max())))
            for r in res["rows"]:
                lams.append(dict(dataset=name, split=seed, lam=r["lam"], bound=r["bound"], seeger=r["seeger"], KL=r["KL"],
                                 neff=1 / float(np.sum(r["Q"] ** 2)),
                                 test_gibbs=pb.gibbs_risk(r["Q"], res["Vte"], res["yte"]),
                                 test_vote=pb.vote_risk(r["Q"], res["Vte"], res["yte"])))
        p, s, t = res["sizes"]
        info.append(dict(dataset=name, n=X.shape[0], d=X.shape[1], prior=p, S=s, test=t, stumps=float(np.mean(n_stumps))))
        print(f"[{name}] fini", flush=True)
    return pd.DataFrame(runs), pd.DataFrame(pairs), pd.DataFrame(lams), pd.DataFrame(info)


# ----------------------------------------------------------------------------------------------------
# Tableaux
# ----------------------------------------------------------------------------------------------------
def ms(x, nd=3):
    x = np.asarray(x, float)
    return f"{num(x.mean(), nd)} ± {num(x.std(ddof=1), nd)}"


def tableaux(runs, pairs, info, n_verifs):
    out = ["# Question 2 : comparaison des bornes A et B sur des stumps (prior uniforme)",
           f"{len(DATASETS)} jeux, {N_SPLITS} découpages chacun. Moyenne ± écart-type sur les découpages. "
           f"Mes résultats ont été vérifiés identiques à ceux de `run_categorical` sur {n_verifs} exécutions.\n"]
    rows = [[r.dataset, r.n, r.d, r.prior, r.S, r.test, num(r.stumps, 0)] for r in info.itertuples()]
    out += ["## Tableau 1 : les jeux utilisés",
            "Tailles de la partie prior (14 %), de l'échantillon de borne S (56 %) et du test (30 %) ; « Stumps » = nombre de "
            "votants (moyenne sur les découpages).\n",
            md_table(["Jeu", "m", "d", "Prior", "S", "Test", "Stumps"], rows), ""]

    def ligne(sub, nom):
        return [nom, ms(sub.bound), ms(sub.seeger), ms(sub.KL, 2), ms(sub.neff, 1), ms(sub.test_gibbs), ms(sub.test_vote)]
    head = ["Borne", "Valeur de la borne", "Certificat de Seeger", "KL", "Votants effectifs", "Risque de Gibbs (test)", "Risque du vote (test)"]

    w = runs[runs.dataset == "wdbc"]
    out += ["## Tableau 2 : wdbc",
            md_table(head, [ligne(w[w.bound_name == k], NAMES[k]) for k in ("A", "B")]), ""]

    rows = []
    for d in DATASETS:
        for k in ("A", "B"):
            rows.append([d] + ligne(runs[(runs.dataset == d) & (runs.bound_name == k)], NAMES[k]))
    out += ["## Tableau 3 : les six jeux",
            md_table(["Jeu"] + head, rows), ""]

    rows = []
    for d in DATASETS:
        p = pairs[pairs.dataset == d]
        a, b = runs[(runs.dataset == d) & (runs.bound_name == "A")], runs[(runs.dataset == d) & (runs.bound_name == "B")]
        rows.append([d, num(p.tv.mean(), 2), num(p.overlap.mean(), 2), f"{int(p.same_top.sum())}/{len(p)}",
                     f"{int(p.same_top_feature.sum())}/{len(p)}", num(p.wmax_A.mean(), 2), num(p.wmax_B.mean(), 2),
                     num(a.neff.mean(), 1), num(b.neff.mean(), 1)])
    out += ["## Tableau 4 : les deux bornes conduisent-elles au même posterior ?",
            "Distance en variation totale entre Q_A et Q_B (0 = identiques, 1 = supports disjoints) ; recouvrement "
            "Σ min(Q_A, Q_B) ; nombre de découpages où le stump de poids maximal est le même, ou porte sur la même variable ; poids "
            "du stump le plus pondéré de chaque posterior ; nombre effectif de votants.\n",
            md_table(["Jeu", "Distance (var. totale)", "Recouvrement", "Même stump principal", "Même variable principale",
                      "Poids max A", "Poids max B", "Votants eff. A", "Votants eff. B"], rows), ""]

    rows = []
    for d in DATASETS:
        a = runs[(runs.dataset == d) & (runs.bound_name == "A")].reset_index(drop=True)
        b = runs[(runs.dataset == d) & (runs.bound_name == "B")].reset_index(drop=True)
        rows.append([d,
                     f"{int((a.seeger >= a.test_gibbs).sum())}/{N_SPLITS}", f"{int((b.seeger >= b.test_gibbs).sum())}/{N_SPLITS}",
                     num((a.seeger - a.test_gibbs).mean(), 3), num((b.seeger - b.test_gibbs).mean(), 3),
                     num((a.bound - a.seeger).mean(), 3), num((b.bound - b.seeger).mean(), 3),
                     f"{int((a.seeger < b.seeger).sum())}/{N_SPLITS}", num((b.seeger - a.seeger).mean(), 4),
                     f"{int((a.test_vote <= 2 * a.test_gibbs + 1e-12).sum() + (b.test_vote <= 2 * b.test_gibbs + 1e-12).sum())}/{2 * N_SPLITS}"])
    out += ["## Tableau 5 : quel certificat reporter ?",
            "« Seeger ≥ test » : nombre de découpages où le certificat de Seeger est supérieur au risque de Gibbs mesuré sur le test "
            "(il doit l'être avec probabilité 95 %). « Écart » = certificat - risque de test (moyenne). « Borne - Seeger » : de combien la "
            "borne minimisée dépasse le certificat de Seeger du même posterior. « Seeger A < Seeger B » : nombre de découpages où le "
            "posterior de A a le meilleur certificat, et écart moyen Seeger(B) - Seeger(A). Dernière colonne : nombre d'exécutions où "
            "R(vote) ≤ 2 R(Gibbs) sur le test.\n",
            md_table(["Jeu", "Seeger ≥ test (A)", "Seeger ≥ test (B)", "Écart (A)", "Écart (B)", "Borne - Seeger (A)",
                      "Borne - Seeger (B)", "Seeger A < Seeger B", "Écart moyen B - A", "Vote ≤ 2 x Gibbs"], rows), ""]

    a, b = runs[runs.bound_name == "A"].reset_index(drop=True), runs[runs.bound_name == "B"].reset_index(drop=True)
    la = runs[runs.bound_name == "A"].groupby("dataset").lam.median()
    out += ["## Tableau 6 : λ choisi par l'algorithme de la borne A",
            md_table(["Jeu", "λ médian choisi"], [[d, f"{la[d]:.0f}"] for d in DATASETS]), "",
            "## Résumé sur les 60 exécutions",
            f"- Certificat de Seeger ≥ risque de Gibbs en test : A {int((a.seeger >= a.test_gibbs).sum())}/{len(a)}, "
            f"B {int((b.seeger >= b.test_gibbs).sum())}/{len(b)}.",
            f"- Le posterior de A a un certificat de Seeger plus petit que celui de B dans {int((a.seeger < b.seeger).sum())}/{len(a)} cas ; "
            f"le risque de Gibbs en test de A est plus petit dans {int((a.test_gibbs < b.test_gibbs).sum())}/{len(a)} cas ; "
            f"le risque du vote de A est plus petit ou égal dans {int((a.test_vote <= b.test_vote).sum())}/{len(a)} cas.",
            f"- Votants effectifs : A {num(a.neff.mean(), 1)} en moyenne, B {num(b.neff.mean(), 1)} ; B concentre tout son poids sur un seul "
            f"stump (votants effectifs < 1,5) dans {int((b.neff < 1.5).sum())}/{len(b)} cas, A dans {int((a.neff < 1.5).sum())}/{len(a)} cas.", ""]
    return out


# ----------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------
def fig_lambda_wdbc(lams, path):
    g = lams[lams.dataset == "wdbc"].groupby("lam")
    mu, sd = g.mean(numeric_only=True), g.std(numeric_only=True)
    x = mu.index.values
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for col, lab, c in [("bound", "borne A(λ)", "tab:blue"), ("seeger", "certificat de Seeger", "tab:green"),
                        ("test_gibbs", "risque de Gibbs (test)", "tab:red"), ("test_vote", "risque du vote (test)", "tab:purple")]:
        ax[0].plot(x, mu[col], "o-", ms=4, color=c, label=lab)
        ax[0].fill_between(x, mu[col] - sd[col], mu[col] + sd[col], color=c, alpha=0.12)
    ax[0].set(xscale="log", xlabel="λ", ylabel="borne / risque", ylim=(0, 0.6), title="Bornes et risques en test")
    ax[0].legend(fontsize=8)
    ax[1].plot(x, mu["KL"], "o-", ms=4, color="tab:orange")
    ax[1].fill_between(x, mu["KL"] - sd["KL"], mu["KL"] + sd["KL"], color="tab:orange", alpha=0.15)
    ax[1].set(xscale="log", xlabel="λ", ylabel="KL(Q‖P)", title="KL du posterior de Gibbs")
    ax[2].plot(x, mu["neff"], "o-", ms=4, color="tab:brown")
    ax[2].set(xscale="log", yscale="log", xlabel="λ", ylabel="votants effectifs", title="Nombre effectif de votants")
    fig.suptitle("wdbc, prior uniforme : borne A en fonction de λ (moyenne ± écart-type sur 10 découpages)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_lambda_tous(lams, path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    for ax, d in zip(axes.ravel(), DATASETS):
        mu = lams[lams.dataset == d].groupby("lam").mean(numeric_only=True)
        for col, lab, c in [("bound", "borne A(λ)", "tab:blue"), ("seeger", "certificat de Seeger", "tab:green"),
                            ("test_gibbs", "risque de Gibbs (test)", "tab:red"), ("test_vote", "risque du vote (test)", "tab:purple")]:
            ax.plot(mu.index, mu[col], "o-", ms=3, color=c, label=lab)
        ax.set(xscale="log", xlabel="λ", ylim=(0, 0.6), title=d)
    axes[0, 0].legend(fontsize=7); axes[0, 0].set_ylabel("borne / risque"); axes[1, 0].set_ylabel("borne / risque")
    fig.suptitle("Borne A en fonction de λ, six jeux (moyenne sur 10 découpages)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_kl_tous(lams, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for d in DATASETS:
        mu = lams[lams.dataset == d].groupby("lam").mean(numeric_only=True)
        ax[0].plot(mu.index, mu["KL"], "o-", ms=3, label=d)
        ax[1].plot(mu.index, mu["neff"], "o-", ms=3, label=d)
    ax[0].set(xscale="log", yscale="log", xlabel="λ", ylabel="KL(Q‖P)", title="KL en fonction de λ")
    ax[1].set(xscale="log", yscale="log", xlabel="λ", ylabel="votants effectifs", title="Votants effectifs en fonction de λ")
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_certificats(runs, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(DATASETS)); w = 0.2
    for k, (key, lab, c) in enumerate([("A", "A", "tab:blue"), ("B", "B", "tab:orange")]):
        sub = runs[runs.bound_name == key].groupby("dataset")
        ax[0].bar(x + (k - 0.5) * w * 2 + 0, sub.seeger.mean().loc[DATASETS], w * 1.8, color=c, label=f"certificat de Seeger, posterior de {lab}")
        ax[0].scatter(x + (k - 0.5) * w * 2, sub.test_gibbs.mean().loc[DATASETS], color="k", marker="_", s=220, zorder=3,
                      label="risque de Gibbs (test)" if k == 0 else None)
    ax[0].set_xticks(x); ax[0].set_xticklabels(DATASETS, rotation=20); ax[0].set(ylabel="certificat / risque", title="Certificats et risque de test réel")
    ax[0].legend(fontsize=7)
    for key, c in (("A", "tab:blue"), ("B", "tab:orange")):
        sub = runs[runs.bound_name == key]
        ax[1].scatter(sub.test_gibbs, sub.seeger, s=14, color=c, label=f"posterior de {key}")
    lim = [0, max(runs.seeger.max(), runs.test_gibbs.max()) * 1.05]
    ax[1].plot(lim, lim, "k--", lw=1, label="certificat = risque")
    ax[1].set(xlabel="risque de Gibbs mesuré sur le test", ylabel="certificat de Seeger", title="Un point = un posterior (120 : 60 par borne)")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    os.makedirs(RESULTS, exist_ok=True); os.makedirs(FIGURES, exist_ok=True)
    t0 = time.perf_counter()
    n = verifier_contre_run_categorical()
    print(f"{n} exécutions identiques à pb.run_categorical", flush=True)
    runs, pairs, lams, info = collecter()
    runs.to_csv(os.path.join(RESULTS, "q2_runs.csv"), index=False)
    pairs.to_csv(os.path.join(RESULTS, "q2_posteriors.csv"), index=False)
    lams.to_csv(os.path.join(RESULTS, "q2_lambda.csv"), index=False)
    with open(os.path.join(RESULTS, "q2_comparaison_tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(tableaux(runs, pairs, info, n)) + "\n")
    fig_lambda_wdbc(lams, os.path.join(FIGURES, "q2_lambda_wdbc.png"))
    fig_lambda_tous(lams, os.path.join(FIGURES, "q2_lambda_tous.png"))
    fig_kl_tous(lams, os.path.join(FIGURES, "q2_kl_neff_lambda.png"))
    fig_certificats(runs, os.path.join(FIGURES, "q2_certificats.png"))
    print(f"terminé en {time.perf_counter() - t0:.0f} s : partie2/results/q2_comparaison_tables.md et partie2/figures/q2_*.png")


if __name__ == "__main__":
    main()
