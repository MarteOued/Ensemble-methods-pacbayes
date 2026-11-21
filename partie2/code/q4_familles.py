"""q4_familles.py : question 4 de la partie 2, changer de famille de distributions

CE QUE DEMANDE LE SUJET
    Partie 2, question 4 : « Distributions catégorielles sur des stumps contre distributions gaussiennes sur des
    classifieurs linéaires : comparez les certificats, les risques de Gibbs et les votes. Vous pouvez ajouter d'autres
    posteriors (un Dirac sur le meilleur stump ; uniforme sur les k meilleurs stumps) avec les fonctions fournies.
    Pourquoi minimiser une borne sur le risque de Gibbs donne-t-il un vote de stumps faible, mais un bon classifieur
    linéaire ? »

CE QUE J'AI FAIT DANS CE FICHIER
    - Mêmes jeux et découpages que les questions 2 et 3 (wdbc, pima, australian, iono, german, spambase ; 10 découpages)
      et le PRIOR APPRIS pour les deux familles (il donne les meilleurs certificats, question 3).
    - Je compare quatre posteriors issus de la minimisation d'une borne (stumps avec A, stumps avec B, linéaire avec A,
      linéaire avec B) et trois posteriors de stumps que je construis moi-même avec les fonctions fournies :
        * un Dirac sur le meilleur stump (le stump de plus petit risque sur S) ;
        * l'uniforme sur les 5 meilleurs stumps ; l'uniforme sur les 20 meilleurs.
      Pour chacun, je calcule le risque empirique de Gibbs sur S, la KL par rapport au prior appris, le certificat de Seeger
      (bound_seeger), et les risques de Gibbs et du vote sur le test.
    - Pour répondre au « pourquoi », je ne veux pas seulement l'affirmer : je le teste. Mon raisonnement est le suivant.
        (1) Le risque de Gibbs est LINÉAIRE en Q : sum_h Q(h) risque(h). Pour des stumps, le minimiser pousse Q à se
            concentrer sur les stumps de plus petit risque, jusqu'au Dirac sur le meilleur ; la KL ne freine que peu. À la question 2,
            j'ai vu que le posterior minimisant A ou B a entre 1 et 4 votants effectifs.
        (2) Le vote d'un posterior concentré sur un seul stump est ce stump : il n'y a plus de vote. Un vote n'est fort que si
            ses votants font des erreurs DIFFÉRENTES (de la diversité) ; le risque de Gibbs ne récompense pas cette diversité.
        (3) Pour des gaussiennes, le posterior est un nuage de classifieurs autour de w ; son vote majoritaire est le classifieur
            central sign<w, x>, qui moyenne le bruit du nuage : son risque est bien plus petit que celui de Gibbs, et minimiser
            le risque de Gibbs du nuage revient à bien placer son centre w.
      Ce que je mesure pour tester cela : (a) l'écart Gibbs - vote pour chaque famille ; (b) une courbe en fonction de k
      (uniforme sur les k meilleurs stumps, k de 1 à tous les stumps) : si le raisonnement tient, le risque de Gibbs augmente
      avec k alors que le risque du vote est minimal pour un k intermédiaire, pas pour k = 1.
    - Je vérifie par assertion que mes posteriors A et B de stumps, avec le prior appris, sont identiques à ceux de
      run_categorical.

À QUOI ÇA ME SERT
    Ce sont les tableaux et figures de la question 4 du rapport, et un fondement pour la question 5 (apprentissage classique
    contre minimisation de borne).

LIMITES QUE J'ASSUME
    - Pour k = tous les stumps, le vote uniforme est toujours nul (les stumps vont par paires opposées h et -h) et le code fourni
      compte les égalités comme des erreurs : je ne trace pas ce vote. Plus généralement vote_risk compte les égalités comme des
      erreurs, ce qui pénalise les k PAIRS (petite bosse à k = 2 sur les courbes) ; c'est une convention du code fourni.
    - Je ne fais qu'un seul choix de k pour les tableaux (1, 5, 20) ; la courbe en k couvre le reste.
    - Je ne règle ni sigma ni les paramètres des priors ; 10 découpages, 6 jeux : des tendances.
    - Un « meilleur k » trouvé en regardant le test est un diagnostic pour comprendre, pas une méthode : je ne le propose pas
      comme algorithme (le choisir demanderait de l'estimer sur des données indépendantes).

Usage : python q4_familles.py   (écrit partie2/results/q4_*.csv, q4_familles_tables.md et partie2/figures/q4_*.png)
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
from q3_prior import gau_run                                           # classifieurs linéaires, prior appris (question 3)

RESULTS = os.path.join(HERE, "..", "results")
FIGURES = os.path.join(HERE, "..", "figures")

DATASETS = ["wdbc", "pima", "australian", "iono", "german", "spambase"]
N_SPLITS = 10
KS_TABLE = [1, 5, 20]
KS_CURVE = [1, 2, 3, 5, 10, 20, 50, 100, 200]          # + « tous les stumps »
POST = {"stumps_A": "Stumps, borne A", "stumps_B": "Stumps, borne B", "dirac": "Dirac sur le meilleur stump",
        "top5": "Uniforme sur les 5 meilleurs", "top20": "Uniforme sur les 20 meilleurs",
        "lin_A": "Linéaire, borne A", "lin_B": "Linéaire, borne B"}


# ----------------------------------------------------------------------------------------------------
# Stumps : prior appris, posteriors A, B, Dirac, uniforme sur les k meilleurs
# ----------------------------------------------------------------------------------------------------
def cat_full(X, y, seed):
    Xp, yp, Xs, ys, Xte, yte = pb.split(X, y, seed)
    H = pb.make_stumps(Xp)
    V, Vte = pb.stump_votes(H, Xs), pb.stump_votes(H, Xte)
    P = np.full(len(H), 1 / len(H))
    P = np.maximum(pb.gibbs_posterior(P, pb.voter_risks(pb.stump_votes(H, Xp), yp), len(yp) / 4), 1e-300)   # prior appris
    P /= P.sum()
    L, m = pb.voter_risks(V, ys), len(ys)
    A, _ = pb.learn_linear_categorical(P, L, m)
    B = pb.learn_mcallester_categorical(P, L, m)

    def evaluate(Q):
        emp, KL = float(Q @ L), pb.kl_categorical(Q, P)
        return dict(seeger=pb.bound_seeger(emp, KL, m), KL=KL, emp=emp, neff=1 / float(np.sum(Q ** 2)),
                    test_gibbs=pb.gibbs_risk(Q, Vte, yte), test_vote=pb.vote_risk(Q, Vte, yte))

    def top_k(k):
        Q = np.zeros(len(H)); Q[order[:k]] = 1.0 / k
        return Q
    order = np.argsort(L, kind="stable")
    posts = {"stumps_A": A["Q"], "stumps_B": B["Q"], "dirac": top_k(1), "top5": top_k(5), "top20": top_k(20)}
    curve = [dict(k=k, all=False, **evaluate(top_k(k))) for k in KS_CURVE if k < len(H)]
    last = dict(k=len(H), all=True, **evaluate(top_k(len(H))))
    last["test_vote"] = np.nan      # stumps par paires (h, -h) : le vote uniforme est toujours nul, donc sans signification
    curve.append(last)
    return {name: evaluate(Q) for name, Q in posts.items()}, curve, (A, B, evaluate)


def verifier():
    n = 0
    for name, seeds in [("wdbc", (0, 1)), ("pima", (2,)), ("iono", (3,))]:
        X, y = pb.real_dataset(name)
        for seed in seeds:
            posts, _, _ = cat_full(X, y, seed)
            ref = pb.run_categorical(X, y, seed=seed, prior="learned")
            for key, label in (("stumps_A", "A (linear)"), ("stumps_B", "B (McAllester)")):
                for k, v in ref[label].items():
                    if k in posts[key]:
                        assert np.isclose(posts[key][k], v, rtol=1e-8, atol=1e-10), (name, seed, key, k, posts[key][k], v)
            n += 1
    return n


def collecter():
    rows, curves = [], []
    for name in DATASETS:
        X, y = pb.real_dataset(name)
        for seed in range(N_SPLITS):
            posts, curve, _ = cat_full(X, y, seed)
            for p, r in posts.items():
                rows.append(dict(dataset=name, split=seed, post=p, **r))
            for c in curve:
                curves.append(dict(dataset=name, split=seed, **c))
            out, _ = gau_run(X, y, seed, True)
            for k in ("A", "B"):
                rows.append(dict(dataset=name, split=seed, post=f"lin_{k}", seeger=out[k]["seeger"], KL=out[k]["KL"],
                                 emp=out[k]["emp"], neff=np.nan, test_gibbs=out[k]["test_gibbs"], test_vote=out[k]["test_vote"]))
        print(f"[{name}] fini", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(curves)


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


def tableaux(rows, curves, n_verifs):
    out = ["# Question 4 : stumps contre classifieurs linéaires, et autres posteriors",
           f"{len(DATASETS)} jeux, {N_SPLITS} découpages, prior appris pour les deux familles. Moyenne ± écart-type sur les découpages. "
           f"Les posteriors de stumps avec les bornes A et B ont été vérifiés identiques à ceux de `run_categorical` (prior appris) "
           f"sur {n_verifs} cas.\n"]
    head = ["Jeu", "Posterior", "Certificat de Seeger", "KL", "Risque de Gibbs (test)", "Risque du vote (test)", "Gibbs - vote"]

    def ligne(d, p):
        s = sel(rows, dataset=d, post=p)
        return [d, POST[p], ms(s.seeger), ms(s.KL, 2), ms(s.test_gibbs), ms(s.test_vote), num((s.test_gibbs - s.test_vote).mean(), 3)]
    out += ["## Tableau 1 : les deux familles, avec une borne minimisée (A ou B)",
            md_table(head, [ligne(d, p) for d in DATASETS for p in ("stumps_A", "stumps_B", "lin_A", "lin_B")]), "",
            "## Tableau 2 : autres posteriors de stumps (Dirac sur le meilleur, uniforme sur les 5 et les 20 meilleurs)",
            "Pour comparaison, le posterior de la borne A. Un Dirac a un vote égal à son stump : « Gibbs - vote » y vaut 0.\n",
            md_table(head, [ligne(d, p) for d in DATASETS for p in ("dirac", "top5", "top20", "stumps_A")]), ""]

    # --- Tableau 3 : l'avantage du vote sur Gibbs -----------------------------------------------------------
    table = []
    for d in DATASETS:
        row = [d]
        for p in ("stumps_A", "lin_A"):
            s = sel(rows, dataset=d, post=p)
            row += [num((s.test_gibbs - s.test_vote).mean(), 3), num((s.test_vote / s.test_gibbs).mean(), 2)]
        table.append(row)
    for lab, ps in (("**moyenne des 6 jeux**", None),):
        row = [lab]
        for p in ("stumps_A", "lin_A"):
            s = rows[rows.post == p]
            row += [num((s.test_gibbs - s.test_vote).mean(), 3), num((s.test_vote / s.test_gibbs).mean(), 2)]
        table.append(row)
    out += ["## Tableau 3 : de combien le vote est-il meilleur que le classifieur de Gibbs ?",
            "Écart Gibbs - vote (positif : le vote est meilleur) et rapport vote / Gibbs (1 = aucun avantage ; 0,5 = vote deux fois meilleur), "
            "avec la borne A.\n",
            md_table(["Jeu", "Stumps : Gibbs - vote", "Stumps : vote / Gibbs", "Linéaire : Gibbs - vote", "Linéaire : vote / Gibbs"], table), ""]

    # --- Tableau 4 : le meilleur k pour le vote ----------------------------------------------------------------
    table = []
    for d in DATASETS:
        c = curves[curves.dataset == d]
        mu = c.groupby("k").agg(vote=("test_vote", "mean"), gibbs=("test_gibbs", "mean"), seeger=("seeger", "mean"), allk=("all", "max"))
        kbest = int(mu.vote.idxmin())
        v1, vbest = mu.loc[1, "vote"], mu.loc[kbest, "vote"]
        vA = sel(rows, dataset=d, post="stumps_A").test_vote.mean()
        vlin = sel(rows, dataset=d, post="lin_A").test_vote.mean()
        lab = "tous" if mu.loc[kbest, "allk"] else str(kbest)
        table.append([d, lab, num(vbest, 3), num(v1, 3), num(vA, 3), num(vlin, 3), num(mu.loc[kbest, "gibbs"], 3), num(mu.loc[1, "gibbs"], 3)])
    out += ["## Tableau 4 : le meilleur k pour le vote (uniforme sur les k meilleurs stumps)",
            "k de la grille qui minimise le risque du vote moyen en test ; ce risque, comparé à celui du Dirac (k = 1), du posterior de la borne A "
            "et du classifieur linéaire (borne A) ; risque de Gibbs à ce k et au Dirac. « tous » = uniforme sur tous les stumps. C'est un diagnostic "
            "(le k est choisi sur le test), pas une méthode.\n",
            md_table(["Jeu", "Meilleur k pour le vote", "Vote à ce k", "Vote du Dirac (k = 1)", "Vote du posterior A", "Vote du linéaire A",
                      "Gibbs à ce k", "Gibbs du Dirac"], table), ""]

    # --- Tableau 5 : quelle famille a le meilleur certificat ? ------------------------------------------------------
    table, tot = [], [0, 0]
    for d in DATASETS:
        s, l = sel(rows, dataset=d, post="stumps_A"), sel(rows, dataset=d, post="lin_A")
        win = int((l.seeger < s.seeger).sum())
        tot[0] += win; tot[1] += len(s)
        table.append([d, num(s.seeger.mean(), 3), num(l.seeger.mean(), 3), f"{win}/{len(s)}", num(s.test_vote.mean(), 3), num(l.test_vote.mean(), 3)])
    out += ["## Tableau 5 : quelle famille a le meilleur certificat et le meilleur vote ? (borne A)",
            md_table(["Jeu", "Certificat stumps", "Certificat linéaire", "Linéaire meilleur certificat", "Vote stumps", "Vote linéaire"], table), "",
            f"Le classifieur linéaire a un meilleur certificat de Seeger que les stumps dans {tot[0]}/{tot[1]} exécutions.", ""]
    return out


# ----------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------
def fig_familles(rows, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    x = np.arange(len(DATASETS)); w = 0.26
    for ax, (col, titre) in zip(axes, (("seeger", "Certificat de Seeger"), ("test_gibbs", "Risque de Gibbs (test)"), ("test_vote", "Risque du vote (test)"))):
        for j, (p, c) in enumerate((("stumps_A", "tab:blue"), ("dirac", "tab:cyan"), ("lin_A", "tab:orange"))):
            ax.bar(x + (j - 1) * w, [sel(rows, dataset=d, post=p)[col].mean() for d in DATASETS], w * 0.95, color=c, label=POST[p])
        ax.set_xticks(x); ax.set_xticklabels(DATASETS, rotation=20); ax.set_title(titre)
    axes[0].legend(fontsize=7)
    fig.suptitle("Stumps contre classifieurs linéaires (prior appris, moyenne sur 10 découpages)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_k(rows, curves, path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    for ax, d in zip(axes.ravel(), DATASETS):
        c = curves[curves.dataset == d]
        mu = c.groupby("k").mean(numeric_only=True)
        x = mu.index.values
        ax.plot(x, mu.seeger, "o-", ms=3, color="tab:green", label="certificat de Seeger")
        ax.plot(x, mu.test_gibbs, "o-", ms=3, color="tab:red", label="risque de Gibbs (test)")
        ax.plot(x, mu.test_vote, "o-", ms=3, color="tab:purple", label="risque du vote (test)")
        ax.axhline(sel(rows, dataset=d, post="stumps_A").test_vote.mean(), color="tab:purple", ls="--", lw=1, label="vote du posterior de la borne A")
        ax.axhline(sel(rows, dataset=d, post="lin_A").test_vote.mean(), color="k", ls=":", lw=1.3, label="vote du classifieur linéaire (borne A)")
        ax.set(xscale="log", xlabel="k (uniforme sur les k meilleurs stumps ; dernier point = tous)", title=d)
    axes[0, 0].legend(fontsize=6.5); axes[0, 0].set_ylabel("risque / certificat"); axes[1, 0].set_ylabel("risque / certificat")
    fig.suptitle("Uniforme sur les k meilleurs stumps : Gibbs, vote et certificat en fonction de k (prior appris)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    os.makedirs(RESULTS, exist_ok=True); os.makedirs(FIGURES, exist_ok=True)
    t0 = time.perf_counter()
    n = verifier()
    print(f"{n} cas identiques à pb.run_categorical (prior appris)", flush=True)
    rows, curves = collecter()
    rows.to_csv(os.path.join(RESULTS, "q4_runs.csv"), index=False)
    curves.to_csv(os.path.join(RESULTS, "q4_k_curve.csv"), index=False)
    with open(os.path.join(RESULTS, "q4_familles_tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(tableaux(rows, curves, n)) + "\n")
    fig_familles(rows, os.path.join(FIGURES, "q4_familles.png"))
    fig_k(rows, curves, os.path.join(FIGURES, "q4_courbe_k.png"))
    print(f"terminé en {time.perf_counter() - t0:.0f} s : partie2/results/q4_familles_tables.md et partie2/figures/q4_*.png")


if __name__ == "__main__":
    main()
