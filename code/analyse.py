"""analyse.py : l'analyse statistique des expériences de la partie 1

CE QUE DEMANDE LE SUJET
    Partie 1, « Analyse » : tableaux et figures de résultats, rangs moyens, éventuellement un test
    statistique pour souligner la significativité d'une méthode, et des conclusions qui répondent à
    quatre questions : (1) quel gain des ensembles par rapport à un arbre seul et à un modèle d'une autre
    famille ? (2) bagging ou boosting, sur quels jeux et pourquoi ? (3) le stacking bat-il le meilleur de
    ses membres, et à quel coût ? (4) sur les jeux déséquilibrés, quelles méthodes souffrent et que change
    la métrique ? Il demande aussi de montrer comment le classement change avec la métrique.

CE QUE J'AI FAIT DANS CE FICHIER
    Je lis results/exp_raw.csv (résultat de run_experiments.py) et je produis, sans rien recopier à la
    main :
      - results/analyse_tables.md : les tableaux du rapport, une section par question du sujet ;
      - results/analyse_annexe.md : les résultats complets (moyenne ± écart-type sur les 10 découpages,
        pour chaque jeu, méthode et métrique) ;
      - figures/cd/ : un diagramme de différences critiques par métrique et par groupe de jeux ;
      - figures/rank_by_metric_imbalanced.png : comment le classement change avec la métrique.

MES CHOIX D'ANALYSE ET POURQUOI
    Une valeur par jeu. Pour chaque jeu et chaque méthode, je prends la moyenne de la métrique sur les 10
    découpages. C'est cette valeur qui entre dans les rangs et les tests : chaque jeu compte donc pour un
    seul point, quelle que soit sa taille.
    Les rangs. Comparer des moyennes brutes entre jeux n'a pas de sens (une erreur de 5 % ne veut pas dire
    la même chose sur un jeu facile et sur un jeu difficile). Sur chaque jeu, je classe donc les méthodes
    (rang 1 = la meilleure, rang moyen en cas d'égalité), puis je moyenne les rangs sur les jeux. C'est la
    démarche de Demšar (2006), citée dans la bibliographie du sujet.
    Test de Friedman, puis Nemenyi. Le test de Friedman répond à « les méthodes sont-elles toutes
    équivalentes ? » à partir des rangs. S'il rejette (p < 0,05), le test de Nemenyi dit quelles paires
    diffèrent : deux méthodes sont significativement différentes si leurs rangs moyens s'écartent de plus
    de la différence critique CD = q * racine(k(k+1) / (6N)), avec k méthodes et N jeux. La valeur q est
    calculée par scipy (loi de l'étendue studentisée) ; elle redonne bien les valeurs du tableau de
    Demšar (par exemple 3,164 pour k = 10). Niveau de risque : 5 %.
    Wilcoxon et Holm pour les comparaisons ciblées. Pour les questions précises (une méthode contre
    l'arbre seul, bagging contre boosting...), j'utilise le test de Wilcoxon par paires sur les 25 jeux
    (ou les 10 déséquilibrés), avec la correction de Holm, qui tient compte du fait qu'on fait plusieurs
    comparaisons à la fois.
    Métrique mise en avant selon le jeu : accuracy pour un jeu équilibré, balanced accuracy pour un jeu
    déséquilibré (voir datasets.py). Les rangs sont calculés pour les 4 métriques, séparément, sur trois
    groupes : tous les jeux, les équilibrés, les déséquilibrés.
    Stacking contre ses membres. Je compare le stacking à ses 4 membres (arbre de profondeur 5, kNN,
    régression logistique, Naive Bayes) : au meilleur membre jeu par jeu (choisi après coup, donc
    favorable aux membres), et au membre qui a la meilleure moyenne sur tous les jeux. Le coût est le
    rapport entre le temps d'entraînement du stacking et la somme de ceux de ses membres.

LIMITES QUE J'ASSUME
    - Les jeux ne sont pas tous indépendants : abalone8 et abalone17 viennent du même fichier, comme
      yeast3 et yeast6. Friedman suppose des jeux indépendants ; je le signale dans le rapport.
    - Avec 10 jeux déséquilibrés seulement, la différence critique est grande : peu de différences
      ressortent comme significatives, même quand elles existent.
    - Les jeux à 7 ou 10 exemples minoritaires dans le test (libras, yeast6) donnent des F1 très bruités.
    - Les hyperparamètres ne sont pas réglés jeu par jeu : un résultat valable pour ces réglages-là.

Usage : python analyse.py
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import datasets as ds
from ensembles import default_base_models
from run_experiments import MAIN_METHODS, MEMBER_METHODS, LABELS
from tables import md_table, num

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIGURES = os.path.join(HERE, "..", "figures")

ALPHA = 0.05
METRICS = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
           "f1_minority": "F1 (minorité)", "auc": "AUC"}
BAGGING_TYPE = ["bagging", "rf"]
BOOSTING_TYPE = ["adaboost_d1", "adaboost_d3", "gb", "hgb"]
STACK_MEMBERS = ["tree_d5", "knn", "logreg", "nb"]
EPS = 1e-9                                   # en dessous, deux moyennes sont considérées égales


# ----------------------------------------------------------------------------------------------------
# Outils statistiques
# ----------------------------------------------------------------------------------------------------
def nemenyi_cd(k, n):
    """Différence critique de Nemenyi pour k méthodes et n jeux (Demšar 2006)."""
    q = stats.studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6 * n)))


def avg_ranks(M):
    """Rang moyen de chaque colonne (méthode) ; rang 1 = valeur la plus haute ; égalités : rang moyen."""
    return M.rank(axis=1, ascending=False, method="average").mean()


def friedman_p(M):
    return float(stats.friedmanchisquare(*[M[c].values for c in M.columns]).pvalue)


def fmt_p(p):
    return "< 0,001" if p < 0.001 else num(p, 3)


def holm(pvals):
    """Correction de Holm : p ajustés, dans l'ordre d'origine."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


def wilcoxon_p(d):
    d = np.asarray(d, float)
    if np.all(np.abs(d) < EPS):
        return 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.wilcoxon(d).pvalue)


def wtl(d):
    """(victoires, égalités, défaites) de la première méthode d'après les différences d."""
    d = np.asarray(d, float)
    return int((d > EPS).sum()), int((np.abs(d) <= EPS).sum()), int((d < -EPS).sum())


def profile_stacking(name="bankmarketing", seed=0):
    """Temps passé par chaque membre du stacking, sur 5 plis, à s'entraîner et à prédire le pli écarté
    (même procédure que MyStacking.fit). Un seul cœur, pour que les temps soient comparables."""
    X, y = ds.load(name)
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=ds.TEST_SIZE, stratify=y, random_state=seed)
    base = [("tree_d5" if n == "tree" else n, est) for n, est in default_base_models(0)]   # mêmes noms que STACK_MEMBERS
    fit_t = {n: 0.0 for n, _ in base}
    oof_t = {n: 0.0 for n, _ in base}
    with threadpool_limits(limits=1):
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, ytr):
            for n, est in base:
                m = clone(est)
                t = time.perf_counter(); m.fit(Xtr[tr], ytr[tr]); fit_t[n] += time.perf_counter() - t
                t = time.perf_counter(); m.predict_proba(Xtr[va]); oof_t[n] += time.perf_counter() - t
    return fit_t, oof_t


# ----------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------
def cd_diagram(ranks, cd, title, path):
    """Diagramme de différences critiques : les méthodes reliées par une barre ne diffèrent pas
    significativement (leurs rangs moyens sont à moins de CD l'un de l'autre)."""
    order = ranks.sort_values()
    k = len(order)
    r = order.values
    fig, ax = plt.subplots(figsize=(9, 3.3))
    ax.set_xlim(-3.2, k + 4.2); ax.axis("off")
    y0 = 0.55
    ax.plot([1, k], [y0, y0], "k", lw=1)
    for t in range(1, k + 1):
        ax.plot([t, t], [y0, y0 + 0.02], "k", lw=1)
        ax.text(t, y0 + 0.04, str(t), ha="center", fontsize=8)
    half = (k + 1) // 2
    for i, (m, rk) in enumerate(order.items()):
        if i < half:                                   # meilleures méthodes : étiquettes à gauche
            y = y0 + 0.08 + 0.07 * i
            ax.plot([rk, rk, 0.7], [y0, y, y], "k", lw=0.7)
            ax.text(0.6, y, f"{LABELS[m]}  ({rk:.2f})", ha="right", va="center", fontsize=8)
        else:                                          # moins bonnes : étiquettes à droite
            y = y0 + 0.08 + 0.07 * (k - 1 - i)
            ax.plot([rk, rk, k + 0.3], [y0, y, y], "k", lw=0.7)
            ax.text(k + 0.4, y, f"({rk:.2f})  {LABELS[m]}", ha="left", va="center", fontsize=8)
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and r[j + 1] - r[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for lvl, (i, j) in enumerate(cliques):
        ax.plot([r[i] - 0.05, r[j] + 0.05], [y0 - 0.07 - 0.05 * lvl] * 2, "tab:red", lw=3)
    ax.set_ylim(y0 - 0.07 - 0.05 * max(len(cliques), 1) - 0.02, 1.0)
    ax.set_title(f"{title}   (CD = {cd:.2f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def rank_by_metric_plot(rank_tbl, path):
    """Le rang moyen de chaque méthode (jeux déséquilibrés) pour chacune des 4 métriques."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(METRICS))
    colors = {}
    for m in MAIN_METHODS:
        y = [rank_tbl[k][m] for k in METRICS]
        colors[m] = ax.plot(x, y, "o-", lw=1.5)[0].get_color()
    # Étiquettes à droite, décalées vers le bas pour ne pas se chevaucher (écart minimal 0,22 en rang).
    ends = sorted(MAIN_METHODS, key=lambda m: rank_tbl[list(METRICS)[-1]][m])
    pos, prev = {}, -np.inf
    for m in ends:
        prev = pos[m] = max(rank_tbl[list(METRICS)[-1]][m], prev + 0.22)
    for m in MAIN_METHODS:
        ax.text(x[-1] + 0.06, pos[m], LABELS[m], fontsize=7, va="center", color=colors[m])
    ax.set_xticks(x); ax.set_xticklabels(list(METRICS.values()))
    ax.set_xlim(-0.2, len(METRICS) - 1 + 1.1)
    ax.invert_yaxis()
    ax.set_ylabel("rang moyen (1 = meilleur)")
    ax.set_title("Jeux déséquilibrés : le classement change avec la métrique")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------
# Construction des tableaux
# ----------------------------------------------------------------------------------------------------
class Analysis:
    def __init__(self):
        raw = pd.read_csv(os.path.join(RESULTS, "exp_raw.csv"))
        self.raw = raw
        self.means = raw.groupby(["dataset", "method"]).mean(numeric_only=True)
        self.stds = raw.groupby(["dataset", "method"]).std(numeric_only=True)
        self.info = ds.describe().set_index("dataset")
        by_rate = self.info.sort_values("minority_rate")
        self.imb = [d for d in by_rate.index if self.info.loc[d, "imbalanced"]]
        self.bal = [d for d in by_rate.index[::-1] if not self.info.loc[d, "imbalanced"]]
        self.groups = {"tous les jeux": self.bal + self.imb, "jeux équilibrés": self.bal, "jeux déséquilibrés": self.imb}
        self.out = []

    # --- accès aux données -------------------------------------------------------------------
    def principal(self, d):
        return "balanced_accuracy" if self.info.loc[d, "imbalanced"] else "accuracy"

    def M(self, metric, methods, datasets):
        """Matrice jeux x méthodes des moyennes d'une métrique."""
        return self.means[metric].unstack("method").loc[datasets, methods]

    def P(self, methods, datasets):
        """Idem avec la métrique mise en avant pour chaque jeu (accuracy ou balanced accuracy)."""
        return pd.DataFrame({d: [self.means.loc[(d, m), self.principal(d)] for m in methods] for d in datasets},
                            index=methods).T

    def add(self, *lines):
        self.out += list(lines)

    # --- section 0 ---------------------------------------------------------------------------
    def sec_intro(self):
        self.add("# Analyse de la partie 1",
                 f"Valeur par jeu et par méthode = moyenne sur {self.raw.split.nunique()} découpages. "
                 f"{len(self.info)} jeux ({len(self.bal)} équilibrés, {len(self.imb)} déséquilibrés). "
                 f"Rangs : 1 = meilleur. Tests au risque de {int(ALPHA * 100)} %. Les 10 méthodes principales "
                 "sont comparées ; l'arbre de profondeur 5 et Naive Bayes ne servent qu'à la section 6.\n")

    # --- section 1 : résultats par jeu ------------------------------------------------------
    def sec_results(self):
        self.add("## 1. Résultats par jeu (métrique mise en avant, en %)",
                 "Accuracy pour les jeux équilibrés, balanced accuracy pour les jeux déséquilibrés. "
                 "En gras : la meilleure méthode du jeu. Résultats complets (4 métriques, écarts-types) "
                 "dans `analyse_annexe.md`.\n")
        for title, dsets in [("Jeux équilibrés : accuracy", self.bal), ("Jeux déséquilibrés : balanced accuracy", self.imb)]:
            P = self.P(MAIN_METHODS, dsets)
            rows = []
            for d in dsets:
                best = P.loc[d].max()
                rows.append([d, self.info.loc[d, "m"]] + [("**%s**" if P.loc[d, m] == best else "%s") % num(100 * P.loc[d, m], 1)
                                                         for m in MAIN_METHODS])
            rows.append(["**rang moyen**", ""] + [num(v, 2) for v in avg_ranks(P)[MAIN_METHODS]])
            self.add(f"### {title}", md_table(["Jeu", "m"] + [LABELS[m] for m in MAIN_METHODS], rows), "")

    # --- section 2 : rangs moyens, Friedman, Nemenyi ---------------------------------------
    def sec_ranks(self):
        self.add("## 2. Rangs moyens, test de Friedman et différence critique de Nemenyi",
                 "En gras : le meilleur rang moyen. Deux méthodes dont les rangs moyens diffèrent de moins "
                 "que CD ne sont pas significativement différentes. Les diagrammes de différences critiques "
                 "sont dans `figures/cd/`.\n")
        os.makedirs(os.path.join(FIGURES, "cd"), exist_ok=True)
        self.ranks = {}
        for gname, dsets in self.groups.items():
            n, k = len(dsets), len(MAIN_METHODS)
            cd = nemenyi_cd(k, n)
            R = {m: avg_ranks(self.M(m, MAIN_METHODS, dsets)) for m in METRICS}
            self.ranks[gname] = R
            rows = []
            for meth in MAIN_METHODS:
                rows.append([LABELS[meth]] + [("**%s**" if R[m][meth] == R[m].min() else "%s") % num(R[m][meth], 2)
                                              for m in METRICS])
            rows.append(["*p de Friedman*"] + [fmt_p(friedman_p(self.M(m, MAIN_METHODS, dsets))) for m in METRICS])
            self.add(f"### {gname.capitalize()} (N = {n}, CD = {num(cd, 2)})",
                     md_table(["Méthode"] + list(METRICS.values()), rows), "")
            tag = {"tous les jeux": "tous", "jeux équilibrés": "equilibres", "jeux déséquilibrés": "desequilibres"}[gname]
            for m in METRICS:
                cd_diagram(R[m], cd, f"{METRICS[m]} - {gname}", os.path.join(FIGURES, "cd", f"cd_{tag}_{m}.png"))

    # --- section 3 : classement selon la métrique (déséquilibrés) ---------------------------
    def sec_imbalanced(self):
        R = self.ranks["jeux déséquilibrés"]
        self.add("## 3. Jeux déséquilibrés : le classement change avec la métrique",
                 "Rang moyen sur les jeux déséquilibrés pour chaque métrique, avec la position entre "
                 "parenthèses. La dernière colonne donne le nombre de places gagnées (+) ou perdues (-) "
                 "quand on passe de l'accuracy à la balanced accuracy. Figure : "
                 "`figures/rank_by_metric_imbalanced.png`.\n")
        pos = {m: R[m].rank(method="min").astype(int) for m in METRICS}
        rows = [[LABELS[meth]] + [f"{num(R[m][meth], 2)} ({pos[m][meth]})" for m in METRICS]
                + [f"{pos['accuracy'][meth] - pos['balanced_accuracy'][meth]:+d}"] for meth in MAIN_METHODS]
        self.add(md_table(["Méthode"] + list(METRICS.values()) + ["Places gagnées (acc. → bal. acc.)"], rows), "")
        keys = list(METRICS)
        rows = []
        for a in keys:
            rows.append([METRICS[a]] + [num(stats.kendalltau(R[a], R[b])[0], 2) for b in keys])
        self.add("Corrélation de Kendall entre les classements de méthodes selon chaque métrique "
                 "(1 = même classement) :\n", md_table([""] + list(METRICS.values()), rows), "")
        rank_by_metric_plot(R, os.path.join(FIGURES, "rank_by_metric_imbalanced.png"))

        self.add("### L'accuracy est trompeuse : comparaison avec la réponse « toujours la classe majoritaire »",
                 "Accuracy d'un classifieur qui répond toujours la classe majoritaire, contre la meilleure "
                 "accuracy et la meilleure balanced accuracy obtenues par les 10 méthodes.\n")
        rows = []
        for d in self.imb:
            acc = self.M("accuracy", MAIN_METHODS, [d]).iloc[0]
            bal = self.M("balanced_accuracy", MAIN_METHODS, [d]).iloc[0]
            rows.append([d, num(100 * self.info.loc[d, "minority_rate"], 1), num(100 * (1 - self.info.loc[d, "minority_rate"]), 1),
                         f"{num(100 * acc.max(), 1)} ({LABELS[acc.idxmax()]})",
                         f"{num(100 * bal.max(), 1)} ({LABELS[bal.idxmax()]})"])
        self.add(md_table(["Jeu", "Minorité (%)", "Acc. toujours-majoritaire (%)", "Meilleure accuracy (%)",
                           "Meilleure bal. acc. (%)"], rows), "")

    # --- tests par paires ---------------------------------------------------------------------
    def pair_rows(self, pairs, settings):
        """pairs : liste de (a, b) ; settings : liste de (métrique, liste de jeux). Pour chaque réglage :
        différence moyenne a - b (en points), victoires/égalités/défaites de a, p de Wilcoxon corrigé (Holm)."""
        cells = {}
        for si, (metric, dsets) in enumerate(settings):
            ps, stats_ = [], []
            for a, b in pairs:
                d = (self.M(metric, [a], dsets)[a] - self.M(metric, [b], dsets)[b]).values
                stats_.append((100 * d.mean(), wtl(d)))
                ps.append(wilcoxon_p(d))
            adj = holm(ps)
            for pi, ((delta, w), p) in enumerate(zip(stats_, adj)):
                cells[(pi, si)] = [f"{delta:+.2f}".replace(".", ","), "%d/%d/%d" % w, fmt_p(p)]
        return [[LABELS[a] + (f" - {LABELS[b]}" if b is not None else "")] + sum((cells[(pi, si)] for si in range(len(settings))), [])
                for pi, (a, b) in enumerate(pairs)]

    def sec_gain(self):
        self.add("## 4. Gain des ensembles par rapport à un arbre seul et à une autre famille de modèles",
                 "Pour chaque ensemble : différence moyenne de la métrique en points (ensemble moins référence), "
                 "victoires/égalités/défaites sur les jeux (G/N/P), et p de Wilcoxon par paires corrigé par "
                 "Holm. Trois situations : accuracy sur les 25 jeux, balanced accuracy sur les 10 jeux "
                 "déséquilibrés, AUC sur les 25 jeux.\n")
        settings = [("accuracy", self.groups["tous les jeux"]), ("balanced_accuracy", self.imb),
                    ("auc", self.groups["tous les jeux"])]
        head = ["Méthode"] + sum(([f"Δ {l}", f"G/N/P {l}", f"p {l}"] for l in ("acc.", "bal. acc.", "AUC")), [])
        ens = ["bagging", "rf", "adaboost_d1", "adaboost_d3", "gb", "hgb", "stacking"]
        for ref, title in [("tree", "Référence : arbre seul"), ("logreg", "Référence : régression logistique"),
                           ("knn", "Référence : kNN")]:
            rows = self.pair_rows([(m, ref) for m in ens], settings)
            rows = [[LABELS[m]] + r[1:] for m, r in zip(ens, rows)]
            self.add(f"### {title}", md_table(head, rows), "")

    # --- section 5 : bagging contre boosting ---------------------------------------------------
    def sec_bag_boost(self):
        self.add("## 5. Bagging ou boosting ?",
                 "### 5.1 Meilleur du type bagging (bagging, forêt aléatoire) contre meilleur du type boosting "
                 "(AdaBoost stumps, AdaBoost prof. 3, gradient boosting, Hist. gradient boosting), par jeu, "
                 "sur la métrique mise en avant. Attention : le boosting a 4 candidats contre 2, ce qui le "
                 "favorise légèrement. Jeux triés par taille.\n")
        dsets = sorted(self.groups["tous les jeux"], key=lambda d: self.info.loc[d, "m"])
        P = self.P(BAGGING_TYPE + BOOSTING_TYPE, dsets)
        rows, winners = [], {}
        for d in dsets:
            bag, boo = P.loc[d, BAGGING_TYPE], P.loc[d, BOOSTING_TYPE]
            diff = boo.max() - bag.max()
            w = "boosting" if diff > EPS else ("bagging" if diff < -EPS else "égalité")
            winners[d] = w
            rows.append([d, self.info.loc[d, "m"], self.info.loc[d, "d"], num(100 * self.info.loc[d, "minority_rate"], 1),
                         f"{num(100 * bag.max(), 1)} ({LABELS[bag.idxmax()]})", f"{num(100 * boo.max(), 1)} ({LABELS[boo.idxmax()]})",
                         f"{100 * diff:+.2f}".replace(".", ","), w])
        self.add(md_table(["Jeu", "m", "d", "Minorité (%)", "Meilleur bagging (%)", "Meilleur boosting (%)",
                           "Boosting - bagging (pts)", "Vainqueur"], rows), "")
        wser = pd.Series(winners)
        size = lambda d: self.info.loc[d, "m"]
        bins = [("m < 1000", lambda d: size(d) < 1000), ("1000 ≤ m < 5000", lambda d: 1000 <= size(d) < 5000),
                ("m ≥ 5000", lambda d: size(d) >= 5000), ("jeux équilibrés", lambda d: d in self.bal),
                ("jeux déséquilibrés", lambda d: d in self.imb)]
        rows = []
        for lab, f in bins:
            sel = [d for d in dsets if f(d)]
            c = wser[sel].value_counts()
            rows.append([lab, len(sel), int(c.get("boosting", 0)), int(c.get("bagging", 0)), int(c.get("égalité", 0))])
        self.add("### 5.2 Qui l'emporte selon la taille et le déséquilibre", md_table(
            ["Groupe", "Nb jeux", "Boosting gagne", "Bagging gagne", "Égalité"], rows), "")
        settings = [("accuracy", self.groups["tous les jeux"]), ("balanced_accuracy", self.imb)]
        pairs = [("bagging", "adaboost_d1"), ("bagging", "adaboost_d3"), ("rf", "gb"), ("rf", "hgb"), ("rf", "adaboost_d3")]
        rows = self.pair_rows(pairs, settings)
        self.add("### 5.3 Comparaisons directes",
                 "Différence (première méthode moins seconde), G/N/P = victoires/égalités/défaites de la "
                 "première, p de Wilcoxon corrigé par Holm. Accuracy sur 25 jeux ; balanced accuracy sur 10 jeux "
                 "déséquilibrés.\n",
                 md_table(["Comparaison", "Δ acc.", "G/N/P acc.", "p acc.", "Δ bal. acc.", "G/N/P bal. acc.", "p bal. acc."], rows), "")

    # --- section 6 : stacking ---------------------------------------------------------------------
    def sec_stacking(self):
        dsets = self.groups["tous les jeux"]
        cols = STACK_MEMBERS + ["stacking"]
        P = self.P(cols, dsets)
        self.add("## 6. Le stacking bat-il le meilleur de ses membres ? À quel coût ?",
                 "### 6.1 Résultats par jeu (métrique mise en avant, en %)\n")
        rows = []
        for d in dsets:
            mem = P.loc[d, STACK_MEMBERS]
            best = mem.max()
            rows.append([d] + [("**%s**" if P.loc[d, c] == P.loc[d, cols].max() else "%s") % num(100 * P.loc[d, c], 1) for c in cols]
                        + [LABELS[mem.idxmax()], f"{100 * (P.loc[d, 'stacking'] - best):+.2f}".replace(".", ",")])
        self.add(md_table(["Jeu"] + [LABELS[c] for c in cols] + ["Meilleur membre", "Stacking - meilleur membre (pts)"], rows), "")
        best_each = P[STACK_MEMBERS].max(axis=1)
        best_avg = avg_ranks(P[STACK_MEMBERS]).idxmin()          # meilleur rang moyen parmi les 4 membres
        rows = []
        for lab, ref in [("meilleur membre de chaque jeu (choisi après coup)", best_each),
                         (f"{LABELS[best_avg]}, membre au meilleur rang moyen", P[best_avg])]:
            d = (P["stacking"] - ref).values
            rows.append([lab, f"{100 * d.mean():+.2f}".replace(".", ","), "%d/%d/%d" % wtl(d), fmt_p(wilcoxon_p(d))])
        self.add("### 6.2 Stacking contre ses membres (25 jeux)",
                 md_table(["Référence", "Stacking - référence (pts)", "G/N/P du stacking", "p de Wilcoxon"], rows), "")
        t = self.means["t_fit"].unstack("method")
        ratio = t["stacking"] / t[STACK_MEMBERS].sum(axis=1)
        tp = self.means["t_pred"].unstack("method")
        ratio_fp = t["stacking"] / (t[STACK_MEMBERS].sum(axis=1) + tp[STACK_MEMBERS].sum(axis=1))
        rows = [["Somme des temps d'entraînement des 4 membres (s), médiane des jeux", num(t[STACK_MEMBERS].sum(axis=1).median(), 4)],
                ["Temps d'entraînement du stacking (s), médiane des jeux", num(t["stacking"].median(), 4)],
                ["Rapport stacking / somme des entraînements des membres, médiane des jeux", num(ratio.median(), 1)],
                ["Rapport stacking / somme des entraînements des membres, bankmarketing", num(ratio["bankmarketing"], 1)],
                ["Rapport stacking / somme (entraînement + prédiction) des membres, médiane des jeux", num(ratio_fp.median(), 1)],
                ["Rapport stacking / somme (entraînement + prédiction) des membres, bankmarketing", num(ratio_fp["bankmarketing"], 1)]]
        self.add("### 6.3 Coût du stacking",
                 "Le stacking entraîne chaque membre 6 fois (5 plis + un réentraînement final) et doit faire "
                 "prédire chaque pli pour obtenir les prédictions out-of-fold. Un rapport de 6 par rapport à la "
                 "somme des entraînements n'est donc qu'un minimum pour des modèles dont le coût est à "
                 "l'entraînement : un modèle « paresseux » comme le kNN ne coûte presque rien à entraîner, "
                 "son coût est dans la prédiction (voir 6.4). C'est pourquoi je donne aussi le rapport à la "
                 "somme entraînement + prédiction.\n", md_table(["Mesure", "Valeur"], rows), "")
        fit_t, oof_t = profile_stacking()
        rows = [[LABELS[m], num(fit_t[m], 3), num(oof_t[m], 3), num(fit_t[m] + oof_t[m], 3)] for m in STACK_MEMBERS]
        rows.append(["**Total des 4 membres**", num(sum(fit_t.values()), 3), num(sum(oof_t.values()), 3),
                     num(sum(fit_t.values()) + sum(oof_t.values()), 3)])
        self.add("### 6.4 Où va le temps du stacking (bankmarketing, un découpage, un seul cœur)",
                 "Temps passé, pour chaque membre, à s'entraîner sur les 5 plis et à prédire les plis écartés "
                 "(les prédictions out-of-fold). Cela exclut le réentraînement final et le méta-modèle.\n",
                 md_table(["Membre", "Entraînement des 5 plis (s)", "Prédictions out-of-fold (s)", "Total (s)"], rows), "")

    # --- section 7 : coûts --------------------------------------------------------------------------
    def sec_cost(self):
        t = self.means["t_fit"].unstack("method")[MAIN_METHODS]
        tp = self.means["t_pred"].unstack("method")[MAIN_METHODS]
        self.add("## 7. Coût de calcul (un seul cœur)",
                 "Médiane sur les 25 jeux du temps moyen par découpage, et temps sur le plus gros jeu "
                 "(bankmarketing, 31 647 exemples d'entraînement).\n")
        rows = [[LABELS[m], num(t[m].median(), 3), num(t[m].median() / t["tree"].median(), 1), num(t.loc["bankmarketing", m], 2),
                 num(tp[m].median(), 4)] for m in MAIN_METHODS]
        self.add(md_table(["Méthode", "Entraînement, médiane (s)", "Rapport à l'arbre seul", "Entraînement, bankmarketing (s)",
                           "Prédiction, médiane (s)"], rows), "")

    # --- annexe --------------------------------------------------------------------------------------
    def annex(self):
        out = ["# Annexe : résultats complets (moyenne ± écart-type sur les 10 découpages, en %)", ""]
        for m, lab in METRICS.items():
            rows = []
            for d in self.groups["tous les jeux"]:
                rows.append([d] + [f"{num(100 * self.means.loc[(d, k), m], 1)} ± {num(100 * self.stds.loc[(d, k), m], 1)}"
                                   for k in MAIN_METHODS])
            out += [f"## {lab}", md_table(["Jeu"] + [LABELS[k] for k in MAIN_METHODS], rows), ""]
        with open(os.path.join(RESULTS, "analyse_annexe.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(out))


def main():
    os.makedirs(FIGURES, exist_ok=True)
    a = Analysis()
    a.sec_intro(); a.sec_results(); a.sec_ranks(); a.sec_imbalanced(); a.sec_gain()
    a.sec_bag_boost(); a.sec_stacking(); a.sec_cost(); a.annex()
    with open(os.path.join(RESULTS, "analyse_tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(a.out) + "\n")
    print("Écrit : results/analyse_tables.md, results/analyse_annexe.md, figures/cd/, figures/rank_by_metric_imbalanced.png")


if __name__ == "__main__":
    main()
