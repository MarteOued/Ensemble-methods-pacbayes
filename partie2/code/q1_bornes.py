"""q1_bornes.py : question 1 de la partie 2, comparer les trois bornes PAC-bayésiennes avec leurs formules

CE QUE DEMANDE LE SUJET
    Partie 2, question 1 : « À l'aide des trois fonctions bound_*, tracez les bornes en fonction de m (pour un
    risque observé de 0,1 et quelques valeurs de la KL) et de la KL (pour quelques m). Quelle borne est la plus
    serrée ? Comment la borne A dépend-elle de lambda, et quel est le prix de la grille ? Reliez ce que vous
    observez aux formules. »

CE QUE J'AI FAIT DANS CE FICHIER
    Je n'utilise aucune donnée : seulement les trois fonctions bound_linear (A), bound_mcallester (B) et
    bound_seeger du fichier fourni (code/pacbayes_tools.py), avec delta = 0,05 et la grille de lambda du sujet
    (1, 2, 4, ..., 16384, soit 15 valeurs). Les bornes ne dépendent que de quatre nombres : le risque observé,
    la KL, la taille m de l'échantillon et delta ; je peux donc les comparer sans rien apprendre.
    - Bornes en fonction de m (risque observé 0,1 ; KL = 1, 5, 20, 50) et de la KL (m = 100, 500, 1000, 10000).
      Pour la borne A, je prends le meilleur lambda de la grille, c'est ce que fait l'algorithme fourni.
    - Borne A en fonction de lambda : on voit une courbe en U, parce que lambda apparaît dans (KL + ln(|grille|/delta)) / lambda
      (qui diminue quand lambda augmente) et dans lambda / (8m) (qui augmente).
    - Le prix de la grille. Pour une valeur fixée de lambda, la borne A est valable avec probabilité 1 - delta.
      Mais on choisit lambda APRÈS avoir vu les données (le meilleur de la grille) : pour que la borne reste
      valable, on fait une borne de l'union sur les |grille| = 15 valeurs, ce qui remplace ln(1/delta) par
      ln(|grille|/delta). Je mesure ce prix, et je le sépare en deux : le coût de l'union (ln 15 en plus) et le
      coût de la discrétisation (le lambda optimal n'est pas dans la grille).
    - Je relie le tout aux formules. Le lambda optimal d'une borne A (sans contrainte de grille) est
      lambda* = racine(8 m (KL + ln(|grille|/delta))) ; avec lui, la borne vaut exactement
      risque + racine((KL + ln(|grille|/delta)) / (2m)), qui a la même forme que McAllester. La borne de Seeger
      est toujours inférieure à McAllester, parce que l'inégalité de Pinsker donne kl(q, p) >= 2 (p - q)^2.
      Ces deux relations sont VÉRIFIÉES par le code (assertions, sur un grand nombre de cas) : si mon
      raisonnement était faux, le script s'arrêterait.

À QUOI ÇA ME SERT
    C'est le tableau et les trois figures de la question 1 du rapport. Comprendre ces formules me sert aussi pour
    la suite : le certificat que je reporterai à la question 2 est une de ces trois bornes évaluée sur un posterior.

LIMITES QUE J'ASSUME
    - Les figures supposent un risque observé de 0,1 (celui du sujet). La borne de Seeger gagne surtout quand le
      risque observé est petit ; je regarde ce point dans le tableau 4, avec le risque observé qui varie.
    - Les bornes sont des majorants valables avec probabilité 95 % ; ce ne sont pas des estimations du vrai risque.

Usage : python q1_bornes.py   (écrit partie2/results/q1_bornes_tables.md et partie2/figures/q1_bornes_*.png)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))              # code fourni par le prof (pacbayes_tools.py)
sys.path.insert(0, os.path.join(HERE, "..", "..", "partie1", "code"))   # tables.py (écriture des tableaux Markdown)
import pacbayes_tools as pb
from tables import md_table, num

RESULTS = os.path.join(HERE, "..", "results")
FIGURES = os.path.join(HERE, "..", "figures")

EMP = 0.1                                    # risque observé fixé par le sujet
KLS = [1, 5, 20, 50]                         # valeurs de la KL pour les courbes en fonction de m
MS = [100, 500, 1000, 10000]                 # valeurs de m pour les courbes en fonction de la KL
C_GRID = np.log(len(pb.GRID) / pb.DELTA)     # ln(|grille| / delta) : constante de la borne A
COLORS = {"A": "tab:blue", "B": "tab:orange", "S": "tab:green"}


# ----------------------------------------------------------------------------------------------------
# Les trois bornes et les quantités théoriques associées
# ----------------------------------------------------------------------------------------------------
def a_grid(emp, KL, m):
    """Borne A avec le meilleur lambda de la grille (ce que fait learn_linear_*) : (valeur, lambda choisi)."""
    vals = [pb.bound_linear(emp, KL, m, lam) for lam in pb.GRID]
    i = int(np.argmin(vals))
    return vals[i], pb.GRID[i]


def a_ideal(emp, KL, m, union=True):
    """Borne A avec le lambda optimal CONTINU : (valeur, lambda*). union=False retire le coût de la grille."""
    a = KL + (C_GRID if union else np.log(1 / pb.DELTA))
    return emp + np.sqrt(a / (2 * m)), float(np.sqrt(8 * m * a))


def b(emp, KL, m):
    return pb.bound_mcallester(emp, KL, m)


def s(emp, KL, m):
    return pb.bound_seeger(emp, KL, m)


def verifier_les_relations():
    """Vérifie par le calcul les relations théoriques que j'utilise pour interpréter les courbes."""
    n = 0
    for m in (50, 100, 500, 1000, 10_000, 100_000):
        for KL in (0, 1, 5, 20, 50, 100):
            for emp in (0.0, 0.01, 0.1, 0.3, 0.5):
                # (1) Pinsker : Seeger <= McAllester, toujours (à la précision de la bissection près).
                assert s(emp, KL, m) <= b(emp, KL, m) + 1e-9, ("Seeger > McAllester", m, KL, emp)
                # (2) la borne A au lambda optimal vaut exactement emp + racine((KL + ln(|grille|/delta)) / (2m)).
                ai, lam = a_ideal(emp, KL, m)
                assert abs(pb.bound_linear(emp, KL, m, lam) - ai) < 1e-12, ("formule de lambda* fausse", m, KL, emp)
                a0, lam0 = a_ideal(emp, KL, m, union=False)
                assert abs(pb.bound_linear(emp, KL, m, lam0, n_grid=1) - a0) < 1e-12, ("formule sans grille fausse",)
                # (3) le meilleur lambda de la grille ne peut pas faire mieux que le lambda optimal continu.
                assert a_grid(emp, KL, m)[0] >= ai - 1e-12, ("la grille bat l'optimum continu ?", m, KL, emp)
                n += 1
    return n


# ----------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------
def fig_en_fonction_de_m(path):
    ms = np.unique(np.round(np.logspace(np.log10(50), 5, 50)).astype(int))
    fig, axes = plt.subplots(1, len(KLS), figsize=(14, 3.8), sharey=True)
    for ax, KL in zip(axes, KLS):
        ax.plot(ms, [a_grid(EMP, KL, m)[0] for m in ms], color=COLORS["A"], label="A (linéaire, meilleur λ de la grille)")
        ax.plot(ms, [b(EMP, KL, m) for m in ms], color=COLORS["B"], label="B (McAllester)")
        ax.plot(ms, [s(EMP, KL, m) for m in ms], color=COLORS["S"], label="Seeger")
        ax.axhline(EMP, color="grey", ls="--", lw=1, label="risque observé (0,1)")
        ax.set(xscale="log", xlabel="m (taille de l'échantillon)", title=f"KL = {KL}", ylim=(0.05, 1.0))
    axes[0].set_ylabel("borne sur le risque de Gibbs")
    axes[0].legend(fontsize=7)
    fig.suptitle("Les trois bornes en fonction de m (risque observé 0,1, δ = 0,05)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_en_fonction_de_kl(path):
    kls = np.linspace(0, 100, 51)
    fig, axes = plt.subplots(1, len(MS), figsize=(14, 3.8), sharey=True)
    for ax, m in zip(axes, MS):
        ax.plot(kls, [a_grid(EMP, K, m)[0] for K in kls], color=COLORS["A"], label="A (linéaire, meilleur λ de la grille)")
        ax.plot(kls, [b(EMP, K, m) for K in kls], color=COLORS["B"], label="B (McAllester)")
        ax.plot(kls, [s(EMP, K, m) for K in kls], color=COLORS["S"], label="Seeger")
        ax.axhline(EMP, color="grey", ls="--", lw=1, label="risque observé (0,1)")
        ax.set(xlabel="KL(Q‖P)", title=f"m = {m}", ylim=(0.05, 1.0))
    axes[0].set_ylabel("borne sur le risque de Gibbs")
    axes[0].legend(fontsize=7)
    fig.suptitle("Les trois bornes en fonction de la KL (risque observé 0,1, δ = 0,05)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_en_fonction_de_lambda(path):
    lams = np.logspace(0, np.log10(2e4), 400)
    cases = [(100, 10), (1000, 10), (10000, 10), (1000, 1), (1000, 10), (1000, 50)]      # (m, KL)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, (m, KL) in zip(axes.ravel(), cases):
        ax.plot(lams, [pb.bound_linear(EMP, KL, m, lam) for lam in lams], color=COLORS["A"], label="borne A(λ)")
        grid_vals = [pb.bound_linear(EMP, KL, m, lam) for lam in pb.GRID]
        ax.plot(pb.GRID, grid_vals, "o", color=COLORS["A"], ms=4, label="valeurs de la grille")
        best, lam_best = a_grid(EMP, KL, m)
        ax.plot([lam_best], [best], "o", color="red", ms=8, mfc="none", mew=2, label=f"meilleur λ de la grille ({lam_best})")
        ax.axvline(a_ideal(EMP, KL, m)[1], color="k", ls=":", lw=1, label=f"λ* = {a_ideal(EMP, KL, m)[1]:.0f}")
        ax.axhline(b(EMP, KL, m), color=COLORS["B"], lw=1.2, label="B (McAllester)")
        ax.axhline(s(EMP, KL, m), color=COLORS["S"], lw=1.2, label="Seeger")
        ax.set(xscale="log", xlabel="λ", title=f"m = {m}, KL = {KL}", ylim=(0.1, 0.6))
        ax.legend(fontsize=6)
    for ax in axes[:, 0]:
        ax.set_ylabel("borne sur le risque de Gibbs")
    fig.suptitle("Borne A en fonction de λ (risque observé 0,1) : haut, effet de m ; bas, effet de la KL")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------------------------------
# Tableaux
# ----------------------------------------------------------------------------------------------------
def tableaux(n_verifs):
    out = ["# Question 1 : les trois bornes PAC-bayésiennes",
           f"δ = {num(pb.DELTA, 2)}, grille de λ : {pb.GRID[0]}, {pb.GRID[1]}, ..., {pb.GRID[-1]} ({len(pb.GRID)} valeurs, "
           f"ln(|grille|/δ) = {num(C_GRID, 2)}). Risque observé = {num(EMP, 1)} sauf mention contraire. "
           f"Les relations théoriques utilisées pour interpréter les résultats (Seeger ≤ McAllester ; formule du λ optimal) "
           f"ont été vérifiées par assertions sur {n_verifs} combinaisons (m, KL, risque observé).\n"]

    # --- Tableau 1 : comparaison ----------------------------------------------------------------
    rows, best_count = [], {"A": 0, "B": 0, "S": 0}
    for m in (100, 1000, 10000):
        for KL in (1, 5, 20, 50):
            av, lam = a_grid(EMP, KL, m); bv = b(EMP, KL, m); sv = s(EMP, KL, m)
            win = min({"A": av, "B": bv, "S": sv}, key=lambda k: {"A": av, "B": bv, "S": sv}[k])
            best_count[win] += 1
            rows.append([m, KL, num(av, 3), lam, num(bv, 3), num(sv, 3), {"A": "A", "B": "B", "S": "Seeger"}[win]])
    out += ["## Tableau 1 : valeur des trois bornes (risque observé 0,1)",
            md_table(["m", "KL", "A (grille)", "λ choisi", "B (McAllester)", "Seeger", "La plus serrée"], rows), ""]

    # --- Tableau 2 : prix de la grille -------------------------------------------------------------
    rows = []
    for m in (100, 1000, 10000):
        for KL in (1, 5, 20, 50):
            a0, lam_star = a_ideal(EMP, KL, m, union=False)
            a1, _ = a_ideal(EMP, KL, m, union=True)
            ag, _ = a_grid(EMP, KL, m)
            rows.append([m, KL, f"{lam_star:.0f}", num(a0, 4), num(a1, 4), num(ag, 4),
                         num(a1 - a0, 4), num(ag - a1, 4), num(ag - a0, 4)])
    out += ["## Tableau 2 : le prix de la grille",
            "Trois versions de la borne A : (a) λ optimal continu, SANS borne de l'union (valeur de référence, mais non "
            "valable car λ dépend des données) ; (b) λ optimal continu AVEC la borne de l'union sur la grille (ln|grille|) ; "
            "(c) meilleur λ de la grille (borne réellement utilisée). Coût de l'union = (b) - (a) ; coût de la "
            "discrétisation = (c) - (b) ; prix total de la grille = (c) - (a).\n",
            md_table(["m", "KL", "λ* théorique", "(a) sans grille", "(b) avec l'union", "(c) grille réelle",
                      "Coût de l'union", "Coût de la discrétisation", "Prix total"], rows), ""]

    # --- Tableau 3 : les constantes de pénalité --------------------------------------------------
    rows = []
    for m in (50, 100, 1000, 10_000, 100_000):
        cb = np.log(2 * np.sqrt(m) / pb.DELTA)
        rows.append([m, num(C_GRID, 2), num(cb, 2), "A" if C_GRID < cb else "B"])
    out += ["## Tableau 3 : la constante de la pénalité, ln(|grille|/δ) pour A et ln(2√m/δ) pour B",
            f"Les deux bornes ont la forme « risque + racine((KL + constante) / (2m)) » (A au λ optimal). La constante de B "
            f"croît avec m ; celle de A est fixe. Elles sont égales quand 2√m = |grille|, c'est-à-dire m = "
            f"{num((len(pb.GRID) / 2) ** 2, 2)}. Au-delà, la constante de A est plus petite.\n",
            md_table(["m", "A : ln(|grille|/δ)", "B : ln(2√m/δ)", "Constante la plus petite"], rows), ""]

    # --- Tableau 4 : effet du risque observé ----------------------------------------------------------
    rows = []
    for emp in (0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5):
        av, _ = a_grid(emp, 5, 1000); bv = b(emp, 5, 1000); sv = s(emp, 5, 1000)
        d = {"A": av, "B": bv, "Seeger": sv}
        rows.append([num(emp, 2), num(av, 3), num(bv, 3), num(sv, 3), min(d, key=d.get)])
    out += ["## Tableau 4 : effet du risque observé (m = 1000, KL = 5)",
            md_table(["Risque observé", "A (grille)", "B (McAllester)", "Seeger", "La plus serrée"], rows), ""]

    # --- Tableau 5 : A en fonction de lambda --------------------------------------------------------
    m, KL = 1000, 10
    rows = [[lam, num(pb.bound_linear(EMP, KL, m, lam), 4)] for lam in pb.GRID]
    best, lam_best = a_grid(EMP, KL, m)
    out += [f"## Tableau 5 : la borne A selon λ (m = {m}, KL = {KL}, risque observé 0,1)",
            f"Meilleur λ de la grille : {lam_best} (borne {num(best, 4)}). λ* théorique = {a_ideal(EMP, KL, m)[1]:.0f}. "
            f"Pour comparaison, B = {num(b(EMP, KL, m), 4)} et Seeger = {num(s(EMP, KL, m), 4)}.\n",
            md_table(["λ", "Borne A(λ)"], rows), ""]

    # --- balayage global --------------------------------------------------------------------------------
    n = tot = 0
    wins = {"A": 0, "B": 0, "Seeger": 0}
    a_lt_b = 0
    for m in np.unique(np.round(np.logspace(np.log10(50), 5, 20)).astype(int)):
        for KL in (0, 1, 2, 5, 10, 20, 50, 100):
            d = {"A": a_grid(EMP, KL, m)[0], "B": b(EMP, KL, m), "Seeger": s(EMP, KL, m)}
            wins[min(d, key=d.get)] += 1
            a_lt_b += d["A"] < d["B"]
            tot += 1
    out += ["## Balayage global (risque observé 0,1)",
            f"Sur {tot} combinaisons (m de 50 à 100 000, KL de 0 à 100) : la borne la plus serrée est Seeger dans "
            f"{wins['Seeger']} cas, A dans {wins['A']} cas, B dans {wins['B']} cas ; A est plus petite que B dans {a_lt_b} cas.\n"]
    return out, wins, tot


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    n = verifier_les_relations()
    out, wins, tot = tableaux(n)
    with open(os.path.join(RESULTS, "q1_bornes_tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    fig_en_fonction_de_m(os.path.join(FIGURES, "q1_bornes_m.png"))
    fig_en_fonction_de_kl(os.path.join(FIGURES, "q1_bornes_kl.png"))
    fig_en_fonction_de_lambda(os.path.join(FIGURES, "q1_bornes_lambda.png"))
    print(f"{n} combinaisons vérifiées (Seeger <= McAllester ; formule de lambda*).")
    print(f"Balayage ({tot} cas) : bornes les plus serrées = {wins}")
    print("Écrit : partie2/results/q1_bornes_tables.md, partie2/figures/q1_bornes_{m,kl,lambda}.png")


if __name__ == "__main__":
    main()
