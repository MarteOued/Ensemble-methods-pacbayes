"""build_tables.py : fabrique les tableaux LaTeX du rapport à partir des tableaux générés par mon code

Ce que fait le fichier : mes scripts écrivent leurs tableaux en Markdown (partie1/results/*.md, partie2/results/*.md). Ce
script les lit, choisit ceux du rapport par leur titre et les convertit en tableaux LaTeX dans rapport/generated/, que
rapport.tex inclut avec \\input. Aucun chiffre n'est retapé à la main : après une nouvelle expérience, il suffit de le
relancer. Si un tableau est introuvable ou un caractère non géré, il s'arrête avec un message (jamais de tableau faux en
silence).

Usage : python3 build_tables.py     (écrit rapport/generated/*.tex)
"""
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(HERE, "generated")
P1 = os.path.join(ROOT, "partie1", "results")
P2 = os.path.join(ROOT, "partie2", "results")

# --------------------------------------------------------------------------------------------------
# Lecture des tableaux Markdown
# --------------------------------------------------------------------------------------------------
def split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def md_tables(path):
    """Liste de (titre, en-tête, lignes) pour chaque tableau à barres verticales du fichier."""
    lines = open(path, encoding="utf-8").read().splitlines()
    out, heading, i = [], "", 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("#"):
            heading = l.lstrip("#").strip()
        if l.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1]):
            header, rows = split_row(l), []
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            for r in rows:
                assert len(r) == len(header), (path, heading, "ligne de longueur différente de l'en-tête", r)
            out.append((heading, header, rows))
            continue
        i += 1
    return out


def pick(path, heading_start, index=0):
    found = [t for t in md_tables(path) if t[0].startswith(heading_start)]
    if len(found) <= index:
        sys.exit(f"ERREUR : aucun tableau « {heading_start} » (n°{index}) dans {os.path.basename(path)}")
    return found[index]


# --------------------------------------------------------------------------------------------------
# Échappement LaTeX
# --------------------------------------------------------------------------------------------------
ESC = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
       "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}
UNI = {"±": r"$\pm$", "λ": r"$\lambda$", "δ": r"$\delta$", "σ": r"$\sigma$", "Δ": r"$\Delta$", "Σ": r"$\Sigma$",
       "‖": r"$\Vert$", "≥": r"$\geq$", "≤": r"$\leq$", "→": r"$\rightarrow$", "×": r"$\times$", "−": r"$-$",
       "–": "--", "⟨": r"$\langle$", "⟩": r"$\rangle$", "…": r"\ldots{}", "√": r"$\sqrt{\phantom{x}}$", "≈": r"$\approx$",
       "Λ": r"$\Lambda$", " ": "~", " ": r"\,", "′": "'"}
PASS = set("àâäçéèêëîïôöùûüœæÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŒÆ«»’")


def fixq(t):
    """Guillemets français -> commandes de babel (la police T1 les affiche mal avec XeTeX)."""
    t = re.sub(r"«\s*", r"\\og ", t)
    return re.sub(r"\s*»", r"\\fg{}", t)


def esc(text):
    """Texte brut (sans marques Markdown) -> LaTeX."""
    out = "".join(ESC.get(c, c) for c in text)
    out = "".join(UNI.get(c, c) for c in out)
    out = fixq(out)
    bad = {c for c in out if ord(c) > 127 and c not in PASS}
    if bad:
        sys.exit(f"ERREUR : caractère(s) non géré(s) {sorted(bad)} dans « {text} » : ajouter-les à UNI dans build_tables.py")
    return out


def cell(text):
    """Cellule Markdown (avec **gras** et *italique*) -> LaTeX."""
    parts = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text)
    out = []
    for p in parts:
        if p.startswith("**") and p.endswith("**") and len(p) > 4:
            out.append(r"\textbf{" + esc(p[2:-2]) + "}")
        elif p.startswith("*") and p.endswith("*") and len(p) > 2:
            out.append(r"\textit{" + esc(p[1:-1]) + "}")
        else:
            out.append(esc(p))
    return "".join(out)


def tabular(header, rows, align=None, bold_header=True):
    n = len(header)
    align = align or "l" + "c" * (n - 1)
    head = " & ".join((r"\textbf{" + cell(h) + "}") if bold_header and h else cell(h) for h in header) + r" \\"
    body = [" & ".join(cell(c) for c in r) + r" \\" for r in rows]
    return "\n".join([r"\begin{tabular}{" + align + "}", r"\toprule", head, r"\midrule", *body, r"\bottomrule", r"\end{tabular}"])


def wrap(label, caption, tab, resize=True, size=r"\scriptsize", pos="tbp"):
    caption = fixq(caption)
    inner = r"\fitwidth{%" + "\n" + tab + "\n}"        # réduit seulement si le tableau dépasse la largeur du texte
    return "\n".join([rf"\begin{{table}}[{pos}]", r"\centering", size, rf"\caption{{{caption}}}", rf"\label{{{label}}}", inner,
                      r"\end{table}", ""])


def write(name, text):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".tex"), "w", encoding="utf-8") as f:
        f.write("% Fichier généré par rapport/build_tables.py : ne pas modifier à la main.\n" + text)


# --------------------------------------------------------------------------------------------------
# La liste des tableaux : (nom, fichier source, début du titre, n° dans ce titre, légende, options)
# --------------------------------------------------------------------------------------------------
A, S, V = (os.path.join(P1, f) for f in ("analyse_tables.md", "sensibilite_tables.md", "verify_tables.md"))
Q = {i: os.path.join(P2, f) for i, f in {1: "q1_bornes_tables.md", 2: "q2_comparaison_tables.md", 3: "q3_prior_tables.md",
                                         4: "q4_familles_tables.md", 5: "q5_classique_tables.md"}.items()}

TABLES = [
    # ---- partie 1 : corps du rapport -----------------------------------------------------------------
    ("verify", V, "Tableau 1 : erreur de test et accord", 0,
     "Vérification de mes implémentations contre scikit-learn (6 jeux $\\times$ 10 découpages, 60 cas par méthode). « Accord » : part des "
     "exemples de test où les deux versions prédisent la même classe ; « Accord réf. » : même mesure entre scikit-learn et lui-même avec une "
     "autre graine.", dict(align="lcccc p{5.4cm}")),
    ("ranks_all", A, "Tous les jeux", 0,
     "Rangs moyens des 10 méthodes sur les 25 jeux, pour chaque métrique (1 = meilleur ; en gras, le meilleur). Dernière ligne : p-valeur du "
     "test de Friedman. Différence critique de Nemenyi : CD = {cd}.", dict(cd=True, resize=False, size=r"\small")),
    ("imb_pos", A, "3. Jeux déséquilibrés", 0,
     "Jeux déséquilibrés : rang moyen de chaque méthode pour chaque métrique, avec sa position entre parenthèses ; dernière colonne : places "
     "gagnées (+) ou perdues ($-$) en passant de l'accuracy à la balanced accuracy.", dict(resize=False, size=r"\small")),
    ("bagboost_pairs", A, "5.3 Comparaisons directes", 0,
     "Bagging contre boosting : différence d'accuracy sur 25 jeux et de balanced accuracy sur 10 jeux déséquilibrés (première méthode moins "
     "seconde, en points), victoires/égalités/défaites de la première, p-valeur de Wilcoxon corrigée par Holm.", dict()),
    ("stack_cost", A, "6.3 Coût du stacking", 0,
     "Coût du stacking par rapport à ses quatre membres (médiane sur les 25 jeux ; « bankmarketing » : le plus grand jeu). Les temps absolus varient d'environ 30 \\% d'une exécution à l'autre ; les rapports sont stables.",
     dict(align="lc", resize=False, size=r"\small")),
    ("sens_noise", S, "Tableau S1 : effet du bruit", 0,
     "Effet du bruit sur les étiquettes d'entraînement : erreur de test moyenne (en \\%) de 6 jeux réels équilibrés ; dernière colonne : hausse "
     "entre 0 \\% et 20 \\% de bruit (points).", dict(resize=False, size=r"\small")),
    ("sens_depth", S, "Tableau S2 : profondeur", 0,
     "Profondeur des arbres de base (200 arbres ou tours) : erreur de test moyenne (en \\%), avec 0 \\% et 20 \\% de bruit. « complet » : profondeur "
     "illimitée.", dict(resize=False, size=r"\small")),
    # ---- partie 2 : corps du rapport -----------------------------------------------------------------
    ("q1_price", Q[1], "Tableau 2 : le prix de la grille", 0,
     "Prix de la grille de la borne A (risque observé 0,1) : (a) $\\lambda$ optimal continu sans borne de l'union ; (b) avec l'union ; (c) "
     "meilleur $\\lambda$ de la grille. Coût de l'union = (b)$-$(a), de la discrétisation = (c)$-$(b), prix total = (c)$-$(a).", dict()),
    ("q2_same", Q[2], "Tableau 4 : les deux bornes", 0,
     "Les bornes A et B mènent-elles au même posterior ? Distance en variation totale entre $Q_A$ et $Q_B$, recouvrement, nombre de découpages "
     "(sur 10) où le stump de poids maximal est le même, poids maximal et nombre de votants effectifs.", dict()),
    ("q3_global", Q[3], "Tableau 4 : résumé global", 0,
     "Effet du prior appris (appris $-$ départ), moyennes sur les 60 exécutions par famille et par borne ; une différence négative de KL ou de "
     "certificat est une amélioration.", dict(resize=False, size=r"\small")),
    ("q4_bestk", Q[4], "Tableau 4 : le meilleur k", 0,
     "Posterior uniforme sur les $k$ meilleurs stumps : $k$ qui minimise le risque du vote en test (diagnostic, pas une méthode), et risque du "
     "vote à ce $k$, pour le Dirac ($k = 1$), pour le posterior de la borne A et pour le classifieur linéaire.", dict()),
    ("q5_err", Q[5], "Tableau 1 : erreur en test", 0,
     "Erreur en test (moyenne $\\pm$ écart-type sur 10 découpages ; vote pour PAC-Bayes). Méthodes classiques entraînées sur 70 \\% des données ; "
     "PAC-Bayes avec le prior appris. Dernières lignes : moyenne et rang moyen sur les 6 jeux. "
     "Les posteriors de la borne B, presque identiques à ceux de A, sont en annexe.", dict(drop=["PAC-Bayes stumps, B", "PAC-Bayes linéaire, B"])),
    ("q5_guar", Q[5], "Tableau 4 : ce que garantit", 0,
     "Garanties : certificat de Seeger (sur le risque de Gibbs), $2\\times$ certificat (sur le vote) et erreur de test du vote pour PAC-Bayes ; "
     "erreur de test et borne « de test » à 95 \\% pour la forêt aléatoire (calculée avec le jeu de test). "
     "Les stumps sont dans le tableau de l'annexe.", dict(drop=["Stumps : certificat", "Stumps : 2 x certificat", "Stumps : erreur du vote"])),
    # ---- annexes ----------------------------------------------------------------------------------------------------
    ("ann_ranks_bal", A, "Jeux équilibrés (N", 0, "Rangs moyens sur les 15 jeux équilibrés (CD = {cd}).", dict(cd=True, resize=False, size=r"\small")),
    ("ann_ranks_imb", A, "Jeux déséquilibrés (N", 0, "Rangs moyens sur les 10 jeux déséquilibrés (CD = {cd}).", dict(cd=True, resize=False, size=r"\small")),
    ("ann_kendall", A, "3. Jeux déséquilibrés", 1, "Corrélation de Kendall entre les classements selon chaque métrique (jeux déséquilibrés).", dict(resize=False, size=r"\small")),
    ("ann_gain_tree", A, "Référence : arbre seul", 0, "Gain des ensembles par rapport à l'arbre seul : différence moyenne (points), victoires/égalités/défaites (G/N/P) et p-valeur de Wilcoxon corrigée par Holm, pour l'accuracy (25 jeux), la balanced accuracy (10 jeux déséquilibrés) et l'AUC (25 jeux).", dict()),
    ("ann_gain_lr", A, "Référence : régression logistique", 0, "Idem, par rapport à la régression logistique.", dict()),
    ("ann_gain_knn", A, "Référence : kNN", 0, "Idem, par rapport au kNN.", dict()),
    ("ann_bagboost_ds", A, "5.1 Meilleur du type bagging", 0, "Meilleur du type bagging (bagging, forêt aléatoire) contre meilleur du type boosting, par jeu (métrique mise en avant, en \\%).", dict(align="lccccccc")),
    ("ann_bagboost_size", A, "5.2 Qui l'emporte", 0, "Qui l'emporte, selon la taille et le déséquilibre.", dict(resize=False, size=r"\small")),
    ("ann_stack_ds", A, "6.1 Résultats par jeu", 0, "Stacking et ses quatre membres, par jeu (métrique mise en avant, en \\%).", dict()),
    ("ann_stack_vs", A, "6.2 Stacking contre ses membres", 0, "Stacking contre ses membres (25 jeux).", dict()),
    ("ann_cost", A, "7. Coût de calcul", 0, "Coût de calcul des méthodes (un seul c\\oe ur) ; les temps absolus varient d'environ 30 \\% d'une exécution à l'autre.", dict()),
    ("ann_sens_curves", S, "Tableau S3", 0, "Erreur de test (\\%) selon le nombre d'arbres (bagging) ou de tours (AdaBoost).", dict()),
    ("ann_sens_bv", S, "Tableau S4", 0, "Biais et variance sur jeux simulés (\\%).", dict(resize=False)),
    ("ann_sens_inc", S, "Hausse de l'erreur entre", 0, "Hausse de l'erreur entre 0 \\% et 20 \\% de bruit, par jeu (points).", dict()),
    ("ann_verify_int", V, "Tableau 2 : contrôles internes", 0, "Vérification : contrôles internes (alpha, epsilon, coefficients).", dict(resize=False)),
    ("ann_q1_vals", Q[1], "Tableau 1 : valeur des trois bornes", 0, "Valeur des trois bornes (risque observé 0,1).", dict(resize=False, size=r"\small")),
    ("ann_q1_const", Q[1], "Tableau 3 : la constante", 0, "Constantes de la pénalité des bornes A et B.", dict(resize=False, size=r"\small")),
    ("ann_q1_emp", Q[1], "Tableau 4 : effet du risque observé", 0, "Effet du risque observé sur les trois bornes ($m = 1000$, KL $= 5$).", dict(resize=False, size=r"\small")),
    ("ann_q2_wdbc", Q[2], "Tableau 2 : wdbc", 0, "Question 2, wdbc : bornes A et B (prior uniforme).", dict()),
    ("ann_q2_all", Q[2], "Tableau 3 : les six jeux", 0, "Question 2 : bornes A et B sur les six jeux (prior uniforme).", dict()),
    ("ann_q2_cert", Q[2], "Tableau 5 : quel certificat", 0, "Question 2 : validité et comparaison des certificats.", dict()),
    ("ann_q3_delta", Q[3], "Tableau 3 : ce qui change", 0, "Question 3 : effet du prior appris, par jeu (appris $-$ départ).", dict()),
    ("ann_q3_prior", Q[3], "Tableau 5 : le prior appris", 0, "Question 3 : le prior appris lui-même, évalué sur le test.", dict()),
    ("ann_q4_fam", Q[4], "Tableau 1 : les deux familles", 0, "Question 4 : stumps et classifieurs linéaires, bornes A et B (prior appris).", dict()),
    ("ann_q4_vote", Q[4], "Tableau 3 : de combien", 0, "Question 4 : avantage du vote sur le classifieur de Gibbs (borne A).", dict()),
    ("ann_q4_extra", Q[4], "Tableau 2 : autres posteriors", 0, "Question 4 : Dirac sur le meilleur stump, uniforme sur les 5 et les 20 meilleurs.", dict()),
    ("ann_q4_cert", Q[4], "Tableau 5 : quelle famille", 0, "Question 4 : certificats et votes des deux familles (borne A).", dict()),
    ("ann_q5_err_full", Q[5], "Tableau 1 : erreur en test", 0, "Question 5 : erreur en test de toutes les méthodes, avec les bornes A et B (vote pour PAC-Bayes).", dict()),
    ("ann_q5_guar_full", Q[5], "Tableau 4 : ce que garantit", 0, "Question 5 : garanties de PAC-Bayes (stumps et linéaire, borne A) et de la forêt aléatoire.", dict()),
    ("ann_q5_pairs", Q[5], "Tableau 3 : différence", 0, "Question 5 : différence d'erreur en test, classique $-$ PAC-Bayes (points ; négatif : le classique est meilleur).", dict()),
    ("ann_q5_pb", Q[5], "Tableau 2 : les classifieurs PAC", 0, "Question 5 : les classifieurs PAC-bayésiens et leurs certificats.", dict()),
    ("ann_q5_price", Q[5], "Tableau 5 : le prix", 0, "Question 5 : hausse de l'erreur (points) quand la méthode classique n'utilise que $S$ (56 \\%) au lieu de 70 \\%.", dict(resize=False, size=r"\small")),
    ("ann_q5_best", Q[5], "Tableau 6 : meilleure", 0, "Question 5 : meilleure méthode de chaque approche, par jeu (choisie sur le test).", dict()),
]


def build_datasets():
    """Tableau des 25 jeux (m, d, proportion de la classe minoritaire), à partir de datasets_table.csv."""
    df = pd.read_csv(os.path.join(P1, "datasets_table.csv"))
    imb = df[df.imbalanced].sort_values("minority_rate")
    bal = df[~df.imbalanced].sort_values("minority_rate")
    fmt = lambda r: [esc(r.dataset), str(int(r.m)), str(int(r.d)), f"{100 * r.minority_rate:.1f}".replace(".", ",")]
    L, R = [fmt(r) for r in imb.itertuples()], [fmt(r) for r in bal.itertuples()]
    L += [["", "", "", ""]] * (len(R) - len(L))
    rows = [" & ".join(l + r) + r" \\" for l, r in zip(L, R)]
    head = (r"\multicolumn{4}{c}{\textbf{Jeux déséquilibrés} ($<15$~\%)} & \multicolumn{4}{c}{\textbf{Jeux équilibrés}} \\" + "\n"
            r"\cmidrule(r){1-4}\cmidrule(l){5-8}" + "\n"
            + " & ".join(r"\textbf{" + h + "}" for h in ["Jeu", "$m$", "$d$", "Min. (\\%)"] * 2) + r" \\")
    tab = "\n".join([r"\begin{tabular}{lrrr@{\hspace{1.2em}}lrrr}", r"\toprule", head, r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])
    cap = ("Les 25 jeux de données : nombre d'exemples $m$, dimension $d$ (après encodage) et proportion de la classe minoritaire (\\%). Un jeu est "
           "dit déséquilibré si cette proportion est inférieure à 15~\\%.")
    write("datasets", wrap("tab:datasets", cap, tab, resize=False, size=r"\small"))


def main():
    os.makedirs(OUT, exist_ok=True)
    build_datasets()
    for name, path, head, idx, caption, opt in TABLES:
        title, header, rows = pick(path, head, idx)
        cap = caption
        if opt.get("cd"):
            m = re.search(r"CD = ([\d,]+)", title)
            if not m:
                sys.exit(f"ERREUR : pas de CD dans le titre « {title} »")
            cap = caption.format(cd=m.group(1))
        keep = [i for i, h in enumerate(header) if h not in opt.get("drop", [])]
        header, rows = [header[i] for i in keep], [[r[i] for i in keep] for r in rows]
        tab = tabular(header, rows, align=opt.get("align"))
        write(name, wrap("tab:" + name, cap, tab, resize=opt.get("resize", True), size=opt.get("size", r"\scriptsize")))
    print(f"{len(TABLES) + 1} tableaux LaTeX écrits dans rapport/generated/")


if __name__ == "__main__":
    main()
