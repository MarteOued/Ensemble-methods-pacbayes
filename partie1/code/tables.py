"""tables.py : mes petites fonctions pour écrire des tableaux lisibles (Markdown)

CE QUE DEMANDE LE SUJET
    Le sujet veut un rapport avec des tableaux de résultats, et des résultats reproductibles : chaque
    tableau du rapport doit pouvoir être régénéré par mon code (« un court README expliquant comment
    reproduire chaque tableau et chaque figure »).

CE QUE J'AI FAIT DANS CE FICHIER
    Les fichiers CSV de results/ sont des données brutes, difficiles à lire. J'ai donc écrit ici trois
    fonctions : md_table pour fabriquer un tableau Markdown, et pct / num pour écrire les nombres à la
    française (virgule décimale, « 12,59 % »). Un tableau Markdown s'affiche comme un vrai tableau
    dans VS Code et sur GitHub, et se copie facilement dans le rapport.

À QUOI ÇA ME SERT
    verify.py, run_experiments.py et analyse.py l'utilisent pour écrire leurs tableaux directement
    depuis les résultats : je ne recopie jamais un chiffre à la main, donc aucun risque d'erreur de
    recopie.
"""
import numpy as np


def num(x, nd=3):
    """Nombre avec virgule décimale ; '–' si la valeur est absente (NaN)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x, nd=2):
    """Proportion -> pourcentage à la française, par exemple 0.1259 -> '12,59 %'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{100 * x:.{nd}f}".replace(".", ",") + " %"


def md_table(header, rows):
    """Tableau Markdown. header : liste de titres ; rows : liste de listes de cellules (déjà en texte)."""
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)
