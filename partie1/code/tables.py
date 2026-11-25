"""tables.py : mes petites fonctions pour écrire des tableaux Markdown lisibles

Ce que fait le fichier : md_table fabrique un tableau Markdown, pct et num écrivent les nombres à la française
(virgule décimale, « 12,59 % »). Les scripts d'expériences l'utilisent pour écrire leurs tableaux directement depuis
les résultats : aucun chiffre n'est recopié à la main.
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
