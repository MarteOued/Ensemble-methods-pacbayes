"""build_mapping.py : table « numéro dans le rapport -> fichier source -> commande qui le produit »

Sujet : le README doit expliquer comment reproduire chaque tableau et chaque figure du rapport.
Ce que fait le fichier : je lis les numéros réellement attribués par LaTeX (rapport.aux), je retrouve pour chaque tableau
et chaque figure son fichier source et le script qui l'a produit, et j'écris le tout dans rapport/correspondance.md. La
table ne peut donc pas se désynchroniser du PDF.

Usage : compiler le rapport en gardant les fichiers intermédiaires (latexmk, ou tectonic -X compile --keep-intermediates),
        puis :  python3 build_mapping.py [chemin/vers/rapport.aux]
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_tables import TABLES  # noqa: E402  (liste des tableaux générés : fichier source, titre, légende)

REL = lambda p: os.path.relpath(p, os.path.join(HERE, "..")).replace(os.sep, "/")

SCRIPT_OF_MD = {"analyse_tables.md": "partie1/code/analyse.py", "sensibilite_tables.md": "partie1/code/sensibilite.py",
                "verify_tables.md": "partie1/code/verify.py", "q1_bornes_tables.md": "partie2/code/q1_bornes.py",
                "q2_comparaison_tables.md": "partie2/code/q2_comparaison.py", "q3_prior_tables.md": "partie2/code/q3_prior.py",
                "q4_familles_tables.md": "partie2/code/q4_familles.py", "q5_classique_tables.md": "partie2/code/q5_classique.py"}
SCRIPT_OF_FIG = [("cd_", "partie1/code/analyse.py"), ("rank_by_metric", "partie1/code/analyse.py"), ("sens_", "partie1/code/sensibilite.py"),
                 ("verify_", "partie1/code/verify.py"), ("datasets_", "partie1/code/datasets.py"), ("q1_", "partie2/code/q1_bornes.py"),
                 ("q2_", "partie2/code/q2_comparaison.py"), ("q3_", "partie2/code/q3_prior.py"), ("q4_", "partie2/code/q4_familles.py"),
                 ("q5_", "partie2/code/q5_classique.py")]


def short(caption, n=95):
    c = caption.replace("\\times", "×").replace("\\pm", "±")
    c = re.sub(r"\$([^$]*)\$", r"\1", c)
    c = re.sub(r"\\[a-zA-Z]+\{?|\\%|[{}\\]", lambda m: "%" if m.group(0) == r"\%" else "", c)
    c = re.sub(r"\s+", " ", c).strip()
    return (c[:n].rstrip() + "…") if len(c) > n else c


def main():
    aux = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "rapport.aux")
    if not os.path.exists(aux):
        sys.exit(f"ERREUR : {aux} introuvable. Compiler d'abord le rapport en gardant les fichiers intermédiaires "
                 "(latexmk, ou tectonic -X compile --keep-intermediates rapport.tex).")
    nums = dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", open(aux, encoding="utf-8").read()))
    tex = open(os.path.join(HERE, "rapport.tex"), encoding="utf-8").read()
    tex_nc = re.sub(r"(?m)^\s*%.*$", "", tex)
    split = tex_nc.index(r"\appendix")
    body = tex_nc[:split]
    gen = lambda name: open(os.path.join(HERE, "generated", name + ".tex"), encoding="utf-8").read()

    rows_t = []
    for name in re.findall(r"\\input\{generated/([^}]+)\}", tex_nc):
        label = "tab:" + name
        if label not in nums:
            sys.exit(f"ERREUR : pas de numéro pour {label}")
        cap = re.search(r"\\caption\{(.*)\}\n\\label", gen(name), flags=re.S).group(1)
        where = "corps" if "generated/" + name + "}" in body else "annexe"
        if name == "datasets":
            src, cmd = "`partie1/results/datasets_table.csv`", "`python3 partie1/code/datasets.py`"
        else:
            entry = next(t for t in TABLES if t[0] == name)
            md = os.path.basename(entry[1])
            src, cmd = f"`{REL(entry[1])}` (tableau « {entry[2]}… »)", f"`python3 {SCRIPT_OF_MD[md]}`"
        rows_t.append((int(nums[label]), where, short(cap), src, cmd))
    rows_f = []
    for m in re.finditer(r"\\fig\{([^}]+)\}\{[^}]*\}\{(.*?)\}\{(fig:[^}]+)\}", tex_nc, flags=re.S):
        f, cap, label = m.groups()
        folder = "partie1/figures/cd/" if f.startswith("cd_") else ("partie1/figures/" if any(f.startswith(p) for p in
                                                                                     ("rank_", "sens_", "verify_", "datasets_")) else "partie2/figures/")
        script = next(s for p, s in SCRIPT_OF_FIG if f.startswith(p))
        rows_f.append((int(nums[label]), "corps" if m.start() < split else "annexe", short(cap), f"`{folder}{f}`", f"`python3 {script}`"))

    out = ["# Correspondance : tableaux et figures du rapport -> fichier source -> commande",
           "",
           "Ce fichier est **généré** par `rapport/build_mapping.py` à partir de la compilation du rapport : les numéros sont ceux du PDF.",
           "Toutes les commandes se lancent depuis la racine du dépôt. Après avoir relancé une expérience, il faut régénérer les tableaux du rapport "
           "(`python3 rapport/build_tables.py`) puis recompiler (voir le README).",
           "",
           "## Tableaux", "",
           "| N° | Où | Contenu | Fichier source | Commande qui le produit |", "|---|---|---|---|---|"]
    out += [f"| {n} | {w} | {c} | {s} | {cmd} |" for n, w, c, s, cmd in sorted(rows_t)]
    out += ["", "Le tableau des hyperparamètres de la section 2.2 n'est pas un résultat : il décrit les réglages écrits dans `partie1/code/run_experiments.py`.",
            "", "## Figures", "", "| N° | Où | Contenu | Fichier image | Commande qui le produit |", "|---|---|---|---|---|"]
    out += [f"| {n} | {w} | {c} | {s} | {cmd} |" for n, w, c, s, cmd in sorted(rows_f)]
    with open(os.path.join(HERE, "correspondance.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"correspondance.md écrit : {len(rows_t)} tableaux, {len(rows_f)} figures")


if __name__ == "__main__":
    main()
