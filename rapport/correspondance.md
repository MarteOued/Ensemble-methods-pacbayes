# Correspondance : tableaux et figures du rapport -> fichier source -> commande

Ce fichier est **généré** par `rapport/build_mapping.py` à partir de la compilation du rapport : les numéros sont ceux du PDF.
Toutes les commandes se lancent depuis la racine du dépôt. Après avoir relancé une expérience, il faut régénérer les tableaux du rapport (`python3 rapport/build_tables.py`) puis recompiler (voir le README).

## Tableaux

| N° | Où | Contenu | Fichier source | Commande qui le produit |
|---|---|---|---|---|
| 1 | corps | Vérification de mes implémentations contre scikit-learn (6 jeux × 10 découpages, 60 cas par mét… | `partie1/results/verify_tables.md` (tableau « Tableau 1 : erreur de test et accord… ») | `python3 partie1/code/verify.py` |
| 2 | corps | Les 25 jeux de données : nombre d'exemples m, dimension d (après encodage) et proportion de la… | `partie1/results/datasets_table.csv` | `python3 partie1/code/datasets.py` |
| 3 | corps | Rangs moyens des 10 méthodes sur les 25 jeux, pour chaque métrique (1 = meilleur ; en gras, le… | `partie1/results/analyse_tables.md` (tableau « Tous les jeux… ») | `python3 partie1/code/analyse.py` |
| 4 | corps | Bagging contre boosting : différence d'accuracy sur 25 jeux et de balanced accuracy sur 10 jeux… | `partie1/results/analyse_tables.md` (tableau « 5.3 Comparaisons directes… ») | `python3 partie1/code/analyse.py` |
| 5 | corps | Effet du bruit sur les étiquettes d'entraînement : erreur de test moyenne (en %) de 6 jeux réel… | `partie1/results/sensibilite_tables.md` (tableau « Tableau S1 : effet du bruit… ») | `python3 partie1/code/sensibilite.py` |
| 6 | corps | Profondeur des arbres de base (200 arbres ou tours) : erreur de test moyenne (en %), avec 0 % e… | `partie1/results/sensibilite_tables.md` (tableau « Tableau S2 : profondeur… ») | `python3 partie1/code/sensibilite.py` |
| 7 | corps | Coût du stacking par rapport à ses quatre membres (médiane sur les 25 jeux ; bankmarketing : le… | `partie1/results/analyse_tables.md` (tableau « 6.3 Coût du stacking… ») | `python3 partie1/code/analyse.py` |
| 8 | corps | Jeux déséquilibrés : rang moyen de chaque méthode pour chaque métrique, avec sa position entre… | `partie1/results/analyse_tables.md` (tableau « 3. Jeux déséquilibrés… ») | `python3 partie1/code/analyse.py` |
| 9 | corps | Prix de la grille de la borne A (risque observé 0,1) : (a) optimal continu sans borne de l'unio… | `partie2/results/q1_bornes_tables.md` (tableau « Tableau 2 : le prix de la grille… ») | `python3 partie2/code/q1_bornes.py` |
| 10 | corps | Les bornes A et B mènent-elles au même posterior ? Distance en variation totale entre Q_A et Q_… | `partie2/results/q2_comparaison_tables.md` (tableau « Tableau 4 : les deux bornes… ») | `python3 partie2/code/q2_comparaison.py` |
| 11 | corps | Effet du prior appris (appris - départ), moyennes sur les 60 exécutions par famille et par born… | `partie2/results/q3_prior_tables.md` (tableau « Tableau 4 : résumé global… ») | `python3 partie2/code/q3_prior.py` |
| 12 | corps | Posterior uniforme sur les k meilleurs stumps : k qui minimise le risque du vote en test (diagn… | `partie2/results/q4_familles_tables.md` (tableau « Tableau 4 : le meilleur k… ») | `python3 partie2/code/q4_familles.py` |
| 13 | corps | Erreur en test (moyenne ± écart-type sur 10 découpages ; vote pour PAC-Bayes). Méthodes classiq… | `partie2/results/q5_classique_tables.md` (tableau « Tableau 1 : erreur en test… ») | `python3 partie2/code/q5_classique.py` |
| 14 | corps | Garanties : certificat de Seeger (sur le risque de Gibbs), 2× certificat (sur le vote) et erreu… | `partie2/results/q5_classique_tables.md` (tableau « Tableau 4 : ce que garantit… ») | `python3 partie2/code/q5_classique.py` |
| 15 | annexe | Rangs moyens sur les 15 jeux équilibrés (CD = 3,50). | `partie1/results/analyse_tables.md` (tableau « Jeux équilibrés (N… ») | `python3 partie1/code/analyse.py` |
| 16 | annexe | Rangs moyens sur les 10 jeux déséquilibrés (CD = 4,28). | `partie1/results/analyse_tables.md` (tableau « Jeux déséquilibrés (N… ») | `python3 partie1/code/analyse.py` |
| 17 | annexe | Corrélation de Kendall entre les classements selon chaque métrique (jeux déséquilibrés). | `partie1/results/analyse_tables.md` (tableau « 3. Jeux déséquilibrés… ») | `python3 partie1/code/analyse.py` |
| 18 | annexe | Gain des ensembles par rapport à l'arbre seul : différence moyenne (points), victoires/égalités… | `partie1/results/analyse_tables.md` (tableau « Référence : arbre seul… ») | `python3 partie1/code/analyse.py` |
| 19 | annexe | Idem, par rapport à la régression logistique. | `partie1/results/analyse_tables.md` (tableau « Référence : régression logistique… ») | `python3 partie1/code/analyse.py` |
| 20 | annexe | Idem, par rapport au kNN. | `partie1/results/analyse_tables.md` (tableau « Référence : kNN… ») | `python3 partie1/code/analyse.py` |
| 21 | annexe | Meilleur du type bagging (bagging, forêt aléatoire) contre meilleur du type boosting, par jeu (… | `partie1/results/analyse_tables.md` (tableau « 5.1 Meilleur du type bagging… ») | `python3 partie1/code/analyse.py` |
| 22 | annexe | Qui l'emporte, selon la taille et le déséquilibre. | `partie1/results/analyse_tables.md` (tableau « 5.2 Qui l'emporte… ») | `python3 partie1/code/analyse.py` |
| 23 | annexe | Stacking et ses quatre membres, par jeu (métrique mise en avant, en %). | `partie1/results/analyse_tables.md` (tableau « 6.1 Résultats par jeu… ») | `python3 partie1/code/analyse.py` |
| 24 | annexe | Stacking contre ses membres (25 jeux). | `partie1/results/analyse_tables.md` (tableau « 6.2 Stacking contre ses membres… ») | `python3 partie1/code/analyse.py` |
| 25 | annexe | Coût de calcul des méthodes (un seul c ur) ; les temps absolus varient d'environ 30 % d'une exé… | `partie1/results/analyse_tables.md` (tableau « 7. Coût de calcul… ») | `python3 partie1/code/analyse.py` |
| 26 | annexe | Erreur de test (%) selon le nombre d'arbres (bagging) ou de tours (AdaBoost). | `partie1/results/sensibilite_tables.md` (tableau « Tableau S3… ») | `python3 partie1/code/sensibilite.py` |
| 27 | annexe | Hausse de l'erreur entre 0 % et 20 % de bruit, par jeu (points). | `partie1/results/sensibilite_tables.md` (tableau « Hausse de l'erreur entre… ») | `python3 partie1/code/sensibilite.py` |
| 28 | annexe | Biais et variance sur jeux simulés (%). | `partie1/results/sensibilite_tables.md` (tableau « Tableau S4… ») | `python3 partie1/code/sensibilite.py` |
| 29 | annexe | Vérification : contrôles internes (alpha, epsilon, coefficients). | `partie1/results/verify_tables.md` (tableau « Tableau 2 : contrôles internes… ») | `python3 partie1/code/verify.py` |
| 30 | annexe | Valeur des trois bornes (risque observé 0,1). | `partie2/results/q1_bornes_tables.md` (tableau « Tableau 1 : valeur des trois bornes… ») | `python3 partie2/code/q1_bornes.py` |
| 31 | annexe | Constantes de la pénalité des bornes A et B. | `partie2/results/q1_bornes_tables.md` (tableau « Tableau 3 : la constante… ») | `python3 partie2/code/q1_bornes.py` |
| 32 | annexe | Effet du risque observé sur les trois bornes (m = 1000, KL = 5). | `partie2/results/q1_bornes_tables.md` (tableau « Tableau 4 : effet du risque observé… ») | `python3 partie2/code/q1_bornes.py` |
| 33 | annexe | Question 2, wdbc : bornes A et B (prior uniforme). | `partie2/results/q2_comparaison_tables.md` (tableau « Tableau 2 : wdbc… ») | `python3 partie2/code/q2_comparaison.py` |
| 34 | annexe | Question 2 : bornes A et B sur les six jeux (prior uniforme). | `partie2/results/q2_comparaison_tables.md` (tableau « Tableau 3 : les six jeux… ») | `python3 partie2/code/q2_comparaison.py` |
| 35 | annexe | Question 2 : validité et comparaison des certificats. | `partie2/results/q2_comparaison_tables.md` (tableau « Tableau 5 : quel certificat… ») | `python3 partie2/code/q2_comparaison.py` |
| 36 | annexe | Question 3 : effet du prior appris, par jeu (appris - départ). | `partie2/results/q3_prior_tables.md` (tableau « Tableau 3 : ce qui change… ») | `python3 partie2/code/q3_prior.py` |
| 37 | annexe | Question 3 : le prior appris lui-même, évalué sur le test. | `partie2/results/q3_prior_tables.md` (tableau « Tableau 5 : le prior appris… ») | `python3 partie2/code/q3_prior.py` |
| 38 | annexe | Question 4 : stumps et classifieurs linéaires, bornes A et B (prior appris). | `partie2/results/q4_familles_tables.md` (tableau « Tableau 1 : les deux familles… ») | `python3 partie2/code/q4_familles.py` |
| 39 | annexe | Question 4 : avantage du vote sur le classifieur de Gibbs (borne A). | `partie2/results/q4_familles_tables.md` (tableau « Tableau 3 : de combien… ») | `python3 partie2/code/q4_familles.py` |
| 40 | annexe | Question 4 : Dirac sur le meilleur stump, uniforme sur les 5 et les 20 meilleurs. | `partie2/results/q4_familles_tables.md` (tableau « Tableau 2 : autres posteriors… ») | `python3 partie2/code/q4_familles.py` |
| 41 | annexe | Question 4 : certificats et votes des deux familles (borne A). | `partie2/results/q4_familles_tables.md` (tableau « Tableau 5 : quelle famille… ») | `python3 partie2/code/q4_familles.py` |
| 42 | annexe | Question 5 : les classifieurs PAC-bayésiens et leurs certificats. | `partie2/results/q5_classique_tables.md` (tableau « Tableau 2 : les classifieurs PAC… ») | `python3 partie2/code/q5_classique.py` |
| 43 | annexe | Question 5 : erreur en test de toutes les méthodes, avec les bornes A et B (vote pour PAC-Bayes… | `partie2/results/q5_classique_tables.md` (tableau « Tableau 1 : erreur en test… ») | `python3 partie2/code/q5_classique.py` |
| 44 | annexe | Question 5 : garanties de PAC-Bayes (stumps et linéaire, borne A) et de la forêt aléatoire. | `partie2/results/q5_classique_tables.md` (tableau « Tableau 4 : ce que garantit… ») | `python3 partie2/code/q5_classique.py` |
| 45 | annexe | Question 5 : différence d'erreur en test, classique - PAC-Bayes (points ; négatif : le classiqu… | `partie2/results/q5_classique_tables.md` (tableau « Tableau 3 : différence… ») | `python3 partie2/code/q5_classique.py` |
| 46 | annexe | Question 5 : hausse de l'erreur (points) quand la méthode classique n'utilise que S (56 %) au l… | `partie2/results/q5_classique_tables.md` (tableau « Tableau 5 : le prix… ») | `python3 partie2/code/q5_classique.py` |
| 47 | annexe | Question 5 : meilleure méthode de chaque approche, par jeu (choisie sur le test). | `partie2/results/q5_classique_tables.md` (tableau « Tableau 6 : meilleure… ») | `python3 partie2/code/q5_classique.py` |

Le tableau des hyperparamètres de la section 2.2 n'est pas un résultat : il décrit les réglages écrits dans `partie1/code/run_experiments.py`.

## Figures

| N° | Où | Contenu | Fichier image | Commande qui le produit |
|---|---|---|---|---|
| 1 | corps | Diagramme de différences critiques, accuracy, 25 jeux : les méthodes reliées par une barre ne d… | `partie1/figures/cd/cd_tous_accuracy.png` | `python3 partie1/code/analyse.py` |
| 2 | corps | Effet du bruit sur les étiquettes d'entraînement (moyenne de 6 jeux réels équilibrés) : erreur… | `partie1/figures/sens_bruit.png` | `python3 partie1/code/sensibilite.py` |
| 3 | corps | Erreur de test selon le nombre d'arbres (bagging) ou de tours (AdaBoost), sans bruit (gauche) e… | `partie1/figures/sens_courbes.png` | `python3 partie1/code/sensibilite.py` |
| 4 | corps | Biais (rouge) et variance nette (bleu) sur les deux jeux simulés, sans bruit et avec 20~% de br… | `partie1/figures/sens_biais_variance.png` | `python3 partie1/code/sensibilite.py` |
| 5 | corps | Jeux déséquilibrés : le rang moyen de chaque méthode change avec la métrique. | `partie1/figures/rank_by_metric_imbalanced.png` | `python3 partie1/code/analyse.py` |
| 6 | corps | Les trois bornes en fonction de m (risque observé 0,1, =0,05) pour quatre valeurs de la KL. | `partie2/figures/q1_bornes_m.png` | `python3 partie2/code/q1_bornes.py` |
| 7 | corps | Les trois bornes en fonction de la KL (risque observé 0,1, =0,05) pour quatre valeurs de m. | `partie2/figures/q1_bornes_kl.png` | `python3 partie2/code/q1_bornes.py` |
| 8 | corps | Borne A en fonction de (courbe), valeurs de la grille (points), meilleur de la grille (rond rou… | `partie2/figures/q1_bornes_lambda.png` | `python3 partie2/code/q1_bornes.py` |
| 9 | corps | wdbc : borne A en fonction de : bornes et risques en test (gauche), KL (milieu), nombre effecti… | `partie2/figures/q2_lambda_wdbc.png` | `python3 partie2/code/q2_comparaison.py` |
| 10 | corps | Certificat de Seeger avec le prior de départ (clair) et le prior appris (foncé), borne A ; les… | `partie2/figures/q3_certificats.png` | `python3 partie2/code/q3_prior.py` |
| 11 | corps | Posterior uniforme sur les k meilleurs stumps : certificat (vert), risque de Gibbs (rouge) et d… | `partie2/figures/q4_courbe_k.png` | `python3 partie2/code/q4_familles.py` |
| 12 | corps | Erreur de test et garanties pour la forêt aléatoire et le linéaire PAC-bayésien : le certificat… | `partie2/figures/q5_garanties.png` | `python3 partie2/code/q5_classique.py` |
| 13 | annexe | Vérification du bagging (partie~1). Gauche : l'erreur out-of-bag de mon bagging contre celle de… | `partie1/figures/verify_oob.png` | `python3 partie1/code/verify.py` |
| 14 | annexe | Question~2. Gauche : certificat de Seeger des posteriors de la borne A et de la borne B, avec l… | `partie2/figures/q2_certificats.png` | `python3 partie2/code/q2_comparaison.py` |
| 15 | annexe | Question~2, borne A, six jeux : KL du posterior (gauche) et nombre effectif de votants (droite)… | `partie2/figures/q2_kl_neff_lambda.png` | `python3 partie2/code/q2_comparaison.py` |
| 16 | annexe | Question~3, borne A : KL du posterior avec le prior de départ (clair) et avec le prior appris (… | `partie2/figures/q3_kl.png` | `python3 partie2/code/q3_prior.py` |
| 17 | annexe | Question~3, borne A : risques de Gibbs et du vote en test avec le prior de départ et avec le pr… | `partie2/figures/q3_risques.png` | `python3 partie2/code/q3_prior.py` |
| 18 | annexe | Question~4 (prior appris) : certificat de Seeger, risque de Gibbs et risque du vote en test pou… | `partie2/figures/q4_familles.png` | `python3 partie2/code/q4_familles.py` |
| 19 | annexe | Question~5 : erreur de test des méthodes classiques (AdaBoost sur stumps, forêt aléatoire, régr… | `partie2/figures/q5_erreurs.png` | `python3 partie2/code/q5_classique.py` |
| 20 | annexe | Question~5 : chaque point est un découpage (60 en tout). Gauche : forêt aléatoire contre vote d… | `partie2/figures/q5_nuage.png` | `python3 partie2/code/q5_classique.py` |
| 21 | annexe | Diagramme de différences critiques, balanced accuracy, 25 jeux (partie~1). | `partie1/figures/cd/cd_tous_balanced_accuracy.png` | `python3 partie1/code/analyse.py` |
| 22 | annexe | Diagramme de différences critiques, balanced accuracy, 10 jeux déséquilibrés (CD=4,28) (partie~… | `partie1/figures/cd/cd_desequilibres_balanced_accuracy.png` | `python3 partie1/code/analyse.py` |
