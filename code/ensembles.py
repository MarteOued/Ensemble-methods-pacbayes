"""ensembles.py : mes implémentations des méthodes d'ensemble (Partie 1 du sujet)

CE QUE DEMANDE LE SUJET
    Le sujet demande d'implémenter moi-même trois méthodes : le bagging d'arbres avec son erreur
    out-of-bag, AdaBoost discret (étiquettes dans {-1,+1}, arbres de profondeur donnée) et le
    stacking (modèles de familles différentes + un méta-modèle entraîné sur des prédictions
    out-of-fold). Il demande ensuite de vérifier chaque implémentation contre son équivalent
    scikit-learn.

CE QUE J'AI FAIT DANS CE FICHIER
    J'ai écrit trois classes, MyBagging, MyAdaBoost et MyStacking. Pour les construire, je n'utilise
    de scikit-learn que l'arbre de décision (et quelques modèles simples pour le stacking). La
    logique d'ensemble elle-même (les bootstraps, la mise à jour des poids, les plis out-of-fold).
    Je les ai écrites à la façon de scikit-learn (méthodes fit, predict, decision_function) pour
    pouvoir les mettre dans les mêmes boucles d'expériences que les autres modèles.

À QUOI ÇA ME SERT ENSUITE
    - verify.py : je compare mes trois classes aux versions de scikit-learn pour montrer qu'elles
      sont correctes.
    - run_experiments.py : je les fais tourner sur les jeux de données avec les autres méthodes
      (arbre seul, forêt aléatoire, gradient boosting, régression logistique, kNN...).
    - analyse.py : j'utilise leurs résultats pour répondre aux questions du sujet (gain par rapport
      à un arbre seul, bagging contre boosting, stacking contre son meilleur membre, jeux
      déséquilibrés).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

CLASSES = np.array([-1, 1])


def _check_labels(y):
    if not np.all(np.isin(y, CLASSES)):
        raise ValueError("les étiquettes doivent être dans {-1, +1}")
    return y.astype(int)


def _sign(s):
    """sign avec 0 -> -1 (même convention que sklearn : égalité = première classe)."""
    return np.where(s > 0, 1, -1)


# ====================================================================================================
# Bagging
# ====================================================================================================
# Mon idée du bagging : un arbre profond colle trop aux données (forte variance), donc son résultat
# change beaucoup si on change un peu l'échantillon. Je tire donc plusieurs échantillons avec remise
# (bootstrap), j'entraîne un arbre sur chacun, et je fais voter les arbres. En moyennant, la
# variance baisse.
# L'erreur out-of-bag : chaque bootstrap laisse de côté environ 37 % des exemples. Pour chaque
# exemple, je ne fais voter que les arbres qui ne l'ont pas vu, ce qui me donne une estimation
# honnête de l'erreur de test, sans avoir besoin de mettre un jeu de validation de côté.

class MyBagging(ClassifierMixin, BaseEstimator):
    """Bagging d'arbres de décision.

    Pour b = 1..B : bootstrap (m tirages avec remise), un arbre par bootstrap.
    Prédiction : vote majoritaire. decision_function = moyenne des votes (dans [-1, 1]).

    Erreur OOB : pour chaque exemple i, on ne fait voter que les arbres dont le bootstrap ne contient
    pas i ; l'erreur OOB est la fraction d'exemples mal classés par ce vote. Attributs après fit :
    oob_decision_ (nan si l'exemple n'est jamais hors-sac), oob_error_, oob_score_ = 1 - oob_error_.
    """

    def __init__(self, n_estimators=200, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth          # None = arbres complets (cas classique du bagging)
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        y = _check_labels(y)
        m = len(y)
        rng = np.random.default_rng(self.random_state)
        self.classes_ = CLASSES
        self.estimators_ = []
        vote_sum = np.zeros(m)              # somme des votes ±1 des arbres pour lesquels i est hors-sac
        vote_cnt = np.zeros(m)              # nombre de ces arbres
        for _ in range(self.n_estimators):
            idx = rng.integers(0, m, size=m)                       # bootstrap
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          random_state=int(rng.integers(2 ** 31 - 1)))
            tree.fit(X[idx], y[idx])
            self.estimators_.append(tree)
            oob = np.ones(m, dtype=bool)
            oob[idx] = False                                       # exemples absents du bootstrap
            if oob.any():
                vote_sum[oob] += self._vote(tree, X[oob])
                vote_cnt[oob] += 1
        seen = vote_cnt > 0
        self.oob_decision_ = np.full(m, np.nan)
        self.oob_decision_[seen] = vote_sum[seen] / vote_cnt[seen]
        self.oob_error_ = float(np.mean(_sign(self.oob_decision_[seen]) != y[seen]))
        self.oob_score_ = 1.0 - self.oob_error_
        self.oob_coverage_ = float(seen.mean())                    # part des exemples hors-sac au moins 1 fois
        return self

    @staticmethod
    def _vote(tree, X):
        """Vote ±1 d'un arbre (les classes de l'arbre sont ⊂ {-1,+1}, cas dégénéré d'un bootstrap pur compris)."""
        return np.where(tree.predict(X) > 0, 1.0, -1.0)

    def decision_function(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        return np.mean([self._vote(t, X) for t in self.estimators_], axis=0)

    def predict(self, X):
        return _sign(self.decision_function(X))


# ====================================================================================================
# AdaBoost discret
# ====================================================================================================
# Mon idée d'AdaBoost : au lieu de faire des arbres indépendants, j'en entraîne un à la suite de
# l'autre. Après chaque arbre, j'augmente le poids des exemples qu'il a mal classés, pour que le
# suivant se concentre dessus. Chaque arbre reçoit un poids alpha d'autant plus grand qu'il est
# précis, et la prédiction finale est le signe de la somme pondérée des arbres.
# Ici les arbres sont très simples (des stumps si la profondeur vaut 1) : chacun est faible, mais
# leur enchaînement réduit le biais. Je gère aussi les cas où l'erreur vaut 0 ou dépasse 1/2.

class MyAdaBoost(ClassifierMixin, BaseEstimator):
    """AdaBoost discret.

    w_i = 1/m. Pour t = 1..T :
        h_t   = arbre de profondeur max_depth entraîné avec sample_weight = w
        eps_t = somme des w_i sur les exemples mal classés
        a_t   = 1/2 ln((1 - eps_t) / eps_t)
        w_i  <- w_i exp(-a_t y_i h_t(x_i)), puis renormalisation
    Classifieur final : sign(sum_t a_t h_t(x)).

    Cas limites : eps_t >= 1/2 (pire que le hasard) -> arrêt (le learner n'est gardé que s'il est le
    premier) ; eps_t = 0 -> eps borné à 1e-10 (alpha grand mais fini) puis arrêt (rien à corriger).
    """

    def __init__(self, n_estimators=200, max_depth=1, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth          # 1 = stumps
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        y = _check_labels(y)
        m = len(y)
        rng = np.random.default_rng(self.random_state)
        self.classes_ = CLASSES
        self.estimators_, self.alphas_, self.errors_ = [], [], []
        w = np.full(m, 1.0 / m)
        for t in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          random_state=int(rng.integers(2 ** 31 - 1)))
            tree.fit(X, y, sample_weight=w)
            pred = np.where(tree.predict(X) > 0, 1, -1)
            eps = float(w[pred != y].sum())
            if eps >= 0.5:                                          # plus faible que le hasard : on s'arrête
                if t == 0:
                    self.estimators_.append(tree); self.alphas_.append(0.0); self.errors_.append(eps)
                break
            eps = max(eps, 1e-10)
            alpha = 0.5 * np.log((1.0 - eps) / eps)
            self.estimators_.append(tree); self.alphas_.append(alpha); self.errors_.append(eps)
            if eps <= 1e-10:                                        # classifieur parfait : plus rien à corriger
                break
            w = w * np.exp(-alpha * y * pred)
            w /= w.sum()
        self.alphas_ = np.array(self.alphas_)
        return self

    def decision_function(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        s = np.zeros(len(X))
        for a, t in zip(self.alphas_, self.estimators_):
            s += a * np.where(t.predict(X) > 0, 1.0, -1.0)
        return s

    def staged_decision_function(self, X):
        """Score après 1, 2, ..., T learners (pour tracer l'erreur en fonction de T)."""
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        s = np.zeros(len(X))
        for a, t in zip(self.alphas_, self.estimators_):
            s = s + a * np.where(t.predict(X) > 0, 1.0, -1.0)
            yield s.copy()

    def predict(self, X):
        return _sign(self.decision_function(X))


# ====================================================================================================
# Stacking
# ====================================================================================================
def default_base_models(random_state=0):
    """Quatre familles différentes. La standardisation est DANS le pipeline : elle est donc réestimée
    sur le train de chaque pli, jamais sur les données qu'on prédit (pas de fuite)."""
    return [
        ("tree", DecisionTreeClassifier(max_depth=5, random_state=random_state)),
        ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))),
        ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
        ("nb", GaussianNB()),
    ]


# Mon idée du stacking : plutôt que de faire voter des modèles différents à égalité, j'entraîne un
# méta-modèle qui apprend à les combiner (par exemple à faire plus confiance à celui qui est meilleur).
# Le piège que j'évite : si le méta-modèle apprend sur des prédictions faites sur les données
# d'entraînement des modèles de base, il leur fait trop confiance (un arbre complet prédit parfaitement
# ses propres données). Je découpe donc le train en plis, et chaque exemple est prédit par un
# modèle qui ne l'a pas vu : ce sont les prédictions out-of-fold.
# Le prix à payer : chaque modèle de base est entraîné (nombre de plis + 1) fois.

class MyStacking(ClassifierMixin, BaseEstimator):
    """Stacking avec prédictions out-of-fold.

    1. Le train est découpé en n_splits plis stratifiés.
    2. Pour chaque pli et chaque modèle de base : entraînement sur les autres plis, prédiction de
       P(y=+1) sur le pli écarté. Chaque exemple du train reçoit donc une prédiction d'un modèle qui
       ne l'a jamais vu -> matrice Z (m x nb de modèles).
    3. Le méta-modèle (régression logistique par défaut) est entraîné sur Z.
    4. Les modèles de base sont ensuite réentraînés sur tout le train ; au test, leurs probabilités
       alimentent le méta-modèle.
    Coût : chaque modèle de base est entraîné n_splits + 1 fois.
    """

    def __init__(self, base_models=None, meta_model=None, n_splits=5, random_state=0):
        self.base_models = base_models      # liste de (nom, estimateur) ; None -> default_base_models
        self.meta_model = meta_model        # None -> LogisticRegression
        self.n_splits = n_splits
        self.random_state = random_state

    @staticmethod
    def _proba_pos(model, X):
        return model.predict_proba(X)[:, list(model.classes_).index(1)]

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        y = _check_labels(y)
        m = len(y)
        self.classes_ = CLASSES
        base = self.base_models if self.base_models is not None else default_base_models(self.random_state)
        self.names_ = [n for n, _ in base]
        Z = np.zeros((m, len(base)))
        skf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, va in skf.split(X, y):
            for j, (_, est) in enumerate(base):
                Z[va, j] = self._proba_pos(clone(est).fit(X[tr], y[tr]), X[va])
        self.oof_ = Z                                               # prédictions out-of-fold (pour analyse)
        self.meta_ = clone(self.meta_model if self.meta_model is not None else LogisticRegression(max_iter=1000))
        self.meta_.fit(Z, y)
        self.base_ = [clone(est).fit(X, y) for _, est in base]      # réentraînement sur tout le train
        return self

    def _meta_features(self, X):
        return np.column_stack([self._proba_pos(b, X) for b in self.base_])

    def decision_function(self, X):
        check_is_fitted(self, "base_")
        return self.meta_.decision_function(self._meta_features(check_array(X)))

    def predict(self, X):
        check_is_fitted(self, "base_")
        return self.meta_.predict(self._meta_features(check_array(X)))
