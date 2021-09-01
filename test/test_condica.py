# Authors: Hugo Richard, Badr Tajini
# License: BSD 3 clause

from condica.main import condica
from picard import picard
from sklearn.preprocessing import QuantileTransformer
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit


def test_condica():
    m_scores = []
    m_scores_aug = []
    n_components = 2
    n_features = 10
    n_samples = 1000
    n_classes = 2
    indexes = np.split(np.arange(n_samples), n_classes)
    seed = 0
    rng = np.random.RandomState(seed)
    S = rng.laplace(size=(n_components, n_samples))
    A = rng.randn(n_features, n_components)
    mu = rng.randn(n_classes, n_components)
    Y = np.zeros(n_samples)
    for i in range(n_classes):
        S[:, indexes[i]] = S[:, indexes[i]] + mu[i][:, None]
        Y[indexes[i]] = i
    S = S.T
    X = S.dot(A.T)

    clf = LinearDiscriminantAnalysis()
    cv = ShuffleSplit(random_state=rng, train_size=0.1, n_splits=20)
    scores_noaug = []
    scores_withaug = []
    for train, test in cv.split(X):
        X_train, X_test = X[train], X[test]
        Y_train, Y_test = Y[train], Y[test]
        X_fakes, Y_fakes = condica(
            A, X_train, Y_train, len(X[train]), n_quantiles=len(X[train])
        )
        scores_noaug.append(clf.fit(X_train, Y_train).score(X_test, Y_test))
        scores_withaug.append(
            clf.fit(
                np.concatenate([X_train, X_fakes]),
                np.concatenate([Y_train, Y_fakes]),
            ).score(X_test, Y_test)
        )
    print(np.mean(scores_noaug), np.mean(scores_withaug))
    assert np.mean(scores_noaug) < np.mean(scores_withaug)
