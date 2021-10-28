# Authors: Hugo Richard, Badr Tajini
# License: BSD 3 clause
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import LedoitWolf


def condica(A, X, Y=None, nb_fakes=10, n_quantiles=1000):
    """
    Computes fake examples using ICA and mixing matrices estimated
    from rest data

    Parameters
    -----------

    A : np array of size (n_features, n_components)
        Mixing matrix obtained from an ICA on rest fMRI data

    X : np array of size (n_samples, n_features)
        Task data

    Y : np array of size (n_samples,)
        Labels

    nb_fakes : int
        Number of fake labels per class

    n_quantiles : int
        Number of quantiles to be computed.
        It corresponds to the number of landmarks
        used to discretize the cumulative
        distribution function. If n_quantiles
        is larger than the number of samples,
        n_quantiles is set to the number of samples
        as a larger number of quantiles does not
        give a better approximation of the
        cumulative distribution function estimator.

    Returns
    -------

    X_fake : np array of size (n_fakes * n_classes, n_features)
        Fake task data, c is the number of unique classes

    Y_fake : np array of size (n_fakes * n_classes,)
        Labels of fake task data
    """
    unique_classes = np.unique(Y)
    _, n_components = A.shape
    quantile_transform = QuantileTransformer(
        output_distribution="normal", n_quantiles=n_quantiles
    )
    S = X.dot(np.linalg.pinv(A).T)
    Z = quantile_transform.fit_transform(S)

    if Y is None:
        lw = LedoitWolf().fit(Z)
        mean = lw.location_
        cov = lw.covariance_
        Z_fakes = np.random.multivariate_normal(mean, cov, size=nb_fakes)
    else:
        # LDA is just used as a mean to compute
        # class specific means and the global covariance.
        # It uses Ledoit Wolf estimator internally.
        lda = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto", store_covariance=True
        )
        lda.fit(Z, Y)
        means = np.array(lda.means_)
        cov = np.array(lda.covariance_)

        Z_fakes = np.zeros((nb_fakes * len(unique_classes), n_components))
        Y_fakes = np.zeros((nb_fakes * len(unique_classes)))

        for i in range(len(unique_classes)):
            class_i = slice(i * nb_fakes, (i + 1) * nb_fakes)
            Z_fakes[class_i, :] = np.random.multivariate_normal(
                means[i], cov, size=nb_fakes
            )
            Y_fakes[class_i] = np.repeat(unique_classes[i], nb_fakes)

    S_fakes = quantile_transform.inverse_transform(Z_fakes)
    X_fakes = np.dot(S_fakes, A.T)

    if Y is None:
        return X_fakes
    else:
        return X_fakes, Y_fakes
