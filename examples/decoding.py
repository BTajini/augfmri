from nilearn.datasets.neurovault import fetch_neurovault_ids
from condica.main import condica
from nilearn.input_data import NiftiMasker
from condica.utils import _assemble, mask_contrasts, fetch_difumo
from joblib import load, dump
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit

nv_ids = {
    "archi": 4339,
    "hcp": 4337,
    "brainomics": 4341,
    "camcan": 4342,
    "la5c": 4343,
    "brainpedia": 1952,
    "henson2010faces": 1811,
}

data = fetch_neurovault_ids([1811], data_dir="../data/")
data = _assemble(data["images"], data["images_meta"], "henson2010faces")
mask_contrasts(data, output_dir="../data/masked", n_jobs=4)

data = load("../data/masked/data_henson2010faces.pt")
X, Y = data

# Download the Difumo atlas to reduce the data
mask = fetch_difumo(dimension=1024, data_dir="../data/").maps
components = (
    NiftiMasker(mask_img="../data/hcp_mask.nii.gz", verbose=1)
    .fit()
    .transform(mask)
)
dump(components, "../data/difumo_atlases/1024/components.pt")
C = load("../data/difumo_atlases/1024/components.pt")
# Reduce the data using the atlas
X = X.dot(C.T)
# Load mixing matrix computed from rest fMRI data
A = np.load("../data/A_rest.npy")

Y = Y["contrast"].values
_, Y = np.unique(Y, return_inverse=True)
clf = LinearDiscriminantAnalysis()
cv = ShuffleSplit(random_state=0, train_size=0.8, n_splits=20)
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
