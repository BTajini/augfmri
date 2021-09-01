# Authors: Hugo Richard, Badr Tajini
# Code heavily inspired from https://github.com/arthurmensch/cogspaces
# and https://github.com/Parietal-INRIA/DiFuMo/blob/master/notebook/fetcher.py
# License: BSD 3 clause

# Useful tools for running the example
import os
import numpy as np
from os.path import join
import pandas as pd
from joblib import load
from nilearn.datasets import fetch_neurovault_ids
from sklearn.utils import gen_batches, Bunch
from nilearn.input_data import NiftiMasker
from joblib import Parallel, delayed, dump, load
from nilearn._utils import check_niimg
from nilearn.datasets.utils import _fetch_files, _get_dataset_dir


def _assemble(images, images_meta, study):
    records = []
    for image, meta in zip(images, images_meta):
        if study == "brainpedia":
            this_study = meta["study"]
            subject = meta["name"].split("_")[-1]
            contrast = "_".join(meta["task"].split("_")[1:])
            task = meta["task"].split("_")[0]
        elif study == "henson2010faces":
            this_study = study
            subject = 0
            contrast = "_".join(meta["task"].split("_")[1:])
            task = meta["task"].split("_")[0]
        else:
            this_study = study
            subject = meta["name"].split("_")[0]
            contrast = meta["contrast_definition"]
            task = meta["task"]
        records.append([image, this_study, subject, task, contrast])
    df = pd.DataFrame(
        records, columns=["z_map", "study", "subject", "task", "contrast"]
    )
    return df


def mask_contrasts(data, output_dir="masked", n_jobs=1):
    batch_size = 10

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    mask = "../data/hcp_mask.nii.gz"
    masker = NiftiMasker(
        smoothing_fwhm=4, mask_img=mask, verbose=0, memory_level=1, memory=None
    ).fit()

    for study, this_data in data.groupby("study"):
        imgs = this_data["z_map"].values
        targets = this_data.reset_index()[["study", "subject", "contrast"]]

        n_samples = this_data.shape[0]
        batches = list(gen_batches(n_samples, batch_size))
        this_data = Parallel(
            n_jobs=n_jobs, verbose=10, backend="multiprocessing", mmap_mode="r"
        )(delayed(single_mask)(masker, imgs[batch]) for batch in batches)
        this_data = np.concatenate(this_data, axis=0)
        dump((this_data, targets), join(output_dir, "data_%s.pt" % study))


def single_mask(
    masker, imgs, confounds=None, root=None, raw_dir=None, save=False
):
    imgs = check_niimg(imgs)
    if save is False:
        return masker.transform(imgs, confounds)

    if imgs.get_filename() is None:
        raise ValueError("Provided Nifti1Image should be linked to a file.")
    filename = imgs.get_filename()
    raw_filename = filename.replace(".nii.gz", ".npy")
    if root is not None and raw_dir is not None:
        raw_filename = raw_filename.replace(root, raw_dir)
    dirname = os.path.dirname(raw_filename)

    all_confounds = []
    for confound in confounds:
        cfd = pd.read_csv(
            confound, header=None, delim_whitespace=True, engine="python"
        )
        all_confounds.append(cfd.values.astype(float))
    all_confounds = np.hstack(all_confounds)

    data = masker.transform(imgs, confounds=all_confounds)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    np.save(raw_filename, data)
    return raw_filename


def fetch_difumo(dimension=64, resolution_mm=2, data_dir=None):
    """Fetch DiFuMo brain atlas
    Parameters
    ----------
    dimension : int
        Number of dimensions in the dictionary. Valid resolutions
        available are {64, 128, 256, 512, 1024}.
    resolution_mm : int
        The resolution in mm of the atlas to fetch. Valid options
        available are {2, 3}.
    data_dir : string, optional
        Path where data should be downloaded. By default,
        files are downloaded in home directory.
    Returns
    -------
    data: sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are :
        - 'maps': str, 4D path to nifti file containing regions definition.
        - 'labels': string list containing the labels of the regions.
    References
    ----------
    Dadi, K., Varoquaux, G., Machlouzarides-Shalit, A., Gorgolewski, KJ.,
    Wassermann, D., Thirion, B., Mensch, A.
    Fine-grain atlases of functional modes for fMRI analysis,
    Paper in preparation
    """
    dic = {
        64: "pqu9r",
        128: "wjvd5",
        256: "3vrct",
        512: "9b76y",
        1024: "34792",
    }
    valid_dimensions = [64, 128, 256, 512, 1024]
    valid_resolution_mm = [2, 3]
    if dimension not in valid_dimensions:
        raise ValueError(
            "Requested dimension={} is not available. Valid "
            "options: {}".format(dimension, valid_dimensions)
        )
    if resolution_mm not in valid_resolution_mm:
        raise ValueError(
            "Requested resolution_mm={} is not available. Valid "
            "options: {}".format(resolution_mm, valid_resolution_mm)
        )
    url = "https://osf.io/{}/download".format(dic[dimension])
    opts = {"uncompress": True}

    csv_file = os.path.join("{0}", "labels_{0}_dictionary.csv")
    if resolution_mm != 3:
        nifti_file = os.path.join("{0}", "2mm", "maps.nii.gz")
    else:
        nifti_file = os.path.join("{0}", "3mm", "maps.nii.gz")

    files = [
        (csv_file.format(dimension), url, opts),
        (nifti_file.format(dimension), url, opts),
    ]

    dataset_name = "difumo_atlases"

    data_dir = _get_dataset_dir(
        data_dir=data_dir, dataset_name=dataset_name, verbose=1
    )

    # Download the zip file, first
    files = _fetch_files(data_dir, files, verbose=2)
    labels = pd.read_csv(files[0])

    # README
    readme_files = [
        ("README.md", "https://osf.io/4k9bf/download", {"move": "README.md"})
    ]
    if not os.path.exists(os.path.join(data_dir, "README.md")):
        _fetch_files(data_dir, readme_files, verbose=2)

    return Bunch(maps=files[1], labels=labels)
