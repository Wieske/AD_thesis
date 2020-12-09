"""
Script with a collection of functions
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataset import NiftiDataGenerator
from sklearn.model_selection import train_test_split, StratifiedKFold


def create_generators(df, mri_shape, pet_shape, to_fit=True, batchsize=1, sampling=None, aug=0, shuffle=True):
    """
    Create data generators for MRI, PET and combi (subjects with both MRI and PET)
    :param df: dataframe with image and subject information
    :param mri_shape: shape of MR images
    :param pet_shape: shape of PET images
    :param to_fit: boolean that indicates if generator should return samples and labels (True) or only samples (False)
    :param batchsize: size of the batch
    :param sampling: indicates if under ("under") or over ("over") sampling should be used to balance classes
    :param aug: augmentation setting (0 for no augmentation, 0.05 for about 5 percent augmentation)
    :param shuffle: boolean that indicates if samples should be shuffled
    :return: data generators for MRI, PET and combi
    """
    df_mri = df[df["modality"] == "MRI"]
    df_pet = df[df["modality"] == "PET"]
    df_combi = df[df.duplicated("subject", keep=False)]

    def f(x):
        d = {"class": x["class"].iloc[0] if len(set(x["class"])) == 1 else "NA",
             "filepath_mri": x["filepath"][x["modality"] == "MRI"].iloc[0],
             "filepath_pet": x["filepath"][x["modality"] == "PET"].iloc[0]}
        return pd.Series(d)

    df_combi = df_combi.groupby("subject").apply(f).reset_index()

    gen_mri = NiftiDataGenerator(df_mri, to_fit=to_fit, batch_size=batchsize, mri_shape=mri_shape, modality="MRI", sampling=sampling, aug=aug, shuffle=shuffle)
    gen_pet = NiftiDataGenerator(df_pet, to_fit=to_fit, batch_size=batchsize, pet_shape=pet_shape, modality="PET", sampling=sampling, aug=aug, shuffle=shuffle)
    gen_combi = NiftiDataGenerator(df_combi, to_fit=to_fit, batch_size=batchsize, mri_shape=mri_shape, pet_shape=pet_shape, modality="combi", sampling=sampling, aug=aug, shuffle=shuffle)

    return gen_mri, gen_pet, gen_combi


def plot_batch(generator, title=None, savepath=None):
    """
    Generate and plot one batch of images and optionally save the figure to file
    :param generator: data generator that should be used to generate a batch of images
    :param title: title of the plot
    :param savepath: path where image should be saved
    """
    X, y = generator.__getitem__(0)
    if y.ndim == 1:
        labels = [{0: "CN", 1: "AD"}[label] for label in y]
    else:
        y = np.argmax(y, axis=1)
        labels = [{0: "CN", 1: "MCI", 2: "AD"}[label] for label in y]
    fig, axes = plt.subplots(3, 4, figsize=(13, 9))
    [ax.set_axis_off() for ax in axes.ravel()]
    fig.set_facecolor('black')
    for i in range(4):
        s = X.shape
        axes[0, i].set_title(labels[i], c="white")
        axes[0, i].imshow(X[i, :, :, s[3]//2, 0].T, origin="lower")
        axes[1, i].imshow(X[i, :, s[2]//2, :, 0].T, origin="lower")
        axes[2, i].imshow(X[i, s[1]//2, :, :, 0].T, origin="lower")
    if title is not None:
        fig.suptitle(title, fontsize=16, c="white")
    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()


def plot_metrics(history, metrics, show=False, savepath=None):
    """
    Create a plot of one or more metrics for training and validation
    :param history: tf.keras history object
    :param metrics: name(s) of the metrics that should be plotted
    :param show: boolean that indicates if figure should be shown
    :param savepath: path where figure should be saved
    """
    for metric in metrics:
        plt.figure()
        plt.plot(history.history[metric])
        plt.plot(history.history['val_' + metric])
        plt.title(metric)
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        if savepath is not None:
            plt.savefig(savepath)
        if show:
            plt.show()
        else:
            plt.close()


def plot_loss(history, show=False, savepath=None):
    """
    Create a plot of the loss for training and validation
    :param history: tf.keras history object
    :param show: boolean that indicates if figure should be shown
    :param savepath: path where figure should be saved
    """
    # Plot the loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()
    else:
        plt.close()


def get_subject_splits(df, nr_splits, rnd):
    """
    Split data in train and test/ validation set, possible using cross validation, stratified by class and modality
    :param df: dataframe with image and subject information
    :param nr_splits: number of splits for cross validation (None for one random split)
    :param rnd: random seed
    :return: subjects, stratification column and test splits
    """
    # Extract subjects and class/modality from dataframe:
    df_sub = df.groupby("subject").agg({"class": "first", "modality": "sum"})
    df_sub["modality"].replace({"PETMRI": "MRIPET"}, inplace=True)
    strat_col = df_sub["class"] + df_sub["modality"]
    subjects = df_sub.index.values
    # Define subjects for test set splits:
    if nr_splits is None:
        _, sub_test = train_test_split(subjects, test_size=100, random_state=rnd, stratify=strat_col)
        sub_test_splits = [sub_test]
    else:
        skf = StratifiedKFold(n_splits=nr_splits, shuffle=True, random_state=rnd)
        sub_test_splits = [subjects[test_idx] for _, test_idx in skf.split(subjects, strat_col)]
    return subjects, strat_col, sub_test_splits


def get_subject_statistics(params):
    """
    Compute statistics for the used subjects
    :param params: dictionary with the parameters for root, df_name, rnd, batchsize, mri_shape, pet_shape, nr_splits,
    dirname, nr_class, augmentation, dropout, model_mri and model_pet
    """
    adni_df = pd.read_csv(params["root"] / "df_ADNI_M_MRIPET.csv", index_col=0)
    adni_df["Group"].replace({"EMCI": "MCI"}, regex=True, inplace=True)
    adni_df = adni_df[adni_df.Registered]
    # Get total number of subjects per class:
    adni_df.groupby(["Subject"]).first().groupby(["Group"]).count()
    # Get total number of MRI per class:
    mri_df = adni_df[adni_df.Modality == "MRI"]
    mri_df.groupby(["Subject"]).first().groupby(["Group"]).count()
    # Get total number of PET per class:
    pet_df = adni_df[adni_df.Modality == "PET"]
    pet_df.groupby(["Subject"]).first().groupby(["Group"]).count()
    # Get number of subjects with MRI & PET per class:
    sub_mod = adni_df.groupby(["Subject"]).agg({"Group": np.unique, "Modality": [np.unique, "nunique"]})
    sub_mod.columns = ["Group", "Modality", "nrmod"]
    sub_mod[sub_mod["nrmod"] == 2].groupby("Group").count()
    # Get age and sex statistics:
    sub_dem = adni_df.groupby(["Subject"]).agg({"Group": np.unique, "Age": np.mean, "Sex": np.unique})
    sub_dem.groupby("Group").agg({"Age": [np.mean, np.std]})
    sub_dem.groupby(["Group", "Sex"]).count()

    # Merge with ADNIMERGE file to get MMSE and CDR:
    adni_merge = pd.read_csv(params["root"] / "Study_data" / "ADNIMERGE.csv", index_col=1)
    adni_merge = adni_merge[adni_merge.VISCODE == "bl"]
    df = adni_df.groupby(["Subject"]).agg({"Group": np.unique, "Sex": np.unique, "Age": np.min})
    df = df.merge(adni_merge, "left", left_index=True, right_index=True)
    df.groupby("Group").agg({"MMSE_bl": [np.mean, np.std]})
