"""
Script with a collection of functions
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataset import NiftiDataGenerator
from sklearn.model_selection import train_test_split, StratifiedKFold


def create_generators(df, mri_shape, pet_shape, to_fit=True, batchsize=1, sampling=None, aug=0, shuffle=True):
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
    gen_pet = NiftiDataGenerator(df_pet, to_fit=to_fit, batch_size=4, pet_shape=pet_shape, modality="PET", sampling=sampling, aug=aug, shuffle=shuffle)
    gen_combi = NiftiDataGenerator(df_combi, to_fit=to_fit, batch_size=batchsize, mri_shape=mri_shape, pet_shape=pet_shape, modality="combi", sampling=sampling, aug=aug, shuffle=shuffle)

    return gen_mri, gen_pet, gen_combi


def plot_batch(generator, title=None, savepath=None):
    X, y = generator.__getitem__(0)
    if y.ndim == 1:
        labels = [{0: "NC", 1: "AD"}[label] for label in y]
    else:
        y = np.argmax(y, axis=1)
        labels = [{0: "NC", 1: "MCI", 2: "AD"}[label] for label in y]
    fig, axes = plt.subplots(3, 4, figsize=(13, 9))
    [ax.set_axis_off() for ax in axes.ravel()]
    for i in range(4):
        axes[0, i].set_title(labels[i])
        axes[0, i].imshow(X[i, :, :, 60, 0].T, origin="lower")
        axes[1, i].imshow(X[i, :, 70, :, 0].T, origin="lower")
        axes[2, i].imshow(X[i, 60, :, :, 0].T, origin="lower")
    if title is not None:
        fig.suptitle(title, fontsize=16)
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()


def plot_metrics(history, metrics, show=False, savepath=None):
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


def create_df_from_adni_csv(root):
    adni_M = pd.read_csv(root / "df_ADNI_M_MRIPET.csv")
    adni_M = adni_M[adni_M["Registered"]]
    df = adni_M[["Subject", "Group", "Modality", "Savedir"]].copy()
    df["Savedir"] = [adni_M["Savedir"][i] + "/" + adni_M["Savename"][i] for i in adni_M.index]
    df.columns = ["subject", "class", "modality", "filepath"]
    df["filepath"].replace({'/media/storage': 'D:'}, regex=True, inplace=True)
    df["class"].replace({"EMCI": "MCI"}, regex=True, inplace=True)
    return df


def get_subject_splits(df, nr_splits, rnd):
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


def extract_from_history(savedir):
    hist_list = []
    for i in range(5):
        hist = []
        for model in ["MRI", "PET", "combi"]:
            df_test = pd.read_csv(savedir / ("split_" + str(i + 1)) / model / "test_results.csv", index_col=0)
            test_acc = np.average(df_test["pred_class"] == df_test["true_class"])
            history = pd.read_csv(savedir / ("split_" + str(i + 1)) / model / "history.csv", index_col=0)
            hist_best = history[history.val_loss == history.val_loss.min()].copy()
            hist_best["test_acc"] = test_acc
            hist_best["epoch"] = hist_best.index
            hist_best["model"] = model
            hist.append(hist_best)
        hist = pd.concat(hist, ignore_index=True)
        hist["split"] = i + 1
        hist_list.append(hist)
    hist = pd.concat(hist_list, ignore_index=True)
    hist.to_csv(savedir / "best_results.csv", index=False)
    return hist


def get_subject_statistics(params):
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
