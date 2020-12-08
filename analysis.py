"""
Script for the analysis of the results
Combine average results for each experiment, create violin plots and apply statistical tests
"""

from pathlib import Path
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.metrics as metrics
import scikit_posthocs as sp


def check_sub_split():
    # Check if the models all use the same test and validation set for each split
    savedir = root / "Models" / "CNN3_run1"
    test = [np.load(savedir / ("split_" + str(n + 1)) / "sub_test.npy", allow_pickle=True) for n in range(5)]
    valid = [np.load(savedir / ("split_" + str(n + 1)) / "sub_valid.npy", allow_pickle=True) for n in range(5)]
    test_equal = []
    valid_equal = []
    for i in range(3, 20):
        for j in [1, 2, 3]:
            dirname = "CNN" + str(i) + "_run" + str(j)
            savedir = root / "Models" / dirname
            for n in range(5):
                savedir_split = savedir / ("split_" + str(n + 1))
                sub_test = np.load(savedir_split / "sub_test.npy", allow_pickle=True)
                sub_valid = np.load(savedir_split / "sub_valid.npy", allow_pickle=True)
                test_equal.append(np.all(sub_test == test[n]))
                valid_equal.append(np.all(sub_valid == valid[n]))
    print("All test sets equal: ", np.all(test_equal))
    print("All validation sets equal: ", np.all(valid_equal))


def combine_history_results(root, expnr, par_reg=False, par_arch=False):
    """
    Combine all history results of multiple experiments
    :param expnr: the numbers of the experiments that should be combined
    :param par_reg: whether to include regularization parameters (default False)
    :param par_arch: whether to include architecture parameters (default False)
    :return: history results of all experiments in expnr
    """
    histlist = []
    for i in expnr:
        # Extract history results from all three runs of this experiment
        for j in [1, 2, 3]:
            dirname = "CNN" + str(i) + "_run" + str(j)
            savedir = root / "Models" / dirname
            # Read the parameter file and extract several parameters:
            with open(savedir / "parameters.txt", "r") as f:
                param = f.read()
                param = ast.literal_eval(param.replace("WindowsPath", ""))
            hist = pd.read_csv(savedir / "best_results.csv")
            hist["exp"] = i
            hist["run"] = j
            hist["batchsize"] = param["batchsize"]
            hist["nr_class"] = param["nr_class"]
            if par_reg:
                hist["augmentation"] = param["augmentation"]
                hist["dropout"] = param["dropout"]
            if par_arch:
                hist["cnn_layers"] = int(param["models"][0])
                hist["start_filters"] = int(param["models"].split("[")[1].split(",")[0])
            histlist.append(hist)
    return pd.concat(histlist, ignore_index=True)


def combine_test_predictions(root, expnr):
    """
    Combine the test predictions of all folds for multiple models
    :param expnr: the numbers of the experiments that should be combined
    :return: test predictions for all experiments in expnr
    """
    testlist = []
    for i in expnr:
        # Extract history results from all three runs of this experiment
        for j in [1, 2, 3]:
            dirname = "CNN" + str(i) + "_run" + str(j)
            savedir = root / "Models" / dirname
            for file in savedir.glob("**/test_results.csv"):
                test = pd.read_csv(file, index_col=0)
                test["model"] = file.parent.name
                test["split"] = int(file.parent.parent.name[-1])
                test["exp"] = i
                test["run"] = j
                testlist.append(test)
    return pd.concat(testlist, ignore_index=True)


def create_violin_plot(history, column, title, labels, names=None, plotmodels=False, savepath=None):
    if names is None:
        names = labels
    runmeans = history.groupby(["model", "exp", "run"]).mean().reset_index()
    if plotmodels:
        data = [history[column][history.model == i].values for i in labels]
        runmeans = [runmeans[column][runmeans.model == i].values for i in labels]
    else:
        data = [history[column][history.exp == i].values for i in labels]
        runmeans = [runmeans[column][runmeans.exp == i].values for i in labels]
    plt.figure()
    pos = np.arange(1, len(names) + 1)
    plt.violinplot(data, positions=pos, showmeans=True, showextrema=True)
    for x, y in zip(pos, runmeans):
        x = np.array([-0.01, 0, 0.01]) + x
        plt.scatter(x, y, c="k", marker="x", linewidths=1, zorder=10)  # [x] * len(y)
    plt.title(title)
    if column == "val_binary_accuracy" or column == "val_categorical_accuracy":
        plt.ylabel("Validation accuracy")
    elif column == "test_acc":
        plt.ylabel("Test accuracy")
    elif column == "test_b_acc":
        plt.ylabel("Balanced test accuracy")
    else:
        plt.ylabel(column)
    plt.xticks(pos, labels=names)
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches="tight")
        plt.close()


def plot_roc(y_true, y_pred, name, savepath):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve %s (area = %0.2f)' % (name, roc_auc))
    plt.savefig(savepath)
    plt.close()


def calculate_metrics(grouped_pred):
    """
    Calculate metrics on predictions, grouped by exp/ model and run
    :param grouped_pred: grouped dataframe containing the predictions
    :return: metrics for grouped dataframe
    """
    x = grouped_pred
    d = {}
    d["test_acc"] = metrics.accuracy_score(x["true_class"], x["pred_class"])
    d["test_b_acc"] = metrics.balanced_accuracy_score(x["true_class"], x["pred_class"])
    if "pred" in x.columns:
        d["test_auc"] = metrics.roc_auc_score(x["true_class"], x["pred"])
        name = x["model"].iloc[0] + "_run" + str(x["run"].iloc[0])
        plot_roc(x["true_class"], x["pred"], name, root / "Images" / "ROC_curve" / (name + ".png"))
        return pd.Series(d, index=["test_acc", "test_b_acc", "test_auc"])
    else:
        d["conf"] = metrics.confusion_matrix(x["true_class"], x["pred_class"])
        return pd.Series(d, index=["test_acc", "test_b_acc", "conf"])


def plot_confusion(test_grouped, models, expnr, title, normalize=True, savepath=None):
    n = 1 if normalize else 3
    fig, axes = plt.subplots(n, len(models), figsize=(5 * len(models), 5.2 * n))
    for name, group in test_grouped:
        exp, model = name
        if model in models.keys() and exp == expnr:
            y = models[model]
            if normalize:
                conf = np.array(np.sum(group["conf"]))
                conf = conf / np.sum(conf, axis=1, keepdims=True)
                confs = [conf]
            else:
                confs = group["conf"]
            for x, conf in enumerate(confs):
                xy = y if normalize else (x, y)
                axes[xy].matshow(conf, cmap="Blues")
                n = conf.shape[0]
                for i in range(n):
                    for j in range(n):
                        if normalize:
                            per = int(np.round(100 * conf[i, j]))
                            ctext = str(per) + " %"
                            color = "k" if per < 50 else "w"
                        else:
                            ctext = conf[i, j]
                            color = "k" if conf[i, j] < (np.max(conf) / 2) else "w"
                        axes[xy].text(j, i, ctext, ha="center", va="center", c=color, fontsize=14)
                labels = ["CN", "MCI", "AD"]
                axes[xy].set(xticks=np.arange(n), yticks=np.arange(n), xticklabels=labels, yticklabels=labels)
                axes[xy].set_xlabel("Predicted class")
                axes[xy].set_ylabel("True class")
                axes[xy].xaxis.set_label_position('top')
                model = "MRI" if model == "MRI_FT" else "PET" if model == "PET_FT" else model
                if normalize:
                    axes[xy].set_title(model)
                else:
                    axes[xy].set_title(model + " run " + str(x + 1))
    fig.suptitle(title)
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath)
        plt.close()


def friedman_test(history, expnr, column="val_binary_accuracy", test="friedman"):
    data = [history[column][history.exp == i].values for i in expnr]
    if test == "friedman":
        stat, pval = stats.friedmanchisquare(*data)
        pvals = sp.posthoc_nemenyi_friedman(history, y_col=column, group_col="exp", block_col="run", melted=True)
        return pval, pvals
    if test == "wilcoxon":
        if len(data) == 2:
            return stats.wilcoxon(*data)
        else:
            n = len(data)
            pvals = np.ones((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        _, pvals[i, j] = stats.wilcoxon(data[i], data[j])
            return pvals


def delete_weights(expnr):
    # Delete the weights of the model numbers in expnr (to make room on disk)
    for i in expnr:
        for j in [1, 2, 3]:
            dirname = "CNN" + str(i) + "_run" + str(j)
            modeldir = root / "Models" / dirname
            for n in [1, 2, 3, 4, 5]:
                splitdir = modeldir / ("split_" + str(n))
                for m in ["MRI", "PET", "combi"]:
                    filename = "weights.data-00000-of-00001"
                    path = splitdir / m / filename
                    if path.exists():
                        path.unlink()
            print("Deleted weights of model", dirname)


# Data augmentation and dropout: [6, 3, 7, 8, 9, 10] -> winner 7 (0.05 aug, 0 drop)
# Batchnormalization: [7, 11] -> winner y (yes)
# strides vs maxpool: [7, 12]
# nr CNN layers & filters: [13, 7, 15, 16, 17]
# (13: 4;8, 7: 5;8, 15: 5;16, 16: 6;8, 17: 6;16;averagepool)
# multi-modal 2: [18, 20] (using all data for 18 and only complete for 20)
# multi-modal 3: [19, 21] (training from scratch for 19 and transfer learning from exp 18 for 21)

root = Path("D:/Wieske/Documents/DataScience/Thesis/Data")

# BATCHNORMALIZATION:
history_bn = combine_history_results(root, [7, 11])
history_bn["batchnorm"] = history_bn.exp == 7
histmean_bn = history_bn.groupby(["model", "exp"]).mean()
histmean_bn.reset_index(inplace=True)
histmean_bn.sort_values(["model", "batchnorm"], inplace=True)
histmean_bn.to_csv(root / "Tables" / "histmean_batchnorm.csv", index=False)

create_violin_plot(history_bn[history_bn.model == "MRI"],
                   column="val_binary_accuracy",
                   title="Batchnormalization results for MRI model",
                   labels=[7, 11],
                   names=["BN", "no BN"],
                   savepath=root / "Images" / "Violin" / "bn_MRI.png")
create_violin_plot(history_bn[history_bn.model == "PET"],
                   column="val_binary_accuracy",
                   title="Batchnormalization results for PET model",
                   labels=[7, 11],
                   names=["BN", "no BN"],
                   savepath=root / "Images" / "Violin" / "bn_PET.png")

stat, pval = friedman_test(history_bn[history_bn.model == "MRI"], [7, 11], test="wilcoxon")
print("Wilcoxon test for MRI batch normalization results: statistic: ", stat, ", p value: ", pval)
stat, pval = friedman_test(history_bn[history_bn.model == "PET"], [7, 11], test="wilcoxon")
print("Wilcoxon test for PET batch normalization results: statistic: ", stat, ", p value: ", pval)


# REGULARIZATION:
history_reg = combine_history_results(root, [6, 3, 7, 8, 9, 10], par_reg=True)
histmean_reg = history_reg.groupby(["model", "exp"]).mean()
histmean_reg.reset_index(inplace=True)
histmean_reg.sort_values(["model", "augmentation", "dropout"], inplace=True)
histmean_reg.to_csv(root / "Tables" / "histmean_regularization.csv", index=False)

create_violin_plot(history_reg[history_reg.model == "MRI"],
                   column="val_binary_accuracy",
                   title="Regularization results for MRI model",
                   labels=[6, 3, 7, 8, 9, 10],
                   names=["A0 D0", "A0 D0.5", "A0.05 D0", "A0.05 D0.5", "A0.1 D0", "A0.1 D0.5"],
                   savepath=root / "Images" / "Violin" / "reg_MRI.png")
create_violin_plot(history_reg[history_reg.model == "PET"],
                   column="val_binary_accuracy",
                   title="Regularization results for PET model",
                   labels=[6, 3, 7, 8, 9, 10],
                   names=["A0 D0", "A0 D0.5", "A0.05 D0", "A0.05 D0.5", "A0.1 D0", "A0.1 D0.5"],
                   savepath=root / "Images" / "Violin" / "reg_PET.png")

stat, pval = friedman_test(history_reg[history_reg.model == "MRI"], [6, 3, 7, 8, 9, 10])
print("Friedman test for MRI regularization results: p value: ", stat)
print(pval)
stat, pval = friedman_test(history_reg[history_reg.model == "PET"], [6, 3, 7, 8, 9, 10])
print("Friedman test for PET regularization results: p value: ", stat)
print(pval)


# MODEL ARCHITECTURE:
history_arch = combine_history_results(root, [13, 7, 15, 16, 17], par_arch=True)
histmean_arch = history_arch.groupby(["model", "exp"]).mean()
histmean_arch.reset_index(inplace=True)
histmean_arch.sort_values(["model", "cnn_layers", "start_filters"], inplace=True)
histmean_arch.to_csv(root / "Tables" / "histmean_architecture.csv", index=False)

create_violin_plot(history_arch[history_arch.model == "MRI"],
                   column="val_binary_accuracy",
                   title="Architecture results for MRI model",
                   labels=[13, 7, 15, 16, 17],
                   names=["4l 8f", "5l 8f", "5l 16f", "6l 8f", "6l 16f"],
                   savepath=root / "Images" / "Violin" / "arch_MRI.png")
create_violin_plot(history_arch[history_arch.model == "PET"],
                   column="val_binary_accuracy",
                   title="Architecture results for PET model",
                   labels=[13, 7, 15, 16, 17],
                   names=["4l 8f", "5l 8f", "5l 16f", "6l 8f", "6l 16f"],
                   savepath=root / "Images" / "Violin" / "arch_PET.png")

stat, pval = friedman_test(history_arch[history_arch.model == "MRI"], [13, 7, 15, 16, 17])
print("Friedman test for MRI architecture results: p value: ", stat)
print(pval)
stat, pval = friedman_test(history_arch[history_arch.model == "PET"], [13, 7, 15, 16, 17])
print("Friedman test for PET architecture results: p value: ", stat)
print(pval)


# MULTI-MODAL MODEL 2 - CN vs AD
pred_mm2 = combine_test_predictions(root, [18, 20])
pred_mm2["model"] = pred_mm2.apply(lambda x: "COMBI_COMPLETE" if x.exp == 20 else x.model, axis=1)
pred_mm2["model"] = pd.Categorical(pred_mm2["model"], ["MRI", "PET", "combi", "finetune", "COMBI_COMPLETE"]).\
    rename_categories(["MRI", "PET", "MM", "MM_FT", "MM_COMPLETE"])
pred_mm2 = pred_mm2[["model", "run", "subject", "class", "pred", "pred_class", "true_class"]]
test_mm2 = pred_mm2.groupby(["model", "run"]).apply(calculate_metrics).reset_index()
test_mm2 = test_mm2.groupby(["model"]).mean().reset_index()

history_mm2 = combine_history_results(root, [18, 20])
history_mm2["model"] = history_mm2.apply(lambda x: "COMBI_COMPLETE" if x.exp == 20 else x.model, axis=1)
histmean_mm2 = history_mm2.groupby(["model", "exp"]).mean()
histmean_mm2.reset_index(inplace=True)
histmean_mm2["model"] = pd.Categorical(histmean_mm2["model"], ["MRI", "PET", "COMBI", "FINETUNE", "COMBI_COMPLETE"]).\
    rename_categories(["MRI", "PET", "MM", "MM_FT", "MM_COMPLETE"])
histmean_mm2 = histmean_mm2.merge(test_mm2[["model", "test_b_acc", "test_auc"]], on="model")
histmean_mm2.sort_values("model", inplace=True)
histmean_mm2 = histmean_mm2[histmean_mm2.model.isin(["MRI", "PET", "MM_FT", "MM_COMPLETE"])]
histmean_mm2.loc[histmean_mm2.model == "MM_FT", "model"] = "MM"
histmean_mm2.to_csv(root / "Tables" / "histmean_multimodal2.csv", index=False)

create_violin_plot(history_mm2,
                   column="test_acc",
                   title="Accuracy for binary classification (CN vs AD)",
                   labels=["MRI", "PET", "FINETUNE", "COMBI_COMPLETE"],
                   names=["MRI", "PET", "MM", "MM_COMPLETE"],
                   plotmodels=True,
                   savepath=root / "Images" / "Violin" / "multimodal2.png")

stat, pval = friedman_test(history_mm2[(history_mm2.exp == 20) | (history_mm2.model == "FINETUNE")],
                           [18, 20], column="test_acc", test="wilcoxon")
print("Wilcoxon test for MM vs MM COMPLETE: statistic: ", stat, ", p value: ", pval)

# MULTI-MODAL MODEL 3 - CN vs MCI vs AD
pred_mm3 = combine_test_predictions(root, [19, 21])
pred_mm3["model"] = pd.Categorical(pred_mm3["model"], ["MRI", "MRI_FT", "PET", "PET_FT", "combi", "finetune"]).\
    rename_categories(["MRI", "MRI_FT", "PET", "PET_FT", "MM", "MM_FT"])
pred_mm3 = pred_mm3[["exp", "run", "split", "model", "subject", "class", "pred_0", "pred_1", "pred_2", "pred_class", "true_class"]]
test_mm3 = pred_mm3.groupby(["exp", "model", "run"], observed=True).apply(calculate_metrics).reset_index()
b_acc = pred_mm3.groupby(["exp", "model", "run", "split"], observed=True).apply(calculate_metrics).reset_index()

grouped_mm3 = test_mm3.groupby(["exp", "model"], observed=True)
test_mm3 = grouped_mm3.mean().reset_index()
plot_confusion(grouped_mm3,
               models={"MRI": 0, "PET": 1, "MM": 2},
               expnr=19,
               title="Confusion matrices for classification of CN vs MCI vs AD",
               savepath=root / "Images" / "mm3_confusion.png")
plot_confusion(grouped_mm3,
               models={"MRI_FT": 0, "PET_FT": 1, "MM": 2},
               expnr=21,
               title="Confusion matrices for classification of CN vs MCI vs AD with transfer learning",
               savepath=root / "Images" / "mm3_confusion_TL.png")

plot_confusion(grouped_mm3,
               models={"MRI": 0, "PET": 1, "MM": 2},
               expnr=19,
               title="Confusion matrices for classification of CN vs MCI vs AD",
               normalize=False,
               savepath=root / "Images" / "mm3_confusion_all.png")
plot_confusion(grouped_mm3,
               models={"MRI_FT": 0, "PET_FT": 1, "MM": 2},
               expnr=21,
               title="Confusion matrices for classification of CN vs MCI vs AD with transfer learning",
               normalize=False,
               savepath=root / "Images" / "mm3_confusion_TL_all.png")

history_mm3 = combine_history_results(root, [19, 21])
history_mm3["model"] = pd.Categorical(history_mm3["model"], ["MRI", "MRI_FT", "PET", "PET_FT", "COMBI", "FINETUNE"]).\
    rename_categories(["MRI", "MRI_FT", "PET", "PET_FT", "MM", "MM_FT"])
history_mm3["transfer_learning"] = history_mm3.exp == 21
histmean_mm3 = history_mm3.groupby(["model", "exp"], observed=True).mean()
histmean_mm3.reset_index(inplace=True)
histmean_mm3 = histmean_mm3.merge(test_mm3[["exp", "model", "test_b_acc"]], how="left", on=["exp", "model"])
histmean_mm3.sort_values(["transfer_learning", "model"], inplace=True)
histmean_mm3 = histmean_mm3[(histmean_mm3.exp == 19) & (histmean_mm3.model.isin(["MRI", "PET", "MM"])) |
                            ((histmean_mm3.exp == 21) & (histmean_mm3.model.isin(["MRI_FT", "PET_FT", "MM"])))]
histmean_mm3.loc[histmean_mm3.model == "MRI_FT", "model"] = "MRI"
histmean_mm3.loc[histmean_mm3.model == "PET_FT", "model"] = "PET"
histmean_mm3.to_csv(root / "Tables" / "histmean_multimodal3.csv", index=False)

history_mm3 = history_mm3.merge(b_acc[["exp", "model", "run", "split", "test_b_acc"]], how="left", on=["exp", "model", "run", "split"])

create_violin_plot(history_mm3[~history_mm3.transfer_learning],
                   column="test_b_acc",
                   title="Accuracy for classification of CN vs MCI vs AD",
                   labels=["MRI", "PET", "MM"],
                   plotmodels=True,
                   savepath=root / "Images" / "Violin" / "multimodal3.png")

create_violin_plot(history_mm3[history_mm3.transfer_learning],
                   column="test_b_acc",
                   title="Accuracy for classification of CN vs MCI vs AD (with TL)",
                   labels=["MRI_FT", "PET_FT", "MM"],
                   names=["MRI", "PET", "MM"],
                   plotmodels=True,
                   savepath=root / "Images" / "Violin" / "multimodal3_tl.png")

stat, pval = friedman_test(history_mm3[((history_mm3.exp == 19) & (history_mm3.model == "MRI")) |
                                       ((history_mm3.exp == 21) & (history_mm3.model == "MRI_FT"))],
                           [19, 21], column="test_b_acc", test="wilcoxon")
print("Wilcoxon test for MRI normal vs transfer learning: statistic: ", stat, ", p value: ", pval)

stat, pval = friedman_test(history_mm3[((history_mm3.exp == 19) & (history_mm3.model == "PET")) |
                                       ((history_mm3.exp == 21) & (history_mm3.model == "PET_FT"))],
                           [19, 21], column="test_b_acc", test="wilcoxon")
print("Wilcoxon test for PET normal vs transfer learning: statistic: ", stat, ", p value: ", pval)

stat, pval = friedman_test(history_mm3[history_mm3.model == "MM"], [19, 21], column="test_b_acc", test="wilcoxon")
print("Wilcoxon test for MM normal vs transfer learning: statistic: ", stat, ", p value: ", pval)

# results = hist.groupby(["model", "exp"]).agg({"test_acc": ["mean", "std"], "batchsize": "first"})
# results.columns = ["test_acc_mean", "test_acc_std", "batchsize"]
# results.to_csv(root / "Tables" / "results_regularization.csv", index=False)
