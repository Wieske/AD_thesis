from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from train import train_evaluate_models, train_only_multimodal, transfer_learning_mci

params = {
    # User parameters:
    "root": Path("D:/Wieske/Documents/DataScience/Thesis/Data"),
    "df_name": "df_sub_class_mod_path.csv",
    "rnd": 1,
    "batchsize": 2,
    "mri_shape": (129, 153, 129),
    "pet_shape": (160, 160, 96),
    "nr_splits": 5,
    # Run parameters:
    "dirname": "CNN21_run3",
    "nr_class": 3,
    "augmentation": 0.05,
    "dropout": 0,
    "model_mri": "6 layers ([16, 32, 64, 128, 256, 512]) with maxpooling and globalaveragepooling",
    "model_pet": "5 layers ([8, 16, 32, 64, 128]) with maxpooling"
}

# hist = train_evaluate_models(params)

from utils import create_generators, plot_batch

df = pd.read_csv(params["root"] / params["df_name"])
mri, pet, mm = create_generators(df, params["mri_shape"], params["pet_shape"], to_fit=True, batchsize=4, sampling="under", aug=0.05, shuffle=True)

#plot_batch(mri, savepath=params["root"] / "Images" / "Augmentation" / "MRI_0.05.png")
plot_batch(pet, savepath=params["root"] / "Images" / "Augmentation" / "PET_0.05.png")
