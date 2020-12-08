from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from train import train_evaluate_models, train_only_multimodal, transfer_learning_mci, finetune_multimodal
from model import cnn_model_combi, cnn_model_mri, cnn_model_pet
from utils import create_generators, get_subject_splits, train_model

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


loaddir = savedir = params["root"] / "Models" / "CNN18_run3"
# hist = train_evaluate_models(params)
# hist = train_only_multimodal(params)
# hist = transfer_learning_mci(params, loaddir)

# hist = finetune_multimodal(loaddir)

mri = cnn_model_mri(params["mri_shape"], nr_class=2)
pet = cnn_model_pet(params["pet_shape"], nr_class=2)
mm = cnn_model_combi(mri_model=mri, pet_model=pet, nr_class=2)
print(mri.summary())

savepath = params["root"] / "Images" / "plot_mri_model.png"
tf.keras.utils.plot_model(mri, to_file=savepath, show_shapes=True, show_layer_names=False)
savepath = params["root"] / "Images" / "plot_pet_model.png"
tf.keras.utils.plot_model(pet, to_file=savepath, show_shapes=True, show_layer_names=False)
savepath = params["root"] / "Images" / "plot_mm_model.png"
tf.keras.utils.plot_model(mm, to_file=savepath, show_shapes=True, show_layer_names=False)

