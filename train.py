"""
Script with function to train the models
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import schedules, Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from model import cnn_model_combi, cnn_model_mri, cnn_model_pet
from utils import plot_metrics, plot_loss, create_generators, get_subject_splits
import ast


def train_model(model, train, valid, test, savedir, nr_class, loss, metrics, learning_rate, nr_ep):
    savedir.mkdir(exist_ok=True)
    callbacks = [ModelCheckpoint(filepath=savedir / "weights", save_best_only=True, save_weights_only=True),
                 EarlyStopping(monitor='val_loss', patience=20)]
    # TensorBoard(log_dir=savedir / "logs")
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=metrics)
    history = model.fit(x=train, steps_per_epoch=len(train), validation_data=valid, validation_steps=len(valid),
                        epochs=nr_ep, verbose=2, callbacks=callbacks, max_queue_size=20, workers=8)

    plot_metrics(history, metrics, savepath=savedir / "accuracy.png")
    plot_loss(history, savepath=savedir / "loss.png")

    model.load_weights(savedir / "weights")
    model.evaluate(x=valid, verbose=1, steps=len(valid))
    df_test = test.df.copy()
    if nr_class == 2:
        df_test["pred"] = model.predict(test)
        df_test["pred_class"] = df_test["pred"].round(0).astype(int)
    else:
        pred = model.predict(test)
        df_test = df_test.join(pd.DataFrame(pred, columns=["pred_" + str(i) for i in range(pred.shape[1])]))
        df_test["pred_class"] = pred.argmax(axis=1)
    df_test["true_class"] = [test.class_indices[label] for label in df_test["class"]]
    confusion = pd.crosstab(df_test["true_class"], df_test["pred_class"], rownames=['Actual'], colnames=['Predicted'])
    test_acc = np.average(df_test["pred_class"] == df_test["true_class"])
    print("Accuracy on test set: ", test_acc)
    print(confusion)

    # Save details to file
    df_test.to_csv(savedir / "test_results.csv")
    history = pd.DataFrame(history.history)
    history.to_csv(savedir / "history.csv")
    hist_best = history[history.val_loss == history.val_loss.min()].copy()
    hist_best["test_acc"] = test_acc
    hist_best["epoch"] = hist_best.index

    return model, hist_best


def train_evaluate_models(params):
    df = pd.read_csv(params["root"] / params["df_name"])
    savedir = params["root"] / "Models" / params["dirname"]
    savedir.mkdir(exist_ok=False)

    # Save parameters to file:
    with open(savedir / "parameters.txt", "w") as f:
        f.write(str(params))

    # Split subjects:
    subjects, strat_col, sub_test_splits = get_subject_splits(df, params["nr_splits"], params["rnd"])

    # Define loss and metrics based on the number of classes
    if params["nr_class"] == 2:
        df = df[df["class"].isin(["CN", "AD"])]
        loss = "binary_crossentropy"
        metrics = ["binary_accuracy"]
    else:
        loss = "categorical_crossentropy"
        metrics = ["categorical_accuracy"]

    hist_list = []

    # Train models
    for i, sub_test in enumerate(sub_test_splits):
        tf.keras.backend.clear_session()
        # Create subdirectory for this split and save the test subjects:
        savedir_split = savedir / ("split_" + str(i + 1))
        savedir_split.mkdir(exist_ok=True)

        # Get samples for training, validation and test set:
        sub_train = np.array([sub for sub in subjects if sub not in sub_test])
        sub_train, sub_valid = train_test_split(sub_train, test_size=100, random_state=params["rnd"], stratify=strat_col[sub_train])
        np.save(savedir_split / "sub_test.npy", sub_test)
        np.save(savedir_split / "sub_valid.npy", sub_valid)
        np.save(savedir_split / "sub_train.npy", sub_train)
        df_train, df_valid, df_test = [df[df["subject"].isin(sub)] for sub in [sub_train, sub_valid, sub_test]]

        # Create data generators:
        train_gen_mri, train_gen_pet, train_gen_combi = create_generators(df_train, params["mri_shape"], params["pet_shape"], to_fit=True, batchsize=params["batchsize"], sampling="under", aug=params["augmentation"], shuffle=True)
        valid_gen_mri, valid_gen_pet, valid_gen_combi = create_generators(df_valid, params["mri_shape"], params["pet_shape"], to_fit=True, batchsize=params["batchsize"], sampling=None, shuffle=False)
        test_gen_mri, test_gen_pet, test_gen_combi = create_generators(df_test, params["mri_shape"], params["pet_shape"], to_fit=False, batchsize=params["batchsize"], sampling=None, shuffle=False)

        # TRAIN MRI MODEL
        print("Start training MRI model for split ", i + 1, " of ", params["nr_splits"])
        lr_schedule = schedules.PolynomialDecay(1e-4, 200*len(train_gen_mri), end_learning_rate=1e-6, power=2.0)
        model_mri = cnn_model_mri(image_shape=params["mri_shape"], nr_class=params["nr_class"], drop_ratio=params["dropout"])
        model_mri, history_mri = train_model(model_mri, train_gen_mri, valid_gen_mri, test_gen_mri, savedir_split / "MRI",
                                             params["nr_class"], loss, metrics, lr_schedule, 200)
        history_mri["model"] = "MRI"

        # TRAIN PET MODEL
        print("Start training PET model for split ", i + 1, " of ", params["nr_splits"])
        lr_schedule = schedules.PolynomialDecay(1e-4, 200*len(train_gen_pet), end_learning_rate=1e-6, power=2.0)
        model_pet = cnn_model_pet(image_shape=params["pet_shape"], nr_class=params["nr_class"], drop_ratio=params["dropout"])
        model_pet, history_pet = train_model(model_pet, train_gen_pet, valid_gen_pet, test_gen_pet, savedir_split / "PET",
                                             params["nr_class"], loss, metrics, lr_schedule, 200)
        history_pet["model"] = "PET"

        # TRAIN COMBI MODEL
        print("Start training COMBI model for split ", i + 1, " of ", params["nr_splits"])
        lr_schedule = schedules.PolynomialDecay(1e-5, 200*len(train_gen_combi), end_learning_rate=1e-7, power=2.0)
        model_combi = cnn_model_combi(model_mri, model_pet, nr_class=params["nr_class"], trainable=False)
        _, history_combi = train_model(model_combi, train_gen_combi, valid_gen_combi, test_gen_combi,
                                       savedir_split / "combi", params["nr_class"], loss, metrics, lr_schedule, 100)
        history_combi["model"] = "COMBI"

        # FINETUNE COMBI MODEL
        print("Start finetuning COMBI model for split ", i + 1, " of ", params["nr_splits"])
        model_combi = cnn_model_combi(model_mri, model_pet, nr_class=params["nr_class"], trainable=True)
        model_combi.load_weights(savedir_split / "combi" / "weights")
        _, history_finetune = train_model(model_combi, train_gen_combi, valid_gen_combi, test_gen_combi,
                                          savedir_split / "finetune", params["nr_class"], loss, metrics, 1e-7, 100)
        history_finetune["model"] = "FINETUNE"

        # SAVE RESULTS
        hist = pd.concat([history_mri, history_pet, history_combi, history_finetune], ignore_index=True)
        hist["split"] = i + 1
        hist_list.append(hist)

    hist = pd.concat(hist_list, ignore_index=True)
    hist.to_csv(savedir / "best_results.csv", index=False)

    return hist


def train_only_multimodal(params):
    df = pd.read_csv(params["root"] / params["df_name"])
    savedir = params["root"] / "Models" / params["dirname"]
    savedir.mkdir(exist_ok=False)

    # Save parameters to file:
    with open(savedir / "parameters.txt", "w") as f:
        f.write(str(params))

    # Split subjects:
    subjects, strat_col, sub_test_splits = get_subject_splits(df, params["nr_splits"], params["rnd"])

    # Define loss and metrics based on the number of classes
    if params["nr_class"] == 2:
        df = df[df["class"].isin(["CN", "AD"])]
        loss = "binary_crossentropy"
        metrics = ["binary_accuracy"]
    else:
        loss = "categorical_crossentropy"
        metrics = ["categorical_accuracy"]

    hist_list = []

    # Train models
    for i, sub_test in enumerate(sub_test_splits):
        tf.keras.backend.clear_session()
        # Create subdirectory for this split and save the test subjects:
        savedir_split = savedir / ("split_" + str(i + 1))
        savedir_split.mkdir(exist_ok=True)

        # Get samples for training, validation and test set:
        sub_train = np.array([sub for sub in subjects if sub not in sub_test])
        sub_train, sub_valid = train_test_split(sub_train, test_size=100, random_state=params["rnd"], stratify=strat_col[sub_train])
        np.save(savedir_split / "sub_test.npy", sub_test)
        np.save(savedir_split / "sub_valid.npy", sub_valid)
        np.save(savedir_split / "sub_train.npy", sub_train)
        df_train, df_valid, df_test = [df[df["subject"].isin(sub)] for sub in [sub_train, sub_valid, sub_test]]

        # Create data generators:
        train_gen_mri, train_gen_pet, train_gen_combi = create_generators(df_train, params["mri_shape"], params["pet_shape"], to_fit=True, batchsize=params["batchsize"], sampling="under", aug=params["augmentation"], shuffle=True)
        valid_gen_mri, valid_gen_pet, valid_gen_combi = create_generators(df_valid, params["mri_shape"], params["pet_shape"], to_fit=True, batchsize=params["batchsize"], sampling=None, shuffle=False)
        test_gen_mri, test_gen_pet, test_gen_combi = create_generators(df_test, params["mri_shape"], params["pet_shape"], to_fit=False, batchsize=params["batchsize"], sampling=None, shuffle=False)

        # TRAIN MULTI MODAL MODEL
        print("Start training COMBI model for split ", i + 1, " of ", params["nr_splits"])
        lr_schedule = schedules.PolynomialDecay(1e-4, 200*len(train_gen_combi), end_learning_rate=1e-6, power=2.0)
        model_mri = cnn_model_mri(image_shape=params["mri_shape"], nr_class=params["nr_class"], drop_ratio=params["dropout"])
        model_pet = cnn_model_pet(image_shape=params["pet_shape"], nr_class=params["nr_class"], drop_ratio=params["dropout"])
        model_combi = cnn_model_combi(model_mri, model_pet, nr_class=params["nr_class"], trainable=True)
        _, hist = train_model(model_combi, train_gen_combi, valid_gen_combi, test_gen_combi,
                              savedir_split / "combi", params["nr_class"], loss, metrics, lr_schedule, 200)
        hist["model"] = "COMBI"

        # SAVE RESULTS
        # hist = pd.concat([history_mri, history_pet, history_combi, history_finetune], ignore_index=True)
        hist["split"] = i + 1
        hist_list.append(hist)

    hist = pd.concat(hist_list, ignore_index=True)
    hist.to_csv(savedir / "best_results.csv", index=False)

    return hist


def transfer_learning_mci(params, loaddir):
    df = pd.read_csv(params["root"] / params["df_name"])
    savedir = params["root"] / "Models" / params["dirname"]
    savedir.mkdir(exist_ok=False)

    # Save parameters to file:
    with open(savedir / "parameters.txt", "w") as f:
        f.write(str(params))

    # Define loss and metrics based on the number of classes
    loss = "categorical_crossentropy"
    metrics = ["categorical_accuracy"]

    hist_list = []

    # Train models
    for i in range(5):
        tf.keras.backend.clear_session()
        # Create subdirectory for this split and save the test subjects:
        savedir_split = savedir / ("split_" + str(i + 1))
        loaddir_split = loaddir / ("split_" + str(i + 1))
        savedir_split.mkdir(exist_ok=True)

        # Load samples for training, validation and test set:
        sub_test = np.load(loaddir_split / "sub_test.npy", allow_pickle=True)
        sub_valid = np.load(loaddir_split / "sub_valid.npy", allow_pickle=True)
        sub_train = np.load(loaddir_split / "sub_train.npy", allow_pickle=True)
        df_train, df_valid, df_test = [df[df["subject"].isin(sub)] for sub in [sub_train, sub_valid, sub_test]]

        # Create data generators:
        train_gen_mri, train_gen_pet, train_gen_combi = create_generators(df_train, params["mri_shape"],
                                                                          params["pet_shape"], to_fit=True,
                                                                          batchsize=params["batchsize"],
                                                                          sampling="under", aug=params["augmentation"],
                                                                          shuffle=True)
        valid_gen_mri, valid_gen_pet, valid_gen_combi = create_generators(df_valid, params["mri_shape"],
                                                                          params["pet_shape"], to_fit=True,
                                                                          batchsize=params["batchsize"], sampling=None,
                                                                          shuffle=False)
        test_gen_mri, test_gen_pet, test_gen_combi = create_generators(df_test, params["mri_shape"],
                                                                       params["pet_shape"], to_fit=False,
                                                                       batchsize=params["batchsize"], sampling=None,
                                                                       shuffle=False)

        # LOAD MRI MODEL AND TRAIN FINAL LAYERS:
        print("Start training MRI model for split ", i + 1, " of ", params["nr_splits"])
        model_mri = cnn_model_mri(image_shape=params["mri_shape"], nr_class=2, drop_ratio=params["dropout"])
        lr_schedule = schedules.PolynomialDecay(1e-4, 200 * 100, end_learning_rate=1e-6, power=2.0)
        model_mri.compile(optimizer=Adam(learning_rate=lr_schedule), loss="binary_crossentropy", metrics=["binary_accuracy"])
        model_mri.load_weights(loaddir_split / "MRI" / "weights")
        model_mri.trainable = False
        basemodel = Model(inputs=model_mri.input, outputs=model_mri.get_layer("mri_features").output, name="mri_features")
        x = basemodel(model_mri.input, training=False)
        x = Dense(units=64, activation="relu")(x)
        predictions = Dense(params["nr_class"], activation="softmax")(x)
        model_mri = Model(inputs=model_mri.input, outputs=predictions)
        lr_schedule = schedules.PolynomialDecay(1e-4, 200 * len(train_gen_mri), end_learning_rate=1e-6, power=2.0)
        model_mri, history_mri = train_model(model_mri, train_gen_mri, valid_gen_mri, test_gen_mri,
                                             savedir_split / "MRI", params["nr_class"], loss, metrics, lr_schedule, 100)
        history_mri["model"] = "MRI"

        # FINETUNE MRI MODEL
        model_mri.trainable = True
        model_mri, history_mri_ft = train_model(model_mri, train_gen_mri, valid_gen_mri, test_gen_mri,
                                                savedir_split / "MRI_FT", params["nr_class"], loss, metrics, 1e-6, 100)
        history_mri_ft["model"] = "MRI_FT"

        # LOAD PET MODEL AND TRAIN FINAL LAYERS:
        print("Start training PET model for split ", i + 1, " of ", params["nr_splits"])
        model_pet = cnn_model_pet(image_shape=params["pet_shape"], nr_class=2, drop_ratio=params["dropout"])
        lr_schedule = schedules.PolynomialDecay(1e-4, 200 * 33, end_learning_rate=1e-6, power=2.0)
        model_pet.compile(optimizer=Adam(learning_rate=lr_schedule), loss="binary_crossentropy", metrics=["binary_accuracy"])
        model_pet.load_weights(loaddir_split / "PET" / "weights")
        model_pet.trainable = False
        basemodel = Model(inputs=model_pet.input, outputs=model_pet.get_layer("pet_features").output, name="pet_features")
        x = basemodel(model_pet.input, training=False)
        x = Dense(units=64, activation="relu")(x)
        predictions = Dense(params["nr_class"], activation="softmax")(x)
        model_pet = Model(inputs=model_pet.input, outputs=predictions)
        lr_schedule = schedules.PolynomialDecay(1e-4, 200 * len(train_gen_pet), end_learning_rate=1e-6, power=2.0)
        model_pet, history_pet = train_model(model_pet, train_gen_pet, valid_gen_pet, test_gen_pet,
                                             savedir_split / "PET", params["nr_class"], loss, metrics, lr_schedule, 100)
        history_pet["model"] = "PET"

        # FINETUNE PET MODEL
        model_pet.trainable = True
        model_pet, history_pet_ft = train_model(model_pet, train_gen_pet, valid_gen_pet, test_gen_pet,
                                                savedir_split / "PET_FT", params["nr_class"], loss, metrics, 1e-6, 100)
        history_pet_ft["model"] = "PET_FT"

        # TRAIN COMBI MODEL
        print("Start training COMBI model for split ", i + 1, " of ", params["nr_splits"])
        lr_schedule = schedules.PolynomialDecay(1e-5, 200 * len(train_gen_combi), end_learning_rate=1e-7, power=2.0)
        model_combi = cnn_model_combi(model_mri, model_pet, nr_class=params["nr_class"], trainable=False)
        _, history_combi = train_model(model_combi, train_gen_combi, valid_gen_combi, test_gen_combi,
                                       savedir_split / "combi", params["nr_class"], loss, metrics, lr_schedule, 100)
        history_combi["model"] = "COMBI"

        # FINETUNE COMBI MODEL
        print("Start finetuning COMBI model for split ", i + 1, " of ", params["nr_splits"])
        model_combi = cnn_model_combi(model_mri, model_pet, nr_class=params["nr_class"], trainable=True)
        model_combi.load_weights(savedir_split / "combi" / "weights")
        _, history_finetune = train_model(model_combi, train_gen_combi, valid_gen_combi, test_gen_combi,
                                          savedir_split / "finetune", params["nr_class"], loss, metrics, 1e-7, 100)
        history_finetune["model"] = "FINETUNE"

        # SAVE RESULTS
        hist = pd.concat([history_mri, history_mri_ft, history_pet, history_pet_ft, history_combi, history_finetune], ignore_index=True)
        hist["split"] = i + 1
        hist_list.append(hist)

    hist = pd.concat(hist_list, ignore_index=True)
    hist.to_csv(savedir / "best_results.csv", index=False)

    return hist


def finetune_multimodal(loaddir):
    with open(loaddir / "parameters.txt", "r") as f:
        params = f.read()
        params = ast.literal_eval(params.replace("WindowsPath", ""))

    df = pd.read_csv(params["root"] / params["df_name"])
    savedir = params["root"] / "Models" / params["dirname"]
    savedir.mkdir(exist_ok=False)

    # Define loss and metrics based on the number of classes
    if params["nr_class"] == 2:
        df = df[df["class"].isin(["CN", "AD"])]
        loss = "binary_crossentropy"
        metrics = ["binary_accuracy"]
    else:
        loss = "categorical_crossentropy"
        metrics = ["categorical_accuracy"]

    hist_list = []

    # Train models
    for i in range(5):
        tf.keras.backend.clear_session()
        # Create subdirectory for this split and save the test subjects:
        savedir_split = savedir / ("split_" + str(i + 1))

        # Load samples for training, validation and test set:
        sub_test = np.load(savedir_split / "sub_test.npy", allow_pickle=True)
        sub_valid = np.load(savedir_split / "sub_valid.npy", allow_pickle=True)
        sub_train = np.load(savedir_split / "sub_train.npy", allow_pickle=True)
        df_train, df_valid, df_test = [df[df["subject"].isin(sub)] for sub in [sub_train, sub_valid, sub_test]]

        # Create data generators:
        train_gen_mri, train_gen_pet, train_gen_combi = create_generators(df_train, params["mri_shape"],
                                                                          params["pet_shape"], to_fit=True,
                                                                          batchsize=params["batchsize"],
                                                                          sampling="under", aug=params["augmentation"],
                                                                          shuffle=True)
        valid_gen_mri, valid_gen_pet, valid_gen_combi = create_generators(df_valid, params["mri_shape"],
                                                                          params["pet_shape"], to_fit=True,
                                                                          batchsize=params["batchsize"], sampling=None,
                                                                          shuffle=False)
        test_gen_mri, test_gen_pet, test_gen_combi = create_generators(df_test, params["mri_shape"],
                                                                       params["pet_shape"], to_fit=False,
                                                                       batchsize=params["batchsize"], sampling=None,
                                                                       shuffle=False)

        # FINETUNE COMBI MODEL
        print("Start finetuning COMBI model for split ", i + 1, " of ", params["nr_splits"])
        model_mri = cnn_model_mri(image_shape=params["mri_shape"], nr_class=2, drop_ratio=params["dropout"])
        model_pet = cnn_model_pet(image_shape=params["pet_shape"], nr_class=2, drop_ratio=params["dropout"])
        model_combi = cnn_model_combi(model_mri, model_pet, nr_class=params["nr_class"], trainable=True)
        lr_schedule = schedules.PolynomialDecay(1e-5, 200 * len(train_gen_combi), end_learning_rate=1e-7, power=2.0)
        model_combi.compile(optimizer=Adam(learning_rate=lr_schedule), loss=loss, metrics=metrics)
        model_combi.load_weights(savedir_split / "combi" / "weights")
        _, hist = train_model(model_combi, train_gen_combi, valid_gen_combi, test_gen_combi,
                              savedir_split / "finetune_2", params["nr_class"], loss, metrics, 1e-7, 100)
        hist["model"] = "FINETUNE_2"

        # SAVE RESULTS
        hist["split"] = i + 1
        hist_list.append(hist)

    hist = pd.concat(hist_list, ignore_index=True)
    hist_old = pd.read_csv(savedir / "best_results.csv")
    hist_old.to_csv(savedir / "best_results_old.csv", index=False)
    hist = pd.concat([hist_old, hist])
    hist.to_csv(savedir / "best_results.csv", index=False)

    return hist
