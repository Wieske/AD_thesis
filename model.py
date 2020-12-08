"""
Script that contains functions to create the model architectures
"""

from tensorflow.keras.layers import Input, Dense, Flatten, concatenate, GlobalAveragePooling3D
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model


def cnn_model_mri(image_shape=(129, 153, 129), nr_class=3, drop_ratio=0):
    """
    Create the MRI model with 6 CNN layers with 16-512 filters and global averagepooling
    :param image_shape: tuple indicating the shape of the MRI images
    :param nr_class: number of classes for the classification task
    :param drop_ratio: Dropout ratio used after the features are flattened
    :return: the MRI model
    """
    inputs = Input(shape=image_shape + (1,))

    nr_filters = [16, 32, 64, 128, 256]

    x = inputs
    for filters in nr_filters:
        x = Conv3D(filters=filters, kernel_size=(3, 3, 3), padding="same", kernel_initializer="he_uniform")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(x)

    x = Conv3D(filters=512, kernel_size=(3, 3, 3), padding="same", kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    dense_1 = GlobalAveragePooling3D(name="mri_features")(x)
    dense_2 = Dense(units=64, activation="relu")(dense_1)

    if nr_class == 2:
        predictions = Dense(1, activation="sigmoid", name="mri_predictions")(dense_2)
    else:
        predictions = Dense(nr_class, activation="softmax", name="mri_predictions")(dense_2)

    return Model(inputs=inputs, outputs=predictions)


def cnn_model_pet(image_shape=(160, 160, 96), nr_class=3, drop_ratio=0):
    """
    Create the PET model with 5 CNN layers with 8-128 filters
    :param image_shape: tuple indicating the shape of the PET images
    :param nr_class: number of classes for the classification task
    :param drop_ratio: Dropout ratio used after the features are flattened
    :return: the PET model
    """
    inputs = Input(shape=image_shape + (1,))

    nr_filters = [8, 16, 32, 64, 128]

    x = inputs
    for filters in nr_filters:
        x = Conv3D(filters=filters, kernel_size=(3, 3, 3), padding="same", kernel_initializer="he_uniform")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(x)

    flatten_layer = Flatten()(x)
    if drop_ratio > 0:
        flatten_layer = Dropout(drop_ratio)(flatten_layer)

    dense_1 = Dense(units=512, activation="relu", name="pet_features")(flatten_layer)
    dense_2 = Dense(units=64, activation="relu")(dense_1)

    if nr_class == 2:
        predictions = Dense(1, activation="sigmoid", name="pet_predictions")(dense_2)
    else:
        predictions = Dense(nr_class, activation="softmax", name="pet_predictions")(dense_2)

    return Model(inputs=inputs, outputs=predictions)


def cnn_model_combi(mri_model, pet_model, nr_class=3, trainable=False):
    """
    Create the multi-modal model
    :param mri_model: the (trained) MRI model
    :param pet_model: the (trained) PET model
    :param nr_class: number of classes for the classification task
    :param trainable: booleon that indicates if the weights in the MRI and PET model should be trainable
    :return: multi-modal model
    """
    mri_model.trainable = trainable
    pet_model.trainable = trainable

    mri_out = mri_model.get_layer("mri_features").output
    pet_out = pet_model.get_layer("pet_features").output
    combi = concatenate([mri_out, pet_out])

    basemodel = Model(inputs=[mri_model.input, pet_model.input], outputs=combi)

    x = basemodel([mri_model.input, pet_model.input], training=False)
    # Training=False is used to keep the batch normalization layers in inference mode
    x = Dense(units=128, activation="relu")(x)

    if nr_class == 2:
        predictions = Dense(1, activation="sigmoid")(x)
    else:
        predictions = Dense(nr_class, activation="softmax")(x)

    return Model(inputs=[mri_model.input, pet_model.input], outputs=predictions)
