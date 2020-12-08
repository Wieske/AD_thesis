"""
Script that contains the DataGenerator and data augmentation functions
"""

from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform


def rnd_trans_matrix(img_shape, angle=0.0, shift=0.0, scale=0.0, around_center=True):
    """
    Create a random transformation matrix for affine transformation of 3D images
    :param img_shape: size of the image (to determine translation)
    :param angle: maximum angle (in radians) to use for rotation
    :param shift: maximum shift (in percentage) to use for translation
    :param scale: maximum scale (in percentage) to use for scaling
    :param around_center: whether to rotate around the center of the image (default: True)
    :return: A 3 x 4 transformation matrix
    """
    # Get random angles for each axis in the range [-angle, angle]:
    a = np.random.uniform(- angle, angle, size=3)
    # Compute rotation matrix:
    rx = np.array([[1, 0, 0],
                   [0, np.cos(a[0]), - np.sin(a[0])],
                   [0, np.sin(a[0]), np.cos(a[0])]])
    ry = np.array([[np.cos(a[1]), 0, np.sin(a[1])],
                   [0, 1, 0],
                   [- np.sin(a[1]), 0, np.cos(a[1])]])
    rz = np.array([[np.cos(a[2]), - np.sin(a[2]), 0],
                   [np.sin(a[2]), np.cos(a[2]), 0],
                   [0, 0, 1]])
    rot_mat = np.dot(np.dot(rz, ry), rx)

    # Get random scaling factors for each axis in the range [1 - scale, 1 + scale]:
    s = np.random.uniform(1 - scale, 1 + scale, size=3)
    # Compute scaling matrix:
    sc_mat = np.diag(s)

    # Compute translation:
    t = np.random.uniform(- shift, shift, size=3) * img_shape

    center_mat = np.eye(4)
    if around_center:
        # Perform a translation that shift the center of the array to the origin
        center = np.array([0.5, 0.5, 0.5]) * img_shape
        t = t + center
        center_mat[:3, -1] = - center

    trans_matrix = np.zeros((3, 4))
    trans_matrix[:, :-1] = np.dot(rot_mat, sc_mat)
    trans_matrix[:, -1] = t

    return np.dot(trans_matrix, center_mat)


class NiftiDataGenerator(Sequence):
    """Sequence based data generator to load batches of 3D MRI and/or PET images to train a keras model.
    """
    def __init__(self, df, to_fit=True, batch_size=4, mri_shape=None, pet_shape=None, modality="MRI", n_channels=1, sampling=None, aug=0, shuffle=True):
        """Initialization
        :param df: dataframe with columns "filepath" and "class"
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param mri_shape: tuple indicating image dimension of mri
        :param pet_shape: tuple indicating image dimension of pet
        :param modality: one of "MRI", "PET" or "combi" that indicates which modalities are used
        :param n_channels: number of image channels
        :param sampling: use "under" for under sampling and "over" for over sampling to create balanced classes
        :param aug: Parameter for on-the-fly data augmentation (0 = no augmentation, 0.05 = ~ 5 percent augmentations)
        :param shuffle: True to shuffle label indexes after every epoch
        """
        # Reset index such that indices of dataframe match indices of lists:
        df.index = np.arange(len(df.index))

        # Initialize parameters:
        self.df = df
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.mri_shape = mri_shape
        self.pet_shape = pet_shape
        self.modality = modality
        self.n_channels = n_channels
        self.sampling = sampling
        self.aug = aug
        self.shuffle = shuffle

        # Convert classes to correct format:
        classes = sorted(set(df["class"].to_list()))
        self.num_class = len(classes)
        # self.class_indices = dict(zip(classes, range(self.num_class)))
        if self.num_class == 2:
            # convert classes to binary format
            self.class_indices = {'CN': 0, 'AD': 1}
            self.classes = [self.class_indices[label] for label in df["class"]]
        else:
            # convert classes to categorical format
            self.class_indices = {'CN': 0, 'MCI': 1, 'AD': 2}
            self.classes = to_categorical([self.class_indices[label] for label in df["class"]])

        # Get indices for first epoch:
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indices of the batch
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        if self.modality == "MRI":
            X = self._generate_X(indices, self.mri_shape, "filepath")
        elif self.modality == "PET":
            X = self._generate_X(indices, self.pet_shape, "filepath")
        elif self.modality == "combi":
            X_mri = self._generate_X(indices, self.mri_shape, "filepath_mri")
            X_pet = self._generate_X(indices, self.pet_shape, "filepath_pet")
            X = [X_mri, X_pet]
        else:
            print("modality should be MRI, PET or combi")
            X = None

        # Return batch
        if self.to_fit:
            y = self._generate_y(indices)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch and optionally performs over or under sampling
        """
        if self.to_fit and self.sampling == "over":
            class_counts = self.df["class"].value_counts()
            nr_per_class = class_counts.max()
            self.indices = []
            for c in class_counts.index:
                if class_counts[c] < nr_per_class:
                    class_idx = self.df[self.df["class"] == c].index
                    self.indices.extend(np.random.choice(class_idx, nr_per_class))
                else:
                    self.indices.extend(self.df[self.df["class"] == c].index)
            np.random.shuffle(self.indices)

        elif self.to_fit and self.sampling == "under":
            class_counts = self.df["class"].value_counts()
            nr_per_class = class_counts.min()
            self.indices = []
            for c in class_counts.index:
                if class_counts[c] > nr_per_class:
                    class_idx = self.df[self.df["class"] == c].index
                    self.indices.extend(np.random.choice(class_idx, nr_per_class, replace=False))
                else:
                    self.indices.extend(self.df[self.df["class"] == c].index)
            np.random.shuffle(self.indices)

        else:
            self.indices = self.df.index.values
            if self.shuffle:
                np.random.shuffle(self.indices)

    def _generate_X(self, idxs, img_shape, col_name):
        """Generates data containing batch_size images
        :param idxs: list of batch indices
        :param img_shape: shape of the images
        :param col_name: name of the column in df that contains the paths to the images
        :return: batch of images
        """
        # initialize
        X = np.empty((len(idxs), *img_shape, self.n_channels), dtype="float32")
        for i, idx in enumerate(idxs):
            # Store sample
            img = nib.load(self.df[col_name][idx])
            data = img.get_fdata()
            data_norm = (data - data.min()) / (data.max() - data.min())
            if self.aug > 0:
                trans_matrix = rnd_trans_matrix(img_shape, angle=self.aug, shift=self.aug, scale=self.aug)
                data_norm = affine_transform(data_norm, trans_matrix, order=2)
            X[i, ] = data_norm.reshape(*img_shape, self.n_channels)
        return X

    def _generate_y(self, idxs):
        """Generates data containing batch_size labels
        :param idxs: list of batch indices
        :return: batch of labels
        """
        # Initialization
        if self.num_class == 2:
            y = np.empty(len(idxs), dtype=int)
        else:
            y = np.empty((len(idxs), self.num_class), dtype=int)
        # Generate data
        for i, idx in enumerate(idxs):
            # Store sample
            y[i] = self.classes[idx]
        return y
