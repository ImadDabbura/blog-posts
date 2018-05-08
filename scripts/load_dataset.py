from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_dataset_catvsdog(path, images_extension='jpg'):
    """
    Loads dataset as numpy objects. It will import only images with ".jpg"
    extension.
    The data will be shuffled to avoid pre-existing ordering.

    Arguments
    ---------
    path : str
        absolute/relative path for the directory of the images.

    Returns
    -------
    X : array
        input data, shape: number of features x number of examples.
    Y : array
        label vector, shape: 1 x number of examples.
    """
    # Set up the path
    path = Path(path)
    
    # Get images that have images_extensions and number of images
    images = [str(fname) for fname in list(path.glob(f'*.{images_extension}'))]
    m = len(images)
    
    # Read the images into numpy arrays
    count = 0
    for img in images:
        x = plt.imread(img).astype('float')
        if count == 0:
            X = x
        else:
            X = np.concatenate([X, x])
        count += 1

    # Derive true label vector
    Y = np.zeros((1, m))
    for i, img in enumerate(path.glob(f'*.{images_extension}')):
        if img.stem.startswith("cat"):
            Y[:, i] = 1
        else:
            Y[:, i] = 0

    # Reshape input X and Shuffle the dataset
    X = X.reshape(m, -1).T
    permutation = np.random.permutation(m)
    X = X[:, permutation]
    Y = Y[:, permutation]

    return X, Y
