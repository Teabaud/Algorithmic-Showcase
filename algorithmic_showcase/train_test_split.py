from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt


@dataclass
class TrainTestData:
    x_train: npt.NDArray[np.float64]
    x_test: npt.NDArray[np.float64]
    y_train: npt.NDArray[np.float64]
    y_test: npt.NDArray[np.float64]


def split_train_test_data(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    test_size: float = 0.2,
    random_seed: Optional[int] = None,
) -> TrainTestData:
    """
    Create a train-test split of the data
    1. Shuffle the data
    2. Maintain alignment between x and y
    3. Handle edge cases (empty arrays, invalid test_size)

    Parameters:
    -----------
    x : array of shape (n_samples, n_features)
        Input features
    y : array of shape (n_samples,)
        Target values
    test_size : float, default=0.2
        Proportion of samples to include in test set
    random_seed : int, default=None
        Random seed for reproducibility

    Returns:
    --------
    split : TrainTestData
        Data split object containing train and test data
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples")

    if x.shape[0] == 0:
        raise ValueError("x and y must have at least one sample")

    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1")

    if random_seed is not None:
        np.random.seed(random_seed)

    indices = np.random.permutation(x.shape[0])
    split_index = int(x.shape[0] * (1 - test_size))

    x_train = x[indices[:split_index]]
    x_test = x[indices[split_index:]]
    y_train = y[indices[:split_index]]
    y_test = y[indices[split_index:]]

    return TrainTestData(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
