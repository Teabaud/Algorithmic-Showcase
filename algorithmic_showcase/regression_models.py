from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from algorithmic_showcase.functions import sigmoid, sigmoid_derivative


class RegressionModel(ABC):
    def __init__(self, params: npt.NDArray[np.float64]) -> None:
        self.params: npt.NDArray[np.float64] = params

    @abstractmethod
    def __call__(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

    @abstractmethod
    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Compute gradient of model with respect to the model parameters

        Parameters:
        -----------
        x : array of shape (n_samples, n_features)
            Input features

        Returns:
        --------
        gradient : array of shape (n_samples, n_features)
            Gradient with respect to the model parameters
        """
        ...


class LinearRegression(RegressionModel):
    def __call__(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return x @ self.params

    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return x


class LogisticRegression(RegressionModel):
    def __call__(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return sigmoid(x @ self.params)

    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return x * sigmoid_derivative(x @ self.params).reshape(-1, 1)
