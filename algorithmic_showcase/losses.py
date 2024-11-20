from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class LossFunction(ABC):
    @abstractmethod
    def __call__(
        self, y_pred: npt.NDArray[np.float64], y_true: npt.NDArray[np.float64]
    ) -> float:
        """
        Compute loss between predictions and true values

        Parameters:
        -----------
        y_pred : array of shape (n_samples,)
            Predicted values
        y_true : array of shape (n_samples,)
            True target values

        Returns:
        --------
        loss : float
            Loss value
        """
        ...

    @abstractmethod
    def gradient(
        self,
        y_pred: npt.NDArray[np.float64],
        y_true: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Compute gradient of loss with respect to predictions

        Parameters:
        -----------
        y_pred : array of shape (n_samples,)
            Predicted values
        y_true : array of shape (n_samples,)
            True target values

        Returns:
        --------
        gradient : array of shape (n_features,)
            Gradient with respect to predictions
        """
        ...


class MSELoss(LossFunction):
    def __call__(
        self, y_pred: npt.NDArray[np.float64], y_true: npt.NDArray[np.float64]
    ) -> float:
        return np.square(y_true - y_pred).mean()

    def gradient(
        self, y_pred: npt.NDArray[np.float64], y_true: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return 2 * (y_pred - y_true) / len(y_true)


class BinaryCrossEntropyLoss(LossFunction):
    def __call__(
        self, y_pred: npt.NDArray[np.float64], y_true: npt.NDArray[np.float64]
    ) -> float:
        np.clip(y_pred, 1e-15, 1 - 1e-15, out=y_pred)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def gradient(
        self, y_pred: npt.NDArray[np.float64], y_true: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        np.clip(y_pred, 1e-15, 1 - 1e-15, out=y_pred)
        return -1 / (1 - y_pred) * (y_true / y_pred - 1) / len(y_true)
