from abc import ABC, abstractmethod
from typing import List

import numpy as np
import numpy.typing as npt

from algorithmic_showcase.losses import LossFunction, MSELoss
from algorithmic_showcase.regression_models import LinearRegression, RegressionModel


class Optimizer(ABC):
    @abstractmethod
    def optimize(
        self,
        model: RegressionModel,
        loss_fn: LossFunction,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> List[float]:
        """
        Optimize model parameters to minimize loss function

        Parameters:
        -----------
        model : RegressionModel
            Model to optimize, parameters are updated in place
        loss_fn : LossFunction
            Loss function to minimize
        x : array of shape (n_samples, n_features)
            Input features
        y : array of shape (n_samples,)
            Target values

        Returns:
        --------
        loss_history : list of floats
            Loss value at each iteration
        """
        ...


class LinearRegressionMSEExactSolver(Optimizer):
    def optimize(
        self,
        model: LinearRegression,
        loss_fn: MSELoss,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> List[float]:
        """
        Fit linear regression model using the MSE analytical solution

        Parameters:
        -----------
        model : LinearRegression
            Model to optimize, parameters are updated in place
        loss_fn : MSELoss
            Loss function to minimize.
        x : array of shape (n_samples, n_features)
            Input features
        y : array of shape (n_samples,)
            Target values

        Returns:
        --------
        loss_history : list of floats
            Loss value at each iteration (only one iteration for exact solver)
        """

        model.params = np.linalg.solve(x.T @ x, x.T @ y)

        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        return [loss]


class GradientDescent(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def optimize(
        self,
        model: RegressionModel,
        loss_fn: LossFunction,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> List[float]:
        y_true = y
        loss_history: List[float] = []

        for _ in range(self.max_iterations):
            y_pred = model(x)
            loss_history.append(loss_fn(y_pred, y_true))

            if len(loss_history) > 1:
                if abs(loss_history[-1] - loss_history[-2]) < self.tolerance:
                    break

            gradient = model.gradient(x).T @ loss_fn.gradient(y_pred, y_true)
            # print(f"{gradient=}, {model.params=}, {loss_history[-1]=}")
            model.params -= self.learning_rate * gradient

        y_pred = model(x)
        loss_history.append(loss_fn(y_pred, y_true))

        return loss_history
