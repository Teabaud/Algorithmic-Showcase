import numpy as np
import numpy.typing as npt


def sigmoid(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return sigmoid(x) * (1 - sigmoid(x))
