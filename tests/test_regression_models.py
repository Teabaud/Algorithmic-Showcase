from typing import Tuple

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from algorithmic_showcase.functions import sigmoid_derivative
from algorithmic_showcase.regression_models import LinearRegression, LogisticRegression

regression_true_params = np.array([0.7, -0.3, 0.5])


@pytest.fixture
def regression_data() -> Tuple[np.ndarray, np.ndarray]:
    """Generate simple regression data"""
    np.random.seed(42)
    x = np.random.randn(100, 3)
    true_params = regression_true_params
    y = x @ true_params + np.random.randn(100) * 0.1
    return x, y


classification_true_params = np.array([10.0, -1.0])


@pytest.fixture
def classification_data() -> Tuple[np.ndarray, np.ndarray]:
    """Generate simple binary classification data"""
    np.random.seed(42)
    x = np.random.randn(100, 2)
    true_params = classification_true_params
    logits = x @ true_params
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(float)
    return x, y


class TestLinearRegression:
    def test_linear_regression_gradient(
        self, regression_data: Tuple[np.ndarray, np.ndarray]
    ):
        """Test linear regression gradient computation"""
        x, _ = regression_data
        model = LinearRegression(np.zeros(x.shape[1]))
        gradient = model.gradient(x)
        # Analytical solution for linear regression gradient
        expected_gradient = x
        assert_array_almost_equal(gradient, expected_gradient)


class TestLogisticRegression:
    def test_logistic_regression_gradient(
        self, classification_data: Tuple[np.ndarray, np.ndarray]
    ):
        """Test logistic regression gradient computation"""
        x, _ = classification_data
        model = LogisticRegression(np.zeros(x.shape[1]))
        gradient = model.gradient(x)
        # Analytical solution for logistic regression gradient
        logits = x @ model.params
        expected_gradient = x * sigmoid_derivative(logits).reshape(-1, 1)
        assert_array_almost_equal(gradient, expected_gradient)
