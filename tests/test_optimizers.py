from typing import Tuple

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from algorithmic_showcase.functions import sigmoid
from algorithmic_showcase.losses import BinaryCrossEntropyLoss, MSELoss
from algorithmic_showcase.optimizers import (
    GradientDescent,
    LinearRegressionMSEExactSolver,
)
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
    probs = sigmoid(x @ true_params)
    y = (probs > 0.5).astype(float)
    return x, y


class TestLinearRegressionMSEExactSolver:
    def test_optimization_mse(self, regression_data: Tuple[np.ndarray, np.ndarray]):
        """Test linear regression with MSE analytical solution"""
        x, y = regression_data
        model = LinearRegression(np.zeros(x.shape[1]))
        loss_fn = MSELoss()
        optimizer = LinearRegressionMSEExactSolver()

        loss_history = optimizer.optimize(model, loss_fn, x, y)

        # Check that loss decreases
        assert len(loss_history) == 1  # Only one iteration

        # Check that parameters give reasonable predictions
        final_predictions = model(x)
        final_loss = loss_fn(y, final_predictions)
        assert final_loss < 0.01  # Reasonable threshold for this dataset

        # check that the final parameters are close to the true parameters
        assert_array_almost_equal(model.params, regression_true_params, decimal=2)


class TestGradientDescent:
    def test_optimization_mse(self, regression_data: Tuple[np.ndarray, np.ndarray]):
        """Test gradient descent with MSE loss"""
        x, y = regression_data
        gd = GradientDescent(learning_rate=0.01, max_iterations=1000)
        loss_fn = MSELoss()
        initial_params = np.zeros(x.shape[1])
        model = LinearRegression(initial_params)

        loss_history = gd.optimize(model, loss_fn, x, y)

        # Check that loss decreases
        assert loss_history[-1] < loss_history[0]

        # Check that parameters give reasonable predictions
        final_predictions = model(x)
        final_loss = loss_fn(y, final_predictions)
        assert final_loss < 1.0  # Reasonable threshold for this dataset

        # Check that the final parameters are close to the true parameters
        assert_array_almost_equal(model.params, regression_true_params, decimal=1)

    def test_optimization_logistic(
        self, classification_data: Tuple[np.ndarray, np.ndarray]
    ):
        """Test gradient descent with logistic classification"""
        x, y = classification_data
        gd = GradientDescent(learning_rate=0.5, max_iterations=1000)
        loss_fn = BinaryCrossEntropyLoss()
        initial_params = np.zeros(x.shape[1])
        model = LogisticRegression(initial_params)

        loss_history = gd.optimize(model, loss_fn, x, y)

        # Check that loss decreases
        assert loss_history[-1] < loss_history[0]

        # Check classification accuracy
        final_predictions = model(x)
        predicted_classes = (final_predictions > 0.5).astype(float)
        accuracy = np.mean(predicted_classes == y)
        assert accuracy > 0.95  # Should achieve reasonable accuracy
