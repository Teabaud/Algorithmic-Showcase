import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from algorithmic_showcase.losses import BinaryCrossEntropyLoss, MSELoss


class TestMSELoss:
    def test_mse_loss_zero(self):
        """Test MSE loss when predictions = targets"""
        mse = MSELoss()
        y_true = np.array([1.0, 2.0, 3.0])
        loss = mse(y_true, y_true)
        assert_almost_equal(loss, 0.0)

    def test_mse_loss_known(self):
        """Test MSE loss for known values"""
        mse = MSELoss()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 4.0])
        # ((1-2)^2 + (2-2)^2 + (3-4)^2) / 3 = (1 + 0 + 1) / 3 = 2/3
        expected_loss = 2 / 3
        loss = mse(y_pred=y_pred, y_true=y_true)
        assert_almost_equal(loss, expected_loss)

    def test_mse_gradient(self):
        """Test MSE gradient computation"""
        mse = MSELoss()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 4.0])
        # Gradient is 2(y_pred - y_true) / n
        expected_gradient = np.array([2.0, 0.0, 2.0]) / 3
        gradient = mse.gradient(y_true=y_true, y_pred=y_pred)
        assert_array_almost_equal(gradient, expected_gradient)


class TestBinaryCrossEntropyLoss:
    def test_logistic_loss_perfect(self):
        """Test logistic loss for perfect predictions"""
        bce_loss = BinaryCrossEntropyLoss()
        y_true = np.array([0.0, 1.0])
        # Use large negative/positive values for clear 0/1 predictions
        y_pred = np.array([1e-10, 1 - 1e-10])
        loss = bce_loss(y_true=y_true, y_pred=y_pred)
        assert loss < 1e-4  # Should be very close to zero

    def test_logistic_loss_known(self):
        """Test logistic loss for known values"""
        bce_loss = BinaryCrossEntropyLoss()
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.5, 0.5])  # predicting 0.5 for both
        # -[0*log(0.5) + 1*log(0.5)] = log(2)
        expected_loss = np.log(2)
        loss = bce_loss(y_true=y_true, y_pred=y_pred)
        assert_almost_equal(loss, expected_loss)

    def test_logistic_gradient(self):
        """Test logistic loss gradient computation"""
        bce_loss = BinaryCrossEntropyLoss()
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.5, 0.5])
        # Gradient is 1 / (1 - y_pred) * (y_true / y_pred - 1) / len(y_true)
        expected_gradient = np.array([1.0, -1.0])
        gradient = bce_loss.gradient(y_true=y_true, y_pred=y_pred)
        assert_array_almost_equal(gradient, expected_gradient)

    def test_numerical_stability(self):
        """Test numerical stability for extreme predictions"""
        bce_loss = BinaryCrossEntropyLoss()
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.0, 1.0])
        # Should not raise any numerical warnings/errors
        loss = bce_loss(y_true=y_true, y_pred=y_pred)
        assert np.isfinite(loss)
        gradient = bce_loss.gradient(y_true=y_true, y_pred=y_pred)
        assert np.all(np.isfinite(gradient))
