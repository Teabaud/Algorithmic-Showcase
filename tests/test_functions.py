from numpy.testing import assert_almost_equal

from algorithmic_showcase.functions import sigmoid, sigmoid_derivative


class TestSigmoid:
    def test_sigmoid_zero(self):
        """Test sigmoid function at zero"""
        assert sigmoid(0) == 0.5

    def test_sigmoid_one(self):
        """Test sigmoid function at one"""
        assert_almost_equal(sigmoid(1), 0.7310585786300049)

    def test_sigmoid_minus_one(self):
        """Test sigmoid function at minus one"""
        assert_almost_equal(sigmoid(-1), 0.2689414213699951)

    def test_sigmoid_known(self):
        """Test sigmoid function for known values"""
        assert_almost_equal(sigmoid(-100), 0.0)
        assert_almost_equal(sigmoid(100), 1.0)


class TestSigmoidDerivative:
    def test_sigmoid_derivative_zero(self):
        """Test sigmoid derivative at zero"""
        assert sigmoid_derivative(0) == 0.25

    def test_sigmoid_derivative_one(self):
        """Test sigmoid derivative at one"""
        assert_almost_equal(sigmoid_derivative(1), 0.19661193324148185)

    def test_sigmoid_derivative_minus_one(self):
        """Test sigmoid derivative at minus one"""
        assert_almost_equal(sigmoid_derivative(-1), 0.19661193324148185)

    def test_sigmoid_derivative_known(self):
        """Test sigmoid derivative for known values"""
        assert_almost_equal(sigmoid_derivative(-100), 0.0)
        assert_almost_equal(sigmoid_derivative(100), 0.0)
