import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as SklearnDTC

from algorithmic_showcase.decision_tree import DecisionTree, SplitCriterion


@pytest.fixture
def classification_data():
    x, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42
    )
    return x, y


@pytest.fixture
def regression_data():
    x, y = make_regression(
        n_samples=1000, n_features=20, n_informative=15, random_state=42
    )
    return x, y


class TestDecisionTree:
    def test_gini_impurity_calculation(self):
        tree = DecisionTree(criterion=SplitCriterion.GINI)

        y = np.array([0, 0, 1, 1])
        assert np.isclose(tree._calculate_impurity(y), 0.5)

        y = np.array([0, 0, 0, 0])
        assert np.isclose(tree._calculate_impurity(y), 0.0)

    def test_entropy_impurity_calculation(self):
        tree = DecisionTree(criterion=SplitCriterion.ENTROPY)

        y = np.array([0, 0, 1, 1])
        assert np.isclose(tree._calculate_impurity(y), 1.0)

        y = np.array([0, 0, 0, 0])
        assert np.isclose(tree._calculate_impurity(y), 0.0)

    def test_mse_impurity_calculation(self):
        tree = DecisionTree(criterion=SplitCriterion.MSE)

        y = np.array([1.0, 2.0, 3.0, 4.0])
        expected_mse = np.var(y)
        assert np.isclose(tree._calculate_impurity(y), expected_mse)

    def test_split_gain_classification(self, classification_data):
        _, y = classification_data
        tree = DecisionTree(criterion=SplitCriterion.GINI)

        # Test with a known good split
        parent = y[:10]
        left = y[:5]
        right = y[5:10]
        gain = tree._calculate_split_gain(parent, left, right)
        assert gain >= 0

        # Test with a perfectly pure split
        gain = tree._calculate_split_gain(
            np.array([0, 0, 1, 1]), np.array([0, 0]), np.array([1, 1])
        )
        assert np.isclose(gain, 0.5)

    def test_find_best_split(self, classification_data):
        x, y = classification_data
        tree = DecisionTree(criterion=SplitCriterion.GINI)

        # Test with small subset
        x_small = x[:10]
        y_small = y[:10]
        split_result = tree._find_best_split(x_small, y_small)

        assert split_result is not None
        assert 0 <= split_result.feature_idx < x.shape[1]
        assert split_result.gain >= 0

    def test_classification_performance(self, classification_data):
        x, y = classification_data

        # Our implementation
        tree = DecisionTree(
            criterion=SplitCriterion.GINI,
            max_depth=5,
        )
        tree.fit(x, y)
        y_pred = tree.predict(x)

        # Sklearn implementation for comparison
        sklearn_tree = SklearnDTC(max_depth=5, random_state=42)
        sklearn_tree.fit(x, y)
        y_pred_sklearn = sklearn_tree.predict(x)

        # Our implementation should be reasonably close to sklearn
        acc_yours = accuracy_score(y, y_pred)
        acc_sklearn = accuracy_score(y, y_pred_sklearn)
        assert acc_yours >= 0.8 * acc_sklearn

    def test_regression_performance(self, regression_data):
        x, y = regression_data

        tree = DecisionTree(
            criterion=SplitCriterion.MSE,
            max_depth=5,
        )
        tree.fit(x, y)
        y_pred = tree.predict(x)

        r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        assert r2 >= 0.5  # Should explain at least 50% of variance

    def test_edge_cases(self):
        tree = DecisionTree()

        # Empty input
        with pytest.raises(ValueError):
            tree.fit(np.array([]), np.array([]))

        # Single sample
        x = np.array([[1.0]])
        y = np.array([1])
        tree.fit(x, y)
        assert tree.predict(x)[0] == y[0]

        # All same values
        x = np.ones((10, 1))
        y = np.ones(10, dtype=int)
        tree.fit(x, y)
        assert np.all(tree.predict(x) == y)
