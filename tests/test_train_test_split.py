import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from algorithmic_showcase.train_test_split import split_train_test_data


class TestTrainTestSplit:
    def test_split_proportions(self):
        """Test that split proportions are correct"""
        x = np.random.randn(100, 3)
        y = np.random.randn(100)
        test_size = 0.2

        split = split_train_test_data(x, y, test_size=test_size)

        assert split.x_train.shape[0] == 80
        assert split.x_test.shape[0] == 20
        assert split.y_train.shape[0] == 80
        assert split.y_test.shape[0] == 20

    def test_data_alignment(self):
        """Test that x and y remain aligned after splitting"""
        x = np.arange(50).reshape(10, 5)
        y = np.arange(10)

        split = split_train_test_data(x, y, test_size=0.3, random_seed=42)

        # Create a mapping of rows from x to corresponding y values
        orig_pairs = {tuple(row): target for row, target in zip(x, y)}

        # Check alignment in train set
        for x_row, y_val in zip(split.x_train, split.y_train):
            assert orig_pairs[tuple(x_row)] == y_val

        # Check alignment in test set
        for x_row, y_val in zip(split.x_test, split.y_test):
            assert orig_pairs[tuple(x_row)] == y_val

    def test_random_seed(self):
        """Test that random seed produces reproducible splits"""
        x = np.random.randn(100, 3)
        y = np.random.randn(100)

        split1 = split_train_test_data(x, y, test_size=0.2, random_seed=42)
        split2 = split_train_test_data(x, y, test_size=0.2, random_seed=42)

        assert_array_almost_equal(split1.x_train, split2.x_train)
        assert_array_almost_equal(split1.x_test, split2.x_test)
        assert_array_almost_equal(split1.y_train, split2.y_train)
        assert_array_almost_equal(split1.y_test, split2.y_test)

    def test_edge_cases(self):
        """Test edge cases and invalid inputs"""
        x = np.random.randn(10, 3)
        y = np.random.randn(10)

        # Test invalid test_size
        with pytest.raises(ValueError):
            split_train_test_data(x, y, test_size=1.5)

        with pytest.raises(ValueError):
            split_train_test_data(x, y, test_size=-0.1)

        # Test empty arrays
        with pytest.raises(ValueError):
            split_train_test_data(np.array([]), np.array([]))

        # Test mismatched dimensions
        with pytest.raises(ValueError):
            split_train_test_data(x, np.random.randn(11))  # y has wrong length
