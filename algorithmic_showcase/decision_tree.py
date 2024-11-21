from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt


class SplitCriterion(Enum):
    GINI = "gini"
    ENTROPY = "entropy"
    MSE = "mse"  # for regression trees


@dataclass
class Node:
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    value: Optional[Union[float, npt.NDArray[np.float64]]] = None

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class SplitResult(NamedTuple):
    feature_idx: int
    threshold: float
    left_indices: npt.NDArray[np.int64]
    right_indices: npt.NDArray[np.int64]
    gain: float


class DecisionTree:
    """
    Parameters:
    max_depth (Optional[int]): The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    min_samples_split (int): The minimum number of samples required to split an internal node.
    min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
    criterion (SplitCriterion): The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: SplitCriterion = SplitCriterion.GINI,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root: Optional[Node] = None

    def fit(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> "DecisionTree":
        """
        Fit the decision tree model according to the given training data.

        Parameters
        ----------
        x : npt.NDArray[np.float64]
            Training data of shape (n_samples, n_features).
        y : npt.NDArray[np.float64]
            Target values of shape (n_samples,).

        Returns
        -------
        DecisionTree
            Fitted decision tree instance.

        Raises
        ------
        ValueError
            If the input data `x` or `y` is empty.

        Notes
        -----
        This method handles both classification and regression tasks based on the specified criterion.
        """
        self.n_classes_ = (
            len(np.unique(y)) if self.criterion != SplitCriterion.MSE else None
        )

        if self.max_depth is None:
            self.max_depth = np.inf

        if (len(x) == 0) or (len(y) == 0):
            raise ValueError("Empty data")

        self.root = self._grow_tree(x, y, depth=0)
        return self

    def _grow_tree(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], depth: int
    ) -> Node:
        """
        Recursively grows the decision tree by finding the best split at each node.

        Parameters:
        -----------
        x : npt.NDArray[np.float64]
            The feature matrix of shape (n_samples, n_features).
        y : npt.NDArray[np.float64]
            The target values of shape (n_samples,).
        depth : int
            The current depth of the tree.

        Returns:
        --------
        Node
            The root node of the grown subtree.

        Notes:
        ------
        This method handles the following stopping criteria:
        1. Maximum depth of the tree.
        2. Minimum number of samples required to split a node.
        3. Pure node (all samples have the same target value).
        """
        if depth == self.max_depth:
            return self._grow_leaf(y)

        if len(y) < self.min_samples_split:
            return self._grow_leaf(y)

        if self._calculate_impurity(y) == 1:
            return self._grow_leaf(y)

        split_result = self._find_best_split(x, y)

        if split_result is None:
            return self._grow_leaf(y)

        left = self._grow_tree(
            x[split_result.left_indices], y[split_result.left_indices], depth + 1
        )
        right = self._grow_tree(
            x[split_result.right_indices], y[split_result.right_indices], depth + 1
        )

        return Node(
            feature_index=split_result.feature_idx,
            threshold=split_result.threshold,
            left=left,
            right=right,
        )

    def _grow_leaf(self, y: npt.NDArray[np.float64]) -> Node:
        """
        Creates a leaf node with the majority class or the mean target value.

        Parameters:
        -----------
        y : npt.NDArray[np.float64]
            The target values of shape (n_samples,).

        Returns:
        --------
        Node
            The leaf node.
        """

        if self.criterion == SplitCriterion.MSE:
            return Node(value=np.mean(y))
        else:
            biggest_class = np.argmax(np.bincount(y))
            return Node(value=biggest_class)

    def _find_best_split(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> Optional[SplitResult]:
        """
        Finds the best split for the given features and target values.
        This method considers all features and possible split points to determine
        the best split that maximizes the information gain. If no valid split is found,
        it returns None.

        Parameters:
        -----------
        x : npt.NDArray[np.float64]
            The feature matrix where each row represents a sample and each column
            represents a feature.
        y : npt.NDArray[np.float64]
            The target values corresponding to the samples in the feature matrix.

        Returns:
        --------
        Optional[SplitResult]
            A SplitResult object containing the details of the best split found,
            including the feature index, threshold, indices of the left and right
            splits, and the gain. Returns None if no valid split is found.
        """
        best_gain = 0
        best_split = None
        for feature_idx in range(x.shape[1]):
            for threshold in np.unique(x[:, feature_idx]):

                left_indices = np.where(x[:, feature_idx] < threshold)[0]
                right_indices = np.where(x[:, feature_idx] >= threshold)[0]
                if (
                    len(left_indices) < self.min_samples_leaf
                    or len(right_indices) < self.min_samples_leaf
                ):
                    continue

                gain = self._calculate_split_gain(y, y[left_indices], y[right_indices])
                if gain > best_gain:
                    best_gain = gain
                    best_split = SplitResult(
                        feature_idx=feature_idx,
                        threshold=threshold,
                        left_indices=left_indices,
                        right_indices=right_indices,
                        gain=gain,
                    )
        return best_split

    def _calculate_impurity(self, y: npt.NDArray[np.float64]) -> float:
        """
        Calculate the impurity of a given set of labels `y` based on the specified criterion.

        Parameters
        ----------
        y : npt.NDArray[np.float64]
            The array of labels for which to calculate the impurity.

        Returns
        -------
        float
            The calculated impurity value.

        Raises
        ------
        ValueError
            If the specified criterion is not one of GINI, ENTROPY, or MSE.

        Notes
        -----
        This method supports three criteria for impurity calculation:
        - GINI: Gini impurity
        - ENTROPY: Entropy impurity
        - MSE: Mean Squared Error impurity
        """
        if self.criterion == SplitCriterion.GINI:
            counts = np.unique(y, return_counts=True)[1]
            p = counts / len(y)
            impurity = 1 - np.sum(p**2)

        elif self.criterion == SplitCriterion.ENTROPY:
            counts = np.unique(y, return_counts=True)[1]
            p = counts / len(y)
            impurity = -np.sum(p * np.log2(p))

        elif self.criterion == SplitCriterion.MSE:
            impurity = np.var(y)

        else:
            raise ValueError(f"Invalid criterion: {self.criterion}")

        return impurity

    def _calculate_split_gain(
        self,
        y_parent: npt.NDArray[np.float64],
        y_left: npt.NDArray[np.float64],
        y_right: npt.NDArray[np.float64],
    ) -> float:
        """
        Calculate the gain from splitting a node into two children nodes.
        This method computes the reduction in impurity (gain) achieved by splitting
        a parent node into two children nodes. It works for both classification and
        regression criteria.

        Parameters:
        ----------
        y_parent (npt.NDArray[np.float64]): The target values of the parent node.
        y_left (npt.NDArray[np.float64]): The target values of the left child node.
        y_right (npt.NDArray[np.float64]): The target values of the right child node.

        Returns:
        --------
        float: The gain from the split, calculated as the difference between the
               impurity of the parent node and the weighted impurity of the children nodes.
        """

        impurity_parent = self._calculate_impurity(y_parent)
        impurity_left = self._calculate_impurity(y_left)
        impurity_right = self._calculate_impurity(y_right)

        n = len(y_parent)
        n_left = len(y_left)
        n_right = len(y_right)

        impurity_children = (n_left / n) * impurity_left + (
            n_right / n
        ) * impurity_right
        gain = impurity_parent - impurity_children
        return gain

    def predict(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Predict the output for the given input data.
        This method handles both classification and regression tasks.

        Parameters
        ----------
        x : npt.NDArray[np.float64]
            Input data array where each row is a sample and each column is a feature.

        Returns
        -------
        npt.NDArray[np.float64]
            Predicted output array where each element corresponds to the prediction for the respective input sample.
        """
        return np.array([self._predict_single(sample) for sample in x])

    def _predict_single(
        self, x: npt.NDArray[np.float64], node: Optional[Node] = None
    ) -> Union[float, npt.NDArray[np.float64]]:
        """
        Predict the target value for a single sample by traversing the decision tree.

        Parameters
        ----------
        x : numpy.ndarray
            Input sample of shape (n_features,) containing feature values
        node : Node, optional
            Current node in the decision tree. If None, starts from root node.
            Default is None.

        Returns
        -------
        Union[float, numpy.ndarray]
            Predicted target value for the input sample. Returns float for regression
            and numpy array for classification tasks.

        Notes
        -----
        This is a recursive method that traverses the decision tree until reaching a leaf node.
        At each internal node, it compares the feature value with the node's threshold to
        decide whether to traverse left or right child.
        """
        if node is None:
            node = self.root

        if node.is_leaf:
            return node.value

        if x[node.feature_index] < node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
