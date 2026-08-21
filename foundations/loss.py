import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        Loss = 0
        n = len(y_true)
        epsilon = 1e-7
        for actual, predicted in zip(y_true, y_pred):
            if actual:
                Loss += actual * np.log(
                        predicted + epsilon
                    ) 
            else:
                Loss += (1 - actual)*np.log(
                        1 - (predicted + epsilon)
                    )
        return np.round(-(Loss/n), 4)


    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        loss = 0
        n = len(y_true)
        epsilon = 1e-7
        for a, p in zip(y_true, y_pred):
            for actual, predicted in zip(a, p):
                loss += actual * np.log(
                    predicted + epsilon
                )
        
        return np.round(-(loss/n), 4)
