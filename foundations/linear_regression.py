import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        predictions = []
        for inputs in X:
            pred = 0
            for inp, weight in zip(inputs, weights):
                pred += inp * weight
            predictions.append(pred)
        return np.round(np.array(predictions), 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        error = 0
        n = len(model_prediction)
        for y_truth, y_pred in zip(ground_truth, model_prediction):
            error += (y_pred[0] - y_truth[0])**2
        
        return np.round(error/n, 5)
        
