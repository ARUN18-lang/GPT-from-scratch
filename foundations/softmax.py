import numpy as np
from numpy.typing import NDArray


class Solution:
    # sigmoid and softmax produces output between [0-1], then why not sigmoid instead of softmax.
    # Sigmoid handles independent binary choices. Softmax handles mutually exclusive multi-class choices by forcing all output scores to add up to exactly 1
    # example -> sigmoid produces 0 or 1 binary choices, softmax produces list of probs, the one with higher probs wins 
    # probs = [0.2, 0.6, 0.1, 0.1] - max(probs) = 0.6
    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        maxi = max(z)
        print(maxi)
        denominator = 0
        for j in z:
            denominator += 2.71828**(j-maxi)
        
        for ind, logit in enumerate(z):
            z[ind] = np.round(2.71828**(logit - maxi) / denominator, 4)
        
        return z
            

        
