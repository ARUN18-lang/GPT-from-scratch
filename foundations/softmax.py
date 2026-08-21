import numpy as np
from numpy.typing import NDArray


class Solution:

    def temperature(self, probs: NDArray[np.float64], T: float) -> NDArray[np.float64]:
        maxi = max(probs)
        denominator = 0
        for j in probs:
            denominator += 2.71828**((j-maxi)/T)
        
        for ind, logit in enumerate(probs):
            probs[ind] = np.round(2.71828**((logit - maxi)/T) / denominator, 4)
        
        return probs

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        maxi = max(z)
        denominator = 0
        for j in z:
            denominator += 2.71828**(j-maxi)
        
        for ind, logit in enumerate(z):
            z[ind] = np.round(2.71828**(logit - maxi) / denominator, 4)
        
        print("Temperature sampling: ", self.temperature(z.copy(), 0.5))
        return z
    


        

