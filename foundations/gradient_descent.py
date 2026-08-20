class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        if iterations == 0:
            return init
        
        # learning formula
        new_weight = init
        for i in range(iterations):
            # weight new = weight old - lr * derivative of loss
            old_weight = new_weight
            new_weight = old_weight -  learning_rate * (2*old_weight)
        
        return round(new_weight, 5)


