import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        
        # dL/dw = dL/dy * dy/dz * dz/dw
        # dL/dy = y_hat - y_true
        # dy/dz = y_hat * (1 - y_hat) # How?
        # dz/dw = x
        
        z = np.dot(x, w) + b # forward pass first

        y_hat = 1.0 / (1.0 + np.exp(-z))

        dL_dy = y_hat - y_true
        dy_dz = y_hat * (1.0 - y_hat)
        dz_dw = x

        dL_dw = np.round(dL_dy * dy_dz * dz_dw, 5)
        dL_db = round(float(dL_dy * dy_dz), 5)

        return (dL_dw, dL_db)
