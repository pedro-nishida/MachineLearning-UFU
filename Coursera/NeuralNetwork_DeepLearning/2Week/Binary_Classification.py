import numpy as np
import matplotlib.pyplot as plt

def simple_neuron(x, w, b):
    """
    #input data
    #(x,y) x belongs to R^2, y belongs to {0,1}
    #w belongs to R^2, b belongs to R
    #output: y_hat belongs to {0,1}
    """
    # Compute the weighted sum of inputs and bias
    z = np.dot(w, x) + b
    # Apply the sigmoid activation function
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat


    