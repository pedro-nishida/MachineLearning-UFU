from classes import MLP
import numpy as np

def test_simple_xor():
    # Create a simple XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create a small network
    mlp = MLP(ninputs=2, nhidden=[4], noutputs=1)
    
    try:
        # Train the network
        mlp.train(X, y, epochs=1000, learning_rate=0.1)
        
        # Test predictions
        predictions = mlp.forward(X)
        print("XOR Test Predictions:")
        for i in range(len(X)):
            print(f"Input: {X[i]}, Target: {y[i]}, Prediction: {predictions[i]:.4f}")
    except ValueError as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    test_simple_xor()
