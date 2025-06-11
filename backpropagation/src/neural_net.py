"""This is a single hidden layer neural net to test my own backpropagation code"""
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNet():
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        self.input_layer_s = input_layer_size
        self.hidden_layer_s = hidden_layer_size
        self.output_layer_s = output_layer_size
        # create random weights to break symmetry
        self.w1 = np.random.randn(self.input_layer_s, self.hidden_layer_s)
        self.w2 = np.random.randn(self.hidden_layer_s, self.output_layer_s)

        self.b1 = np.zeros((1, self.hidden_layer_s))
        self.b2 = np.zeros((1, self.output_layer_s))
        
    def forward_step(self, X: np.ndarray):
        
        # first layer
        self.hidden_activation = sigmoid(X @ self.w1 + self.b1)

        # second layer
        return sigmoid(self.hidden_activation @ self.w2 + self.b2)


    def backprop(self, X, y, yhat, alpha):
        output_error = y - yhat

        output_delta = 2 * output_error * sigmoid_derivative(yhat)

        delta_w2 = self.hidden_activation.T @ output_delta
        
        delta_w1 = X.T.dot( output_delta.dot( self.w2.T) * sigmoid_derivative(self.hidden_activation))
    

        self.w1 += alpha * delta_w1
        self.b1 += alpha * np.sum(output_delta @ self.w2.T * sigmoid_derivative(self.hidden_activation), axis=0, keepdims=True)
        self.w2 += alpha * delta_w2
        self.b2 += np.sum(output_delta, axis=0, keepdims=True) * alpha

    def train(self, X, y, epochs, learning_rate, save_errors):
        error_history = []
        for i in range(epochs):
            # Perform a feedforward pass
            output = self.forward_step(X)
            # Perform backpropagation and update weights
            self.backprop(X, y, output, learning_rate)
            # Print the error at every 1000th epoch to observe training progress
            if i % 1000 == 0:
                print(f"Error at epoch {i}: {np.mean(np.square(y - output))}")

            if save_errors:
                if i % 100 == 0:
                    error = np.mean(np.square(y - output))
                    error_history.append(error)

        return error_history
