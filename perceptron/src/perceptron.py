import numpy as np

class Perceptron():
    def __init__(self, num_iterations = 100, gamma = 0.1):
        self.iters = num_iterations
        self.gamma = gamma

    def train(self, X, y, num_iterations = None):
        if num_iterations != None:
            self.iters = num_iterations
        self.weights = np.zeros(X.shape[1])
        self.bias = 0


        for _ in range(self.iters):
            num_changes = 0
            for j,ans in enumerate(y):
                if (ans-self.bias) * self.weights.T @ X[j] <= 0:
                    num_changes += 1
                    self.weights = self.weights + self.gamma*ans*X[j]
                    self.bias = self.bias + self.gamma * ans

            if num_changes == 0:
                break

        print(self.weights)
        print(self.bias)

    def response(self, x):
        return np.where(x @ self.weights + self.bias >= 0.0, 1, -1)