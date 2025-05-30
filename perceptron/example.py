import numpy as np
from src.perceptron import Perceptron
import matplotlib.pyplot as plt

if __name__ == "__main__":
    percept = Perceptron(10000)

    num_data_points = 100

    # Class 1
    # np.random.seed(42) # for reproducibility
    X1 = np.random.randn(num_data_points, 2) + 3  # 50 samples, 2 features, centered around (2,2)
    y1 = np.ones(num_data_points)                # Labels for class 1 are 1

    # Class 2
    X2 = np.random.randn(num_data_points, 2) -1  # 50 samples, 2 features, centered around (-2,-2)
    y2 = np.ones(num_data_points) * -1           # Labels for class 2 are -1

    X = np.vstack((X1, X2)) # Combine features
    y = np.concatenate((y1, y2)) # Combine labels

    # Shuffle the data to ensure training isn't biased by order
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    percept.train(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = percept.response(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Perceptron Decision Boundary')
    plt.show()