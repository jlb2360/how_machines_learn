from sklearn.datasets import make_blobs
from src.SVM2 import SVM2
from src.SVM import SVM
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 1. Create a simple, linearly separable dataset
    # Class 0: Points generally in the lower-left quadrant
    np.random.seed(42) # for reproducibility
    num_samples_class0 = 50
    X0 = np.random.randn(num_samples_class0, 2) * 0.8 + np.array([-2, -2])
    y0 = np.zeros(num_samples_class0) # Label for class 0

    # Class 1: Points generally in the upper-right quadrant
    num_samples_class1 = 50
    X1 = np.random.randn(num_samples_class1, 2) * 0.8 + np.array([2, 2])
    y1 = np.ones(num_samples_class1) # Label for class 1

    # Combine the datasets
    X = np.vstack((X0, X1))
    y = np.hstack((y0, y1))
    y[np.where(y==0)] = -1

    # 2. Visualize the dataset (optional, but highly recommended)
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50, edgecolors='k')
    plt.title('Linearly Separable Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    svm_smo = SVM2()
    svm_smo.fit(X, y, C=2.0, tol=1e-3, max_pass=100)

    print("\n--- Results ---")
    print(f"w: {svm_smo.w}")
    print(f"b: {svm_smo.b}")
    print(f"Alphas: {svm_smo.alphas}")

    # Make predictions
    predictions = svm_smo.predict(X)
    print(f"Predictions on training data: {predictions}")
    print(f"Actual labels:              {y}")
    print(f"Accuracy: {np.mean(predictions == y) * 100:.2f}%")

    plt.figure(figsize=(10, 7))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='coolwarm', edgecolors='k', zorder=10)

    if svm_smo.w is not None and svm_smo.b is not None:
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 50)
        yy = np.linspace(ylim[0], ylim[1], 50)
        YY, XX = np.meshgrid(yy, xx)
        xy_grid = np.vstack([XX.ravel(), YY.ravel()]).T
        
        # Decision boundary
        Z = np.dot(xy_grid, svm_smo.w) - svm_smo.b
        Z = Z.reshape(XX.shape)
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.8,
                   linestyles=['--', '-', '--'])

        # Highlight support vectors (points with alpha > tolerance)
        if svm_smo.support_vectors_X is not None and len(svm_smo.support_vectors_X) > 0:
            plt.scatter(svm_smo.support_vectors_X[:, 0], svm_smo.support_vectors_X[:, 1], s=200,
                        linewidth=1.5, facecolors='none', edgecolors='k', label='Support Vectors')
    
    plt.title('SVM (Linear Kernel)')
    plt.xlabel('X1')
    plt.ylabel('X2')
    if svm_smo.support_vectors_X is not None and len(svm_smo.support_vectors_X) > 0 : plt.legend()
    plt.show()