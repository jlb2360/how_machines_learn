import numpy as np
from src.neural_net import NeuralNet
import matplotlib.pyplot as plt

def XOR_Data():
    X = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]])

    y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    nn = NeuralNet(input_layer_size=2, hidden_layer_size=4, output_layer_size=1)

    # --- 3. Training the Network ---
    # Train the neural network for 10,000 epochs with a learning rate of 0.1
    print("Starting training...")
    nn.train(X, y, epochs=10000, learning_rate=0.1, save_errors=False)
    print("Training finished.")

    # --- 4. Testing the Network ---
    print("\n--- Testing the trained network ---")
    for data_point in X:
        prediction = nn.forward_step(data_point)
        print(f"Input: {data_point} -> Predicted Output: {prediction} (Rounded: {np.round(prediction)})")


def random_clusters():
    # Create a synthetic dataset with two classes (clusters of points)
    np.random.seed(0) # for reproducibility
    X1 = np.random.rand(50, 2) * 0.4 
    y1 = np.zeros((50, 1))
    X2 = np.random.rand(50, 2) * 0.4 + 0.6
    y2 = np.ones((50, 1))
    
    # Combine the data
    X = np.vstack((X1, X2))
    y = np.vstack((y1, y2))


    # --- 2. Neural Network Initialization ---
    nn = NeuralNet(input_layer_size=2, hidden_layer_size=10, output_layer_size=1)

    # --- 3. Training the Network ---
    print("Starting training...")
    errors = nn.train(X, y, epochs=20000, learning_rate=0.05, save_errors = True)
    print("Training finished.")

    # --- 4. Plotting the Results ---
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(errors)), errors)
    plt.title('Training Error over Epochs')
    plt.xlabel('Epochs (x100)')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    
    # Plot 2: Decision Boundary
    plt.subplot(1, 2, 2)
    h = 0.01 # step size in the mesh
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict the function value for the whole grid
    Z = nn.forward_step(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.RdYlBu, edgecolors='k')
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    XOR_Data()
    random_clusters()



    