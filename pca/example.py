from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()

X = iris['data']
y = iris['target']
y_names = iris['target_names']

cov_matrix = X.T@X

eigen_vals, eigen_vecs = np.linalg.eig(cov_matrix)

w_r = eigen_vecs[:, 0:2]

T = X @ w_r

T_setosa = T[np.where(y==0)]
T_versicolor = T[np.where(y==1)]
T_virginica = T[np.where(y==2)]

plt.scatter(T_setosa[:,0], T_setosa[:,1], label="Iris setosa")
plt.scatter(T_versicolor[:,0], T_versicolor[:,1], label="Iris versicolor")
plt.scatter(T_virginica[:,0], T_virginica[:,1], label="Iris virginica")

plt.title("Iris Data after principal component analysis ")
plt.xlabel("pca 1")
plt.ylabel("pca 2")

plt.legend()
plt.show()