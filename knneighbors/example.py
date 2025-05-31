import numpy as np
import pandas as pd
from src.KNN import KNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def accuracy_score(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    correct_predictions = np.sum(y_true == y_pred)
    return correct_predictions / len(y_true)


np.random.seed(42)
means_howler = [10, 60, 70, 80, 15]
means_spider = [7, 50, 80, 120, 10]
means_macaque = [8, 55, 30, 70, 12]
std_dev = [1.5, 5, 8, 10, 2]
n_samples_per_class = 100

data_howler = np.random.normal(loc=means_howler, scale=std_dev, size=(n_samples_per_class, 5))
data_spider = np.random.normal(loc=means_spider, scale=std_dev, size=(n_samples_per_class, 5))
data_macaque = np.random.normal(loc=means_macaque, scale=std_dev, size=(n_samples_per_class, 5))

X = np.vstack([data_howler, data_spider, data_macaque])
y = np.array(['Howler'] * n_samples_per_class + ['Spider Monkey'] * n_samples_per_class + ['Macaque'] * n_samples_per_class)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"--- Data Split ---")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print("\n")

classifier = KNN()

classifier.fit(X_train, y_train)

print("--- Model Training Complete ---")
print(f"Learned Classes: {classifier.classes}")
print("\n")

y_pred = classifier.predict(X_test, 7)

for i in range(min(5, len(X_test))):
    print(f"Actual: {y_test[i]}, Predicted: {y_pred[i]}")
print("\n")

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred, labels=classifier.classes)

print(f"--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens',
            xticklabels=classifier.classes, yticklabels=classifier.classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
print("\n")