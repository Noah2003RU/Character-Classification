import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import digit


# Labels for each image, assuming all are the letter 'a'
# labels = ["a", "d", "m", "n", "o", "p", "q", "r", "u", "w"]
labels = ["d", "d", "d", "d", "d", "d", "d", "d", "d", "d"]

# Convert lists to NumPy arrays for easier manipulation
hu_moments_data = np.array(digit.features)
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    hu_moments_data, labels, test_size=0.2, random_state=42
)

# Create a k-nearest neighbors classifier (you can use other classifiers based on your needs)
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn_classifier.fit(X_train, y_train)

# Predict the labels for the test set
predictions = knn_classifier.predict(X_test)

# Evaluate the accuracy
predictions = knn_classifier.predict(X_test)

# Print Predicted and Actual Labels:
for pred, actual in zip(predictions, y_test):
    print(f"Predicted: {pred}, Actual: {actual}")

# Evaluate Accuracy:
accuracy = np.mean(predictions == y_test)
print("Accuracy:", accuracy)
