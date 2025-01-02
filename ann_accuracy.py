#import libraries
import numpy as np
import pandas as pd
import warnings

#ignore warnings
warnings.filterwarnings('ignore')

#load csv file
data = pd.read_csv('kidney-stone-dataset.csv')
data.isna().any()

# drop the unwanted columns
X = data.drop(['Unnamed: 0', 'target'], axis=1)
y = data['target']

class NeuralNetwork:
    def __init__(self):
        np.random.seed(1)
        self.weights = {}
        self.biases = {}

    def add_layer(self, input_size, output_size):
        layer_index = len(self.weights)
        self.weights[layer_index] = np.random.randn(input_size, output_size) * 0.01
        self.biases[layer_index] = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, X):
        self.layer_outputs = []
        self.activations = [X]

        for i in range(len(self.weights)):
            weighted_sum = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            activation = self.sigmoid(weighted_sum)
            self.layer_outputs.append(weighted_sum)
            self.activations.append(activation)

        return self.activations[-1]

    def backward_propagation(self, y_true, output):
        deltas = [output - y_true]
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[-1], self.weights[i].T) * self.sigmoid_derivative(self.activations[i])
            deltas.append(delta)
        deltas.reverse()

        self.weight_gradients = {}
        self.bias_gradients = {}

        for i in range(len(self.weights)):
            self.weight_gradients[i] = np.dot(self.activations[i].T, deltas[i])
            self.bias_gradients[i] = np.sum(deltas[i], axis=0, keepdims=True)

    def update_weights(self, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.weight_gradients[i]
            self.biases[i] -= learning_rate * self.bias_gradients[i]


# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values.reshape(-1, 1), test_size=0.5, random_state=42)

# Initialize neural network
model = NeuralNetwork()

# Add layers
model.add_layer(X_train.shape[1], 16)
model.add_layer(16, 8)
model.add_layer(8, 1)

# Train the model
epochs = 1000
learning_rate = 0.01

for epoch in range(epochs):
    # Forward propagation
    output = model.forward_propagation(X_train)

    # Backward propagation
    model.backward_propagation(y_train, output)

    # Update weights
    model.update_weights(learning_rate)

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        loss = np.mean((y_train - output) ** 2)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Forward propagation on test data
predictions = model.forward_propagation(X_test)

# Threshold predictions (0.5 for binary classification)
threshold = 0.5
binary_predictions = (predictions > threshold).astype(int)

# Calculate accuracy
accuracy = np.mean(binary_predictions == y_test)
print("Accuracy:", accuracy)
