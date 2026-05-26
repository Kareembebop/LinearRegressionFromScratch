import numpy as np

class LinearRegressionFromScratch:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None  # Represents 'm'
        self.bias = None     # Represents 'b'

    def fit(self, X, y):
        # n_samples = number of rows, n_features = number of columns
        n_samples, n_features = X.shape
        
        # Initialize weights to zeros and bias to zero
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Gradient Descent Loop
        for epoch in range(self.epochs):
            # 1. Predict y values using current weights and bias
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # 2. Calculate gradients (derivatives)
            dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (2 / n_samples) * np.sum(y_predicted - y)
            
            # 3. Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Optional: Print loss every 100 epochs to see it decrease
            if epoch % 100 == 0:
                loss = np.mean((y - y_predicted) ** 2)
                print(f"Epoch {epoch}: Loss (MSE) = {loss:.4f}")

    def predict(self, X):
        # Use the trained weights and bias to make new predictions
        return np.dot(X, self.weights) + self.bias
    

import matplotlib.pyplot as plt

# 1. Generate random, roughly linear data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X.squeeze() + np.random.randn(100) # True relationship: y = 3x + 4 + noise

# 2. Initialize and train our scratch model
model = LinearRegressionFromScratch(learning_rate=0.1, epochs=500)
model.fit(X, y)

# 3. Print the learned parameters
print(f"\nFinal Learned Weight (m): {model.weights[0]:.4f} (Expected close to 3)")
print(f"Final Learned Bias (b): {model.bias:.4f} (Expected close to 4)")

# 4. Make predictions for plotting
X_test = np.array([[0], [2]])
y_preds = model.predict(X_test)

# 5. Plot the results
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X_test, y_preds, color='red', linewidth=2, label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()    