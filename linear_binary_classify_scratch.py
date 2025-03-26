import numpy as np

class BinaryClassifier:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        # Clip z to avoid overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calculate and print loss every 100 iterations
            if _ % 100 == 0:
                loss = -np.mean(y * np.log(predictions + 1e-15) + 
                              (1-y) * np.log(1 - predictions + 1e-15))
                print(f"Iteration {_}, Loss: {loss:.4f}")
    
    def predict_proba(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    
    # Create synthetic dataset
    N = 1000  # number of samples
    d = 5     # number of features
    
    # Generate features
    X = np.random.randn(N, d)
    
    # Generate true weights and bias
    true_weights = np.array([1.5, -2.0, 0.5, 3.0, 2.5])
    true_bias = 5.0
    
    # Generate labels with some noise
    noise = np.random.randn(N) * 2.0
    y = (X @ true_weights + true_bias + noise > 0).astype(int)
    
    # Initialize and train the classifier
    classifier = BinaryClassifier(learning_rate=0.01, n_iterations=1000)
    classifier.fit(X, y)
    
    # Make predictions
    y_pred = classifier.predict(X)
    
    # Print results
    print("First 5 true labels: ", y[:5])
    print("First 5 predicted labels: ", y_pred[:5])