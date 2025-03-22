import numpy as np

class SoftmaxClassifier:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def softmax(self, z):
        # Numerically stable softmax
        exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Initialize parameters
        self.weights = np.random.randn(n_features, n_classes) * 0.01
        self.bias = np.zeros((1, n_classes))
        
        # Convert y to one-hot encoded matrix
        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), y] = 1
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.softmax(z)
            
            # Compute gradients
            dz = y_pred - y_onehot
            dw = (1/n_samples) * np.dot(X.T, dz)
            db = (1/n_samples) * np.sum(dz, axis=0, keepdims=True)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calculate and print loss every 100 iterations
            if _ % 100 == 0:
                loss = -np.mean(np.sum(y_onehot * np.log(y_pred + 1e-15), axis=1))
                print(f"Iteration {_}, Loss: {loss:.4f}")
    
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.softmax(z)
        return np.argmax(y_pred, axis=1)

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    
    # Create synthetic dataset with 3 classes
    n_samples = 300
    X = np.random.randn(n_samples, 2)
    y = np.random.randint(0, 3, n_samples)
    
    # Add some separation between classes
    X += 3 * y.reshape(-1, 1)
    
    # Create and train the classifier
    classifier = SoftmaxClassifier(learning_rate=0.1, n_iterations=1000)
    classifier.fit(X, y)
    
    # Make predictions
    y_pred = classifier.predict(X)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y)
    print(f"\nAccuracy: {accuracy:.4f}")