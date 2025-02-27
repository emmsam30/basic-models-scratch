import numpy as np

N = 10_000  # number of samples (rows)
d = 5       # number of features (columns)

X = np.random.rand(N, d) * 10   # random feature matrix of size (N, d)

true_w = np.array([1.5, -2.0, 0.5, 3.0, 2.5])    # weights
true_b = 5.0     # bias

noise = np.random.randn(N) * 2.0    # generate noise to make data more realistic

y = X @ true_w + true_b + noise

w = np.array([0.5, 0.5, 0.5, 0.5, 0.5])     # start with arbitrary weights, not learned
b = 0.0     # same with bias

y_pred = X @ w + b

print("X shape: ", X.shape)
print("y shape: ", y.shape)
print("y_pred shape: ", y_pred.shape)

print("First 5 rows or true labels (y):     ", y[:5])
print("First 5 rows or predicted labels (y_pred):     ", y_pred[:5])