#!/usr/bin/env python3

import numpy as np

# -------------------------
# 1️⃣ Weight Initialization
# -------------------------
def initialize_parameters(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(1 / layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
    return parameters

# -------------------------
# 2️⃣ Forward Pass with Dropout
# -------------------------
def forward_with_dropout(A_prev, W, b, keep_prob=1.0):
    Z = np.dot(W, A_prev) + b
    A = np.tanh(Z)
    
    if keep_prob < 1.0:
        D = np.random.rand(*A.shape) < keep_prob
        A = np.multiply(A, D) / keep_prob
    else:
        D = np.ones_like(A)
    
    cache = (A_prev, W, b, Z, D)
    return A, cache

# -------------------------
# 3️⃣ Backward Pass with Dropout
# -------------------------
def backward_with_dropout(dA, cache, keep_prob=1.0):
    A_prev, W, b, Z, D = cache
    
    if keep_prob < 1.0:
        dA = np.multiply(dA, D) / keep_prob
    
    dZ = dA * (1 - np.tanh(Z) ** 2)
    dW = np.dot(dZ, A_prev.T) / A_prev.shape[1]
    db = np.sum(dZ, axis=1, keepdims=True) / A_prev.shape[1]
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

# -------------------------
# 4️⃣ Full Forward Through Network
# -------------------------
def forward_network(X, parameters, keep_prob=1.0):
    caches = []
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L+1):
        A_prev = A
        A, cache = forward_with_dropout(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], keep_prob)
        caches.append(cache)
    
    return A, caches

# -------------------------
# 5️⃣ Example Usage
# -------------------------
if __name__ == "__main__":
    layer_dims = [64, 32, 16, 16, 16, 8, 8, 4, 1]  # example
    X = np.random.randn(64, 100)  # 100 examples
    parameters = initialize_parameters(layer_dims)
    
    A_final, caches = forward_network(X, parameters, keep_prob=0.8)
    
    # Print first layer weights as sanity check
    print("W1", parameters['W1'])
    print("b1", parameters['b1'])
