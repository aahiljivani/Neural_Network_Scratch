import numpy as np

class Node:
    def __init__(self, input_size, output_size, weight=None, bias=None):
        # weight is a matrix of shape (input_size, output_size)
        self.weight = np.random.randn(input_size, output_size) if weight is None else weight
        # bias is a vector of shape (output_size,1)
        self.bias = np.random.randn(output_size) if bias is None else bias
        
    def forward(self, x):
        self.last_input = x  # Store input for backward pass
        # this is just the linear transformation
        return np.dot(x, self.weight) + self.bias
    
    def backward(self, grad_from_next_layer, alpha=0.01):
        # Compute gradients using stored input
        # For weights: gradient is input.T @ grad_output (averaged over batch)
        weight_gradient = np.dot(self.last_input.T, grad_from_next_layer) / len(grad_from_next_layer)
        # For bias: gradient is mean of grad_output across batch dimension
        bias_gradient = np.mean(grad_from_next_layer, axis=0)
        
        # Update parameters
        self.weight -= alpha * weight_gradient
        self.bias -= alpha * bias_gradient
        
        # Return gradient for previous layer
        grad_for_previous = np.dot(grad_from_next_layer, self.weight.T)
        return grad_for_previous


class Relu:
    def __init__(self):
        pass
        
    def forward(self, x):
        self.last_input = x  # Store input for backward pass to preserve chain rule
        return np.maximum(0, x)
    
    def backward(self, grad_from_next_layer):
        # Compute ReLU derivative using stored input as float makes it 1 or 0
        relu_derivative = (self.last_input > 0).astype(float)
        
        # Apply chain rule
        grad_for_previous = grad_from_next_layer * relu_derivative # (1 or o in this case)
        return grad_for_previous


class MSE:
    def __init__(self):
        pass
        
    def forward(self, target, y_pred):
        self.last_target = target    # Store for backward pass
        self.last_y_pred = y_pred    # Store for backward pass
        return np.mean((target - y_pred) ** 2)
    
    def backward(self):
        # Compute gradient of loss with respect to predictions
        grad = 2 * (self.last_y_pred - self.last_target) / len(self.last_y_pred)
        return grad