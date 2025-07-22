# Neural Networks From Scratch with NumPy

A ground-up implementation of neural networks with backpropagation, built to understand the fundamentals.



**Core Implementation (My Work):**
- Matrix-based neural network architecture with scalable layers
- Backpropagation algorithm with proper gradient flow
- ReLU activation and MSE loss functions
- Debugging methodology for gradient flow issues
- Function approximation that captures non-linear patterns

**Key Insights I Discovered:**
- How averaging gradients affects learning (MSE vs parameter updates)
- Matrix dimension requirements for multi-layer networks
- The relationship between network capacity and function approximation
- Debugging dead neurons and initialization issues

## 🤝 Collaboration & Learning Process

**What I Figured Out:**
- Gradient flow debugging (traced shapes, identified averaging issues)
- Matrix dimension logic ((50,1) @ (1,10) = (50,10))
- Overfitting diagnosis (too many parameters vs data points)
- Network capacity scaling (10 → 50 neurons for better curves)

**Implementation Help Received from Claude:**
- Syntax fixes (`np.random.randn()` parameter format)
- Matrix multiplication in backprop (`input.T @ grad`)
- Data generation and plotting code
- This README structure

## 🚀 Results

<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/936f0ce8-5ad3-426c-b720-6e47173ab2dd" />


**Neuron Activation Image:**

- each neuron learned:
<img width="1489" height="489" alt="image" src="https://github.com/user-attachments/assets/c00888e2-1b06-4b8a-aa00-f90758c71fb2" />


**Performance:**
- Successfully learned sin(2x) + 0.3x² function
- Training MSE: [your number]
- Demonstrates proper non-linear function approximation
