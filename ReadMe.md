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

**Implementation Help Received:**
- Syntax fixes (`np.random.randn()` parameter format)
- Matrix multiplication in backprop (`input.T @ grad`)
- Data generation and plotting code
- This README structure

## 🚀 Results

image.png

**Neuron Activation Image:**

- each neuron learned:
image.png

**Performance:**
- Successfully learned sin(2x) + 0.3x² function
- Training MSE: [your number]
- Demonstrates proper non-linear function approximation

## 💡 Why This Approach?

I wanted to build genuine understanding, not just copy working code. The collaboration helped with syntax while I drove the conceptual understanding and debugging process.