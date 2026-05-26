# Linear Regression from Scratch

A clean, object-oriented implementation of Linear Regression built entirely from scratch in Python using **NumPy** for vectorized matrix mathematics. 

The entire project—including the model architecture, gradient descent training loop, synthetic data generation, and visualization—is contained within a single self-contained script for seamless execution.

## 🚀 Key Features

- **Single-File Architecture:** Zero complex modular imports or directory dependencies; copy and run instantly.
- **Matrix Vectorization:** Scaled efficiently via NumPy matrix operations utilizing vector extensions rather than nested iterative loops.
- **Versatile Dimension Handling:** Natively accommodates simple data inputs ($y = wx + b$) and scales automatically to multivariate column structures ($y = Xw + b$).
- **Clean API Design:** Built around industry-standard `.fit()` and `.predict()` paradigms.

---

## 📊 Theoretical Mechanics

### 1. Underlying Hypothesis Matrix
Predictions ($\hat{y}$) are derived via standard inner dot-products of features combined with optimized parameter intercepts:

```math
\hat{y} = Xw + b
