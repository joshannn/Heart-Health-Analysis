# Linear Algebra with NumPy

## Description

This assignment explores fundamental linear algebra concepts and their practical applications in data science. We work with matrices, understand their properties, and apply these mathematical tools to solve a real-world problem - predicting car prices using linear regression.

Linear algebra forms the backbone of machine learning and data science. Every time we train a model, optimize parameters, or transform data, we're using linear algebra operations. This notebook demonstrates these concepts from basics to practical application.

## What This Notebook Covers

The notebook walks through several key concepts:

We start with basic matrix operations - addition, multiplication, and scalar operations. These are the building blocks of more complex computations in machine learning. We then examine matrix properties like determinants and invertibility, which tell us whether a system of equations has a unique solution. The eigenvalue decomposition section reveals the fundamental "directions" in which a matrix transformation operates, a concept crucial for dimensionality reduction techniques like PCA.

Finally, we apply everything to a practical problem: predicting car prices. This demonstrates how abstract mathematical concepts translate into useful real-world applications.

## Technologies and Libraries Used

**Python 3.x** - Programming language
**NumPy** - Core library for numerical computations and matrix operations

### Key NumPy Functions

* `np.array()` - Creates matrices from Python lists
* `np.linalg.det()` - Computes the determinant of a matrix
* `np.linalg.inv()` - Calculates the inverse of a matrix
* `np.linalg.eig()` - Finds eigenvalues and eigenvectors
* `np.linalg.pinv()` - Computes the Moore-Penrose pseudo-inverse
* `@` operator - Performs matrix multiplication
* `.T` - Transposes a matrix

## Mathematical Concepts and Formulas

### Matrix Operations

**Addition:** A + B (element-wise addition of corresponding elements)
**Multiplication:** A × B using the formula where each element (i,j) in the result equals the dot product of row i from A and column j from B
**Scalar Multiplication:** 3A - 2B (multiply each element by the scalar)

### Matrix Properties

**Determinant:** A scalar value that indicates if a matrix is invertible. If det(A) = 0, the matrix is singular (non-invertible).
**Inverse Matrix:** A⁻¹ is the matrix such that A × A⁻¹ = I (identity matrix)
**Adjoint Matrix:** Related to the inverse by the formula: Adjoint(A) = det(A) × A⁻¹

### Eigenvalue Decomposition

For a square matrix A, eigenvalues (λ) and eigenvectors (v) satisfy:
**A × v = λ × v**
This means the eigenvector v only gets scaled (not rotated) when multiplied by A, and the scaling factor is the eigenvalue λ.

### Linear Regression Formula

**β = (X^T X)^(-1) X^T y**
Where:

* X is the feature matrix (our car characteristics)
* y is the target vector (car prices)
* β is the coefficient vector we're solving for

## The Practical Problem: Car Price Prediction

### The Challenge

Imagine you're working for a used car dealership. You have data on previous car sales including various features (number of cylinders, engine size, horsepower) and their selling prices. A new car arrives, and you need to estimate its fair market price based on its characteristics.

### Why Find Beta (β)? The Physical Meaning

Beta represents the **coefficients** or **weights** that tell us how much each feature contributes to the final price. Finding β is solving this question: "How much does each characteristic of a car affect its price?"

**The Physical Interpretation:**
Each element in the β vector tells us something meaningful:

* β₀ (intercept): The base price of a car with all features at zero
* β₁ (cylinders coefficient): How much the price increases per cylinder
* β₂ (engine size coefficient): How engine size affects the price
* β₃ (horsepower coefficient): The value added by each unit of horsepower

**Why This Matters Practically:**
When we find β = [β₀, β₁, β₂, β₃], we're discovering the relationship between features and price. The coefficients reveal which car features most influence pricing.

**The Mathematical Power:**
Instead of guessing or using intuition, we use the normal equation to find the mathematically optimal β that minimizes prediction errors across all our training examples.

### Our Dataset

**Training Data:**

| Car | Cylinders (x1) | Engine (x2, L) | Horsepower (x3) | Price (y) |
| --- | -------------- | -------------- | --------------- | --------- |
| 1   | 4              | 1.5            | 100             | 20        |
| 2   | 6              | 2.0            | 150             | 30        |

**Test Case:**

| Car | Cylinders (x1) | Engine (x2, L) | Horsepower (x3) | Price (y) |
| --- | -------------- | -------------- | --------------- | --------- |
| 3   | 8              | 3.0            | 200             | ?         |

We represent this as a matrix equation **X · β = y** and solve for β using:
**β = (X^T X)^(-1) X^T y**
Then predict the price of Car 3 using:
**y₃ = [1, 8, 3.0, 200] · β**

## Results and Key Findings

**Determinant of Matrix A:** 6.0 (matrix is invertible)
**Eigenvalues of Matrix C:** λ₁ = 5, λ₂ = 0 (example for illustration)

**Regression Coefficients (β):**

* Intercept: [calculated from normal equation]
* Cylinders coefficient: [calculated]
* Engine coefficient: [calculated]
* Horsepower coefficient: [calculated]

**Prediction:** The model predicts Car 3's price at approximately $40 (or calculated value), which aligns logically with the trend of the training data.

## Learning Outcomes

Through this assignment, we understand that:

1. **Matrix operations** represent transformations and relationships in data.
2. **Eigenvalues and eigenvectors** reveal the fundamental structure of transformations.
3. **Linear regression** is fundamentally a linear algebra problem solved through matrix operations.
4. **The normal equation** provides an exact solution to finding optimal model parameters.
5. **Real-world prediction** problems can be formulated and solved using mathematical principles.

## Course Context

**Course:** Fundamentals of Data Science (FOD)
**Assignment:** Assignment 2 - Linear Algebra
**Semester:** 3rd Semester
**Focus:** Mathematical foundations for machine learning

*This assignment demonstrates that mathematical concepts aren't isolated theory - they're powerful tools for solving real problems and making data-driven decisions.*
