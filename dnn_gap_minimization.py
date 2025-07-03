import numpy as np
import cvxpy as cp
from scipy.optimize import minimize

# Define data
x = np.array([-1.8, -0.3, 0.7, 1.5])
m = len(x)
mu = 1

    # F(x) = x^2 + cos(x)
grad_F = 2 * x - np.sin(x)
G = np.outer(grad_F, grad_F)
M = np.outer(x, x)
H = G / mu**2 - M
d = x**2

    # Primal (nonconvex QP) using scipy
def objective(alpha):
    return alpha @ d + alpha @ H @ alpha

def constraint_sum_to_one(alpha):
    return np.sum(alpha) - 1

constraints = [
    {'type': 'eq', 'fun': constraint_sum_to_one},
    {'type': 'ineq', 'fun': lambda a: a}  # alpha >= 0
]

alpha0 = np.ones(m) / m
res = minimize(objective, alpha0, constraints=constraints, method='SLSQP')
primal_val = res.fun
alpha_primal = res.x





    # DNN relaxation via CVXPY
alpha = cp.Variable((m, 1))  # Column vector
Lambda = cp.Variable((m, m), PSD=True)

constraints = [
    cp.sum(alpha) == 1,
    alpha >= 0,
    cp.bmat([[cp.reshape(cp.Constant(1.0), (1, 1)), alpha.T],
             [alpha, Lambda]]) >> 0,
    Lambda >= 0,
    cp.trace(Lambda) <= 1
]
for i in range(m):
    for j in range(m):
        constraints.append(Lambda[i, j] <= alpha[i, 0])
        constraints.append(Lambda[i, j] <= alpha[j, 0])

objective = cp.Minimize(cp.trace(H @ Lambda) + cp.sum(cp.multiply(d.reshape((m, 1)), alpha)))
prob = cp.Problem(objective, constraints)
prob.solve()

# Output
print("\n--- Results ---")
print("Eigenvalues of H:", np.linalg.eigvalsh(H))
print("Original primal value (scipy):", primal_val)
print("DNN relaxation value (cvxpy):", prob.value)
print("Optimal alpha (primal):", alpha_primal)
print("Optimal alpha (DNN):", alpha.value.flatten())
