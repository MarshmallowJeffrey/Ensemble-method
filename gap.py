import numpy as np
from numpy.random import multivariate_normal as mn
import gurobipy as gp
from gurobipy import GRB

    #Data matrix: X[i] ~ N(0, I_n)
    #Data vector: d
    #Gradient matrix: H = GradF/(mu**2)-M
def Data_generator(m, n, X_distri, F_form):
        # Data matrix
    X = np.zeros((m, n))
    if X_distri == 'normal':
        for i in range(m):
            X[i, :] = mn([0], [[1]], n)[:, 0]
        # Data vector
    d = np.zeros(m)
    for i in range(m):
        d[i] = np.linalg.norm(X[i, :]) ** 2
        #L-smooth and \mu-strongly convex coeffs
    L = np.inf; mu = np.inf

        #Form of objective function
    if F_form == "quadratic":
            # How to choose c???
        c = 0.2 * np.ones(n)
        A = mn([0], [[1]], (n, n))[:, :, 0]
        Q = A.T @ A
        eigsQ = np.linalg.eigvals(Q)
        L = np.max(eigsQ)
        mu = np.min(eigsQ)
        GradF = np.zeros((m, n))
        for i in range(m):
            GradF[i, :] = Q @ X[i, :] + c

        # Gradient matrix:
    H = np.zeros((m, m))
    for i in range(m):
        for j in range(n):
            H[i, j] = GradF[i, :].T @ GradF[j, :] / (mu ** 2) - X[i, :].T @ X[j, :]
    return (X, d, H)


    #(Data dimension, Number of samples)
n = 2; m = 4
    #Generate data
(X, d, H) = Data_generator(m, n, X_distri='normal', F_form='quadratic')
    #Solve QP on a unit simplex
model = gp.Model("QPoverSimplex")
alpha = model.addMVar(shape=m, lb=np.zeros(m), vtype=GRB.CONTINUOUS, name="alpha")
model.addConstr(gp.quicksum(alpha) == 1, "simplex constraint")
model.setObjective(alpha.T@d + alpha.T@H@alpha, GRB.MINIMIZE)
model.optimize()
    #Solvability
if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    print(f"alpha = {alpha.x}")
    print(f"Objective value = {model.objVal}")
else:
    print("No solution found or optimization was not successful.")


