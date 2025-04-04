import numpy as np
import gurobipy as gp
from gurobipy import GRB
#np.random.seed(12)

#Data_generator: generates data needed for the duality gap optimization problem
    #Inputs:
        #m: number of data points (x_i, grad F(x_i))
        #n: vector length
        #X_distri:
            #Default: Randomly generates m data points X_i ~ N(0, I_n)
            #first_order_GD: X_0 ~ N(0, I_n), and each subsequent point X_k is updated by GD
        #F_form: F(x):= f_lambda1 * f + g_lambda1 * g
            #quadratic: f(x) = (0.5 * x^TQx + c^Tx); g(x) = 0.5 * ||x||^2
        #lambda1s: coefficients for functions f and g
    #Outputs:
        #X: Data matrix of dimension m x n
        #d: Data vector
        #c: Linear objective coefficient of dimension n
        #f: L_f smooth and mu-strongly convex function
        #g: L_g smooth and mu-strongly convex function
def Data_generator(m, n, X_distri, F_form, lambda1s):
        # Form of objective function
    if F_form == "quadratic":
        c = np.random.randn(n)
        A = np.random.randn(n, n)
        Q = A.T @ A
        f = Q; g = np.identity(n)
        f_lambda1 = lambda1s[0]; g_lambda1 = lambda1s[1]
        GradF_lambda1 = f_lambda1 * f + g_lambda1 * g
        eigs = np.linalg.eigvals(GradF_lambda1)
        L = np.max(eigs)

        # Data matrix
    X = np.random.randn(m, n)
    if X_distri == 'first_order_GD':
        X_opt = - np.linalg.inv(GradF_lambda1) @ (f_lambda1 * c) #Optimal x vector
        X_cur = X[0,:]
        t = 1
        while np.linalg.norm(X_cur - X_opt) > eps and t < m:
            GradF_cur = GradF_lambda1 @ X_cur + (f_lambda1 * c)
            X_cur = X_cur - GradF_cur / L
            X[t, :] = X_cur
            t += 1
    #print('X:', X)
    d = np.linalg.norm(X, axis=1) ** 2 # Data vector: d[i] = ||x_i||^2
    return (X, d, c, f, g)

#gap: computes the duality gap U_F - L_F
    #Inputs:
        #alpha: weights for data points
        #GradF: gradient matrix for computing H
        #d: data vector
        #L: smoothness constant
        #mu: strongly convex constant
        #X: data matrix
    #Outputs: duality gap
def gap(alpha, GradF, d, L, mu, X):
        # Gradient matrix: H = G/(mu**2)-M
    H = GradF @ GradF.T / (mu ** 2) - X @ X.T
    return (L-mu)/2 * (alpha.T @ d + alpha.T @ H @ alpha)

#gap_solver: Use Gurobi to solve the gap optimization problem, i.e: Minimize the gap on the unit simplex
    #Inputs:
        #m: number of data points (x_i, grad F(x_i))
        #n: vector length
        #eps: error tolerance
        #X_distri:
            #Default: Randomly generates m data points X_i ~ N(0, I_n)
            #first_order_GD: X_0 ~ N(0, I_n), and each subsequent point X_k is updated by GD
        #lambda2s: Trade-off coefficients at new point lambda2
    #Outputs:
        #Gurobi running time
        #duality gap
        #sparsity of alpha
def gap_solver(m, n, eps, X_distri, lambda2s):
        # Generate data
    (X, d, c, f, g) = Data_generator(m, n, X_distri=X_distri, F_form='quadratic', lambda1s=(1, 1))
    #print(X)
    f_lambda2 = lambda2s[0]; g_lambda2 = lambda2s[1]
    GradF = X @ (f_lambda2 * f.T + g_lambda2 * g.T) + (f_lambda2 * c)
    eigs = np.linalg.eigvals(f_lambda2 * f + g_lambda2 * g)
    L = np.max(eigs); mu = np.min(eigs)
    #print(L,mu)

        # Solve QP on the unit simplex
    model = gp.Model("QPoverSimplex")
    model.setParam("OutputFlag", 0)
    alpha = model.addMVar(shape=m, lb=np.zeros(m), vtype=GRB.CONTINUOUS, name="alpha")
    model.addConstr(gp.quicksum(alpha) == 1, "simplex constraint")
    model.setObjective(gap(alpha, GradF, d, L, mu, X), GRB.MINIMIZE)
    model.optimize()
        # Solvability
    if model.status == GRB.OPTIMAL:
        #print("Optimal solution found:")
        alpha = alpha.x
        #print(f"alpha = ", alpha)
        alpha_non_zero = sum(1 for i in alpha if (i > eps))
        return (model.Runtime, model.objVal, alpha_non_zero)
    else:
        print("No solution found or optimization was not successful.")

# Function to generate a nested LaTeX table given a list of values.
# Function to create an inner LaTeX table for a given (n, m) pair.
# The inner table has columns for the lambda tuple and its corresponding metrics.
def create_inner_table(lambda_tuples, times, gaps, sparsities):
    lines = [
        "\\begin{tabular}{cccc}",
        "  \\hline",
        "   $(\lambda_2, 2-\lambda_2)$  & Time & Duality Gap & Alpha Sparsity \\\\",
        "  \\hline"
    ]
    for lam, t, gap, sparsity in zip(lambda_tuples, times, gaps, sparsities):
        lam_str = f"({lam[0]:.3f}, {lam[1]:.3f})"
        lines.append(f"  {lam_str} & {t:.3f} & {gap:.3f} & {sparsity:.3f} \\\\")
    lines.append("  \\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)

#table_generator: Generates a table in latex displaying statistics (model running time, duality gap, alpha sparsity)
#   for different data dimensions
def table_generator(file_name, m_start, m_end, n_start, n_end, X_distri, lambda2s):
    with open(file_name, "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("  \\centering\n")
        # Outer table has three columns: n, m, and the nested metrics table.
        f.write("  \\begin{tabular}{ccc}\n")
        f.write("    \\hline\n")
        f.write("    $n$ & $m$ & Metrics \\\\\n")
        f.write("    \\hline\n")

            #(Data dimension, Number of samples)
        for n in range(n_start, n_end):
            for m in range(m_start, m_end):
                time_values = []
                gap_values = []
                sparsity_values = []
                #f_lambda2s = []
                for (f_lambda2, g_lambda2) in lambda2s:
                    timeSum = 0
                    duality_gapSum = 0
                    alpha_sparsitySum = 0
                    #Iterate for 3 times
                    for k in range(3):
                        time, duality_gap, alpha_sparsity = gap_solver(m, n, eps, X_distri, (f_lambda2, g_lambda2))
                        timeSum += time
                        duality_gapSum += duality_gap
                        alpha_sparsitySum += alpha_sparsity
                    time = timeSum / 3
                    duality_gap = duality_gapSum / 3
                    alpha_sparsity = alpha_sparsitySum / 3
                        #Append each value
                    time_values.append(time)
                    gap_values.append(duality_gap)
                    sparsity_values.append(alpha_sparsity)
                    #f_lambda2s.append(())

                    # Create the nested inner table.
                inner_table = create_inner_table(lambda2s, time_values, gap_values, sparsity_values)
                f.write(f"    {n} & {m} & {inner_table} \\\\\n")

                print('finished:',(n,m))

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Table of computation time, duality gap, and alpha sparsity for various (n, m) values.}\n")
        f.write("\\label{tab:metrics}\n")
        f.write("\\end{table}\n")
    print("LaTeX table has been written to table.tex")


    #Generate statistics
eps = 10 ** (-5)
#lambda2s = []
#for i in range(3):
#    lambda2s.append((1, 2-1))
#table_generator("alpha_table_random.tex", 5, 10, 5, 8, 'normal', lambda2s)
#table_generator("alpha_table_FOGD.tex", 20, 30, 10, 15, 'first_order_GD', lambda2s)
lambda2s = []
for i in np.arange(1, 1.4, 0.1):
    lambda2s.append((i, 2-i))
table_generator("alpha_table_approximateGD.tex", 7, 10, 5, 8, 'first_order_GD', lambda2s)

