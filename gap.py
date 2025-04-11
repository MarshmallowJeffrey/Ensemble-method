import numpy as np
import gurobipy as gp
from gurobipy import GRB
np.random.seed(12)

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
    f_lambda1 = lambda1s[0]; g_lambda1 = lambda1s[1]
        # Form of objective function
    if F_form == "quadratic":
        c = np.random.randn(n)
        A = np.random.randn(n, n)
            #Counter example for sparsity = 1
        #for k in range(100):
        #    A += np.random.randn(n, n)
        #A /= 100
        Q = A.T @ A
        f = Q; g = np.identity(n)
        GradF_lambda1 = f_lambda1 * f + g_lambda1 * g
        eigs = np.linalg.eigvals(GradF_lambda1)
        L = np.max(eigs); mu = np.min(eigs)
            # Data matrix
        X_opt = - np.linalg.inv(GradF_lambda1) @ (f_lambda1 * c)  # Optimal x vector
        # print('X_opt:', X_opt)
            # Sampling with mean the optimal solution
        X = np.random.randn(m, n) + X_opt
            #Stopping criterion
        GradFx_lambda1 = GradF_lambda1 @ X[0, :] + (f_lambda1 * c)
    elif F_form == "logistic":
            #Form of objective function
        p = 10
        y = np.random.choice([-1, 1], size=p, replace=True)
        Z = np.random.normal(loc=y[:, None], scale=1, size=(len(y), n))
        mu = g_lambda1; sigma_max = np.linalg.norm(Z, ord=2); L = f_lambda1 * (sigma_max**2) / (4*len(y)) + g_lambda1
            #Data matrix(no closed-form optimal solution)
        X = np.random.randn(m, n)
            # Stopping criterion
        sigmoid_sum = 0
        for i in range(p):
            sigmoid_sum += - (1 - 1/(1+np.exp(-y[i]*Z[i].T @ X[0,:])))*y[i]*Z[i]
        GradFx_lambda1 = sigmoid_sum / p + X[0,:]

        # Initializations
    X_cur = X[0, :]
    t = 1
    W_cur = X_cur
    tau = np.sqrt(mu / L)
    stop_cri = np.linalg.norm(GradFx_lambda1) > eps
        # Apply first-order algorithms: GD and AGD, to solve it
    while stop_cri and t < m:
        if X_distri == 'first_order_GD':
            if F_form == "quadratic":
                GradFx_lambda1 = GradF_lambda1 @ X_cur + (f_lambda1 * c)
            elif F_form == "logistic":
                sigmoid_sum = 0
                for i in range(p):
                    sigmoid_sum += - (1 - 1 / (1 + np.exp(-y[i] * Z[i].T @ X_cur))) * y[i] * Z[i]
                GradFx_lambda1 = sigmoid_sum / p + X_cur
            X_cur = X_cur - GradFx_lambda1 / L
        elif X_distri == 'AGD':
                # Mixing sequence Y_t
            Y_cur = 1 / (1 + tau) * X_cur + tau / (1 + tau) * W_cur
                # Dual sequence W_t
            if F_form == "quadratic":
                GradFx_lambda1 = GradF_lambda1 @ Y_cur + (f_lambda1 * c)
            elif F_form == "logistic":
                sigmoid_sum = 0
                for i in range(p):
                    sigmoid_sum += - (1 - 1 / (1 + np.exp(-y[i] * Z[i].T @ Y_cur))) * y[i] * Z[i]
                GradFx_lambda1 = sigmoid_sum / p + Y_cur
            W_cur = (1 - tau) * W_cur + tau * (Y_cur - GradFx_lambda1 / mu)
                # Primal sequence X_t
            X_cur = Y_cur - GradFx_lambda1 / L
        else:
            break
        X[t, :] = X_cur
        t += 1
        stop_cri = np.linalg.norm(GradFx_lambda1) > eps

    #print('X:', X)
    d = np.linalg.norm(X, axis=1) ** 2 # Data vector: d[i] = ||x_i||^2
    if F_form == "quadratic":
        return (X, d, c, f, g)
    elif F_form == "logistic":
        return (X, d, y, Z)

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
def gap_solver(m, n, eps, X_distri, F_form, lambda2s):
    f_lambda2 = lambda2s[0]; g_lambda2 = lambda2s[1]
        # Generate data
    if F_form == 'quadratic':
        (X, d, c, f, g) = Data_generator(m, n, X_distri=X_distri, F_form=F_form, lambda1s=(1, 1))
        GradF = X @ (f_lambda2 * f.T + g_lambda2 * g.T) + (f_lambda2 * c)
        eigs = np.linalg.eigvals(f_lambda2 * f + g_lambda2 * g); L = np.max(eigs); mu = np.min(eigs)
    elif F_form == 'logistic':
        (X, d, y, Z) = Data_generator(m, n, X_distri=X_distri, F_form=F_form, lambda1s=(1, 1))
        GradF = np.zeros([m, m])
        GradFMat = np.zeros([m, n])
        for i in range(m):
            sigmoidX_i_sum = 0
            for k in range(len(y)):
                sigmoidX_i_sum += - (1 - 1 / (1 + np.exp(-y[k] * Z[k].T @ X[i, :]))) * y[k] * Z[k]
            GradFMat[i] = f_lambda2 * sigmoidX_i_sum + g_lambda2 * X[i, :]

             #Update GradF(i,j)
        for i in range(m):
            for j in range(m):
                GradF[i,j] = GradFMat[i].T @ GradFMat[j]
        mu = g_lambda2; sigma_max = np.linalg.norm(Z, ord=2); L = f_lambda2 * (sigma_max ** 2) / (4 * len(y)) + g_lambda2
    #print(X)

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
def create_inner_table(lambda_tuples, times, gaps, sparsities, param):
    if param:
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
    else:
        lines = []
        lam_str = f"({lambda_tuples[0][0]:.3f}, {lambda_tuples[0][1]:.3f})"
        lines.append(f"  {lam_str} & {times:.3f} & {gaps:.3f} & {sparsities:.3f} \\\\")
    return "\n".join(lines)

#table_generator: Generates a table in latex displaying statistics (model running time, duality gap, alpha sparsity)
#   for different data dimensions
def table_generator(file_name, m_start, m_end, n_start, n_end, X_distri, F_form, lambda2s, param):
    with open(file_name, "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("  \\centering\n")
        # Outer table has three columns: n, m, and the nested metrics table.
        if param:
            f.write("  \\begin{tabular}{ccc}\n")
            f.write("    \\hline\n")
            f.write("    $n$ & $m$ & Metrics \\\\\n")
        else:
            f.write("  \\begin{tabular}{cccccc}\n")
            f.write("    \\hline\n")
            f.write("    $n$ & $m$ & $(\lambda_2, 2-\lambda_2)$ & Time & Duality Gap & Alpha sparsity \\\\\n")
        f.write("    \\hline\n")

            #(Data dimension, Number of samples)
        for n in range(n_start, n_end):
            for m in range(m_start, m_end):
                time_values = []
                gap_values = []
                sparsity_values = []
                    #No parametrization
                if not param:
                    timeSum = 0
                    duality_gapSum = 0
                    alpha_sparsitySum = 0
                    for k in range(3):
                        time, duality_gap, alpha_sparsity = gap_solver(m, n, eps, X_distri, F_form, (lambda2s[0][0], lambda2s[0][1]))
                        timeSum += time
                        duality_gapSum += duality_gap
                        alpha_sparsitySum += alpha_sparsity
                    time = timeSum / 3
                    duality_gap = duality_gapSum / 3
                    alpha_sparsity = alpha_sparsitySum / 3
                        # Create the nested inner table.
                    inner_table = create_inner_table(lambda2s, time, duality_gap, alpha_sparsity, False)

                    #Parametrization
                else:
                    for (f_lambda2, g_lambda2) in lambda2s:
                        timeSum = 0
                        duality_gapSum = 0
                        alpha_sparsitySum = 0
                        #Iterate for 3 times
                        for k in range(3):
                            time, duality_gap, alpha_sparsity = gap_solver(m, n, eps, X_distri, F_form,(f_lambda2, g_lambda2))
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
                        # Create the nested inner table.
                    inner_table = create_inner_table(lambda2s, time_values, gap_values, sparsity_values, True)
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
        #random + first-order-GD
lambda2s = []
for i in range(3):
    lambda2s.append((1, 2-1))
#table_generator("alpha_table_random.tex", 5, 10, 5, 6, 'normal', 'logistic', lambda2s, False)
table_generator("alpha_table_FOGD.tex", 10, 13, 5, 10, 'first_order_GD', 'logistic', lambda2s, False)
#table_generator("alpha_table_AGD.tex", 10, 13, 5, 10, 'AGD', 'quadratic', lambda2s, False)

        #Parametrization optimization
#lambda2s = []
#for i in np.arange(1, 1.4, 0.1):
#   lambda2s.append((i, 2-i))
#table_generator("alpha_table_approximateAGD.tex", 7, 10, 5, 8, 'AGD', 'quadratic', lambda2s, True)

#time, duality_gap, alpha_sparsity = gap_solver(10, 10, eps, 'first_order_GD', (1, 1))

