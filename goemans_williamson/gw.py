from .utils import get_matrix
import cvxpy as cp
import scipy as sp
import numpy as np
import networkx as nx
import pdb
import time


def gw(args, graph):
    W = get_matrix(args, graph)
    t0 = time.time()
    num_of_nodes = args.n
    X = cp.Variable((num_of_nodes, num_of_nodes), symmetric=True)

    constraints = [X >> 0]
    constraints += [X[i, i] == 1 for i in range(num_of_nodes)]
    objective = cp.Minimize(sum(X[i, j] * W[i, j] for i, j in graph.edges))
    prob = cp.Problem(objective, constraints)
    prob.solve()
    X_solution = X.value
    x_projected = sp.linalg.sqrtm(X_solution)
    u = np.random.randn(num_of_nodes)
    cut = np.sign(x_projected @ u)
    cut = cut.real
    cut = cut * 0.5 + 0.5
    runtime = time.time() - t0
    # Return (solution, time_seconds) instead of writing a file (match ros)
    return cut, float(runtime)
