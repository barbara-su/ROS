from .utils import get_matrix, sgd, sample_discrete_matrix_choices
import numpy as np
from scipy.sparse import csr_matrix, linalg, identity
import logging
import time


def f(X, A):
    Inter = np.trace(X @ A @ X.T)
    return Inter


def f_and_g(X, A):
    f_C = f(X, A)
    grad_C = 2 * X @ A
    return f_C, np.array(grad_C)


def gp(args, graph):
    W = get_matrix(args, graph)
    
    t0 = time.time()
    X0 = np.random.rand(args.k, args.n)
    X0 = X0 / X0.sum(axis=0)
    X0 = X0 / args.soft
    
    now = f(X0, W)
    best_X = X0
    L0 = linalg.norm(W + W.T, ord='fro')

    X_final = sgd(args, X0, W, L0)
    if type(X_final) == float:
        runtime = time.time() - t0
        logging.info("Time: {}".format(np.inf))
        if args.save:
            with open("./res/gp_time.txt", "a") as ff:
                ff.write("numpy.inf, ")
        return np.inf

    # mod3_inter_Exp = np.trace(X_final @ W @ X_final.T)
    # logging.info("Expectation = " + str(mod3_inter_Exp))

    max_iter = args.max_iter
    best_inter = np.inf
    for iter in range(max_iter):
        xt = sample_discrete_matrix_choices(X_final)
        Xt = np.zeros((args.k, args.n))
        if X_final.shape[1] != 0:
            Xt[xt, np.arange(args.n)] = 1
        mod3_inter = np.trace(Xt @ W@ Xt.T)
        if mod3_inter < best_inter:
            best_inter = mod3_inter
            best_X = Xt

    runtime = time.time() - t0
    logging.info("Time: {}".format(runtime))
    if args.save:
        with open("./res/gp_time.txt", "a") as ff:
            ff.write(str(round(runtime, 3)) + ", ")
    
    best_x = best_X.argmax(axis=0)
    return best_x