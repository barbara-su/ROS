from .utils import get_matrix, mirror_gd, sample_discrete_matrix_choices
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


def md(args, graph):
    W = get_matrix(args, graph)
    
    t0 = time.time()
    X0 = np.random.rand(args.k, args.n)
    X0 = X0 / X0.sum(axis=0)
    X0 = X0 / args.soft
    
    now = f(X0, W)
    best = now
    best_X = X0
    obj = lambda x: f(x, W)
    f_and_grad = lambda x: f_and_g(x, W)
    L0 = linalg.norm(W + W.T, ord='fro')

    X_final = mirror_gd(args, obj, f_and_grad, X0, W, L0)


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
        with open("./res/md_time.txt", "a") as ff:
            ff.write(str(runtime) + ", ")
    best_x = best_X.argmax(axis=0)
    return best_x