import numpy as np
import time
import logging
from .utils import get_gnn, run_gnn_training, gen_q_matrix
import torch

def pignn(args, graph):
    q_torch = gen_q_matrix(args, graph)
    gnn_start = time.time()
    if args.n > 1e5:
        dim_embedding = round(np.sqrt(args.n))
    else:
        dim_embedding = round(np.cbrt(args.n))
    hidden_dim = round(dim_embedding / 2)
    opt_params = {'lr': args.lr}
    gnn_hypers = {
        'dim_embedding': dim_embedding,
        'hidden_dim': hidden_dim,
        'dropout': 0.0,
        'number_classes': 1,
        'number_epochs': args.epochs,
        'tolerance': args.tol,
        'patience': args.patience
    }
    print(opt_params, gnn_hypers)
    net, optimizer, edges, edges_weight, inputs = get_gnn(args, gnn_hypers, opt_params, args.TORCH_DEVICE, args.TORCH_DTYPE, graph)
    logging.info("Running GNN...")
    _, epoch, final_bitstring, best_bitstring, best_val = run_gnn_training(
    q_torch, inputs, graph, edges, edges_weight, net, optimizer, gnn_hypers['number_epochs'],
    gnn_hypers['tolerance'], gnn_hypers['patience'])


    gnn_time = time.time() - gnn_start

    # Return (solution, time_seconds) instead of writing anything
    return best_bitstring, float(gnn_time)