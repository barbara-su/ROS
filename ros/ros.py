import time
from .utils import get_gnn_tuning, get_matrix, run_gnn_tuning, sample_one_hot
import torch


def ros(args, graph):
    W = get_matrix(args, graph)
    gnn_start = time.time()
    net, optimizer, edges, edges_weight, inputs = get_gnn_tuning(args, graph)
    print("Load gcn_model_ood_k" + str(args.k) + ".pth")
    state = torch.load(
        "gcn_model_ood_k" + str(args.k) + ".pth",
        weights_only=True,
        map_location=args.TORCH_DEVICE,
    )
    net.load_state_dict(state)
    best_solution_relaxed = run_gnn_tuning(args, W, edges, edges_weight, net, optimizer, args.epochs, args.tol, args.patience, inputs)
    # Expectation = torch.trace(best_solution_relaxed @ W @ best_solution_relaxed.T)
    # print("Expectation = " + str((W.sum() - Expectation).item()))
    best_val = torch.inf
    for _ in range(args.max_iter):
        Xt = sample_one_hot(best_solution_relaxed)
        val = torch.trace(Xt @ W @ Xt.T)
        if val < best_val:
            best_val = val
            best_solution = Xt
    best_solution = best_solution.argmax(axis=0)
    gnn_time = time.time() - gnn_start
    if args.save:
        with open("./res/ros_time.txt", "a") as ff:
            ff.write(str(round(gnn_time, 3)) + ", ")
    return best_solution
    
