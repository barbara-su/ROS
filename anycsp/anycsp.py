import torch
import numpy as np
import sys
# sys.path.append("/data1/qyq/Benchmark-for-Max-k-Cut/anycsp")
from anycsp.src.csp.csp_data import CSP_Data
from anycsp.src.model.model import ANYCSP

from argparse import ArgumentParser
from anycsp.src.data.dataset import File_Dataset, File_Dataset_fromgraph




def ac(args, graph):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dict_args = vars(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    name = 'model' if args.checkpoint is None else f'{args.checkpoint}'
    model = ANYCSP.load_model(args, args.model_dir, name)
    model.eval()
    model.to(device)

    dataset = File_Dataset_fromgraph(args, graph)

    num_solved = 0
    num_total = len(dataset)
    import time
    t0 = time.time()
    for data in dataset:
        max_val = data.constraints['ext'].cst_neg_mask.int().sum().cpu().numpy()
        if args.num_boost > 1:
            data = CSP_Data.collate([data for _ in range(args.num_boost)])
        data.to(device)

        if args.verbose:
            print(f'Solving {args.gset}:')
        #with torch.cuda.amp.autocast():
        with torch.inference_mode():
            data = model(
                data,
                args.network_steps,
                return_all_assignments=True,
                return_log_probs=False,
                stop_early=True,
                verbose=args.verbose,
                keep_time=True,
                timeout=args.timeout,
            )
        best_per_run = data.best_num_unsat
        mean_best = best_per_run.mean()
        best = best_per_run.min().cpu().numpy()
        solved = best == 0
        num_solved += int(solved)
        best_cut_val = max_val - best
    print(f'Solved {100 * num_solved / num_total:.2f}%')
    elapsed = time.time() - t0
    A = data.best_assignment if hasattr(data, "best_assignment") else data.all_assignments[:, [0]]
    if A.ndim == 2: A = A[:, 0]  # handle [num_var, T] case
    b = int(torch.argmin(data.best_num_unsat).item()) if args.num_boost > 1 else 0
    if A.numel() == data.num_val: A = data.hard_assign_max(A.float()).view(-1)  # only if value-level
    sol = A.view(-1)[data.batch == b].detach().cpu().numpy()  # keep best boosted instance only
    
    # if args.save: open("./res/" + args.alg + "_time.txt", "a").write(f"{elapsed:.3f}, ")
    return sol, float(elapsed)