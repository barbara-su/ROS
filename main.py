import torch
import numpy as np
import networkx as nx
import logging
import sys
from utils import generate_graph, postprocess, set_random_seed
from add_parser import add_parse
from pignn.pignn import pignn
from optimization.md import md
from optimization.gp import gp
from ros_vanilla.ros_vanilla import ros_vanilla 
from goemans_williamson.gw import gw
from ros.ros import ros
from genetic.genetic import genetic
from bqp.bqp import bqp


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.getLogger().setLevel(logging.INFO)
    args = add_parse()
    set_random_seed(args.seed)
    args.TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.TORCH_DTYPE = torch.float32
    graph = generate_graph(args)
    
    if args.alg == "pignn":
        result = pignn(args, graph)
    elif args.alg == "md":
        result = md(args, graph)
    elif args.alg == "gp":
        result = gp(args, graph)
    elif args.alg == "ros_vanilla":
        result = ros_vanilla(args, graph)
    elif args.alg == "gw":
        result = gw(args, graph)
    elif args.alg == "ros":
        result = ros(args, graph)
    elif args.alg == "genetic":
        result = genetic(args, graph)
    elif args.alg == "from_file":
        result = torch.load(args.sol_dir)
    elif args.alg == "bqp":
        result = bqp(args, graph)
    elif args.alg == "ANYCSP":
        from anycsp.anycsp import ac
        result = ac(args, graph)
    else:
        print(args.alg)
        raise "Not Implemented Algorithm"
    
    if type(result) == float:
        if args.save:
            with open("./res/" + args.alg + "_value.txt", "a") as f:
                f.write("numpy.inf, ")
        print("FINAL RESULT: " + str(np.inf))
    else:    
        maxcut = postprocess(result, graph)
        if args.save:
            with open("./res/" + args.alg + "_value.txt", "a") as f:
                f.write(str(maxcut) + ", ")
        print("FINAL RESULT: " + str(maxcut))
