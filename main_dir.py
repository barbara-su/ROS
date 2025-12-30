#!/usr/bin/env python3
"""
main_dir.py

Batch runner for MAX-CUT solvers (including ROS).

- Scans ONE directory (non-recursive) for files whose basename starts with q_prefix (default "Q")
  and ends with q_ext (default ".npy").
- Loads each Q, builds a weighted networkx graph (assumes Laplacian: w_ij = -Q_ij for i!=j).
- Runs the chosen algorithm.
- Computes maxcut via postprocess(result, graph).
- Writes one JSON per Q file into output_dir (exactly as passed).

Example:
  python main_dir.py --input_dir graphs_dir --output_dir out_dir --alg ros --seed 42
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import networkx as nx

from utils import postprocess, set_random_seed
from add_parser import add_parse

from pignn.pignn import pignn
from optimization.md import md
from optimization.gp import gp
from ros_vanilla.ros_vanilla import ros_vanilla
from goemans_williamson.gw import gw
from ros.ros import ros
from genetic.genetic import genetic
from bqp.bqp import bqp


def load_q(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(str(path))
    raise ValueError(f"Unsupported Q file extension: {path} (expected .npy)")


def q_to_nx_graph(Q: np.ndarray) -> nx.Graph:
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError(f"Q must be square. Got shape={Q.shape}")

    n = Q.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            w = float(-Q[i, j])
            if w != 0.0:
                G.add_edge(i, j, weight=w)

    return G


def run_alg(args, graph: nx.Graph):
    if args.alg == "pignn":
        return pignn(args, graph)
    if args.alg == "md":
        return md(args, graph)
    if args.alg == "gp":
        return gp(args, graph)
    if args.alg == "ros_vanilla":
        return ros_vanilla(args, graph)
    if args.alg == "gw":
        return gw(args, graph)
    if args.alg == "ros":
        return ros(args, graph)  # may return (solution, time_seconds)
    if args.alg == "genetic":
        return genetic(args, graph)
    if args.alg == "from_file":
        return torch.load(args.sol_dir)
    if args.alg == "bqp":
        return bqp(args, graph)
    if args.alg == "ANYCSP":
        from anycsp.anycsp import ac
        return ac(args, graph)

    raise NotImplementedError(f"Not Implemented Algorithm: {args.alg}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.getLogger().setLevel(logging.INFO)
    log = logging.getLogger("main_dir")

    batch = argparse.ArgumentParser(add_help=False)
    batch.add_argument("--input_dir", type=str, required=True, help="Directory containing Q* graph files")
    batch.add_argument("--output_dir", type=str, required=True, help="Directory to write results")
    batch.add_argument("--q_prefix", type=str, default="Q", help="Only process files whose basename starts with this")
    batch.add_argument("--q_ext", type=str, default=".npy", help="Only process files with this extension")

    batch_args, remaining = batch.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = add_parse()

    # simplest path handling: DO NOT resolve(), just expand ~ and use as-is
    in_dir = Path(batch_args.input_dir).expanduser()
    out_dir = Path(batch_args.output_dir).expanduser()

    if not in_dir.is_dir():
        raise ValueError(f"Not a directory: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Fixed seed from CLI (you will pass --seed 42 in sbatch)
    set_random_seed(args.seed)

    args.TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.TORCH_DTYPE = torch.float32

    pattern = f"{batch_args.q_prefix}*{batch_args.q_ext}"
    files = sorted(in_dir.glob(pattern))
    files = [p for p in files if p.is_file() and p.name.startswith(batch_args.q_prefix)]

    if not files:
        log.warning(f"No files found under {in_dir} matching prefix={batch_args.q_prefix} ext={batch_args.q_ext}")
        return

    log.info(f"Found {len(files)} Q files to process.")
    log.info(f"Algorithm: {args.alg}")
    log.info(f"Output directory: {out_dir}")
    log.info(f"Using fixed seed: {args.seed}")

    for idx, q_path in enumerate(files, start=1):
        log.info("============================================================")
        log.info(f"[{idx}/{len(files)}] Q file: {q_path}")
        out_path = out_dir / f"{q_path.stem}.json"
        
        if out_path.exists():
            log.info(f"[skip existing] {out_path}")
            continue
        
        try:
            Q = load_q(q_path)
            args.n = int(Q.shape[0])
            graph = q_to_nx_graph(Q)
            result = run_alg(args, graph)
            

            # support (solution, time_seconds)
            alg_time_seconds = None
            if isinstance(result, tuple) and len(result) == 2:
                result, alg_time_seconds = result

            if isinstance(result, float):
                maxcut = float(np.inf)
            else:
                maxcut = float(postprocess(result, graph))

            payload = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "alg": str(args.alg),
                "seed": int(args.seed),
                "q_file": str(q_path),
                "n": int(Q.shape[0]),
                "maxcut": maxcut,
            }
            if alg_time_seconds is not None:
                payload["alg_time_seconds"] = float(alg_time_seconds)

            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)

            log.info(f"FINAL RESULT: {maxcut}")
            log.info(f"Wrote: {out_path}")

        except Exception as e:
            log.exception(f"Failed on {q_path}: {e}")

    log.info("Batch complete.")


if __name__ == "__main__":
    main()
