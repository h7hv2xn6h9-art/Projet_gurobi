#!/usr/bin/env python3
"""
videos.py - Solve the Hash Code 2017 "Streaming Videos" cache placement problem
using a MIP model with gurobipy only.

The script:
  * reads an input dataset file
  * builds and solves a MIP model
  * writes videos.mps with the model
  * writes videos.out with the solution (cache -> videos)
"""

import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple
import gurobipy as gp
from gurobipy import GRB


@dataclass
class Request:
    video: int
    endpoint: int
    count: int


@dataclass
class Data:
    nb_videos: int
    nb_endpoints: int
    nb_requests: int
    nb_caches: int
    cache_capacity: int
    video_sizes: List[int]
    dc_latency: List[int]
    endpoint_caches: List[List[Tuple[int, int]]]
    requests: List[Request]


def parse_input(path: str) -> Data:
    print(f"Reading input file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        # First line: V E R C X
        first = f.readline().strip().split()
        nb_videos = int(first[0])
        nb_endpoints = int(first[1])
        nb_requests = int(first[2])
        nb_caches = int(first[3])
        cache_capacity = int(first[4])

        # Second line: sizes of all videos
        sizes = list(map(int, f.readline().strip().split()))
        if len(sizes) != nb_videos:
            raise ValueError("Unexpected number of video sizes")

        # Endpoints description
        dc_latency = [0] * nb_endpoints
        endpoint_caches: List[List[Tuple[int, int]]] = [[] for _ in range(nb_endpoints)]

        for e in range(nb_endpoints):
            parts = f.readline().strip().split()
            if len(parts) != 2:
                raise ValueError("Invalid endpoint header line")
            l_de = int(parts[0])
            k_e = int(parts[1])
            dc_latency[e] = l_de
            conn = []
            for _ in range(k_e):
                c_str, l_str = f.readline().strip().split()
                conn.append((int(c_str), int(l_str)))
            endpoint_caches[e] = conn

        # Requests: possibly several lines with same (video, endpoint)
        raw_requests: Dict[Tuple[int, int], int] = {}
        for _ in range(nb_requests):
            line = f.readline()
            if not line:
                break
            v_str, e_str, n_str = line.strip().split()
            v = int(v_str)
            e = int(e_str)
            n = int(n_str)
            key = (v, e)
            raw_requests[key] = raw_requests.get(key, 0) + n

    requests = [Request(video=v, endpoint=e, count=n)
                for (v, e), n in raw_requests.items()]

    print(
        f"Parsed V={nb_videos}, E={nb_endpoints}, R={len(requests)}, "
        f"C={nb_caches}, X={cache_capacity}"
    )

    return Data(
        nb_videos=nb_videos,
        nb_endpoints=nb_endpoints,
        nb_requests=len(requests),
        nb_caches=nb_caches,
        cache_capacity=cache_capacity,
        video_sizes=sizes,
        dc_latency=dc_latency,
        endpoint_caches=endpoint_caches,
        requests=requests,
    )


def build_model(data: Data):
    print("Building optimization model with gurobipy...")
    m = gp.Model("videos_streaming")

    V = range(data.nb_videos)
    C = range(data.nb_caches)

    # Decision variables: x[v, c] = 1 if video v is stored in cache c
    x = m.addVars(
        data.nb_videos,
        data.nb_caches,
        vtype=GRB.BINARY,
        name="x",
    )

    # Capacity constraints for each cache
    for c in C:
        m.addConstr(
            gp.quicksum(data.video_sizes[v] * x[v, c] for v in V)
            <= data.cache_capacity,
            name=f"cap_{c}",
        )

    # For each request and each "useful" cache, create y[r, c] and linking constraints
    y = {}
    weights = {}

    for r_idx, req in enumerate(data.requests):
        e = req.endpoint
        v = req.video
        n_req = req.count
        lat_dc = data.dc_latency[e]

        candidate_caches = []
        for c, lat_ec in data.endpoint_caches[e]:
            saving = lat_dc - lat_ec
            if saving <= 0:
                continue  # never useful to route through this cache
            y_var = m.addVar(vtype=GRB.BINARY, name=f"y_{r_idx}_{c}")
            y[(r_idx, c)] = y_var
            weights[(r_idx, c)] = n_req * saving
            candidate_caches.append(c)

            # Linking: can serve from cache c only if v is stored on c
            m.addConstr(y_var <= x[v, c], name=f"link_{r_idx}_{c}")

        if candidate_caches:
            # At most one cache chosen for this request
            m.addConstr(
                gp.quicksum(y[(r_idx, c)] for c in candidate_caches) <= 1,
                name=f"req_{r_idx}",
            )

    # Objective: maximize total latency saving
    if weights:
        obj = gp.quicksum(weights[idx] * y[idx] for idx in y.keys())
    else:
        obj = 0.0

    m.setObjective(obj, GRB.MAXIMIZE)

    # Ask for a good optimality gap
    m.Params.MIPGap = 5e-3

    m.update()

    print("Writing MPS file videos.mps ...")
    m.write("videos.mps")

    return m, x


def solve_model(m: gp.Model):
    print("Optimizing model...")
    m.optimize()

    if m.status == GRB.OPTIMAL:
        print("Optimal solution found.")
    elif m.status == GRB.SUBOPTIMAL:
        print("Suboptimal solution (time limit or gap) but model finished.")
    elif m.status == GRB.INFEASIBLE:
        print("Model infeasible.")
    else:
        print(f"Gurobi finished with status {m.status}.")

    # Print final gap for information
    try:
        gap = m.MIPGap
        print(f"Final MIP gap: {gap:.6f}")
    except AttributeError:
        pass


def extract_solution(data: Data, x) -> Dict[int, List[int]]:
    print("Extracting solution (cache -> videos)...")
    cache_videos: Dict[int, List[int]] = {c: [] for c in range(data.nb_caches)}

    for v in range(data.nb_videos):
        for c in range(data.nb_caches):
            val = x[v, c].X
            if val > 0.5:
                cache_videos[c].append(v)

    # Remove caches with no videos
    cache_videos = {c: vids for c, vids in cache_videos.items() if vids}
    return cache_videos


def write_output(cache_videos: Dict[int, List[int]], path: str = "videos.out"):
    print(f"Writing solution file: {path}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(len(cache_videos)) + "\n")
        for c in sorted(cache_videos.keys()):
            vids = sorted(cache_videos[c])
            line = " ".join(str(v) for v in vids)
            f.write(f"{c} {line}\n")


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("Usage: python videos.py path/to/dataset.in")
        return 1

    dataset_path = argv[1]

    data = parse_input(dataset_path)
    model, x = build_model(data)
    solve_model(model)
    cache_videos = extract_solution(data, x)
    write_output(cache_videos, "videos.out")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
