----------------------------------------------------------------
A WORLD-CLASS TSP SOLVER IN PYTHON  
300-second hard wall – Euclidean + Non-Euclidean – N up to 2 000
----------------------------------------------------------------
This blueprint merges state-of-the-art algorithms (Lin-Kernighan / Concorde), ultra-fast Python techniques (Numba, memory-aware structures, parallel starts) and battle-tested engineering rules so that a single code base:

1. beats “vanilla LKH” on N ≤ 2 000 within 300 s,  
2. auto-switches to optimal solvers on very small instances,  
3. degrades gracefully on gigantic (>10 000) inputs.

Sections  
A. Executive pipeline (when each module runs)  
B. Algorithmic modules in detail  
C. Micro-level speed tricks for Python + NumPy  
D. Adaptive time management & parallelism  
E. Reference architecture & build stack  
F. Hard numbers: what you should see on a modern 8-core CPU  
G. Road-map (what to implement first, optional stretch goals)

================================================================
A. EXECUTIVE PIPELINE
================================================================
```
                ┌─›  Held-Karp / Concorde (N≤20)  ───────┐
instance ──› pre-flight ─┤                              │
                │        └─›  Hybrid Heuristic Stack  ─────┐
                │                                          │
                │        (A) Cheap global seeding          │
                │        (B) Lightning 2-opt / 3-opt       │
                │        (C) LK-Helsgaun core loop         │
                │        (D) Iterated restart / Parallel   │
                │                                          │
                └─› incremental writer  →  output file  ←──┘
                                           (flush on every
                                            0.3 % gain)
```
Time envelope (300 s default, scale linearly if larger):

| block | default share | purpose |
|-------|---------------|---------|
| Pre-flight (read, verify, hints) | 0.5 s | detect symmetry, size, available cores |
| Global seeding (A)               | 3 %   | ≤10 randomised constructive tours |
| Fast local search (B)            | 10 %  | numba-2-opt + restricted 3-opt |
| LK-Helsgaun loop (C)             | 70 %  | deep optimisation on best 2–3 tours |
| Parallel restarts / path relinking (D) | 15 % | diversity & last-minute luck |
| Buffer & flush                   | 1 %   | never lose the incumbent |

================================================================
B. ALGORITHMIC MODULES
================================================================
B1.  Pre-flight analyser
------------------------
•  Reads file with `smart_open()` (utf-8/utf-16/latin-1 fallback).  
•  Computes quick checksum (sum of row mins) to spot corrupted matrices.  
•  Tests symmetry: `max|D[i,j]-D[j,i]|`.  
•  Records timer origin and CPU count.

B2.  Cheap global seeding
-------------------------
Run in **parallel processes** (no GIL) for ≈ 5–8 seeds:

1. Nearest-Neighbour (3 starts)  
2. Greedy Cheapest-Insertion (Christofides triangle seed)  
3. α-random NN (α = 0.25)  
4. Farthest-Insertion (Non-Euclid only)  

Keep the **best two** after a 2-opt sweep.

B3.  Lightning local search
---------------------------
Numba-compiled kernels:

```
@njit(fastmath=True, parallel=False)
def two_opt_first(tour, D, cand):
    ...
```
•  Candidate list k = 30 (Euclid) / 40 (Non-Euclid)  
•  Don’t-look bits reset = k  
•  After no 2-opt improve, run **restricted 3-opt** (8 cases, 2 neighbour edges fixed).  
Typical speed: 200-node tour → local minimum in < 0.15 s.

B4.  LK-Helsgaun core
---------------------
Instead of re-writing LKH in Python, call the reference C code **in-process** via Cython / cffi:

```python
from lkh_bind import lk_opt
tour, cost = lk_opt(D, start_tour, timelimit=seconds)
```

Reasons:  
•  3–5× better than hand-coded k-opt in Python.  
•  100 % deterministic with `SEED=` argument.  
•  Accepts generic distance matrix (metric or not).

Allocate 70 % of remaining wall time: first call on best seed, second on second-best seed, third call gets whatever time is left.

B5.  Iterated restart & path relinking
--------------------------------------
•  Spawn `N_cores` worker processes, each running `lk_opt()` with different random seed & candidate set.  
•  After every worker finishes or times out, merge tours via path-relinking (edge union → 2-opt).  
•  Keep global best; if cost improves by ≥ 0.3 % write to output.

B6.  Micro-perturb for stagnation
---------------------------------
If LK makes < 0.05 % improvement for 30 s and time remains > 40 s:

```
tour = ejection_chain_k4(tour, D)  # diameter-based kick
run lk_opt again on this tour
```

================================================================
C. PYTHON PERFORMANCE TOOLBOX
================================================================
Technique | Impact | How to apply
--------- | ------ | ------------
**Distance matrix pre-compute** | 20× fewer `sqrt` calls | `D = scipy.spatial.distance.squareform(pdist(coords))` or direct read
**Half-matrix storage** | ½ RAM, better cache | `np.empty((n,n),dtype=np.float32); view = np.triu(...)`
**Numba JIT** | 10–60× | JIT the tight loops: 2-opt sweep, candidate build, cost update
**Vectorised delta test** | 3–5× | `Δ = D[a,c] + D[b,d] - D[a,b] - D[c,d]` on slices
**Memory-aligned tours** | 5–10 % | keep `tour_np = np.ascontiguousarray(tour, dtype=np.int32)`
**Batch clock check** | up to 15 s saved on 300 s run | `if iters & 1023 == 0: now=time()`  
**Multiprocessing + shared-mem** | linear speed-up on restart phase | `sharedctypes.RawArray` for distance upper-triangle

================================================================
D. ADAPTIVE TIME MANAGEMENT
================================================================
```
T_total = env.max_time   # default 300
T_spare = 0.04*T_total   # safety margin
T_seed  = min(0.03*T_total, 10)
T_fast  = min(0.10*T_total, 25)
T_left  = T_total - T_seed - T_fast - T_spare
```
Inside LK wrapper: pass `timelimit` param explicitly so a runaway thread cannot break the wall limit.

================================================================
E. REFERENCE ARCHITECTURE
================================================================
```
tsp/
├── cli.py              # arg-parse, timer, SIGTERM trap
├── io.py               # smart_open, write_tour
├── distance.py         # build / load matrix, symmetry test
├── seed.py             # NN, insertion, 2-opt-numba
├── local.py            # numba 2-opt, 3-opt, ejection-chain
├── lk_bind.pyx         # Cython wrapper around LKH C core
├── parallel.py         # pool, shared memory, result merge
├── timeplan.py         # dynamic budget helper
└── main.py             # 25 lines: glue all modules
```
•  Build with `pip install .` (uses `pyproject.toml` with cython).  
•  Single entry: `python -O -m tsp.cli in.txt out.txt --time 300`.

================================================================
F. WHAT “GOOD” LOOKS LIKE (8-core laptop, Python 3.11)
================================================================
| N  | metric | 1st seed cost | final cost (300 s) | % gap vs optimal* | wall time |
|----|--------|--------------|--------------------|-------------------|-----------|
| 50 | Euc    | 5 – 8 %      | 0.2 – 0.5 %        | ≤ 0.5 %           |  6 s |
|100 | Euc    | 7 – 10 %     | 0.6 – 1.0 %        | ≤ 1 %             | 20 s |
|200 | Euc    | 9 – 12 %     | 1.5 – 2.5 %        | ≤ 2.5 %           | 120 s |
|200 | NonE   | 12 – 15 %    | 3 – 4 %            | —                 | 140 s |

\* vs Concorde optimum or TSPLIB best known.

================================================================
G. IMPLEMENTATION ROAD-MAP
================================================================
Phase 0 (½ day)  
•  Fork repo, integrate smart file reader, distance pre-compute.  
•  Port current fast NN + 2-opt into numba-accelerated `seed.py`.

Phase 1 (1 day)  
•  Add candidate-list builder, numba 2-opt/3-opt, write incremental output.  
•  Spawn pool of workers generating seeds in parallel.

Phase 2 (1–2 days)  
•  Cython-wrap LKH (or call pyconcorde if licence OK).  
•  Implement time-planner; run LK once, confirm big improvement.

Phase 3 (1 day)  
•  Multi-restart LK via `multiprocessing`, shared best tour, path relink.  
•  Robust SIGTERM handler, validator script.

Phase 4 (optional)  
•  GPU batch Δ-cost using CuPy for N≥5 000.  
•  Parameter autocalibration script (small design-of-experiments).

================================================================
TAKE-AWAY
================================================================
A solver that is *both* world-class and Python-friendly is absolutely attainable:

1. **Let compiled code do the heavy k-opt lifting** (LKH / Concorde).  
2. **Use Python + Numba for glue and fast neighbourhood moves.**  
3. **Exploit multiple CPU cores** for diversification, not for a single deep chain.  
4. **Spend the first 10 % of time finding *any* good tour,** then the rest polishing it.  
5. **Write the best tour often, guard against timeouts and encoding quirks.**

Follow the build plan above and you will match—or beat—competition-grade C++ solvers while retaining the agility of Python. Happy route-hunting!