"""
QAOA feasibility study: Max-Cut on a small weighted graph.

Real-world framing: split 5 delivery stops between 2 drivers. Total pairwise distance
is constant, so MAXIMIZING the distance between the two groups = MINIMIZING the distance
inside each group = two compact routes.

Stages:
  python qaoa_experiments.py sim        # classical baselines + noiseless + noisy simulation
  python qaoa_experiments.py hardware   # run best parameters on IBM hardware
  python qaoa_experiments.py plot       # figures + results table
"""
import argparse
import itertools
import json
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

# ----------------------------------------------------------------------------
# CONFIG: REPLACE THESE PLACEHOLDER DISTANCES (km) WITH YOUR REAL INSTANCE
# ----------------------------------------------------------------------------
STOPS = ["A", "B", "C", "D", "E"]
DIST = np.array([
    [0.0, 4.2, 7.9, 3.1, 9.4],
    [4.2, 0.0, 5.5, 6.0, 8.1],
    [7.9, 5.5, 0.0, 9.2, 3.8],
    [3.1, 6.0, 9.2, 0.0, 10.5],
    [9.4, 8.1, 3.8, 10.5, 0.0],
])
N = len(STOPS)
W = DIST / DIST.max()  # normalize weights to [0, 1]
EDGES = [(i, j, float(W[i, j])) for i in range(N) for j in range(i + 1, N)]


# ----------------------------------------------------------------------------
# Classical baselines
# ----------------------------------------------------------------------------
def cut_value(bits):
    return sum(w for i, j, w in EDGES if bits[i] != bits[j])


def brute_force():
    best, best_set = -1.0, set()
    for bits in itertools.product([0, 1], repeat=N):
        v = cut_value(bits)
        key = "".join(map(str, bits))
        if v > best + 1e-9:
            best, best_set = v, {key}
        elif abs(v - best) <= 1e-9:
            best_set.add(key)
    return best, best_set


OPT_VAL, OPT_SET = brute_force()


def local_search(restarts=20, seed=0):
    """Heuristic baseline: random-restart one-flip hill climbing."""
    rng = np.random.default_rng(seed)
    best = 0.0
    for _ in range(restarts):
        b = [int(x) for x in rng.integers(0, 2, N)]
        v = cut_value(b)
        improved = True
        while improved:
            improved = False
            for i in range(N):
                b[i] ^= 1
                nv = cut_value(b)
                if nv > v + 1e-12:
                    v, improved = nv, True
                else:
                    b[i] ^= 1
        best = max(best, v)
    return best


# ----------------------------------------------------------------------------
# QAOA
# ----------------------------------------------------------------------------
def build_qaoa(p):
    gam = ParameterVector("g", p)
    bet = ParameterVector("b", p)
    qc = QuantumCircuit(N)
    qc.h(range(N))
    for k in range(p):
        for i, j, w in EDGES:
            qc.rzz(gam[k] * w, i, j)
        qc.rx(2 * bet[k], range(N))
    qc.measure_all()
    return qc, list(gam) + list(bet)


def evaluate(counts):
    """Return (approximation ratio from expected cut, probability of an optimal bitstring)."""
    total = sum(counts.values())
    exp_cut, p_opt = 0.0, 0.0
    for s, c in counts.items():
        bits = s[::-1]  # Qiskit is little-endian: reverse to qubit order
        exp_cut += cut_value(bits) * c
        if bits in OPT_SET:
            p_opt += c
    return exp_cut / total / OPT_VAL, p_opt / total


def run_counts(backend, tqc, params, x, shots, seed):
    bound = tqc.assign_parameters(dict(zip(params, x)))
    return backend.run(bound, shots=shots, seed_simulator=seed).result().get_counts()


def optimize(backend, p, seed, shots_opt=1024, shots_final=8192, maxiter=200):
    rng = np.random.default_rng(seed)
    qc, params = build_qaoa(p)
    tqc = transpile(qc, backend, optimization_level=1, seed_transpiler=seed)
    x0 = np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)])
    hist = []

    def objective(x):
        r, _ = evaluate(run_counts(backend, tqc, params, x, shots_opt, seed))
        hist.append(float(r))
        return -r

    res = minimize(objective, x0, method="COBYLA", options={"maxiter": maxiter})
    ratio, p_opt = evaluate(run_counts(backend, tqc, params, res.x, shots_final, seed))
    ops = tqc.count_ops()
    return {
        "p": p, "seed": seed, "params": res.x.tolist(),
        "ratio": float(ratio), "p_opt": float(p_opt), "history": hist,
        "depth": int(tqc.depth()),
        "twoq": int(sum(ops.get(g, 0) for g in ("cx", "ecr", "cz"))),
    }


def noisy_backend(name=None):
    """Noisy simulator. Default: synthetic depolarizing noise. Optionally copy a real IBM backend."""
    if name:
        from qiskit_ibm_runtime import QiskitRuntimeService
        return AerSimulator.from_backend(QiskitRuntimeService().backend(name))
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(1e-3, 1), ["sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(1e-2, 2), ["cx"])
    nm.add_all_qubit_readout_error(ReadoutError([[0.98, 0.02], [0.02, 0.98]]))
    return AerSimulator(noise_model=nm)


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------
def stage_sim(args):
    out = {
        "stops": STOPS, "n_qubits": N, "opt_value": float(OPT_VAL),
        "opt_bitstrings": sorted(OPT_SET),
        "heuristic_ratio": float(local_search() / OPT_VAL),
        "random_ratio": float(0.5 * sum(w for _, _, w in EDGES) / OPT_VAL),
        "runs": [],
    }
    print(f"Classical optimum cut = {OPT_VAL:.4f}, optimal bitstrings = {sorted(OPT_SET)}")
    print(f"Heuristic ratio = {out['heuristic_ratio']:.3f}, random-guess ratio = {out['random_ratio']:.3f}")
    # empty NoiseModel => same gate set (rz, sx, cx) as the noisy run, so depth/2Q counts are comparable
    envs = [("noiseless", AerSimulator(noise_model=NoiseModel())), ("noisy", noisy_backend(args.noise_backend))]
    for env, be in envs:
        for p in args.depths:
            for seed in range(args.seeds):
                r = optimize(be, p, seed)
                r["env"] = env
                out["runs"].append(r)
                print(f"{env:9s} p={p} seed={seed} ratio={r['ratio']:.3f} P(opt)={r['p_opt']:.3f} "
                      f"depth={r['depth']} 2q={r['twoq']}")
    json.dump(out, open("sim_results.json", "w"), indent=2)
    print("Saved sim_results.json")


def stage_hardware(args):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    sim = json.load(open("sim_results.json"))
    service = QiskitRuntimeService()  # needs a saved IBM account (see instructions)
    backend = (service.backend(args.backend) if args.backend
               else service.least_busy(operational=True, simulator=False, min_num_qubits=N))
    print("Using backend:", backend.name)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

    pubs, meta = [], []
    for p in args.depths:
        best = max((r for r in sim["runs"] if r["env"] == "noiseless" and r["p"] == p),
                   key=lambda r: r["ratio"])
        qc, params = build_qaoa(p)
        isa = pm.run(qc.assign_parameters(dict(zip(params, best["params"]))))
        ops = isa.count_ops()
        for rep in range(args.repeats):
            pubs.append(isa)
            meta.append({"p": p, "rep": rep, "depth": int(isa.depth()),
                         "twoq": int(sum(ops.get(g, 0) for g in ("cx", "ecr", "cz")))})

    job = SamplerV2(mode=backend).run(pubs, shots=4096)  # ONE job keeps queue time and quota low
    print("Job id:", job.job_id())
    result = job.result()
    runs = []
    for m, pub_result in zip(meta, result):
        counts = pub_result.data.meas.get_counts()
        ratio, p_opt = evaluate(counts)
        runs.append({**m, "ratio": float(ratio), "p_opt": float(p_opt), "counts": counts})
        print(f"p={m['p']} rep={m['rep']} ratio={ratio:.3f} P(opt)={p_opt:.3f}")
    json.dump({"backend": backend.name, "job_id": job.job_id(), "shots": 4096, "runs": runs},
              open("hw_results.json", "w"), indent=2)
    print("Saved hw_results.json")


def stage_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sim = json.load(open("sim_results.json"))
    try:
        hw = json.load(open("hw_results.json"))
    except FileNotFoundError:
        hw = None

    ps = sorted({r["p"] for r in sim["runs"]})
    rows = ["| Environment | p | Approx. ratio (mean ± std) | Best ratio | P(optimal) | Depth | 2Q gates |",
            "|---|---|---|---|---|---|---|"]

    fig, ax = plt.subplots(figsize=(6, 4))
    for env in ("noiseless", "noisy"):
        means, stds = [], []
        for p in ps:
            rs = [r for r in sim["runs"] if r["env"] == env and r["p"] == p]
            v = [r["ratio"] for r in rs]
            means.append(np.mean(v))
            stds.append(np.std(v))
            rows.append(f"| {env} simulator | {p} | {np.mean(v):.3f} ± {np.std(v):.3f} | {np.max(v):.3f} | "
                        f"{np.mean([r['p_opt'] for r in rs]):.3f} | {rs[0]['depth']} | {rs[0]['twoq']} |")
        ax.errorbar(ps, means, yerr=stds, marker="o", capsize=3, label=f"{env} simulator")
    if hw:
        hps = sorted({r["p"] for r in hw["runs"]})
        means, stds = [], []
        for p in hps:
            rs = [r for r in hw["runs"] if r["p"] == p]
            v = [r["ratio"] for r in rs]
            means.append(np.mean(v))
            stds.append(np.std(v))
            rows.append(f"| IBM {hw['backend']} | {p} | {np.mean(v):.3f} ± {np.std(v):.3f} | {np.max(v):.3f} | "
                        f"{np.mean([r['p_opt'] for r in rs]):.3f} | {rs[0]['depth']} | {rs[0]['twoq']} |")
        ax.errorbar(hps, means, yerr=stds, marker="s", capsize=3, label=f"IBM {hw['backend']}")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="classical optimum")
    ax.axhline(sim["random_ratio"], color="gray", ls=":", lw=1, label="random guessing")
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("Approximation ratio")
    ax.set_xticks(ps)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_ratio_vs_depth.png", dpi=200)

    # Convergence curves (best noiseless seed per depth)
    fig, ax = plt.subplots(figsize=(6, 4))
    for p in ps:
        r = max((r for r in sim["runs"] if r["env"] == "noiseless" and r["p"] == p), key=lambda r: r["ratio"])
        ax.plot(r["history"], label=f"p={p}")
    ax.set_xlabel("Optimizer iteration")
    ax.set_ylabel("Approximation ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig_convergence.png", dpi=200)

    # Outcome histogram: noiseless vs hardware at the deepest hardware p
    if hw:
        p = max(r["p"] for r in hw["runs"])
        best = max((r for r in sim["runs"] if r["env"] == "noiseless" and r["p"] == p),
                   key=lambda r: r["ratio"])
        qc, params = build_qaoa(p)
        be = AerSimulator()
        ideal = run_counts(be, transpile(qc, be), params, best["params"], 8192, 0)
        hwc = {}
        for r in hw["runs"]:
            if r["p"] == p:
                for k, v in r["counts"].items():
                    hwc[k] = hwc.get(k, 0) + v
        top = [k for k, _ in sorted(ideal.items(), key=lambda kv: -kv[1])[:12]]
        ti, th = sum(ideal.values()), sum(hwc.values())
        x = np.arange(len(top))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - 0.2, [ideal.get(k, 0) / ti for k in top], 0.4, label="noiseless simulator")
        ax.bar(x + 0.2, [hwc.get(k, 0) / th for k in top], 0.4, label=f"IBM {hw['backend']}")
        ax.set_xticks(x)
        ax.set_xticklabels(top, rotation=60, fontsize=8)
        ax.set_ylabel("Probability")
        ax.set_title(f"Top outcomes, p={p}")
        ax.legend()
        fig.tight_layout()
        fig.savefig("fig_histogram.png", dpi=200)

    table = "\n".join(rows)
    open("results_table.md", "w", encoding="utf-8").write(table + "\n")
    print(table)
    print(f"\nClassical optimum cut = {sim['opt_value']:.4f}; heuristic ratio = {sim['heuristic_ratio']:.3f}; "
          f"random-guess ratio = {sim['random_ratio']:.3f}")
    print("Saved fig_*.png and results_table.md")


def stage_compare(args):
    """Fair comparison: evaluate the SAME best noiseless parameters in every environment."""
    sim = json.load(open("sim_results.json"))
    try:
        hw = json.load(open("hw_results.json"))
    except FileNotFoundError:
        hw = None
    rand_popt = len(OPT_SET) / 2 ** N
    rows = ["| Environment (same parameters) | p | Approx. ratio (mean ± std) | P(optimal) |",
            "|---|---|---|---|"]
    series = {}  # name -> (depths, means, stds) for the figure
    envs = [("noiseless simulator", AerSimulator(noise_model=NoiseModel())),
            ("noisy simulator" + (f" ({args.noise_backend} noise)" if args.noise_backend else " (synthetic noise)"),
             noisy_backend(args.noise_backend))]
    for name, be in envs:
        for p in args.depths:
            best = max((r for r in sim["runs"] if r["env"] == "noiseless" and r["p"] == p),
                       key=lambda r: r["ratio"])
            qc, params = build_qaoa(p)
            tqc = transpile(qc, be, optimization_level=1, seed_transpiler=0)
            vals = [evaluate(run_counts(be, tqc, params, best["params"], 8192, s)) for s in range(5)]
            series.setdefault(name, ([], [], []))
            series[name][0].append(p)
            series[name][1].append(float(np.mean([v[0] for v in vals])))
            series[name][2].append(float(np.std([v[0] for v in vals])))
            rows.append(f"| {name} | {p} | {np.mean([v[0] for v in vals]):.3f} ± {np.std([v[0] for v in vals]):.3f} | "
                        f"{np.mean([v[1] for v in vals]):.3f} |")
    if hw:
        for p in sorted({r["p"] for r in hw["runs"]}):
            rs = [r for r in hw["runs"] if r["p"] == p]
            series.setdefault(f"IBM {hw['backend']} (real)", ([], [], []))
            series[f"IBM {hw['backend']} (real)"][0].append(p)
            series[f"IBM {hw['backend']} (real)"][1].append(float(np.mean([r['ratio'] for r in rs])))
            series[f"IBM {hw['backend']} (real)"][2].append(float(np.std([r['ratio'] for r in rs])))
            rows.append(f"| IBM {hw['backend']} (real) | {p} | {np.mean([r['ratio'] for r in rs]):.3f} ± "
                        f"{np.std([r['ratio'] for r in rs]):.3f} | {np.mean([r['p_opt'] for r in rs]):.3f} |")
    rows.append(f"| random guessing | - | {sim['random_ratio']:.3f} | {rand_popt:.3f} |")
    table = "\n".join(rows)
    open("compare_table.md", "w", encoding="utf-8").write(table + "\n")
    print(table)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (xs, ms, ss) in series.items():
        ax.errorbar(xs, ms, yerr=ss, marker="o", capsize=3, label=name)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="classical optimum")
    ax.axhline(sim["random_ratio"], color="gray", ls=":", lw=1, label="random guessing")
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("Approximation ratio (same parameters)")
    ax.set_xticks(sorted({x for v in series.values() for x in v[0]}))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_compare.png", dpi=200)
    print("Saved compare_table.md and fig_compare.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sim", "hardware", "plot", "compare"])
    ap.add_argument("--seeds", type=int, default=10, help="random initializations per (env, p)")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--repeats", type=int, default=3, help="hardware repeats per depth")
    ap.add_argument("--backend", default=None, help="IBM backend name for hardware (default: least busy)")
    ap.add_argument("--noise-backend", default=None,
                    help="IBM backend name to copy noise from for the noisy simulator (needs IBM account)")
    a = ap.parse_args()
    {"sim": stage_sim, "hardware": stage_hardware, "plot": stage_plot, "compare": stage_compare}[a.stage](a)