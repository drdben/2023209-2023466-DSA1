"""Question 1: Gaussian random projections and k-means."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from data_preprocessing import fetch_and_preprocess_kddcup99


K = 15
MS = [5, 10, 15, 20, 25]
SEED = 42
BLOCK = 4096


def gaussian_projection(rng, m, d):
    # N(0, 1/m) has standard deviation 1/sqrt(m).
    return rng.standard_normal((m, d), dtype=np.float32) / np.float32(np.sqrt(m))


def cluster_means(X, labels):
    """Use float64 sums for accuracy, without making a float64 copy of X."""
    centers = np.zeros((K, X.shape[1]), dtype=np.float64)
    counts = np.bincount(labels, minlength=K)
    for j in range(K):
        if counts[j]:
            centers[j] = X[labels == j].mean(axis=0, dtype=np.float64)
        else:
            # A duplicate valid center has no effect on nearest-center cost.
            centers[j] = X[0]
    return centers


def center_cost(X, centers):
    """Assign every row to its nearest center, using small temporary arrays."""
    total = 0.0
    for start in range(0, len(X), BLOCK):
        rows = X[start:start + BLOCK].astype(np.float64)
        best = np.full(len(rows), np.inf)
        for center in centers:
            diff = rows - center
            distances = np.einsum("ij,ij->i", diff, diff)
            np.minimum(best, distances, out=best)
        total += best.sum()
    return float(total)


def partition_scatter(X, labels):
    """R.T @ R where R contains residuals from the partition's exact means."""
    centers = cluster_means(X, labels)
    scatter = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    for start in range(0, len(X), BLOCK):
        residual = X[start:start + BLOCK].astype(np.float64)
        residual -= centers[labels[start:start + BLOCK]]
        scatter += residual.T @ residual
    return scatter


def sample_pairs(X, rng, count=10000):
    left, right = [], []
    # Reject equal feature vectors, including duplicates at different indices.
    if np.all(X == X[0]):
        raise ValueError("Cannot sample unequal pairs from identical rows")
    while len(left) < count:
        pairs = rng.integers(0, len(X), size=(count - len(left), 2))
        valid = np.any(X[pairs[:, 0]] != X[pairs[:, 1]], axis=1)
        left.extend(pairs[valid, 0].tolist())
        right.extend(pairs[valid, 1].tolist())
    return np.array(left), np.array(right)


def run_q1(D, output, seed=SEED):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Independent streams keep pair sampling separate from the clustering runs.
    streams = np.random.SeedSequence(seed).spawn(3)
    rng, pair_rng, distortion_rng = [np.random.default_rng(s) for s in streams]
    D = np.ascontiguousarray(D, dtype=np.float32)

    baseline = KMeans(n_clusters=15, n_init=5, random_state=seed)
    start = perf_counter()
    baseline.fit(D)
    baseline_seconds = perf_counter() - start
    baseline_cost = center_cost(D, baseline.cluster_centers_)
    np.savez(output / "baseline.npz", centers=baseline.cluster_centers_,
             labels=baseline.labels_)
    metadata = dict(n=len(D), d=D.shape[1], dtype=str(D.dtype), seed=seed,
                    n_clusters=K, n_init=5, baseline_cost=baseline_cost,
                    sklearn_inertia=float(baseline.inertia_),
                    baseline_seconds=baseline_seconds,
                    numpy_version=np.__version__, sklearn_version=sklearn.__version__)
    (output / "baseline.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Baseline: cost={baseline_cost:.6f}, fit={baseline_seconds:.3f}s", flush=True)

    trials = []
    for m in MS:
        for trial in range(1, 6):
            A = gaussian_projection(rng, m, D.shape[1])
            km_seed = int(rng.integers(0, 2**31 - 1))
            model = KMeans(n_clusters=15, n_init=5, random_state=km_seed)
            start = perf_counter()
            E = D @ A.T
            model.fit(E)
            seconds = perf_counter() - start
            averaged = cluster_means(D, model.labels_)
            inverse = model.cluster_centers_ @ np.linalg.pinv(A).T
            average_cost = center_cost(D, averaged)
            inverse_cost = center_cost(D, inverse)
            trials.append(dict(m=m, trial=trial, kmeans_seed=km_seed,
                               averaged_cost=average_cost,
                               averaged_ratio=average_cost / baseline_cost,
                               inverse_cost=inverse_cost,
                               inverse_ratio=inverse_cost / baseline_cost,
                               projection_fit_seconds=seconds))
            # Save after each run so completed results survive interruptions.
            pd.DataFrame(trials).to_csv(output / "trials.csv", index=False)
            print(f"m={m}, trial={trial}: averaged ratio={average_cost / baseline_cost:.6f}",
                  flush=True)
            del E

    summary = pd.DataFrame(trials).groupby("m").agg(
        mean_cost=("averaged_cost", "mean"),
        mean_ratio=("averaged_ratio", "mean"),
        min_ratio=("averaged_ratio", "min"), max_ratio=("averaged_ratio", "max"),
        inverse_mean_ratio=("inverse_ratio", "mean"),
        inverse_min_ratio=("inverse_ratio", "min"),
        inverse_max_ratio=("inverse_ratio", "max"),
        mean_seconds=("projection_fit_seconds", "mean"))
    summary.to_csv(output / "summary.csv")
    fig, ax = plt.subplots()
    ax.plot(MS, summary.mean_seconds, "o-", label="Projection + projected k-means")
    ax.axhline(baseline_seconds, color="tab:red", linestyle="--", label="Baseline fit")
    ax.set(xlabel="Projection dimension m", ylabel="Wall-clock time (seconds)", xticks=MS)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "runtime.png", dpi=200)
    plt.close(fig)

    i, l = sample_pairs(D, pair_rng)
    np.savez(output / "pairs.npz", i=i, l=l)
    differences = D[i].astype(np.float64) - D[l]
    denominators = np.einsum("ij,ij->i", differences, differences)
    scatter = partition_scatter(D, baseline.labels_)
    partition_cost = float(np.trace(scatter))
    pair_results, partition_results = {}, {}
    for m in MS:
        A = gaussian_projection(distortion_rng, m, D.shape[1])
        projected_diff = differences @ A.T
        pair_results[str(m)] = np.einsum("ij,ij->i", projected_diff, projected_diff) / denominators
        partition_results[str(m)] = []
        for _ in range(20):
            A = gaussian_projection(distortion_rng, m, D.shape[1]).astype(np.float64)
            # sum ||A r_i||^2 = trace(A (R.T R) A.T).
            projected_cost = np.trace(A @ scatter @ A.T)
            partition_results[str(m)].append(float(projected_cost / partition_cost))
    pd.DataFrame(pair_results).to_csv(output / "pairwise_ratios.csv", index=False)
    pd.DataFrame(partition_results).to_csv(output / "partition_ratios.csv", index=False)
    metadata["baseline_partition_cost"] = partition_cost
    (output / "baseline.json").write_text(json.dumps(metadata, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, results, title in zip(axes, [pair_results, partition_results],
                                  ["10,000 unequal row pairs", "20 fixed-partition projections"]):
        ax.boxplot(list(results.values()), tick_labels=MS, showfliers=True)
        ax.axhline(1, color="gray", linestyle="--")
        ax.set(xlabel="Projection dimension m", title=title)
    axes[0].set_ylabel("Projected / original squared-distance cost")
    fig.tight_layout()
    fig.savefig(output / "distortion.png", dpi=200)
    plt.close(fig)
    print(summary.to_string())
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/q1")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--threads", type=int, default=1,
                        help="Thread limit used for both baseline and projected runs")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Exercise all steps on synthetic data; not assignment results")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.smoke_test:
        D = np.random.default_rng(args.seed).random((300, 30), dtype=np.float32)
        output = Path(args.output) / "smoke_test"
    else:
        D, _, names = fetch_and_preprocess_kddcup99()
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "feature_names.json").write_text(json.dumps(names, indent=2) + "\n")
    with threadpool_limits(limits=args.threads):
        result = run_q1(D, output, args.seed)
    result["threads"] = args.threads
    result["synthetic_smoke_test"] = args.smoke_test
    (output / "baseline.json").write_text(json.dumps(result, indent=2) + "\n")
