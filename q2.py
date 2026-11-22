"""Question 2: CountSketch for least-squares regression."""

import os
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_preprocessing import fetch_and_preprocess_kddcup99

MS = [50, 100, 200, 500, 1000, 2000, 5000]
OUT = "outputs/q2"
rng = np.random.default_rng(42)


def loss(X, y, a):
    r = X @ a - y
    return r @ r


def solve(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def countsketch(X, m):
    """Return S @ X without storing S: row i of X, times s[i], goes to row h[i]."""
    h = rng.integers(0, m, len(X))
    s = rng.choice([-1.0, 1.0], len(X))
    SX = np.zeros((m, X.shape[1]))
    np.add.at(SX, h, s[:, None] * X)
    return SX


os.makedirs(OUT, exist_ok=True)
D, labels, _ = fetch_and_preprocess_kddcup99()
D = D.astype(np.float64)
y = (labels != b"normal.").astype(float)
# Sketch D and y together, so the same S is applied to both.
Dy = np.column_stack((D, y))
rank = np.linalg.matrix_rank(D)
print(f"n={len(D)}, d={D.shape[1]}, rank(D)={rank}")

# (a) full least squares
start = perf_counter()
b = solve(D, y)
full_time = perf_counter() - start
full_loss = loss(D, y, b)
print(f"\n(a) Loss(D,y,b)={full_loss:.4f}  time={full_time:.3f}s")

# (b), (c) sketched least squares, a fresh S in each of 5 runs per m
print("\n(b)     m run  Loss(D,y,a~)    ratio  sketched loss  time(s)")
ratios, times = [], []
for m in MS:
    for run in range(1, 6):
        start = perf_counter()
        S = countsketch(Dy, m)
        a = solve(S[:, :-1], S[:, -1])
        times.append(perf_counter() - start)
        L = loss(D, y, a)
        ratios.append(L / full_loss)
        print(f"{m:9d} {run:3d} {L:13.4f} {L / full_loss:8.4f} "
              f"{loss(S[:, :-1], S[:, -1], a):14.4e} {times[-1]:8.3f}")
ratios = np.array(ratios).reshape(len(MS), 5)
times = np.array(times).reshape(len(MS), 5).mean(axis=1)

print("\n(b),(c)  m  mean ratio   min ratio   max ratio  mean time(s)")
for m, r, t in zip(MS, ratios, times):
    print(f"{m:9d} {r.mean():11.4f} {r.min():11.4f} {r.max():11.4f} {t:12.3f}")
print(f"full solve time: {full_time:.3f}s")

mean = ratios.mean(axis=1)
plt.errorbar(MS, mean, yerr=[mean - ratios.min(axis=1), ratios.max(axis=1) - mean],
             fmt="o-", capsize=4, label="mean ratio (bars: min, max of 5)")
plt.axvline(rank, color="gray", linestyle="--", label=f"rank(D) = {rank}")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("sketch size m")
plt.ylabel("Loss(D, y, ã) / Loss(D, y, b)")
plt.legend()
plt.savefig(f"{OUT}/ratio.png", dpi=200)
plt.close()

# (d) how well one sketch preserves the norm of the full residual
v = D @ b - y
print("\n(d)     m    mean      std")
plt.figure(figsize=(12, 3.5))
for k, m in enumerate([50, 500, 5000]):
    r = np.array([(countsketch(v[:, None], m) ** 2).sum() for _ in range(100)]) / (v @ v)
    print(f"{m:9d} {r.mean():8.4f} {r.std():8.4f}")
    plt.subplot(1, 3, k + 1)
    plt.hist(r, bins=20)
    plt.title(f"m = {m}")
    plt.xlabel("‖Sv‖² / ‖v‖²")
plt.tight_layout()
plt.savefig(f"{OUT}/residual_hist.png", dpi=200)
plt.close()

# (e) CountSketch vs dense Gaussian sketch on one fixed 50,000-row subset
rows = np.random.default_rng(0).choice(len(D), 50000, replace=False)
D2, y2, Dy2 = D[rows], y[rows], Dy[rows]
best = loss(D2, y2, solve(D2, y2))
print("\n(e) m=500     time(s)    ratio")
for name in ("CountSketch", "Gaussian"):
    t = r = 0
    for _ in range(5):
        start = perf_counter()
        if name == "CountSketch":
            S = countsketch(Dy2, 500)
        else:
            S = (rng.standard_normal((500, 50000)) / np.sqrt(500)) @ Dy2
        t += perf_counter() - start
        r += loss(D2, y2, solve(S[:, :-1], S[:, -1])) / best
    print(f"{name:12s} {t / 5:8.3f} {r / 5:8.4f}")
