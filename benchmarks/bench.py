"""
NB: THIS FILE IS 100% AI GENERATED. TAKE THAT AS YOU WISH.

Python counterpart of the znumerics benchmarks.

Mirrors benchmarks/mat.zig and benchmarks/eigen.zig as closely as possible:
same deterministic fill, same sizes, same iteration/warmup counts, results
allocated fresh each iteration (like the Zig code), and a checksum consumed
each iteration so nothing can be skipped.

Two Python flavours are measured where feasible:
  - pure Python (list-of-lists, naive loops)  -> what "Python" costs
  - NumPy                                     -> what C-backed Python costs

Run:  python benchmarks/bench.py
"""

import time

import numpy as np

# ---------------------------------------------------------------- helpers


def fill_deterministic_np(rows, cols):
    # v = ((i*131 + j*17) % 1000) * 0.001  -- same as fillDeterministic in mat.zig
    i = np.arange(rows).reshape(-1, 1)
    j = np.arange(cols).reshape(1, -1)
    return (((i * 131 + j * 17) % 1000) * 0.001).astype(np.float64)


def fill_deterministic_py(rows, cols):
    return [
        [((i * 131 + j * 17) % 1000) * 0.001 for j in range(cols)]
        for i in range(rows)
    ]


def fill_symmetric_np(n):
    # Same as fillSymmetric in eigen.zig
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            off = ((i * 131 + j * 17) % 100) * 0.05
            v = off + (i + 1) if i == j else off
            m[i, j] = v
            m[j, i] = v
    return m


def fill_complex_spectrum_np(n):
    # Same as fillComplexSpectrum in benchmarks/eigen.zig: scaled 2x2 rotation
    # blocks on the diagonal, deterministic filler above. Block triangular, so
    # the spectrum is exactly n/2 complex conjugate pairs.
    m = np.zeros((n, n))
    k = 0
    while k + 1 < n:
        idx = k // 2
        r = 1.0 + 0.3 * idx
        th = 0.4 + 0.15 * idx
        m[k, k] = r * np.cos(th)
        m[k, k + 1] = -r * np.sin(th)
        m[k + 1, k] = r * np.sin(th)
        m[k + 1, k + 1] = r * np.cos(th)
        k += 2
    if n % 2 == 1:
        m[n - 1, n - 1] = 0.5
    for i in range(n):
        for j in range(i + 2, n):
            m[i, j] = ((i * 131 + j * 17) % 100) * 0.01
    return m


def ns_per_iter(ns_total, iters):
    return ns_total / iters


# ---------------------------------------------------------------- add 512x512


def bench_add():
    R = C = 512
    iters = 10

    A_np = fill_deterministic_np(R, C)
    B_np = fill_deterministic_np(R, C)
    A_py = fill_deterministic_py(R, C)
    B_py = fill_deterministic_py(R, C)

    # Warmup (matches the 10 warmup rounds in mat.zig)
    for _ in range(10):
        (A_np + B_np).sum()

    # --- NumPy ---
    s = 0.0
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        tmp = A_np + B_np          # fresh allocation each iter, like Mat.add
        s += tmp.sum()             # checksum, result gets "used"
    ns_np = time.perf_counter_ns() - t0

    # --- pure Python ---
    s2 = 0.0
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        tmp = [
            [a + b for a, b in zip(ra, rb)]
            for ra, rb in zip(A_py, B_py)
        ]
        s2 += sum(map(sum, tmp))
    ns_py = time.perf_counter_ns() - t0

    assert abs(s - s2) < 1e-6 * abs(s)

    print(f"\n[bench] Mat {R}x{C}, iters={iters}")
    print(f"  numpy add   : {ns_np} ns total, {ns_per_iter(ns_np, iters):.0f} ns/iter")
    print(f"  python add  : {ns_py} ns total, {ns_per_iter(ns_py, iters):.0f} ns/iter")


# ---------------------------------------------------------------- matmul 64x64


def bench_matmul():
    N = 64
    iters = 20

    A_np = fill_deterministic_np(N, N)
    B_np = fill_deterministic_np(N, N)
    A_py = fill_deterministic_py(N, N)
    B_py = fill_deterministic_py(N, N)

    # Warmup (matches the 3 warmup rounds in mat.zig)
    for _ in range(3):
        (A_np @ B_np).sum()

    # --- NumPy (BLAS) ---
    s = 0.0
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        Cm = A_np @ B_np
        s += Cm.sum()
    ns_np = time.perf_counter_ns() - t0

    # --- pure Python, i-k-j like matMult ---
    s2 = 0.0
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        out = [[0.0] * N for _ in range(N)]
        for i in range(N):
            Ai = A_py[i]
            Oi = out[i]
            for k in range(N):
                a = Ai[k]
                Bk = B_py[k]
                for j in range(N):
                    Oi[j] += a * Bk[j]
        s2 += sum(map(sum, out))
    ns_py = time.perf_counter_ns() - t0

    assert abs(s - s2) < 1e-6 * abs(s)

    print(f"\n[bench] matmul {N}x{N} * {N}x{N}, iters={iters}")
    print(f"  numpy matmul  : {ns_np} ns total, {ns_per_iter(ns_np, iters):.0f} ns/iter")
    print(f"  python matmul : {ns_py} ns total, {ns_per_iter(ns_py, iters):.0f} ns/iter")


# ---------------------------------------------------------------- eigenvalues


def bench_eigen():
    N = 20
    reps = 25

    A = fill_symmetric_np(N)

    # Warmup (matches the 3 warmup rounds in eigen.zig)
    for _ in range(3):
        np.linalg.eigvalsh(A)
        np.linalg.eigvals(A)

    # --- eigvalsh: symmetric solver, closest analogue of the shifted-QR route ---
    s = 0.0
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        e = np.linalg.eigvalsh(A)
        s += e.sum()
    ns_sym = time.perf_counter_ns() - t0

    # --- eigvals: general solver (Hessenberg + QR, like the Arnoldi pipeline) ---
    s2 = 0.0
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        e = np.linalg.eigvals(A)
        s2 += e.real.sum()
    ns_gen = time.perf_counter_ns() - t0

    assert abs(s - s2) < 1e-6 * abs(s)

    print(f"\n[bench] Eigenvalues, symmetric {N}x{N}, reps={reps}")
    print(f"  numpy eigvalsh : {ns_sym} ns total, {ns_per_iter(ns_sym, reps):.0f} ns/call")
    print(f"  numpy eigvals  : {ns_gen} ns total, {ns_per_iter(ns_gen, reps):.0f} ns/call")


# ------------------------------------------------- complex eigenvalues


def bench_eigen_complex():
    N = 20
    reps = 25

    A = fill_complex_spectrum_np(N)

    # Warmup (matches the 3 warmup rounds in qrComplexBench)
    for _ in range(3):
        np.linalg.eigvals(A)

    s = 0.0
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        e = np.linalg.eigvals(A)
        s += e.real.sum()
    ns = time.perf_counter_ns() - t0

    # Sanity: the eigenvalue sum equals the trace, conjugate pairs cancel.
    assert abs(s / reps - np.trace(A)) < 1e-8 * abs(np.trace(A))

    print(f"\n[bench] Complex eigenvalues, non-symmetric {N}x{N} (10 conjugate pairs), reps={reps}")
    print(f"  numpy eigvals  : {ns} ns total, {ns_per_iter(ns, reps):.0f} ns/call")


if __name__ == "__main__":
    print(f"numpy {np.__version__}")
    bench_add()
    bench_matmul()
    bench_eigen()
    bench_eigen_complex()
