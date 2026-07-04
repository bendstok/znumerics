const std = @import("std");
const znum = @import("znumerics");
const Mat = znum.Mat;

/// Deterministic symmetric fill (real spectrum), with a distinct, dominant
/// diagonal so the shifted QR converges quickly.
fn fillSymmetric(m: *Mat) void {
    for (0..m.rows) |i| {
        for (i..m.cols) |j| {
            const off = @as(f64, @floatFromInt((i * 131 + j * 17) % 100)) * 0.05;
            const v = if (i == j) off + @as(f64, @floatFromInt(i + 1)) else off;
            m.setUnsafe(i, j, v);
            m.setUnsafe(j, i, v);
        }
    }
}

fn sumSlice(s: []const f64) f64 {
    var acc: f64 = 0.0;
    for (s) |x| acc += x;
    return acc;
}

/// Compares the two eigenvalue routes on the same dense symmetric matrix:
///   - qrAlgorithm: shifted QR directly on the dense matrix
///   - eigenvalues: Arnoldi reduction to Hessenberg, then shifted QR
pub fn qrDirectVsPipeline(alloc: std.mem.Allocator, io: std.Io) !void {
    const N: usize = 20;
    const reps: usize = 25;
    const max_iter: usize = 1000;
    const tol: f64 = 1e-12;

    var A = try Mat.initZero(alloc, N, N);
    defer A.deinit();
    fillSymmetric(&A);

    // Iteration counts (one call each; neither route mutates A).
    var it_direct: usize = 0;
    var it_pipe: usize = 0;
    {
        const e1 = try znum.eigen.qrAlgorithm(alloc, A, max_iter, tol, &it_direct);
        alloc.free(e1);
        const e2 = try znum.eigenvalues(alloc, A, max_iter, tol, &it_pipe);
        alloc.free(e2);
    }

    // Warmup.
    {
        var w: usize = 0;
        while (w < 3) : (w += 1) {
            const e1 = try znum.eigen.qrAlgorithm(alloc, A, max_iter, tol, null);
            alloc.free(e1);
            const e2 = try znum.eigenvalues(alloc, A, max_iter, tol, null);
            alloc.free(e2);
        }
    }

    // --- time qrAlgorithm (direct on dense) ---
    var sum_direct: f64 = 0.0;
    const t_direct = std.Io.Timestamp.now(io, .awake);
    {
        var k: usize = 0;
        while (k < reps) : (k += 1) {
            const e = try znum.eigen.qrAlgorithm(alloc, A, max_iter, tol, null);
            sum_direct += sumSlice(e);
            alloc.free(e);
        }
    }
    const ns_direct: u64 = @intCast(t_direct.untilNow(io, .awake).toNanoseconds());
    std.mem.doNotOptimizeAway(sum_direct);

    // --- time eigenvalues (Arnoldi -> Hessenberg -> QR) ---
    var sum_pipe: f64 = 0.0;
    const t_pipe = std.Io.Timestamp.now(io, .awake);
    {
        var k: usize = 0;
        while (k < reps) : (k += 1) {
            const e = try znum.eigenvalues(alloc, A, max_iter, tol, null);
            sum_pipe += sumSlice(e);
            alloc.free(e);
        }
    }
    const ns_pipe: u64 = @intCast(t_pipe.untilNow(io, .awake).toNanoseconds());
    std.mem.doNotOptimizeAway(sum_pipe);

    // Sanity: both routes recover the same spectrum (equal eigenvalue sums).
    try std.testing.expectApproxEqAbs(sum_direct, sum_pipe, 1e-6 * @abs(sum_direct));

    const direct_per = @as(f64, @floatFromInt(ns_direct)) / @as(f64, @floatFromInt(reps));
    const pipe_per = @as(f64, @floatFromInt(ns_pipe)) / @as(f64, @floatFromInt(reps));

    std.debug.print(
        "\n[bench] Eigenvalues, symmetric {d}x{d}, reps={d}\n  qrAlgorithm (direct) : {d} iters, {d} ns total, {d} ns/call\n  eigenvalues (Arnoldi): {d} iters, {d} ns total, {d} ns/call\n  speedup              : {d}x\n",
        .{
            N,         N,         reps,
            it_direct, ns_direct, direct_per,
            it_pipe,   ns_pipe,   pipe_per,
            direct_per / pipe_per,
        },
    );
}

/// Deterministic non-symmetric fill with a known complex spectrum: scaled
/// 2x2 rotation blocks on the diagonal (eigenvalues r*cos(t) ± i*r*sin(t),
/// distinct moduli per block for reliable convergence) and deterministic
/// filler above. Block triangular, so the spectrum is exactly the blocks'.
fn fillComplexSpectrum(m: *Mat) void {
    const n = m.rows;
    m.setAll(0.0);
    var k: usize = 0;
    while (k + 1 < n) : (k += 2) {
        const idx = @as(f64, @floatFromInt(k / 2));
        const r = 1.0 + 0.3 * idx;
        const th = 0.4 + 0.15 * idx;
        m.setUnsafe(k, k, r * @cos(th));
        m.setUnsafe(k, k + 1, -r * @sin(th));
        m.setUnsafe(k + 1, k, r * @sin(th));
        m.setUnsafe(k + 1, k + 1, r * @cos(th));
    }
    if (n % 2 == 1) m.setUnsafe(n - 1, n - 1, 0.5); // odd leftover: one real eigenvalue
    // Filler above the blocks (does not change the spectrum)
    for (0..n) |i| {
        var j = i + 2;
        while (j < n) : (j += 1) {
            m.setUnsafe(i, j, @as(f64, @floatFromInt((i * 131 + j * 17) % 100)) * 0.01);
        }
    }
}

/// Times qrAlgorithmComplex on a dense non-symmetric matrix whose spectrum
/// is 10 complex conjugate pairs. Compare against numpy's eigvals on the
/// same matrix via benchmarks/bench.py.
pub fn qrComplexBench(alloc: std.mem.Allocator, io: std.Io) !void {
    const N: usize = 20;
    const reps: usize = 25;
    const max_iter: usize = 2000;
    const tol: f64 = 1e-12;

    var A = try Mat.initZero(alloc, N, N);
    defer A.deinit();
    fillComplexSpectrum(&A);

    // Iteration count (one call).
    var it: usize = 0;
    {
        const e = try znum.eigen.qrAlgorithmComplex(alloc, A, max_iter, tol, &it);
        alloc.free(e);
    }

    // Warmup.
    {
        var w: usize = 0;
        while (w < 3) : (w += 1) {
            const e = try znum.eigen.qrAlgorithmComplex(alloc, A, max_iter, tol, null);
            alloc.free(e);
        }
    }

    var sum_re: f64 = 0.0;
    var sum_im: f64 = 0.0;
    const t0 = std.Io.Timestamp.now(io, .awake);
    {
        var k: usize = 0;
        while (k < reps) : (k += 1) {
            const e = try znum.eigen.qrAlgorithmComplex(alloc, A, max_iter, tol, null);
            for (e) |z| {
                sum_re += z.re;
                sum_im += z.im;
            }
            alloc.free(e);
        }
    }
    const ns: u64 = @intCast(t0.untilNow(io, .awake).toNanoseconds());
    std.mem.doNotOptimizeAway(sum_re);
    std.mem.doNotOptimizeAway(sum_im);

    // Sanity: the eigenvalue sum equals the trace, conjugate pairs cancel.
    const tr = try A.trace();
    const reps_f = @as(f64, @floatFromInt(reps));
    try std.testing.expectApproxEqAbs(tr * reps_f, sum_re, 1e-6 * @abs(tr) * reps_f);
    try std.testing.expectApproxEqAbs(0.0, sum_im, 1e-6);

    const per = @as(f64, @floatFromInt(ns)) / reps_f;
    std.debug.print(
        "\n[bench] Complex eigenvalues, non-symmetric {d}x{d} (10 conjugate pairs), reps={d}\n  qrAlgorithmComplex   : {d} iters, {d} ns total, {d} ns/call\n",
        .{ N, N, reps, it, ns, per },
    );
}
