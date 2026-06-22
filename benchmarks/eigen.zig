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
