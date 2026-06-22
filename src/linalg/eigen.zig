const std = @import("std");
const mat = @import("../core/mat.zig");
const vec = @import("../core/vec.zig");
const err_mod = @import("../error.zig");
const qr_mod = @import("qrdecomposition.zig");

const Vec = vec.Vec;
const Mat = mat.Mat;

const arnoldi_result = struct {
    Q: Mat, // Coloumns are an orthonormal basis of the Krylov subspace
    h: Mat, // A on basis Q. It is upper Hessenberg.
};

/// Builds an orthonormal basis of the order-'iter' Krylov subspace of A,
/// starting from 'init_vec'. Returns an 'arnoldi_result' with 'Q' (columns
/// are the orthonormal basis) and 'h' (the upper Hessenberg projection QᵀAQ).
///
/// Uses modified Gram-Schmidt orthogonalisation.
///
/// A must be square; 'init_vec' must match its dimension and is normalised
/// in place. Works on any real square matrix (symmetric or not).
///
/// Stops early if the Krylov subspace becomes invariant (a "lucky breakdown"),
/// producing fewer than 'iter' basis vectors.
pub fn arnoldi_iteration(A: Mat, init_vec: Vec, iter: usize, alloc: std.mem.Allocator) !arnoldi_result {
    const eps: f64 = 1e-12;
    var res: arnoldi_result = undefined;
    var h = try Mat.initZero(alloc, iter + 1, iter);
    errdefer h.deinit();
    var Q = try Mat.initZero(alloc, A.cols, iter + 1);
    errdefer Q.deinit();

    init_vec.normalize();
    try Q.setCol(0, init_vec.data); // Use the first vector as first Krylov vector

    for (1..iter + 1) |k| {
        var x = try Q.getCol(k - 1, alloc);
        defer x.deinit();
        var v = try mat.matVec(alloc, A, x); // Generate new candidate vector
        defer v.deinit();

        for (0..k) |j| { // Subtract projections onto previous basis vectors
            var cur_vec = try Q.getCol(j, alloc);
            defer cur_vec.deinit();
            v.colvec = false;
            const proj = try vec.dot(cur_vec, v);
            h.setUnsafe(j, k - 1, proj);
            cur_vec.multConstUnsafe(proj);
            try v.subInPlace(cur_vec);
        }

        h.setUnsafe(k, k - 1, v.norm());
        if (h.atUnsafe(k, k - 1) > eps) { // Add the produced vector to the basis
            v.normalize();
            try Q.setCol(k, v.data); // setCol copies, so v may be freed by defer
        } else { // Breakdown: Krylov subspace is invariant, stop early
            res.Q = Q;
            res.h = h;
            return res;
        }
    }
    res.Q = Q;
    res.h = h;
    return res;
}

/// Returns the eigenvalues of A as a slice (the diagonal after reduction).
/// Caller owns the returned slice. If 'iters' is non-null, the iteration
/// count is written to it. 'iters' COUNTS a deflation event as an iteration.
///
/// Uses the QR algorithm with Wilkinson shift and deflation.
///
/// LIMITATION: only matrices with a real spectrum converge (e.g. symmetric /
/// Hermitian, or any matrix known to have real eigenvalues). Matrices with
/// complex-conjugate eigenvalues will NOT converge — the iteration leaves 2x2
/// blocks on the diagonal; complex eigenvalues are not supported.
///
/// Stops once every subdiagonal has deflated, or after 'max_iter' sweeps
/// (best effort: the current diagonal is returned either way).
///
/// Returns a BadShape error if A is not square.
pub fn qrAlgorithm(alloc: std.mem.Allocator, A: Mat, max_iter: usize, tolerance: f64, iters: ?*usize) ![]f64 {
    if (A.rows != A.cols) return err_mod.Common.BadShape;
    const n = A.rows;

    var Ak = try A.clone();
    defer Ak.deinit();

    var iter: usize = 0;
    var m: usize = n; // Active leading block for deflation

    while (m > 1 and iter < max_iter) : (iter += 1) {
        // Converged once all subdiagonal entries are negligible.
        var off_max: f64 = 0;
        for (0..n - 1) |i| off_max = @max(off_max, @abs(Ak.atUnsafe(i + 1, i)));
        if (off_max < tolerance) break;

        // Deflation: Is the bottom subdiagonal of the active block negligble?
        const sub = @abs(Ak.atUnsafe(m - 1, m - 2));
        const scale = @abs(Ak.atUnsafe(m - 2, m - 2)) + @abs(Ak.atUnsafe(m - 1, m - 1));
        if (sub <= tolerance * scale) { // lock in Ak[m-1,m-1] as a converged eigenvalue
            m -= 1;
            continue;
        }

        // Wilkinson shift ACTIVE block
        const a = Ak.atUnsafe(m - 2, m - 2);
        const b = Ak.atUnsafe(m - 2, m - 1);
        const c = Ak.atUnsafe(m - 1, m - 2);
        const d = Ak.atUnsafe(m - 1, m - 1);
        const delta = (a - d) / 2.0;
        const bc = b * c;
        var mu = d; // Initial guess, since we dont support complex numbers!
        const denom = @abs(delta) + @sqrt(@max(delta * delta + bc, 0.0));
        if (denom != 0) {
            const sign: f64 = if (delta >= 0) 1.0 else -1.0;
            mu = d - sign * bc / denom;
        }

        // Shifted QR step on the leading mxm block
        var B = try Mat.init(alloc, m, m);
        defer B.deinit();
        for (0..m) |i| {
            for (0..m) |j| {
                B.setUnsafe(i, j, Ak.atUnsafe(i, j));
            }
        }
        for (0..m) |i| B.setUnsafe(i, i, B.atUnsafe(i, i) - mu); // Shift

        const qr = try qr_mod.qrDecomposition(alloc, B);
        var Q = qr[0];
        defer Q.deinit();
        var R = qr[1];
        defer R.deinit();

        var next = try mat.matMult(alloc, R, Q);
        defer next.deinit();

        for (0..m) |i| {
            for (0..m) |j| Ak.setUnsafe(i, j, next.atUnsafe(i, j)); // write block back
        }
        for (0..m) |i| Ak.setUnsafe(i, i, Ak.atUnsafe(i, i) + mu); // + muI

    }

    if (iters) |p| p.* = iter;

    const eigs = try alloc.alloc(f64, n);
    for (0..n) |i| eigs[i] = Ak.atUnsafe(i, i);
    return eigs;
}

/// Returns the 'm' eigenvalues of an m×m matrix A. Caller owns the returned
/// slice. If 'iters' is non-null, the QR iteration count is written to it. 'iters' COUNTS a deflation event as an iteration.
///
/// Reduces A to upper Hessenberg form with a full Arnoldi run, then extracts
/// the eigenvalues with 'qrAlgorithm'.
///
/// LIMITATION: same real-spectrum restriction as 'qrAlgorithm' — intended for
/// symmetric / real-eigenvalue matrices; complex eigenvalues are not supported.
/// A fixed all-ones start vector is used, so for an input it fails to excite
/// the Arnoldi process may break down early and return only a partial spectrum.
///
/// Returns a BadShape error if A is not square.
pub fn eigenvalues(alloc: std.mem.Allocator, A: Mat, max_iter: usize, tolerance: f64, iters: ?*usize) ![]f64 {
    if (A.rows != A.cols) return err_mod.Common.BadShape;
    const m = A.rows;

    // Starting vector b = ones: has a component along every eigenvector for a
    // generic matrix, which avoids an early (deficient) Arnoldi breakdown.
    var b = try Vec.initZero(alloc, m, true);
    defer b.deinit();
    for (0..m) |i| b.setUnsafe(i, 1.0);

    // Arnoldi to full dimension (iter = m) reduces A to Hessenberg form.
    var ar = try arnoldi_iteration(A, b, m, alloc);
    defer ar.Q.deinit();
    defer ar.h.deinit();

    // Square Hessenberg block H = h[0:m, 0:m] (drop the trailing residual row).
    var H = try Mat.initZero(alloc, m, m);
    defer H.deinit();
    for (0..m) |i| {
        for (0..m) |j| H.setUnsafe(i, j, ar.h.atUnsafe(i, j));
    }

    return qrAlgorithm(alloc, H, max_iter, tolerance, iters);
}

// ---------------------------------------------------------------------------
// Tests — mirror reference/arnoldi_reference_test.py
// ---------------------------------------------------------------------------
const testing = std.testing;
const tol: f64 = 1e-10;

/// Columns 0..ncols-1 of Q are orthonormal: Q_c^T Q_c ≈ I.
fn expectOrthonormal(Q: Mat, ncols: usize, alloc: std.mem.Allocator) !void {
    for (0..ncols) |i| {
        var qi = try Q.getCol(i, alloc);
        defer qi.deinit();
        for (0..ncols) |j| {
            var qj = try Q.getCol(j, alloc);
            defer qj.deinit();
            qj.colvec = false;
            const d = try vec.dot(qi, qj);
            const expected: f64 = if (i == j) 1.0 else 0.0;
            try testing.expectApproxEqAbs(expected, d, tol);
        }
    }
}

/// Arnoldi relation: A * Q[:, k] == Q * h[:, k] for k = 0..n-1.
fn expectArnoldiRelation(A: Mat, Q: Mat, h: Mat, n: usize, alloc: std.mem.Allocator) !void {
    for (0..n) |k| {
        var qk = try Q.getCol(k, alloc);
        defer qk.deinit();
        var lhs = try mat.matVec(alloc, A, qk); // A * q_k          (length m)
        defer lhs.deinit();

        var hk = try h.getCol(k, alloc); // h[:, k]                 (length n+1)
        defer hk.deinit();
        var rhs = try mat.matVec(alloc, Q, hk); // Q * h[:, k]      (length m)
        defer rhs.deinit();

        for (0..lhs.len()) |i| {
            try testing.expectApproxEqAbs(lhs.atUnsafe(i), rhs.atUnsafe(i), tol);
        }
    }
}

/// h must be upper Hessenberg: zero below the first subdiagonal.
fn expectHessenberg(h: Mat, n: usize) !void {
    for (0..n) |j| {
        for (j + 2..n + 1) |i| {
            try testing.expectApproxEqAbs(@as(f64, 0.0), h.atUnsafe(i, j), tol);
        }
    }
}

fn setMat(A: Mat, rows: anytype) void {
    for (0..A.rows) |i| {
        for (0..A.cols) |j| A.setUnsafe(i, j, rows[i][j]);
    }
}

test "Arnoldi: orthonormal basis, Hessenberg, and Arnoldi relation" {
    const alloc = testing.allocator;
    const m = 4;
    const n = 3;

    // Same matrix as the Python walkthrough (symmetric tridiagonal -> Lanczos).
    var A = try Mat.initZero(alloc, m, m);
    defer A.deinit();
    setMat(A, [_][4]f64{
        .{ 4, 1, 0, 0 },
        .{ 1, 3, 1, 0 },
        .{ 0, 1, 2, 1 },
        .{ 0, 0, 1, 1 },
    });

    var b = try Vec.initZero(alloc, m, true);
    defer b.deinit();
    b.setUnsafe(0, 1.0); // b = e1

    var result = try arnoldi_iteration(A, b, n, alloc);
    defer result.Q.deinit();
    defer result.h.deinit();

    try testing.expectEqual(@as(usize, n + 1), result.Q.cols);
    try testing.expectEqual(@as(usize, n), result.h.cols);
    try expectOrthonormal(result.Q, n + 1, alloc);
    try expectHessenberg(result.h, n);
    try expectArnoldiRelation(A, result.Q, result.h, n, alloc);
}

test "Arnoldi: lucky breakdown on invariant subspace" {
    const alloc = testing.allocator;
    const m = 4;
    const n = 3;

    // Diagonal A with b = e1: e1 is an eigenvector, so the Krylov space is
    // 1-dimensional and the iteration must break down at the first step.
    var A = try Mat.initZero(alloc, m, m);
    defer A.deinit();
    setMat(A, [_][4]f64{
        .{ 2, 0, 0, 0 },
        .{ 0, 3, 0, 0 },
        .{ 0, 0, 5, 0 },
        .{ 0, 0, 0, 11 },
    });

    var b = try Vec.initZero(alloc, m, true);
    defer b.deinit();
    b.setUnsafe(0, 1.0);

    var result = try arnoldi_iteration(A, b, n, alloc);
    defer result.Q.deinit();
    defer result.h.deinit();

    // First subdiagonal entry vanishes -> breakdown detected.
    try testing.expect(result.h.atUnsafe(1, 0) < 1e-10);
}

test "Arnoldi: n=1 shapes and relation" {
    const alloc = testing.allocator;

    var A = try Mat.initZero(alloc, 2, 2);
    defer A.deinit();
    setMat(A, [_][2]f64{
        .{ 4, 1 },
        .{ 2, 3 },
    });

    var b = try Vec.initZero(alloc, 2, true);
    defer b.deinit();
    b.setUnsafe(0, 1.0);

    var result = try arnoldi_iteration(A, b, 1, alloc);
    defer result.Q.deinit();
    defer result.h.deinit();

    try testing.expectEqual(@as(usize, 2), result.Q.cols);
    try testing.expectEqual(@as(usize, 1), result.h.cols);
    try expectOrthonormal(result.Q, 2, alloc);
    try expectArnoldiRelation(A, result.Q, result.h, 1, alloc);
}

test "QR algorithm: eigenvalues of symmetric matrices" {
    const alloc = testing.allocator;

    // 2x2: [[2,1],[1,2]] -> eigenvalues {3, 1}
    {
        var A = try Mat.initZero(alloc, 2, 2);
        defer A.deinit();
        setMat(A, [_][2]f64{ .{ 2, 1 }, .{ 1, 2 } });

        const eigs = try qrAlgorithm(alloc, A, 500, 1e-12, null);
        defer alloc.free(eigs);
        std.mem.sort(f64, eigs, {}, std.sort.desc(f64));

        try testing.expectApproxEqAbs(@as(f64, 3.0), eigs[0], 1e-9);
        try testing.expectApproxEqAbs(@as(f64, 1.0), eigs[1], 1e-9);
    }

    // 4x4 symmetric tridiagonal (the Arnoldi walkthrough matrix).
    // Reference eigenvalues from numpy.linalg.eigvalsh.
    {
        var A = try Mat.initZero(alloc, 4, 4);
        defer A.deinit();
        setMat(A, [_][4]f64{
            .{ 4, 1, 0, 0 },
            .{ 1, 3, 1, 0 },
            .{ 0, 1, 2, 1 },
            .{ 0, 0, 1, 1 },
        });

        const eigs = try qrAlgorithm(alloc, A, 1000, 1e-12, null);
        defer alloc.free(eigs);
        std.mem.sort(f64, eigs, {}, std.sort.desc(f64));

        const expected = [_]f64{ 4.74528124, 3.17728292, 1.82271708, 0.25471876 };
        for (0..4) |i| try testing.expectApproxEqAbs(expected[i], eigs[i], 1e-6);
    }
}

test "eigenvalues: Arnoldi + QR pipeline (symmetric)" {
    const alloc = testing.allocator;

    // Dense symmetric 3x3 (distinct eigenvalues) -> exercises the reduction.
    {
        var A = try Mat.initZero(alloc, 3, 3);
        defer A.deinit();
        setMat(A, [_][3]f64{
            .{ 2, 1, 1 },
            .{ 1, 3, 2 },
            .{ 1, 2, 4 },
        });

        const eigs = try eigenvalues(alloc, A, 1000, 1e-12, null);
        defer alloc.free(eigs);
        std.mem.sort(f64, eigs, {}, std.sort.desc(f64));

        const expected = [_]f64{ 6.04891734, 1.64310413, 1.30797853 };
        for (0..3) |i| try testing.expectApproxEqAbs(expected[i], eigs[i], 1e-6);
    }

    // 4x4 symmetric tridiagonal (reference values from numpy.linalg.eigvalsh).
    {
        var A = try Mat.initZero(alloc, 4, 4);
        defer A.deinit();
        setMat(A, [_][4]f64{
            .{ 4, 1, 0, 0 },
            .{ 1, 3, 1, 0 },
            .{ 0, 1, 2, 1 },
            .{ 0, 0, 1, 1 },
        });

        const eigs = try eigenvalues(alloc, A, 1000, 1e-12, null);
        defer alloc.free(eigs);
        std.mem.sort(f64, eigs, {}, std.sort.desc(f64));

        const expected = [_]f64{ 4.74528124, 3.17728292, 1.82271708, 0.25471876 };
        for (0..4) |i| try testing.expectApproxEqAbs(expected[i], eigs[i], 1e-6);
    }
}

test "Arnoldi: no leaks on allocation failure" {
    const m = 4;
    const n = 3;
    const a_data = [_][4]f64{
        .{ 4, 1, 0, 0 },
        .{ 1, 3, 1, 0 },
        .{ 0, 1, 2, 1 },
        .{ 0, 0, 1, 1 },
    };

    // Force OOM at each successive allocation; every early-exit path must free
    // whatever it had already allocated, or the testing allocator reports a leak.
    for (0..40) |fail_index| {
        var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const alloc = fa.allocator();

        const A_or = Mat.initZero(alloc, m, m);
        if (A_or) |A_| {
            var A = A_;
            defer A.deinit();
            setMat(A, a_data);

            const b_or = Vec.initZero(alloc, m, true);
            if (b_or) |b_| {
                var b = b_;
                defer b.deinit();
                b.setUnsafe(0, 1.0);

                const res_or = arnoldi_iteration(A, b, n, alloc);
                if (res_or) |res_| {
                    var res = res_;
                    res.Q.deinit();
                    res.h.deinit();
                } else |e| {
                    try std.testing.expect(e == error.OutOfMemory);
                }
            } else |e| {
                try std.testing.expect(e == error.OutOfMemory);
            }
        } else |e| {
            try std.testing.expect(e == error.OutOfMemory);
        }
    }
}
