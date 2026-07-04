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

    var v = try Vec.initZero(alloc, A.rows, true);
    defer v.deinit();

    for (1..iter + 1) |k| {
        // v = A * Q[:, k-1]
        for (0..A.rows) |i| {
            var s: f64 = 0;
            for (0..A.cols) |j| s += A.atUnsafe(i, j) * Q.atUnsafe(j, k - 1);
            v.setUnsafe(i, s);
        }

        for (0..k) |j| { // Subtract projections onto previous basis vectors
            // proj = dot(Q[:, j], v)
            var proj: f64 = 0;
            for (0..A.cols) |i| proj += Q.atUnsafe(i, j) * v.atUnsafe(i);
            h.setUnsafe(j, k - 1, proj);
            for (0..A.cols) |i| v.setUnsafe(i, v.atUnsafe(i) - proj * Q.atUnsafe(i, j));
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

const Givens = struct { c: f64, s: f64 };

fn givens(a: f64, b: f64) Givens {
    if (b == 0) return .{ .c = 1, .s = 0 };
    const r = @sqrt(a * a + b * b);
    return .{ .c = a / r, .s = -b / r };
}

// left-apply to rows i and i+1, across the active block's columns [0, m)
fn rotRows(H: Mat, i: usize, g: Givens, m: usize) void {
    for (0..m) |j| {
        const p = H.atUnsafe(i, j);
        const q = H.atUnsafe(i + 1, j);
        H.setUnsafe(i, j, g.c * p - g.s * q);
        H.setUnsafe(i + 1, j, g.s * p + g.c * q);
    }
}

// right-apply (Gᵀ) to columns i and i+1, down the active block's rows [0, m)
fn rotCols(H: Mat, i: usize, g: Givens, m: usize) void {
    for (0..m) |r| {
        const p = H.atUnsafe(r, i);
        const q = H.atUnsafe(r, i + 1);
        H.setUnsafe(r, i, g.c * p - g.s * q);
        H.setUnsafe(r, i + 1, g.s * p + g.c * q);
    }
}

/// Returns the eigenvalues of A as a slice (the diagonal after reduction).
/// Caller owns the returned slice. If 'iters' is non-null, the QR sweep count
/// is written to it.
///
/// Reduces A to upper Hessenberg form once (Householder similarity), then runs
/// the QR algorithm with Wilkinson shift and deflation, applying each shifted
/// sweep as in-place Givens rotations. For a symmetric A the reduction yields a tridiagonal matrix.
///
/// Accepts any real square matrix.
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

    // Reduce A to upper Hessenberg form once (Householder similarity), so the
    // single-Givens-per-column sweep below is valid and each sweep is O(n²).
    // For a symmetric A this produces a tridiagonal matrix. Eigenvalues are
    // preserved (Ak := HₖAkHₖ), which is all we read out at the end.
    if (n > 2) {
        const v = try alloc.alloc(f64, n);
        defer alloc.free(v);
        for (0..n - 2) |k| {
            var nrm: f64 = 0;
            for (k + 1..n) |i| nrm += Ak.atUnsafe(i, k) * Ak.atUnsafe(i, k);
            nrm = @sqrt(nrm);
            if (nrm == 0) continue;
            const x0 = Ak.atUnsafe(k + 1, k);
            const alpha: f64 = if (x0 >= 0) -nrm else nrm; // stable sign
            v[k + 1] = x0 - alpha;
            for (k + 2..n) |i| v[i] = Ak.atUnsafe(i, k);
            var vv: f64 = 0;
            for (k + 1..n) |i| vv += v[i] * v[i];
            if (vv == 0) continue;
            const beta = 2.0 / vv;
            // Left:  Ak := (I - beta v vᵀ) Ak
            for (0..n) |j| {
                var dt: f64 = 0;
                for (k + 1..n) |i| dt += v[i] * Ak.atUnsafe(i, j);
                const w = beta * dt;
                for (k + 1..n) |i| Ak.setUnsafe(i, j, Ak.atUnsafe(i, j) - v[i] * w);
            }
            // Right: Ak := Ak (I - beta v vᵀ)
            for (0..n) |i| {
                var dt: f64 = 0;
                for (k + 1..n) |j| dt += Ak.atUnsafe(i, j) * v[j];
                const w = beta * dt;
                for (k + 1..n) |j| Ak.setUnsafe(i, j, Ak.atUnsafe(i, j) - w * v[j]);
            }
        }
    }

    const rotation = try alloc.alloc(Givens, n);
    defer alloc.free(rotation);

    var iter: usize = 0;
    var m: usize = n; // Active leading block for deflation

    while (m > 1 and iter < max_iter) : (iter += 1) {
        // Deflation: is the bottom subdiagonal of the active block negligible?
        const sub = @abs(Ak.atUnsafe(m - 1, m - 2));
        const scale = @abs(Ak.atUnsafe(m - 2, m - 2)) + @abs(Ak.atUnsafe(m - 1, m - 1));
        if (sub <= tolerance * scale) { // lock in Ak[m-1,m-1] as a converged eigenvalue
            m -= 1;
            continue;
        }

        // Wilkinson shift from the trailing 2x2 of the active block.
        const a = Ak.atUnsafe(m - 2, m - 2);
        const b = Ak.atUnsafe(m - 2, m - 1);
        const c = Ak.atUnsafe(m - 1, m - 2);
        const d = Ak.atUnsafe(m - 1, m - 1);
        const delta = (a - d) / 2.0;
        const bc = b * c;
        var mu = d;
        const denom = @abs(delta) + @sqrt(@max(delta * delta + bc, 0.0));
        if (denom != 0) {
            const sign: f64 = if (delta >= 0) 1.0 else -1.0;
            mu = d - sign * bc / denom;
        }

        // Shifted Givens QR step on the active mxm block.
        for (0..m) |i| Ak.setUnsafe(i, i, Ak.atUnsafe(i, i) - mu); // shift

        for (0..m - 1) |i| { // forward sweep: zero each subdiagonal, store rotation
            rotation[i] = givens(Ak.atUnsafe(i, i), Ak.atUnsafe(i + 1, i));
            rotRows(Ak, i, rotation[i], m);
        }
        for (0..m - 1) |i| { // right-apply Gᵀ: restores Hessenberg form
            rotCols(Ak, i, rotation[i], m);
        }

        for (0..m) |i| Ak.setUnsafe(i, i, Ak.atUnsafe(i, i) + mu); // unshift
    }

    if (iters) |p| p.* = iter;

    const eigs = try alloc.alloc(f64, n);
    for (0..n) |i| eigs[i] = Ak.atUnsafe(i, i);
    return eigs;
}

/// Returns the 'm' eigenvalues of an m×m matrix A. Caller owns the returned
/// slice. If 'iters' is non-null, the QR sweep count is written to it.
///
/// Reduces A to upper Hessenberg form with a full Arnoldi run, then extracts
/// the eigenvalues with 'qrAlgorithm'.
///
/// NOTE: 'qrAlgorithm' already performs its own Hessenberg reduction, so for a
/// dense full-spectrum problem this path reduces twice and is slower than
/// calling 'qrAlgorithm' directly — prefer that here. This pipeline exists to
/// exercise the Arnoldi reduction; its real niche is large, sparse matrices
/// where only a few eigenvalues are wanted (run Arnoldi to a small dimension).
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

/// LEGACY: superseded by 'qrAlgorithm', kept only for reference/comparison.
///
/// Returns the eigenvalues of A as a slice (the diagonal after reduction).
/// Caller owns the returned slice. If 'iters' is non-null, the iteration
/// count is written to it.
///
/// Uses the QR algorithm with Wilkinson shift and deflation, forming each QR
/// step explicitly via the Householder 'qrDecomposition' plus 'matMult'. It
/// works on any dense real-spectrum matrix, but is slow.
///
/// LIMITATION: only matrices with a real spectrum converge (e.g. symmetric /
/// Hermitian). Matrices with complex-conjugate eigenvalues will NOT converge —
/// the iteration leaves 2x2 blocks on the diagonal; complex is not supported.
///
/// Stops once every subdiagonal has deflated, or after 'max_iter' sweeps
/// (best effort: the current diagonal is returned either way).
///
/// Returns a BadShape error if A is not square.
pub fn qrAlgorithm_LEGACY(alloc: std.mem.Allocator, A: Mat, max_iter: usize, tolerance: f64, iters: ?*usize) ![]f64 {
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

    // Dense symmetric 3x3 (NOT Hessenberg: entry (2,0) != 0) -> exercises the
    // Householder Hessenberg reduction inside qrAlgorithm.
    {
        var A = try Mat.initZero(alloc, 3, 3);
        defer A.deinit();
        setMat(A, [_][3]f64{
            .{ 2, 1, 1 },
            .{ 1, 3, 2 },
            .{ 1, 2, 4 },
        });

        const eigs = try qrAlgorithm(alloc, A, 1000, 1e-12, null);
        defer alloc.free(eigs);
        std.mem.sort(f64, eigs, {}, std.sort.desc(f64));

        const expected = [_]f64{ 6.04891734, 1.64310413, 1.30797853 };
        for (0..3) |i| try testing.expectApproxEqAbs(expected[i], eigs[i], 1e-6);
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
