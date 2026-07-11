const std = @import("std");
const mat = @import("../core/mat.zig");
const vec = @import("../core/vec.zig");
const err_mod = @import("../error.zig");
const qr_mod = @import("qrdecomposition.zig");
const sclr = @import("../core/scalar.zig");

const Vec = vec.Vec;
const Mat = mat.Mat;
const Complex = std.math.Complex(f64);

pub const EigenError = error{
    NotSquare,
} || err_mod.Common;

pub fn ArnoldiResult(comptime T: type) type {
    return struct {
        Q: mat.Matrix(T), // Coloumns are an orthonormal basis of the Krylov subspace
        h: mat.Matrix(T), // A on basis Q. It is upper Hessenberg.
    };
}

/// Builds an orthonormal basis of the order-'iter' Krylov subspace of A,
/// starting from 'init_vec'. Returns an 'ArnoldiResult' with 'Q' (columns
/// are the orthonormal basis) and 'h' (the upper Hessenberg projection QᴴAQ,
/// which is QᵀAQ for real T).
///
/// Uses modified Gram-Schmidt orthogonalisation.
///
/// A must be square; 'init_vec' must match its dimension and is normalised
/// in place. Works on any square matrix (symmetric/Hermitian or not).
///
/// Stops early if the Krylov subspace becomes invariant (a "lucky breakdown"),
/// producing fewer than 'iter' basis vectors.
pub fn arnoldi_iteration(A: anytype, init_vec: vec.Vector(mat.ElementOf(@TypeOf(A))), iter: usize, alloc: std.mem.Allocator) (err_mod.Common || std.mem.Allocator.Error)!ArnoldiResult(mat.ElementOf(@TypeOf(A))) {
    const T = mat.ElementOf(@TypeOf(A));

    // Breakdown tolerance, scaled to the element precision
    // (~2e-12 for f64, ~1e-3 for f32; a fixed 1e-12 is below f32 eps).
    const eps: sclr.Real(T) = std.math.floatEps(sclr.Real(T)) * 1e4;

    var h = try mat.Matrix(T).initZero(alloc, iter + 1, iter);
    errdefer h.deinit();
    var Q = try mat.Matrix(T).initZero(alloc, A.cols, iter + 1);
    errdefer Q.deinit();

    init_vec.normalize();
    try Q.setCol(0, init_vec.data); // Use the first vector as first Krylov vector

    var v = try vec.Vector(T).initZero(alloc, A.rows, true);
    defer v.deinit();

    for (1..iter + 1) |k| {
        // v = A * Q[:, k-1]
        for (0..A.rows) |i| {
            var s = sclr.zero(T);
            for (0..A.cols) |j| s = sclr.add(s, sclr.mul(A.atUnsafe(i, j), Q.atUnsafe(j, k - 1)));
            v.setUnsafe(i, s);
        }

        for (0..k) |j| { // Subtract projections onto previous basis vectors
            // proj = dot(Q[:, j], v)
            var proj = sclr.zero(T);
            for (0..A.cols) |i| proj = sclr.add(proj, sclr.mul(sclr.conj(Q.atUnsafe(i, j)), v.atUnsafe(i)));
            h.setUnsafe(j, k - 1, proj);
            for (0..A.cols) |i| v.setUnsafe(i, sclr.sub(v.atUnsafe(i), sclr.mul(proj, Q.atUnsafe(i, j))));
        }

        const hn = v.norm(); // Real(T)
        h.setUnsafe(k, k - 1, sclr.fromReal(T, hn));
        if (hn > eps) { // Add the produced vector to the basis
            v.normalize();
            try Q.setCol(k, v.data); // setCol copies, so v may be freed by defer
        } else { // Breakdown: Krylov subspace is invariant, stop early
            return .{ .Q = Q, .h = h };
        }
    }
    return .{ .Q = Q, .h = h };
}

const Givens = struct { c: f64, s: f64 };

fn givens(a: f64, b: f64) Givens {
    if (b == 0) return .{ .c = 1, .s = 0 };
    const r = @sqrt(a * a + b * b);
    return .{ .c = a / r, .s = -b / r };
}

/// Balances A in place with a diagonal similarity transform (Parlett-Reinsch),
/// equalising the 1-norms of matching rows and columns. Scale factors are
/// powers of 2, so no rounding error is introduced. Eigenvalues are unchanged
/// and the transform is not recorded, since only eigenvalues are read out.
/// The zero pattern is preserved (diagonal similarity), so a Hessenberg
/// matrix stays Hessenberg.
///
/// Badly scaled matrices (hand-edited state matrices, companion matrices of
/// polynomials with widely varying coefficients) are exactly where the QR
/// iteration loses accuracy; balancing restores it at negligible cost.
fn balance(A: Mat) void {
    const n = A.rows;
    const radix: f64 = 2.0;
    var done = false;
    while (!done) {
        done = true;
        for (0..n) |i| {
            var c: f64 = 0.0;
            var r: f64 = 0.0;
            for (0..n) |j| {
                if (j == i) continue;
                c += @abs(A.atUnsafe(j, i));
                r += @abs(A.atUnsafe(i, j));
            }
            if (c == 0.0 or r == 0.0) continue;
            const s = c + r;
            var f: f64 = 1.0;
            while (c < r / radix) {
                c *= radix * radix;
                f *= radix;
            }
            while (c > r * radix) {
                c /= radix * radix;
                f /= radix;
            }
            if ((c + r) / f < 0.95 * s) {
                done = false;
                const g = 1.0 / f;
                for (0..n) |j| A.setUnsafe(i, j, A.atUnsafe(i, j) * g); // row i /= f
                for (0..n) |j| A.setUnsafe(j, i, A.atUnsafe(j, i) * f); // col i *= f
            }
        }
    }
}

/// Reduces A in place to upper Hessenberg form by Householder similarity
/// transforms. Deterministic: no start vector and no breakdown (unlike the
/// Arnoldi reduction). For a symmetric A the result is tridiagonal.
/// Eigenvalues are preserved (A := HₖAHₖ).
fn hessenbergReduce(alloc: std.mem.Allocator, A: Mat) std.mem.Allocator.Error!void {
    const n = A.rows;
    if (n <= 2) return;
    const v = try alloc.alloc(f64, n);
    defer alloc.free(v);
    for (0..n - 2) |k| {
        var nrm: f64 = 0;
        for (k + 1..n) |i| nrm += A.atUnsafe(i, k) * A.atUnsafe(i, k);
        nrm = @sqrt(nrm);
        if (nrm == 0) continue;
        const x0 = A.atUnsafe(k + 1, k);
        const alpha: f64 = if (x0 >= 0) -nrm else nrm; // stable sign
        v[k + 1] = x0 - alpha;
        for (k + 2..n) |i| v[i] = A.atUnsafe(i, k);
        var vv: f64 = 0;
        for (k + 1..n) |i| vv += v[i] * v[i];
        if (vv == 0) continue;
        const beta = 2.0 / vv;
        // Left:  A := (I - beta v vᵀ) A
        for (0..n) |j| {
            var dt: f64 = 0;
            for (k + 1..n) |i| dt += v[i] * A.atUnsafe(i, j);
            const w = beta * dt;
            for (k + 1..n) |i| A.setUnsafe(i, j, A.atUnsafe(i, j) - v[i] * w);
        }
        // Right: A := A (I - beta v vᵀ)
        for (0..n) |i| {
            var dt: f64 = 0;
            for (k + 1..n) |j| dt += A.atUnsafe(i, j) * v[j];
            const w = beta * dt;
            for (k + 1..n) |j| A.setUnsafe(i, j, A.atUnsafe(i, j) - w * v[j]);
        }
    }
}

fn eig2x2(a: f64, b: f64, c: f64, d: f64) [2]Complex {
    const tr = a + d;
    const det = a * d - b * c;
    const disc = tr * tr / 4.0 - det;
    if (disc >= 0) { // We have two real eigenvalues
        const sq = @sqrt(disc);
        return .{ Complex.init(tr / 2.0 + sq, 0), Complex.init(tr / 2.0 - sq, 0) };
    } else { // We have to imaginary eigenvalues
        const im = @sqrt(-disc);
        return .{ Complex.init(tr / 2.0, im), Complex.init(tr / 2.0, -im) };
    }
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
/// Balances A (Parlett-Reinsch) and reduces it to upper Hessenberg form once
/// (Householder similarity), then runs the QR algorithm with Wilkinson shift
/// and deflation, applying each shifted sweep as in-place Givens rotations.
/// For a symmetric A the reduction yields a tridiagonal matrix.
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
/// Returns an EigenError.NotSquare if A is not square.
pub fn qrAlgorithm(alloc: std.mem.Allocator, A: Mat, max_iter: usize, tolerance: f64, iters: ?*usize) (EigenError || std.mem.Allocator.Error)![]f64 {
    if (A.rows != A.cols) return EigenError.NotSquare;
    const n = A.rows;

    var Ak = try A.clone();
    defer Ak.deinit();

    // Balance, then reduce to upper Hessenberg form once (Householder
    // similarity), so the single-Givens-per-column sweep below is valid and
    // each sweep is O(n²). For a symmetric A the reduction produces a
    // tridiagonal matrix. Both steps preserve the eigenvalues, which is all
    // we read out at the end.
    balance(Ak);
    try hessenbergReduce(alloc, Ak);

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

/// Returns the eigenvalues of A as a slice (the diagonal after reduction).
/// Caller owns the returned slice. If 'iters' is non-null, the QR sweep count
/// is written to it.
///
/// Balances A (Parlett-Reinsch) and reduces it to upper Hessenberg form once
/// (Householder similarity), then runs the QR algorithm with Wilkinson shift
/// and deflation, applying each shifted sweep as in-place Givens rotations.
/// For a symmetric A the reduction yields a tridiagonal matrix.
///
/// Accepts any real square matrix, including matrices with complex-conjugate
/// eigenvalue pairs. The iteration stays in real arithmetic (real Schur form):
/// a converged 2x2 block is deflated as a unit and its eigenvalue pair is
/// extracted analytically. An exceptional ad-hoc shift is applied every 10
/// stalled sweeps, since the true shift of a complex block is complex.
///
/// Stops once every block has deflated, or after 'max_iter' sweeps
/// (best effort: the remaining diagonal/blocks are extracted either way).
///
/// Returns an EigenError.NotSquare if A is not square.
pub fn qrAlgorithmComplex(alloc: std.mem.Allocator, A: Mat, max_iter: usize, tolerance: f64, iters: ?*usize) (EigenError || std.mem.Allocator.Error)![]Complex {
    if (A.rows != A.cols) return EigenError.NotSquare;
    const n = A.rows;

    var Ak = try A.clone();
    defer Ak.deinit();

    // Balance, then reduce to upper Hessenberg form once (Householder
    // similarity), so the single-Givens-per-column sweep below is valid and
    // each sweep is O(n²). For a symmetric A the reduction produces a
    // tridiagonal matrix. Both steps preserve the eigenvalues, which is all
    // we read out at the end.
    balance(Ak);
    try hessenbergReduce(alloc, Ak);

    const rotation = try alloc.alloc(Givens, n);
    defer alloc.free(rotation);

    const eigs = try alloc.alloc(Complex, n);
    errdefer alloc.free(eigs);

    var iter: usize = 0;
    var m: usize = n; // Active leading block for deflation
    var stall: usize = 0; // sweeps since the last deflation

    while (m > 0 and iter < max_iter) : (iter += 1) {
        if (m == 1) { // Last 1x1 block
            eigs[0] = Complex.init(Ak.atUnsafe(0, 0), 0);
            m = 0;
            break;
        }

        // Deflation: is the bottom subdiagonal of the active block negligible?
        const sub = @abs(Ak.atUnsafe(m - 1, m - 2));
        const scale = @abs(Ak.atUnsafe(m - 2, m - 2)) + @abs(Ak.atUnsafe(m - 1, m - 1));
        if (sub <= tolerance * scale) { // lock in Ak[m-1,m-1] as a converged eigenvalue
            eigs[m - 1] = Complex.init(Ak.atUnsafe(m - 1, m - 1), 0);
            m -= 1;
            stall = 0;
            continue;
        }

        const block_done = if (m == 2) true else blk: {
            const sub2 = @abs(Ak.atUnsafe(m - 2, m - 3));
            const scale2 = @abs(Ak.atUnsafe(m - 3, m - 3)) + @abs(Ak.atUnsafe(m - 2, m - 2));
            break :blk sub2 <= tolerance * scale2;
        };
        if (block_done) {
            const pair = eig2x2(Ak.atUnsafe(m - 2, m - 2), Ak.atUnsafe(m - 2, m - 1), Ak.atUnsafe(m - 1, m - 2), Ak.atUnsafe(m - 1, m - 1));
            eigs[m - 1] = pair[1];
            eigs[m - 2] = pair[0];
            m -= 2;
            stall = 0;
            continue;
        }
        var mu: f64 = undefined;
        if (stall > 0 and stall % 10 == 0) {
            // Breaks cycles  the real Wilkinson shift can't,
            // since the shift is complex in nature
            mu = @abs(Ak.atUnsafe(m - 1, m - 2)) +
                (if (m > 2) @abs(Ak.atUnsafe(m - 2, m - 3)) else 0.0);
        } else {
            // Wilkinson shift from the trailing 2x2 of the active block.
            const a = Ak.atUnsafe(m - 2, m - 2);
            const b = Ak.atUnsafe(m - 2, m - 1);
            const c = Ak.atUnsafe(m - 1, m - 2);
            const d = Ak.atUnsafe(m - 1, m - 1);
            const delta = (a - d) / 2.0;
            const bc = b * c;
            const denom = @abs(delta) + @sqrt(@max(delta * delta + bc, 0.0));
            mu = d; // Fallback so mu is never left undefined
            if (denom != 0) {
                const sign: f64 = if (delta >= 0) 1.0 else -1.0;
                mu = d - sign * bc / denom;
            }
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
        stall += 1;
    }

    if (iters) |p| p.* = iter;

    // Best effort if max_iter was exhausted: extract whatever remains of the
    // active block, pairwise where a subdiagonal has not converged.
    var i: usize = m;
    while (i > 0) {
        if (i == 1) {
            eigs[0] = Complex.init(Ak.atUnsafe(0, 0), 0);
            i = 0;
        } else if (@abs(Ak.atUnsafe(i - 1, i - 2)) <=
            tolerance * (@abs(Ak.atUnsafe(i - 2, i - 2)) + @abs(Ak.atUnsafe(i - 1, i - 1))))
        {
            eigs[i - 1] = Complex.init(Ak.atUnsafe(i - 1, i - 1), 0);
            i -= 1;
        } else {
            const pair = eig2x2(Ak.atUnsafe(i - 2, i - 2), Ak.atUnsafe(i - 2, i - 1), Ak.atUnsafe(i - 1, i - 2), Ak.atUnsafe(i - 1, i - 1));
            eigs[i - 2] = pair[0];
            eigs[i - 1] = pair[1];
            i -= 2;
        }
    }

    return eigs;
}

/// Returns the eigenvalues of an n×n real matrix A. Caller owns the returned
/// slice. If 'iters' is non-null, the QR sweep count is written to it.
///
/// Delegates to 'qrAlgorithm': balancing, then a deterministic Householder
/// Hessenberg reduction (no start vector, no breakdown), then shifted QR
/// with deflation.
///
/// LIMITATION: same real-spectrum restriction as 'qrAlgorithm' — matrices
/// with complex-conjugate eigenvalue pairs will not converge; use
/// 'eigenvaluesComplex' for those.
///
/// Returns an EigenError.NotSquare if A is not square.
pub fn eigenvalues(alloc: std.mem.Allocator, A: Mat, max_iter: usize, tolerance: f64, iters: ?*usize) (EigenError || std.mem.Allocator.Error)![]f64 {
    return qrAlgorithm(alloc, A, max_iter, tolerance, iters);
}

/// Returns the eigenvalues of an n×n real matrix A, including
/// complex-conjugate pairs. Caller owns the returned slice. If 'iters' is
/// non-null, the QR sweep count is written to it.
///
/// Delegates to 'qrAlgorithmComplex': balancing, then a deterministic
/// Householder Hessenberg reduction (no start vector, no breakdown), then
/// shifted QR in real arithmetic (real Schur form) with 2x2 blocks
/// extracted analytically.
///
/// Returns an EigenError.NotSquare if A is not square.
pub fn eigenvaluesComplex(alloc: std.mem.Allocator, A: Mat, max_iter: usize, tolerance: f64, iters: ?*usize) (EigenError || std.mem.Allocator.Error)![]Complex {
    return qrAlgorithmComplex(alloc, A, max_iter, tolerance, iters);
}

/// Returns the 'm' eigenvalues of an m×m matrix A. Caller owns the returned
/// slice. If 'iters' is non-null, the QR sweep count is written to it.
///
/// Reduces A to upper Hessenberg form with a full Arnoldi run, then extracts
/// the eigenvalues with 'qrAlgorithm'.
///
/// NOTE: 'qrAlgorithm' already performs its own Hessenberg reduction, so for a
/// dense full-spectrum problem this path reduces twice and is slower than
/// calling 'qrAlgorithm' (or 'eigenvalues') directly — prefer those here.
/// This pipeline exists to exercise the Arnoldi reduction; its real niche is
/// large, sparse matrices where only a few eigenvalues are wanted (run
/// Arnoldi to a small dimension).
///
/// LIMITATION: same real-spectrum restriction as 'qrAlgorithm' — intended for
/// symmetric / real-eigenvalue matrices; complex eigenvalues are not supported.
/// A fixed all-ones start vector is used, so for an input it fails to excite
/// the Arnoldi process may break down early and return only a partial spectrum.
///
/// Returns an EigenError.NotSquare if A is not square.
pub fn eigenvaluesArnoldi(alloc: std.mem.Allocator, A: Mat, max_iter: usize, tolerance: f64, iters: ?*usize) (EigenError || std.mem.Allocator.Error)![]f64 {
    if (A.rows != A.cols) return EigenError.NotSquare;
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

/// Returns the 'm' eigenvalues of an m×m matrix A. Caller owns the returned
/// slice. If 'iters' is non-null, the QR sweep count is written to it.
///
/// Reduces A to upper Hessenberg form with a full Arnoldi run, then extracts
/// the eigenvalues with 'qrAlgorithm'.
///
/// NOTE: 'qrAlgorithmComplex' already performs its own Hessenberg reduction,
/// so for a dense full-spectrum problem this path reduces twice and is slower
/// than calling 'qrAlgorithmComplex' (or 'eigenvaluesComplex') directly —
/// prefer those here. This pipeline exists to exercise the Arnoldi reduction;
/// its real niche is large, sparse matrices where only a few eigenvalues are
/// wanted (run Arnoldi to a small dimension).
///
/// Complex-conjugate eigenvalue pairs are supported via 'qrAlgorithmComplex'.
/// A fixed all-ones start vector is used, so for an input it fails to excite
/// the Arnoldi process may break down early and return only a partial spectrum.
///
/// Returns an EigenError.NotSquare if A is not square.
pub fn eigenvaluesComplexArnoldi(alloc: std.mem.Allocator, A: Mat, max_iter: usize, tolerance: f64, iters: ?*usize) (EigenError || std.mem.Allocator.Error)![]Complex {
    if (A.rows != A.cols) return EigenError.NotSquare;
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

    return qrAlgorithmComplex(alloc, H, max_iter, tolerance, iters);
}

/// Returns the roots of a real polynomial as the eigenvalues of its
/// companion matrix (via 'qrAlgorithmComplex'). Caller owns the returned
/// slice, which has length = the degree of the polynomial.
///
/// 'coeffs' are in descending degree, matching 'charPoly':
///   coeffs[0]·xⁿ + coeffs[1]·xⁿ⁻¹ + … + coeffs[n]
///
/// Leading zero coefficients are stripped. Trailing zero coefficients
/// (roots at x = 0) are deflated exactly instead of going through QR.
/// A constant, empty, or all-zero polynomial has no roots: an empty slice
/// is returned.
pub fn roots(alloc: std.mem.Allocator, coeffs: []const f64, max_iter: usize, tolerance: f64) (EigenError || std.mem.Allocator.Error)![]Complex {
    // Strip leading zeros to find the true degree.
    var lead: usize = 0;
    while (lead < coeffs.len and coeffs[lead] == 0.0) lead += 1;
    const p = coeffs[lead..];
    if (p.len < 2) return alloc.alloc(Complex, 0);
    const n = p.len - 1; // true degree

    // Deflate roots at zero exactly (cheaper and more accurate than QR).
    var tz: usize = 0;
    while (tz < n and p[p.len - 1 - tz] == 0.0) tz += 1;
    const q = p[0 .. p.len - tz];
    const m = q.len - 1; // degree after removing the roots at zero

    const out = try alloc.alloc(Complex, n);
    errdefer alloc.free(out);
    for (m..n) |i| out[i] = Complex.init(0, 0);
    if (m == 0) return out;
    if (m == 1) {
        out[0] = Complex.init(-q[1] / q[0], 0);
        return out;
    }

    // Companion matrix, top-row form. Balancing inside qrAlgorithmComplex
    // takes care of widely scaled coefficients.
    var C = try Mat.initZero(alloc, m, m);
    defer C.deinit();
    for (0..m) |j| C.setUnsafe(0, j, -q[j + 1] / q[0]);
    for (1..m) |i| C.setUnsafe(i, i - 1, 1.0);

    const eigs = try qrAlgorithmComplex(alloc, C, max_iter, tolerance, null);
    defer alloc.free(eigs);
    @memcpy(out[0..m], eigs);
    return out;
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
/// Returns an EigenError.NotSquare if A is not square.
pub fn qrAlgorithm_LEGACY(alloc: std.mem.Allocator, A: Mat, max_iter: usize, tolerance: f64, iters: ?*usize) (EigenError || qr_mod.QRError || std.mem.Allocator.Error)![]f64 {
    if (A.rows != A.cols) return EigenError.NotSquare;
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

/// Columns 0..ncols-1 of Q are orthonormal: Q_c^H Q_c ≈ I.
fn expectOrthonormal(Q: anytype, ncols: usize, alloc: std.mem.Allocator) !void {
    const T = mat.ElementOf(@TypeOf(Q));
    for (0..ncols) |i| {
        var qi = try Q.getCol(i, alloc);
        defer qi.deinit();
        for (0..ncols) |j| {
            var qj = try Q.getCol(j, alloc);
            defer qj.deinit();
            const d = try vec.dot(qi, qj);
            const expected = if (i == j) sclr.one(T) else sclr.zero(T);
            try testing.expect(sclr.approxEq(d, expected, tol));
        }
    }
}

/// Arnoldi relation: A * Q[:, k] == Q * h[:, k] for k = 0..n-1.
fn expectArnoldiRelation(A: anytype, Q: @TypeOf(A), h: @TypeOf(A), n: usize, alloc: std.mem.Allocator) !void {
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
            try testing.expect(sclr.approxEq(lhs.atUnsafe(i), rhs.atUnsafe(i), tol));
        }
    }
}

/// h must be upper Hessenberg: zero below the first subdiagonal.
fn expectHessenberg(h: anytype, n: usize) !void {
    for (0..n) |j| {
        for (j + 2..n + 1) |i| {
            try testing.expect(sclr.isZeroApprox(h.atUnsafe(i, j), tol));
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

test "Arnoldi: Complex Hermitian matrix (Lanczos structure, breakdown)" {
    const alloc = testing.allocator;
    const Cx = std.math.Complex(f64);

    // A = [[2, i], [-i, 2]] (Hermitian). Hand-computed with b = e1:
    //   q0 = e1, v = (2, -i)^T, h00 = 2, h10 = 1, q1 = (0, -i)^T
    //   v = A*q1 = (1, -2i)^T, h01 = 1, h11 = 2, residual = 0 -> breakdown
    var A = try mat.CMat.initZero(alloc, 2, 2);
    defer A.deinit();
    try A.setRow(0, [_]Cx{ Cx.init(2, 0), Cx.init(0, 1) });
    try A.setRow(1, [_]Cx{ Cx.init(0, -1), Cx.init(2, 0) });

    var b = try vec.Vector(Cx).initZero(alloc, 2, true);
    defer b.deinit();
    b.setUnsafe(0, Cx.init(1, 0));

    var result = try arnoldi_iteration(A, b, 2, alloc);
    defer result.Q.deinit();
    defer result.h.deinit();

    // The first 2 basis vectors span C^2; requires the conjugating inner product.
    try expectOrthonormal(result.Q, 2, alloc);
    try expectArnoldiRelation(A, result.Q, result.h, 2, alloc);

    // Hermitian input -> h is Hermitian tridiagonal with real diagonal.
    try testing.expect(sclr.approxEq(result.h.atUnsafe(0, 0), Cx.init(2, 0), tol));
    try testing.expect(sclr.approxEq(result.h.atUnsafe(1, 0), Cx.init(1, 0), tol));
    try testing.expect(sclr.approxEq(result.h.atUnsafe(0, 1), Cx.init(1, 0), tol));
    try testing.expect(sclr.approxEq(result.h.atUnsafe(1, 1), Cx.init(2, 0), tol));

    // Krylov space is exhausted at m = 2 -> lucky breakdown.
    try testing.expect(sclr.abs(result.h.atUnsafe(2, 1)) < tol);
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

test "eigenvaluesArnoldi: Arnoldi + QR pipeline (symmetric)" {
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

        const eigs = try eigenvaluesArnoldi(alloc, A, 1000, 1e-12, null);
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

        const eigs = try eigenvaluesArnoldi(alloc, A, 1000, 1e-12, null);
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

// Sort order for complex results in tests: by real part, then imaginary part.
fn lessComplex(_: void, x: Complex, y: Complex) bool {
    if (x.re != y.re) return x.re < y.re;
    return x.im < y.im;
}

// Order-independent check: the expected eigenvalue must appear somewhere in
// the result. Positional comparison after sorting is brittle when distinct
// eigenvalues share a real part (the sort order then depends on FP noise).
fn expectEigContains(eigs: []const Complex, re: f64, im: f64, tol_: f64) !void {
    for (eigs) |e| {
        if (@abs(e.re - re) <= tol_ and @abs(e.im - im) <= tol_) return;
    }
    return error.ExpectedEigenvalueMissing;
}

test "qrAlgorithmComplex: rotation matrix -> cos +- i*sin" {
    const alloc = testing.allocator;
    const theta: f64 = std.math.pi / 3.0;
    const co: f64 = @cos(theta);
    const si: f64 = @sin(theta);

    var A = try Mat.initZero(alloc, 2, 2);
    defer A.deinit();
    setMat(A, [_][2]f64{
        .{ co, -si },
        .{ si, co },
    });

    const eigs = try qrAlgorithmComplex(alloc, A, 1000, 1e-12, null);
    defer alloc.free(eigs);
    std.mem.sort(Complex, eigs, {}, lessComplex);

    try testing.expectApproxEqAbs(co, eigs[0].re, 1e-9);
    try testing.expectApproxEqAbs(-si, eigs[0].im, 1e-9);
    try testing.expectApproxEqAbs(co, eigs[1].re, 1e-9);
    try testing.expectApproxEqAbs(si, eigs[1].im, 1e-9);
}

test "qrAlgorithmComplex: companion of (x-1)(x^2-2x+5) -> 1, 1 +- 2i" {
    const alloc = testing.allocator;

    // p(x) = x^3 - 3x^2 + 7x - 5, companion (top-row form)
    var A = try Mat.initZero(alloc, 3, 3);
    defer A.deinit();
    setMat(A, [_][3]f64{
        .{ 3, -7, 5 },
        .{ 1, 0, 0 },
        .{ 0, 1, 0 },
    });

    const eigs = try qrAlgorithmComplex(alloc, A, 1000, 1e-12, null);
    defer alloc.free(eigs);

    // All real parts are 1, so a sorted positional compare is FP-noise
    // dependent; check for membership instead.
    try expectEigContains(eigs, 1.0, 0.0, 1e-8);
    try expectEigContains(eigs, 1.0, 2.0, 1e-8);
    try expectEigContains(eigs, 1.0, -2.0, 1e-8);
}

test "qrAlgorithmComplex: two complex pairs, companion of (x^2-2x+5)(x^2-4x+13)" {
    const alloc = testing.allocator;

    // p(x) = x^4 - 6x^3 + 26x^2 - 46x + 65, roots 1 ± 2i and 2 ± 3i.
    var A = try Mat.initZero(alloc, 4, 4);
    defer A.deinit();
    setMat(A, [_][4]f64{
        .{ 6, -26, 46, -65 },
        .{ 1, 0, 0, 0 },
        .{ 0, 1, 0, 0 },
        .{ 0, 0, 1, 0 },
    });

    const eigs = try qrAlgorithmComplex(alloc, A, 1000, 1e-12, null);
    defer alloc.free(eigs);
    std.mem.sort(Complex, eigs, {}, lessComplex);

    const expected_re = [_]f64{ 1.0, 1.0, 2.0, 2.0 };
    const expected_im = [_]f64{ -2.0, 2.0, -3.0, 3.0 };
    for (0..4) |i| {
        try testing.expectApproxEqAbs(expected_re[i], eigs[i].re, 1e-8);
        try testing.expectApproxEqAbs(expected_im[i], eigs[i].im, 1e-8);
    }
}

test "qrAlgorithmComplex: symmetric matrix regression, all imaginary parts zero" {
    const alloc = testing.allocator;

    // Same 4x4 tridiagonal as the real qrAlgorithm test; must give the same
    // spectrum with zero imaginary parts.
    var A = try Mat.initZero(alloc, 4, 4);
    defer A.deinit();
    setMat(A, [_][4]f64{
        .{ 4, 1, 0, 0 },
        .{ 1, 3, 1, 0 },
        .{ 0, 1, 2, 1 },
        .{ 0, 0, 1, 1 },
    });

    const eigs = try qrAlgorithmComplex(alloc, A, 1000, 1e-12, null);
    defer alloc.free(eigs);
    std.mem.sort(Complex, eigs, {}, lessComplex);

    const expected = [_]f64{ 0.25471876, 1.82271708, 3.17728292, 4.74528124 };
    for (0..4) |i| {
        try testing.expectApproxEqAbs(expected[i], eigs[i].re, 1e-6);
        try testing.expectApproxEqAbs(0.0, eigs[i].im, 1e-9);
    }
}

test "eigenvaluesComplexArnoldi: Arnoldi + complex QR pipeline" {
    const alloc = testing.allocator;

    // Same 4x4 companion as above (all-ones start vector is not an
    // eigenvector of it, so Arnoldi runs to full dimension).
    var A = try Mat.initZero(alloc, 4, 4);
    defer A.deinit();
    setMat(A, [_][4]f64{
        .{ 6, -26, 46, -65 },
        .{ 1, 0, 0, 0 },
        .{ 0, 1, 0, 0 },
        .{ 0, 0, 1, 0 },
    });

    const eigs = try eigenvaluesComplexArnoldi(alloc, A, 1000, 1e-12, null);
    defer alloc.free(eigs);
    std.mem.sort(Complex, eigs, {}, lessComplex);

    const expected_re = [_]f64{ 1.0, 1.0, 2.0, 2.0 };
    const expected_im = [_]f64{ -2.0, 2.0, -3.0, 3.0 };
    for (0..4) |i| {
        try testing.expectApproxEqAbs(expected_re[i], eigs[i].re, 1e-6);
        try testing.expectApproxEqAbs(expected_im[i], eigs[i].im, 1e-6);
    }
}

test "qrAlgorithmComplex: no leaks on allocation failure" {
    const a_data = [_][3]f64{
        .{ 3, -7, 5 },
        .{ 1, 0, 0 },
        .{ 0, 1, 0 },
    };

    for (0..60) |fail_index| {
        var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const alloc = fa.allocator();

        const A_or = Mat.initZero(alloc, 3, 3);
        if (A_or) |A_| {
            var A = A_;
            defer A.deinit();
            setMat(A, a_data);

            const res = qrAlgorithmComplex(alloc, A, 1000, 1e-12, null);
            if (res) |eigs| {
                alloc.free(eigs);
            } else |e| {
                try std.testing.expect(e == error.OutOfMemory);
            }
        } else |e| {
            try std.testing.expect(e == error.OutOfMemory);
        }
    }
}

test "balancing: badly scaled matrix recovers accurate eigenvalues" {
    const alloc = testing.allocator;

    // D C D⁻¹ where C is the companion of (x-1)(x-2)(x-3) and
    // D = diag(1, 1e6, 1e12). Entries span 1e-12 .. 1e12; without balancing
    // the QR iteration loses roughly half the significant digits here.
    // The spectrum is exactly {1, 2, 3}.
    var A = try Mat.initZero(alloc, 3, 3);
    defer A.deinit();
    setMat(A, [_][3]f64{
        .{ 6.0, -1.1e-5, 6.0e-12 },
        .{ 1.0e6, 0.0, 0.0 },
        .{ 0.0, 1.0e6, 0.0 },
    });

    const eigs = try qrAlgorithm(alloc, A, 1000, 1e-12, null);
    defer alloc.free(eigs);
    std.mem.sort(f64, eigs, {}, std.sort.desc(f64));

    try testing.expectApproxEqAbs(@as(f64, 3.0), eigs[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 2.0), eigs[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 1.0), eigs[2], 1e-6);

    // Same matrix through the complex path.
    const ceigs = try qrAlgorithmComplex(alloc, A, 1000, 1e-12, null);
    defer alloc.free(ceigs);
    try expectEigContains(ceigs, 1.0, 0.0, 1e-6);
    try expectEigContains(ceigs, 2.0, 0.0, 1e-6);
    try expectEigContains(ceigs, 3.0, 0.0, 1e-6);
}

test "eigenvalues / eigenvaluesComplex: delegate to the QR algorithms" {
    const alloc = testing.allocator;

    // Real spectrum via the friendly wrapper.
    {
        var A = try Mat.initZero(alloc, 2, 2);
        defer A.deinit();
        setMat(A, [_][2]f64{ .{ 2, 1 }, .{ 1, 2 } });

        const eigs = try eigenvalues(alloc, A, 500, 1e-12, null);
        defer alloc.free(eigs);
        std.mem.sort(f64, eigs, {}, std.sort.desc(f64));
        try testing.expectApproxEqAbs(@as(f64, 3.0), eigs[0], 1e-9);
        try testing.expectApproxEqAbs(@as(f64, 1.0), eigs[1], 1e-9);
    }

    // Complex pair via the friendly wrapper. This companion matrix makes the
    // all-ones-start Arnoldi pipeline unreliable; the Householder route has
    // no such failure mode.
    {
        var A = try Mat.initZero(alloc, 3, 3);
        defer A.deinit();
        setMat(A, [_][3]f64{
            .{ 3, -7, 5 },
            .{ 1, 0, 0 },
            .{ 0, 1, 0 },
        });

        const eigs = try eigenvaluesComplex(alloc, A, 1000, 1e-12, null);
        defer alloc.free(eigs);
        try expectEigContains(eigs, 1.0, 0.0, 1e-8);
        try expectEigContains(eigs, 1.0, 2.0, 1e-8);
        try expectEigContains(eigs, 1.0, -2.0, 1e-8);
    }
}

test "roots: real and complex roots, descending coefficients" {
    const alloc = testing.allocator;

    // x² - 3x + 2 = (x-1)(x-2)
    {
        const rs = try roots(alloc, &[_]f64{ 1, -3, 2 }, 1000, 1e-12);
        defer alloc.free(rs);
        try testing.expectEqual(@as(usize, 2), rs.len);
        try expectEigContains(rs, 1.0, 0.0, 1e-9);
        try expectEigContains(rs, 2.0, 0.0, 1e-9);
    }

    // Non-monic: 2x² - 6x + 4 has the same roots.
    {
        const rs = try roots(alloc, &[_]f64{ 2, -6, 4 }, 1000, 1e-12);
        defer alloc.free(rs);
        try testing.expectEqual(@as(usize, 2), rs.len);
        try expectEigContains(rs, 1.0, 0.0, 1e-9);
        try expectEigContains(rs, 2.0, 0.0, 1e-9);
    }

    // (x²-2x+5)(x²-4x+13) = x⁴ - 6x³ + 26x² - 46x + 65, roots 1±2i, 2±3i.
    {
        const rs = try roots(alloc, &[_]f64{ 1, -6, 26, -46, 65 }, 1000, 1e-12);
        defer alloc.free(rs);
        try testing.expectEqual(@as(usize, 4), rs.len);
        try expectEigContains(rs, 1.0, 2.0, 1e-8);
        try expectEigContains(rs, 1.0, -2.0, 1e-8);
        try expectEigContains(rs, 2.0, 3.0, 1e-8);
        try expectEigContains(rs, 2.0, -3.0, 1e-8);
    }
}

test "roots: degenerate shapes (leading/trailing zeros, constants, linear)" {
    const alloc = testing.allocator;

    // Leading zeros are stripped: [0, 0, 1, -3, 2] is still x² - 3x + 2.
    {
        const rs = try roots(alloc, &[_]f64{ 0, 0, 1, -3, 2 }, 1000, 1e-12);
        defer alloc.free(rs);
        try testing.expectEqual(@as(usize, 2), rs.len);
        try expectEigContains(rs, 1.0, 0.0, 1e-9);
        try expectEigContains(rs, 2.0, 0.0, 1e-9);
    }

    // Trailing zeros are exact roots at 0: x³ - x² = x²(x - 1).
    {
        const rs = try roots(alloc, &[_]f64{ 1, -1, 0, 0 }, 1000, 1e-12);
        defer alloc.free(rs);
        try testing.expectEqual(@as(usize, 3), rs.len);
        var zeros: usize = 0;
        for (rs) |r| {
            if (r.re == 0.0 and r.im == 0.0) zeros += 1;
        }
        try testing.expectEqual(@as(usize, 2), zeros);
        try expectEigContains(rs, 1.0, 0.0, 1e-9);
    }

    // Pure powers of x: x² -> two exact zero roots, no QR involved.
    {
        const rs = try roots(alloc, &[_]f64{ 1, 0, 0 }, 1000, 1e-12);
        defer alloc.free(rs);
        try testing.expectEqual(@as(usize, 2), rs.len);
        for (rs) |r| {
            try testing.expectEqual(@as(f64, 0.0), r.re);
            try testing.expectEqual(@as(f64, 0.0), r.im);
        }
    }

    // Linear: 2x - 4 -> 2, solved analytically.
    {
        const rs = try roots(alloc, &[_]f64{ 2, -4 }, 1000, 1e-12);
        defer alloc.free(rs);
        try testing.expectEqual(@as(usize, 1), rs.len);
        try testing.expectApproxEqAbs(@as(f64, 2.0), rs[0].re, 1e-12);
        try testing.expectApproxEqAbs(@as(f64, 0.0), rs[0].im, 1e-12);
    }

    // Constant, empty, and all-zero polynomials have no roots.
    {
        const c = try roots(alloc, &[_]f64{5.0}, 1000, 1e-12);
        defer alloc.free(c);
        try testing.expectEqual(@as(usize, 0), c.len);

        const e = try roots(alloc, &[_]f64{}, 1000, 1e-12);
        defer alloc.free(e);
        try testing.expectEqual(@as(usize, 0), e.len);

        const z = try roots(alloc, &[_]f64{ 0, 0, 0 }, 1000, 1e-12);
        defer alloc.free(z);
        try testing.expectEqual(@as(usize, 0), z.len);
    }
}

test "roots: charPoly of a companion round-trips (root locus path)" {
    const alloc = testing.allocator;

    // The app's hottest path: charPoly -> roots. Build A with known spectrum
    // {1, 2, 3}, take its characteristic polynomial (descending, monic), and
    // recover the spectrum as roots.
    var A = try Mat.initZero(alloc, 3, 3);
    defer A.deinit();
    setMat(A, [_][3]f64{
        .{ 6, -11, 6 },
        .{ 1, 0, 0 },
        .{ 0, 1, 0 },
    });

    var coeffs: [4]f64 = undefined;
    try mat.charPoly(alloc, A, &coeffs);

    const rs = try roots(alloc, &coeffs, 1000, 1e-12);
    defer alloc.free(rs);
    try testing.expectEqual(@as(usize, 3), rs.len);
    try expectEigContains(rs, 1.0, 0.0, 1e-8);
    try expectEigContains(rs, 2.0, 0.0, 1e-8);
    try expectEigContains(rs, 3.0, 0.0, 1e-8);
}

test "roots: no leaks on allocation failure" {
    const coeffs = [_]f64{ 1, -6, 26, -46, 65 };

    for (0..60) |fail_index| {
        var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const alloc = fa.allocator();

        const res = roots(alloc, &coeffs, 1000, 1e-12);
        if (res) |rs| {
            alloc.free(rs);
        } else |e| {
            try std.testing.expect(e == error.OutOfMemory);
        }
    }
}
