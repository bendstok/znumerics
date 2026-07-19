const std = @import("std");
const mat = @import("../core/mat.zig");
const err_mod = @import("../error.zig");
const vec = @import("../core/vec.zig");
const sclr = @import("../core/scalar.zig");

const Mat = mat.Mat;

pub fn SVDResult(comptime T: type) type {
    return struct {
        const Self = @This();
        S: mat.Matrix(T),
        V: mat.Matrix(T),
        U: mat.Matrix(T),
        alloc: std.mem.Allocator,

        pub fn deinit(self: *Self) void {
            self.S.deinit();
            self.V.deinit();
            self.U.deinit();
            self.* = undefined;
        }

        pub fn print(self: *Self) !void {
            std.debug.print("S: \n", .{});
            try self.S.printMat();
            std.debug.print("V: \n", .{});
            try self.V.printMat();
            std.debug.print("U: \n", .{});
            try self.U.printMat();
        }
    };
}

/// Thin SVD via one-sided Jacobi (Hestenes): A = U * S * V^T.
///
/// U is m x n with orthonormal columns, S is n x n diagonal with the
/// singular values sorted descending (all >= 0), V is n x n orthogonal.
/// The returned SVDResult owns its memory; call .deinit() when done.
///
/// Cyclic sweeps rotate column pairs of A until every pair is orthogonal;
/// max_iter caps the number of full sweeps (a handful is typically enough).
/// A pair is skipped once |a_p . a_q| <= 1e-15 * ||a_p|| * ||a_q||, so
/// sweeps after convergence only cost the dot products.
///
/// Intended for m >= n (for m < n, factor A^T and swap U and V).
///
/// NB: for rank-deficient A the trailing singular values are ~0 and their
/// U columns are left as-is (near zero, not orthonormal); U^T * U = I
/// holds on the leading rank columns only.
pub fn svdJacobi(alloc: std.mem.Allocator, U: anytype, max_iter: usize) !SVDResult(mat.ElementOf(@TypeOf(U))) {
    const T = mat.ElementOf(@TypeOf(U));
    const u_r = U.rows;
    const u_c = U.cols;

    var S = try mat.Matrix(T).initZero(alloc, u_c, u_c);
    errdefer S.deinit();
    var V = try mat.Matrix(T).initIdentity(alloc, u_c, u_c);
    errdefer V.deinit();

    var A_k = try mat.Matrix(T).initZero(alloc, u_r, u_c);
    errdefer A_k.deinit();
    try mat.copyMat(U, A_k);

    for (0..max_iter) |_| {

        // cyclic pivot
        for (0..A_k.cols - 1) |p| {
            for (p + 1..A_k.cols) |q| {
                var a: T = sclr.zero(T);
                var b: T = sclr.zero(T);
                var c: T = sclr.zero(T);
                for (0..A_k.rows) |i| {
                    const ap = A_k.atUnsafe(i, p);
                    const aq = A_k.atUnsafe(i, q);
                    a = sclr.add(a, sclr.mul(ap, ap));
                    b = sclr.add(b, sclr.mul(aq, aq));
                    c = sclr.add(c, sclr.mul(ap, aq));
                }

                //if (@abs(c) <= 1e-15 * @sqrt(a * b)) std.debug.print("UH OH! c <= 1e-15!! \n", .{});
                if (@abs(c) <= 1e-15 * @sqrt(a * b)) continue; // guard

                //std.debug.print("Pivot found: p:{}, q:{} \n", .{ p, q });

                const alpha = sclr.div(sclr.sub(b, a), sclr.mul(c, sclr.fromReal(T, 2)));
                const t = if (sclr.abs(alpha) < 1e-300) sclr.one(T) else sclr.div(sclr.fromReal(T, std.math.sign(alpha)), sclr.add(sclr.abs(alpha), sclr.sqrt(sclr.add(sclr.mul(alpha, alpha), sclr.fromReal(T, 1)))));
                const cs = sclr.div(sclr.fromReal(T, 1), sclr.sqrt(sclr.add(sclr.mul(t, t), sclr.fromReal(T, 1))));
                const sn = sclr.mul(cs, t);

                //std.debug.print("A_K before: \n", .{});
                //try A_k.printMat();
                for (0..A_k.rows) |i| {
                    const aip = A_k.atUnsafe(i, p);
                    const aiq = A_k.atUnsafe(i, q);
                    A_k.setUnsafe(i, p, sclr.sub(sclr.mul(cs, aip), sclr.mul(sn, aiq)));
                    A_k.setUnsafe(i, q, sclr.add(sclr.mul(sn, aip), sclr.mul(cs, aiq)));
                }
                //std.debug.print("A_K be after\n", .{});
                //try A_k.printMat();
                for (0..V.rows) |i| {
                    const vip = V.atUnsafe(i, p);
                    const viq = V.atUnsafe(i, q);

                    V.setUnsafe(i, p, sclr.sub(sclr.mul(cs, vip), sclr.mul(sn, viq)));
                    V.setUnsafe(i, q, sclr.add(sclr.mul(sn, vip), sclr.mul(cs, viq)));
                }
                //std.debug.print("V: \n", .{});
                //try V.printMat();
            }
        }
    }

    const n = U.cols;
    const m = U.rows;
    //sigma + sort (selection sort is fine for a few columns)
    const sigma = try alloc.alloc(f64, n);
    errdefer alloc.free(sigma);
    for (0..n) |j| {
        var s: f64 = 0;
        for (0..m) |i| s += A_k.atUnsafe(i, j) * A_k.atUnsafe(i, j);
        sigma[j] = @sqrt(s);
    }
    for (0..n) |j| { // descending, permuting A_k and V columns alongside
        var best = j;
        for (j + 1..n) |k| if (sigma[k] > sigma[best]) {
            best = k;
        };
        if (best != j) {
            std.mem.swap(f64, &sigma[j], &sigma[best]);
            try A_k.swapCol(j, best);
            try V.swapCol(j, best);
        }
    }

    // 2. A_k -> U: normalize each column
    for (0..n) |j| {
        if (sigma[j] > 1e-300) {
            for (0..m) |i| A_k.setUnsafe(i, j, A_k.atUnsafe(i, j) / sigma[j]);
        }
    }

    for (0..n) |j| S.setUnsafe(j, j, sigma[j]);
    alloc.free(sigma);

    return .{ .U = A_k, .S = S, .V = V, .alloc = alloc };
}

/// Moore-Penrose pseudoinverse via the SVD: A+ = V * S+ * U^T, where S+
/// takes the reciprocal of each singular value above the cutoff
/// eps * max(m, n) * sigma_max and zeroes the rest (MATLAB/Octave pinv
/// convention). Returns an n x m matrix for an m x n input; caller owns it.
///
/// Works for any shape and rank: wide inputs are factored through A^T
/// (the SVD of A^T gives A+ = U_B * S+ * V_B^T directly, no final
/// transpose), and rank-deficient inputs are handled by the cutoff.
///
/// For square invertible A this equals inverse(A). For tall full-rank A,
/// pinv(A) * b is the least-squares solution of Ax = b.
pub fn pinv(alloc: std.mem.Allocator, A: anytype, max_iter: usize) !mat.Matrix(mat.ElementOf(@TypeOf(A))) {
    const wide = A.rows < A.cols; // svdJacobi wants m >= n

    var At = if (wide) try mat.transpose(A, alloc) else undefined;
    defer if (wide) At.deinit();

    var svdres = try svdJacobi(alloc, if (wide) At else A, max_iter);
    defer svdres.deinit();

    // MATLAB/Octave tolerance: t = eps * max(m, n) * sigma_max
    const tol = std.math.floatEps(f64) *
        @as(f64, @floatFromInt(@max(A.rows, A.cols))) * svdres.S.atUnsafe(0, 0);

    // S -> S+ in place: reciprocal above tol, zero below
    for (0..svdres.S.cols) |j| {
        const s = svdres.S.atUnsafe(j, j);
        svdres.S.setUnsafe(j, j, if (s > tol) 1.0 / s else 0.0);
    }

    // tall/square: A+ = V * S+ * U^T; wide: A+ = U_B * S+ * V_B^T
    const left = if (wide) svdres.U else svdres.V;
    const right = if (wide) svdres.V else svdres.U;

    var LS = try mat.matMult(alloc, left, svdres.S);
    defer LS.deinit();
    var Rt = try mat.transpose(right, alloc);
    defer Rt.deinit();
    return try mat.matMult(alloc, LS, Rt);
}

/// The singular values of A, sorted descending (all >= 0). Works for any
/// shape (wide inputs are transposed internally; the singular values of
/// A^T equal those of A). Caller owns the returned slice.
pub fn singularValues(alloc: std.mem.Allocator, A: anytype, max_iter: usize) ![]f64 {
    const wide = A.rows < A.cols;
    var At = if (wide) try mat.transpose(A, alloc) else undefined;
    defer if (wide) At.deinit();

    var svdres = try svdJacobi(alloc, if (wide) At else A, max_iter);
    defer svdres.deinit();

    const k = svdres.S.rows;
    const s = try alloc.alloc(f64, k);
    for (0..k) |j| s[j] = svdres.S.atUnsafe(j, j);
    return s;
}

/// Numerical rank: the number of singular values above tol * sigma_max.
/// Pass null for tol to get the MATLAB/Octave default eps * max(m, n).
pub fn rank(alloc: std.mem.Allocator, A: anytype, max_iter: usize, tol: ?f64) !usize {
    const s = try singularValues(alloc, A, max_iter);
    defer alloc.free(s);

    const rel = tol orelse (std.math.floatEps(f64) * @as(f64, @floatFromInt(@max(A.rows, A.cols))));
    const t = rel * s[0];
    var r: usize = 0;
    for (s) |x| {
        if (x > t) r += 1;
    }
    return r;
}

/// 2-norm condition number: sigma_max / sigma_min.
///
/// Returns inf if sigma_min falls below the rank tolerance
/// eps * max(m, n) * sigma_max — i.e. the matrix is numerically singular.
/// (A computed sigma_min below that level is round-off noise, so the ratio
/// would be a meaningless ~1e16 rather than the true value, inf.)
pub fn cond2(alloc: std.mem.Allocator, A: anytype, max_iter: usize) !f64 {
    const s = try singularValues(alloc, A, max_iter);
    defer alloc.free(s);

    const t = std.math.floatEps(f64) * @as(f64, @floatFromInt(@max(A.rows, A.cols))) * s[0];
    const smin = s[s.len - 1];
    if (smin <= t) return std.math.inf(f64);
    return s[0] / smin;
}

/// Spectral norm ||A||_2 = sigma_max: the largest factor by which A can
/// stretch a vector.
pub fn norm2(alloc: std.mem.Allocator, A: anytype, max_iter: usize) !f64 {
    const s = try singularValues(alloc, A, max_iter);
    defer alloc.free(s);
    return s[0];
}
