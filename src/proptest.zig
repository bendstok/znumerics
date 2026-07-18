//! Property-based tests on random matrices.
//!
//! Each test checks a mathematical identity that must hold (up to floating
//! point error) for *any* matrix, evaluated over a grid of seeds and sizes.

const std = @import("std");
const mat = @import("core/mat.zig");
const vec = @import("core/vec.zig");
const sclr = @import("core/scalar.zig");
const lu_mod = @import("linalg/lu.zig");
const qr_mod = @import("linalg/qrdecomposition.zig");
const chol_mod = @import("linalg/cholesky.zig");
const eigen_mod = @import("linalg/eigen.zig");
const svd_mod = @import("linalg/svd.zig");

const Cx = std.math.Complex(f64);
const Mat = mat.Mat;

const seeds = [_]u64{ 1, 2, 3, 4, 5 };
const sizes = [_]usize{ 2, 3, 5, 8 };

// ---------------------------------------------------------------- helpers

fn randMat(comptime T: type, alloc: std.mem.Allocator, n: usize, seed: u64) !mat.Matrix(T) {
    if (comptime sclr.isComplex(T)) {
        return mat.Matrix(T).initRandom(alloc, n, n, seed, T.init(-1.0, -1.0), T.init(1.0, 1.0));
    }
    return mat.Matrix(T).initRandom(alloc, n, n, seed, -1.0, 1.0);
}

fn randVec(comptime T: type, alloc: std.mem.Allocator, n: usize, seed: u64) !vec.Vector(T) {
    if (comptime sclr.isComplex(T)) {
        return vec.Vector(T).initRandom(alloc, n, true, seed, T.init(-1.0, -1.0), T.init(1.0, 1.0));
    }
    return vec.Vector(T).initRandom(alloc, n, true, seed, -1.0, 1.0);
}

fn maxAbsDiff(comptime T: type, A: mat.Matrix(T), B: mat.Matrix(T)) sclr.Real(T) {
    var m: sclr.Real(T) = 0;
    for (A.data, B.data) |a, b| {
        const d = sclr.abs(sclr.sub(a, b));
        if (d > m) m = d;
    }
    return m;
}

fn maxAbsDiffVec(comptime T: type, a: vec.Vector(T), b: vec.Vector(T)) sclr.Real(T) {
    var m: sclr.Real(T) = 0;
    for (a.data, b.data) |x, y| {
        const d = sclr.abs(sclr.sub(x, y));
        if (d > m) m = d;
    }
    return m;
}

fn maxAbsVec(comptime T: type, a: vec.Vector(T)) sclr.Real(T) {
    var m: sclr.Real(T) = 0;
    for (a.data) |x| {
        const d = sclr.abs(x);
        if (d > m) m = d;
    }
    return m;
}

// ------------------------------------------------------- A * inv(A) == I

fn checkInverse(comptime T: type, alloc: std.mem.Allocator, n: usize, seed: u64) !void {
    var A = try randMat(T, alloc, n, seed);
    defer A.deinit();
    var Ainv = try mat.inverse(alloc, A);
    defer Ainv.deinit();
    var prod = try mat.matMult(alloc, A, Ainv);
    defer prod.deinit();
    var I = try mat.Matrix(T).initIdentity(alloc, n, n);
    defer I.deinit();

    // ||A*inv(A) - I|| <= c * eps * cond(A); 1e-12 leaves ~4 digits headroom.
    const tol = 1e-12 * (1.0 + A.norm1() * Ainv.norm1());
    try std.testing.expect(maxAbsDiff(T, prod, I) < tol);
}

test "property: A * inv(A) == I (f64 and complex)" {
    const alloc = std.testing.allocator;
    for (sizes) |n| for (seeds) |seed| {
        try checkInverse(f64, alloc, n, seed);
        try checkInverse(Cx, alloc, n, seed);
    };
}

// ------------------------------------------------- LU: L * U == P * A

fn checkLU(comptime T: type, alloc: std.mem.Allocator, n: usize, seed: u64) !void {
    var A = try randMat(T, alloc, n, seed);
    defer A.deinit();
    var res = try lu_mod.lu(alloc, A);
    defer res.deinit();

    // Unpack L (implicit unit diagonal) and U from the packed factor.
    var L = try mat.Matrix(T).initIdentity(alloc, n, n);
    defer L.deinit();
    var U = try mat.Matrix(T).initZero(alloc, n, n);
    defer U.deinit();
    for (0..n) |r| for (0..n) |c| {
        if (r > c) L.setUnsafe(r, c, res.lu.atUnsafe(r, c)) else U.setUnsafe(r, c, res.lu.atUnsafe(r, c));
    };

    // P*A: row i of the factored system is row perm[i] of A.
    var PA = try mat.Matrix(T).initZero(alloc, n, n);
    defer PA.deinit();
    for (0..n) |r| for (0..n) |c| {
        PA.setUnsafe(r, c, A.atUnsafe(res.perm[r], c));
    };

    var LU = try mat.matMult(alloc, L, U);
    defer LU.deinit();

    const tol = 1e-12 * @as(f64, @floatFromInt(n)) * (1.0 + A.norm1());
    try std.testing.expect(maxAbsDiff(T, LU, PA) < tol);

    // det from the factorization must agree with determinant().
    const d1 = res.det();
    const d2 = try mat.determinant(alloc, A);
    try std.testing.expect(sclr.abs(sclr.sub(d1, d2)) < 1e-9 * (1.0 + sclr.abs(d1)));
}

test "property: LU reconstructs P*A and det matches (f64 and complex)" {
    const alloc = std.testing.allocator;
    for (sizes) |n| for (seeds) |seed| {
        try checkLU(f64, alloc, n, seed);
        try checkLU(Cx, alloc, n, seed);
    };
}

// ------------------------------------------------------ solve: A*x == b

fn checkSolve(comptime T: type, alloc: std.mem.Allocator, n: usize, seed: u64) !void {
    var A = try randMat(T, alloc, n, seed);
    defer A.deinit();
    var b = try randVec(T, alloc, n, seed + 1000);
    defer b.deinit();

    var x = try lu_mod.solve(alloc, A, b);
    defer x.deinit();
    var Ax = try mat.matVec(alloc, A, x);
    defer Ax.deinit();

    // Backward stability: ||A*x - b|| <= c * eps * ||A|| * ||x||.
    const tol = 1e-12 * (1.0 + A.norm1() * maxAbsVec(T, x));
    try std.testing.expect(maxAbsDiffVec(T, Ax, b) < tol);
}

test "property: solve gives A*x == b (f64 and complex)" {
    const alloc = std.testing.allocator;
    for (sizes) |n| for (seeds) |seed| {
        try checkSolve(f64, alloc, n, seed);
        try checkSolve(Cx, alloc, n, seed);
    };
}

// ------------------------------------------- QR: Q*R == A, Q^H * Q == I

fn checkQR(comptime T: type, alloc: std.mem.Allocator, n: usize, seed: u64) !void {
    var A = try randMat(T, alloc, n, seed);
    defer A.deinit();
    var qr = try qr_mod.qrDecomposition(alloc, A);
    defer qr[0].deinit();
    defer qr[1].deinit();
    const Q = qr[0];
    const R = qr[1];

    var QR = try mat.matMult(alloc, Q, R);
    defer QR.deinit();
    const tol = 1e-12 * @as(f64, @floatFromInt(n)) * (1.0 + A.norm1());
    try std.testing.expect(maxAbsDiff(T, QR, A) < tol);

    // Unitarity: Q^H * Q == I.
    var Qh = try mat.transpose(Q, alloc);
    defer Qh.deinit();
    for (Qh.data) |*e| e.* = sclr.conj(e.*);
    var QhQ = try mat.matMult(alloc, Qh, Q);
    defer QhQ.deinit();
    var I = try mat.Matrix(T).initIdentity(alloc, n, n);
    defer I.deinit();
    try std.testing.expect(maxAbsDiff(T, QhQ, I) < tol);

    // R upper triangular (below-diagonal entries are numerically zero).
    var below: sclr.Real(T) = 0;
    for (1..n) |r| for (0..r) |c| {
        const d = sclr.abs(R.atUnsafe(r, c));
        if (d > below) below = d;
    };
    try std.testing.expect(below < tol);
}

test "property: QR: Q*R == A, Q unitary, R triangular (f64 and complex)" {
    const alloc = std.testing.allocator;
    for (sizes) |n| for (seeds) |seed| {
        try checkQR(f64, alloc, n, seed);
        try checkQR(Cx, alloc, n, seed);
    };
}

// ------------------------------------- Cholesky: L * L^T == A (SPD)

test "property: Cholesky reconstructs SPD matrix" {
    const alloc = std.testing.allocator;
    for (sizes) |n| for (seeds) |seed| {
        var M = try randMat(f64, alloc, n, seed);
        defer M.deinit();
        var Mt = try mat.transpose(M, alloc);
        defer Mt.deinit();
        // A = M*M^T + n*I is symmetric positive definite.
        var A = try mat.matMult(alloc, M, Mt);
        defer A.deinit();
        for (0..n) |i| A.setUnsafe(i, i, A.atUnsafe(i, i) + @as(f64, @floatFromInt(n)));

        var L = try chol_mod.cholesky(alloc, A);
        defer L.deinit();
        var Lt = try mat.transpose(L, alloc);
        defer Lt.deinit();
        var LLt = try mat.matMult(alloc, L, Lt);
        defer LLt.deinit();

        const tol = 1e-12 * @as(f64, @floatFromInt(n)) * (1.0 + A.norm1());
        try std.testing.expect(maxAbsDiff(f64, LLt, A) < tol);
    };
}

// ------------------------------------------- eigenvalue sum == trace

fn checkEigenTrace(comptime T: type, alloc: std.mem.Allocator, n: usize, seed: u64) !void {
    var A = try randMat(T, alloc, n, seed);
    defer A.deinit();

    const eigs = try eigen_mod.eigenvaluesComplex(alloc, A, 50_000, 1e-12, null);
    defer alloc.free(eigs);
    try std.testing.expectEqual(n, eigs.len);

    var sum = Cx.init(0.0, 0.0);
    for (eigs) |e| sum = sum.add(e);

    const tr = try A.trace();
    const trc = if (comptime sclr.isComplex(T)) tr else Cx.init(tr, 0.0);

    // Similarity transforms preserve the trace exactly, but the solver stops
    // after max_iter regardless of convergence, so the achievable accuracy is
    // limited by accumulated QR-iteration round-off. 1e-6 still catches any
    // wrong or missing eigenvalue (an O(1) shift of the sum).
    const diff = sclr.abs(sclr.sub(sum, trc));
    const tol = 1e-6 * (1.0 + sum.magnitude() + trc.magnitude());
    if (!(diff < tol)) {
        return error.TestUnexpectedResult;
    }
}

test "property: eigenvalue sum == trace (f64 and complex)" {
    const alloc = std.testing.allocator;
    for (sizes) |n| for (seeds) |seed| {
        try checkEigenTrace(f64, alloc, n, seed);
        try checkEigenTrace(Cx, alloc, n, seed);
    };
}

// --------------------------------------- det(A*B) == det(A) * det(B)

test "property: det is multiplicative" {
    const alloc = std.testing.allocator;
    for (sizes) |n| for (seeds) |seed| {
        var A = try randMat(f64, alloc, n, seed);
        defer A.deinit();
        var B = try randMat(f64, alloc, n, seed + 1000);
        defer B.deinit();
        var AB = try mat.matMult(alloc, A, B);
        defer AB.deinit();

        const dA = try mat.determinant(alloc, A);
        const dB = try mat.determinant(alloc, B);
        const dAB = try mat.determinant(alloc, AB);

        try std.testing.expect(@abs(dAB - dA * dB) < 1e-9 * (1.0 + @abs(dA * dB)));
    };
}

// -------------------------------------------------- transpose identities

test "property: (A^T)^T == A and (A*B)^T == B^T * A^T" {
    const alloc = std.testing.allocator;
    for (sizes) |n| for (seeds) |seed| {
        var A = try randMat(f64, alloc, n, seed);
        defer A.deinit();
        var B = try randMat(f64, alloc, n, seed + 1000);
        defer B.deinit();

        // (A^T)^T == A, exactly.
        var At = try mat.transpose(A, alloc);
        defer At.deinit();
        var Att = try mat.transpose(At, alloc);
        defer Att.deinit();
        try std.testing.expectEqualSlices(f64, A.data, Att.data);

        // (A*B)^T == B^T * A^T.
        var AB = try mat.matMult(alloc, A, B);
        defer AB.deinit();
        var ABt = try mat.transpose(AB, alloc);
        defer ABt.deinit();
        var Bt = try mat.transpose(B, alloc);
        defer Bt.deinit();
        var BtAt = try mat.matMult(alloc, Bt, At);
        defer BtAt.deinit();
        try std.testing.expect(maxAbsDiff(f64, ABt, BtAt) < 1e-13);
    };
}

// ------------------------------------------- matMultSIMD == matMult

test "property: matMultSIMD agrees with matMult" {
    const alloc = std.testing.allocator;
    for (sizes) |n| for (seeds) |seed| {
        var A = try randMat(f64, alloc, n, seed);
        defer A.deinit();
        var B = try randMat(f64, alloc, n, seed + 1000);
        defer B.deinit();

        var C1 = try mat.matMult(alloc, A, B);
        defer C1.deinit();
        var C2 = try mat.matMultSIMD(alloc, A, B);
        defer C2.deinit();

        // Different summation order, so not bitwise equal — but close.
        const tol = 1e-13 * @as(f64, @floatFromInt(n));
        try std.testing.expect(maxAbsDiff(f64, C1, C2) < tol);
    };
}

// ----------------------------------- SVD (thin): U * S * V^T == A

fn checkSVD(alloc: std.mem.Allocator, m: usize, n: usize, seed: u64) !void {
    var A = try mat.Matrix(f64).initRandom(alloc, m, n, seed, -1.0, 1.0);
    defer A.deinit();

    var res = try svd_mod.svdJacobi(alloc, A, 100);
    defer res.deinit();

    // Thin shapes: U is m x n, S and V are n x n.
    try std.testing.expectEqual(m, res.U.rows);
    try std.testing.expectEqual(n, res.U.cols);
    try std.testing.expectEqual(n, res.S.rows);
    try std.testing.expectEqual(n, res.S.cols);
    try std.testing.expectEqual(n, res.V.rows);
    try std.testing.expectEqual(n, res.V.cols);

    const tol = 1e-10 * @as(f64, @floatFromInt(n)) * (1.0 + A.norm1());

    // Reconstruction: U * S * V^T == A.
    var US = try mat.matMult(alloc, res.U, res.S);
    defer US.deinit();
    var Vt = try mat.transpose(res.V, alloc);
    defer Vt.deinit();
    var USVt = try mat.matMult(alloc, US, Vt);
    defer USVt.deinit();
    try std.testing.expect(maxAbsDiff(f64, USVt, A) < tol);

    // Orthonormal factors (random A is full rank with probability 1).
    var I = try mat.Matrix(f64).initIdentity(alloc, n, n);
    defer I.deinit();
    var Ut = try mat.transpose(res.U, alloc);
    defer Ut.deinit();
    var UtU = try mat.matMult(alloc, Ut, res.U);
    defer UtU.deinit();
    try std.testing.expect(maxAbsDiff(f64, UtU, I) < tol);
    var VtV = try mat.matMult(alloc, Vt, res.V);
    defer VtV.deinit();
    try std.testing.expect(maxAbsDiff(f64, VtV, I) < tol);

    // Singular values nonnegative and descending.
    for (0..n) |j| {
        const s = res.S.atUnsafe(j, j);
        try std.testing.expect(s >= 0);
        if (j > 0) try std.testing.expect(res.S.atUnsafe(j - 1, j - 1) >= s);
    }

    // Cross-check against the eigensolver: sigma_j^2 == eig_j(A^T A).
    var At = try mat.transpose(A, alloc);
    defer At.deinit();
    var AtA = try mat.matMult(alloc, At, A);
    defer AtA.deinit();
    const lams = try eigen_mod.eigenvalues(alloc, AtA, 50_000, 1e-12, null);
    defer alloc.free(lams);
    std.mem.sort(f64, lams, {}, std.sort.desc(f64));
    const scale = 1.0 + @sqrt(@max(lams[0], 0.0));
    for (0..n) |j| {
        const sig_ref = @sqrt(@max(lams[j], 0.0));
        try std.testing.expect(@abs(res.S.atUnsafe(j, j) - sig_ref) < 1e-7 * scale);
    }
}

test "property: SVD (thin): reconstruction, orthonormality, sigma order" {
    const alloc = std.testing.allocator;
    for (sizes) |n| for (seeds) |seed| {
        try checkSVD(alloc, n, n, seed); // square
    };
    for (seeds) |seed| { // tall
        try checkSVD(alloc, 5, 3, seed);
        try checkSVD(alloc, 8, 2, seed);
    }
}

test "property: SVD handles rank-deficient input" {
    const alloc = std.testing.allocator;
    const m = 5;
    const n = 3;
    for (seeds) |seed| {
        var x = try vec.Vector(f64).initRandom(alloc, m, true, seed, -1.0, 1.0);
        defer x.deinit();
        var y = try vec.Vector(f64).initRandom(alloc, n, true, seed + 1000, -1.0, 1.0);
        defer y.deinit();

        // A = x * y^T has rank 1: sigma_0 = ||x||*||y||, the rest are 0.
        var A = try mat.Matrix(f64).initZero(alloc, m, n);
        defer A.deinit();
        for (0..m) |i| for (0..n) |j| {
            A.setUnsafe(i, j, x.atUnsafe(i) * y.atUnsafe(j));
        };

        var res = try svd_mod.svdJacobi(alloc, A, 100);
        defer res.deinit();

        var nx: f64 = 0;
        for (x.data) |v| nx += v * v;
        var ny: f64 = 0;
        for (y.data) |v| ny += v * v;
        const expected = @sqrt(nx) * @sqrt(ny);

        const tol = 1e-10 * (1.0 + expected);
        try std.testing.expectApproxEqAbs(expected, res.S.atUnsafe(0, 0), tol);
        for (1..n) |j| try std.testing.expect(res.S.atUnsafe(j, j) < tol);

        // Reconstruction holds even though U's trailing columns are unnormalized.
        var US = try mat.matMult(alloc, res.U, res.S);
        defer US.deinit();
        var Vt = try mat.transpose(res.V, alloc);
        defer Vt.deinit();
        var USVt = try mat.matMult(alloc, US, Vt);
        defer USVt.deinit();
        try std.testing.expect(maxAbsDiff(f64, USVt, A) < 1e-10 * (1.0 + A.norm1()));
    }
}
