const std = @import("std");
const mat = @import("../core/mat.zig");
const err_mod = @import("../error.zig");
const vec = @import("../core/vec.zig");
const sclr = @import("../core/scalar.zig");

const Mat = mat.Mat;
const Vec = vec.Vec;

pub const QRError = error{
    NotSquare,
} || err_mod.Common;

/// QR Decomposes matrix A into [Q, R].
///
/// Uses the Householder Reflections strategy.
///
/// Returns [Q , R] on success.
///
/// Returns a QRError.NotSquare if A is not square, and a
/// QRError.BadShape if A is 1x1 (nothing to decompose).
pub fn qrDecomposition(alloc: std.mem.Allocator, A: anytype) (QRError || std.mem.Allocator.Error)![2]mat.Matrix(mat.ElementOf(@TypeOf(A))) {
    const T = mat.ElementOf(@TypeOf(A));

    if (A.rows != A.cols) return QRError.NotSquare;
    const m = A.rows;
    if (m <= 1) return QRError.BadShape;

    var Q = try mat.Matrix(T).initIdentity(alloc, m, m);
    errdefer Q.deinit();

    var q_initialized: usize = 0;
    var Q_vec = try alloc.alloc(mat.Matrix(T), m - 1);
    defer {
        for (0..q_initialized) |i| {
            Q_vec[i].deinit();
        }
        alloc.free(Q_vec);
    }

    for (0..Q_vec.len) |idx| {
        Q_vec[idx] = try mat.Matrix(T).initZero(alloc, m, m);
        q_initialized += 1;
    }

    var R = try mat.Matrix(T).initZero(alloc, m, m);
    errdefer R.deinit();
    var work = try A.clone();
    defer work.deinit();

    for (0..m - 1) |i| {
        var ai = try vec.Vector(T).initZero(alloc, m, true);
        defer ai.deinit();
        // Set a to be the coloumn vector
        for (i..m) |v_i| {
            ai.setUnsafe(v_i, work.atUnsafe(v_i, i));
        }
        // ||a_i|| * e_i == [zeros_0->i-1, alfa_i, zeros_i+1->m] ^ T
        var ei = try vec.Vector(T).initZero(alloc, m, true);
        defer ei.deinit();
        ei.setUnsafe(i, sclr.one(T));

        // alfa = -sign(x_i) * ||a_i||, where sign is the phase x/|x| (sign(0) = 1).
        // For real T this reduces to the usual cancellation-avoiding sign choice.
        const nrm = ai.norm();
        const xi = ai.atUnsafe(i);
        const alfa: T = blk: {
            const axi = sclr.abs(xi);
            if (axi == 0) break :blk sclr.fromReal(T, -nrm);
            const phase = sclr.div(xi, sclr.fromReal(T, axi));
            break :blk sclr.neg(sclr.mul(phase, sclr.fromReal(T, nrm)));
        };

        ei.multConst(alfa);

        // u = X - alfa * e_i
        // v = u / ||u||
        // x = a_i
        var u = try vec.Vector(T).initZero(alloc, m, true);
        defer u.deinit();
        for (0..m) |idx_u| {
            u.setUnsafe(idx_u, sclr.sub(ai.atUnsafe(idx_u), ei.atUnsafe(idx_u)));
        }

        const norm_u = u.norm();

        var v = try vec.Vector(T).initZero(alloc, m, true);
        defer v.deinit();
        for (0..m) |idx_v| {
            v.setUnsafe(idx_v, sclr.div(u.atUnsafe(idx_v), sclr.fromReal(T, norm_u)));
        }

        // Q_i = I - 2 * v * v^H (v^T for real T, since conj is the identity there)
        var I = try mat.Matrix(T).initIdentity(alloc, m, m);
        defer I.deinit();
        var v_conj = try vec.Vector(T).initZero(alloc, m, false);
        defer v_conj.deinit();
        for (0..m) |k| {
            v_conj.setUnsafe(k, sclr.conj(v.atUnsafe(k)));
        }
        var v_v = try vec.vecMult(alloc, v, v_conj);
        defer v_v.deinit();
        v_v.multAll(sclr.fromReal(T, 2.0));

        try I.subInPlace(v_v);

        try mat.copyMat(I, Q_vec[i]);
        var work_next = try mat.matMult(alloc, I, work);
        work_next.setUnsafe(0, 0, work_next.atUnsafe(0, 0));
        work.deinit();
        work = work_next;
    }

    // Idea:
    // Q = Q_1^H * ... Q_i^H
    // Each Householder reflector is Hermitian (symmetric for real T),
    // so Q_i^H == Q_i and no transpose is needed.
    var idx: u16 = 0;
    var I_loop = try mat.Matrix(T).initIdentity(alloc, m, m);
    defer I_loop.deinit();
    while (idx < Q_vec.len) {
        var temp = try mat.matMult(alloc, I_loop, Q_vec[idx]);
        defer temp.deinit();
        try mat.copyMat(temp, I_loop);

        idx += 1;
    }
    try mat.copyMat(I_loop, Q);

    // R:
    // R = Q^H * A (Q^T for real T, since conj is the identity there)
    var Q_t = try mat.transpose(Q, alloc);
    defer Q_t.deinit();
    for (Q_t.data) |*e| {
        e.* = sclr.conj(e.*);
    }
    var temp = try mat.matMult(alloc, Q_t, A);
    defer temp.deinit();

    try mat.copyMat(temp, R);

    const retObj = [2]mat.Matrix(T){ Q, R };
    return retObj;
}

test "QR Decomposition: Verification & OOM" {
    const alloc_normal = std.testing.allocator;
    var A1 = try Mat.initZero(alloc_normal, 3, 3);
    defer A1.deinit();

    try A1.setRow(0, [_]f64{ 12.0, -51.0, 4.0 });
    try A1.setRow(1, [_]f64{ 6.0, 167.0, -68.0 });
    try A1.setRow(2, [_]f64{ -4, 24, -42 });

    const qr_decomposition1 = try qrDecomposition(alloc_normal, A1);

    var Q1 = qr_decomposition1[0];

    var R1 = qr_decomposition1[1];

    var A1_close = try mat.matMult(alloc_normal, Q1, R1);
    for (0..A1.rows) |r| {
        for (0..A1.cols) |c| {
            try std.testing.expectApproxEqAbs(A1.atUnsafe(r, c), A1_close.atUnsafe(r, c), 1e-12);
        }
    }

    A1_close.deinit();
    Q1.deinit();
    R1.deinit();

    // Each constructor does one alloc; fail_index 0 should trigger OOM.
    for (0..50) |fail_index| {
        var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const alloc = fa.allocator();
        // Not testing this
        const A = Mat.initZero(alloc, 3, 3);
        if (A) |A_| {
            var C = A_;
            defer C.deinit();
            try C.setRow(0, [_]f64{ 12.0, -51.0, 4.0 });
            try C.setRow(1, [_]f64{ 6.0, 167.0, -68.0 });
            try C.setRow(2, [_]f64{ -4, 24, -42 });

            const qr_decomposition = qrDecomposition(alloc, C);
            if (qr_decomposition) |qr| {
                var Q = qr[0];
                var R = qr[1];
                Q.deinit();
                R.deinit();
            } else |e| {
                try std.testing.expect(e == error.OutOfMemory);
            }
        } else |e| {
            try std.testing.expect(e == error.OutOfMemory);
        }
    }
}

test "QR Decomposition: Complex (QR == A, Q unitary, R upper triangular)" {
    const alloc = std.testing.allocator;
    const Cx = std.math.Complex(f64);
    const tol: f64 = 1e-10;

    var A = try mat.CMat.initZero(alloc, 3, 3);
    defer A.deinit();
    try A.setRow(0, [_]Cx{ Cx.init(1, 0), Cx.init(0, 1), Cx.init(2, 0) });
    try A.setRow(1, [_]Cx{ Cx.init(0, -1), Cx.init(3, 0), Cx.init(0, 0) });
    try A.setRow(2, [_]Cx{ Cx.init(2, 0), Cx.init(0, 0), Cx.init(1, 1) });

    const qr = try qrDecomposition(alloc, A);
    var Q = qr[0];
    defer Q.deinit();
    var R = qr[1];
    defer R.deinit();

    // Q * R == A
    var QR = try mat.matMult(alloc, Q, R);
    defer QR.deinit();
    for (0..3) |r| for (0..3) |c| {
        try std.testing.expect(sclr.approxEq(QR.atUnsafe(r, c), A.atUnsafe(r, c), tol));
    };

    // Q^H * Q == I (unitary; plain Q^T * Q == I does NOT hold for complex Q)
    var Qh = try mat.transpose(Q, alloc);
    defer Qh.deinit();
    for (Qh.data) |*e| {
        e.* = sclr.conj(e.*);
    }
    var QhQ = try mat.matMult(alloc, Qh, Q);
    defer QhQ.deinit();
    for (0..3) |r| for (0..3) |c| {
        const expected = if (r == c) sclr.one(Cx) else sclr.zero(Cx);
        try std.testing.expect(sclr.approxEq(QhQ.atUnsafe(r, c), expected, tol));
    };

    // R is upper triangular
    try std.testing.expect(mat.isUpperTriangular(R, tol));
}
