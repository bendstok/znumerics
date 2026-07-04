const std = @import("std");
const mat = @import("../core/mat.zig");
const err_mod = @import("../error.zig");
const vec = @import("../core/vec.zig");

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
pub fn qrDecomposition(alloc: std.mem.Allocator, A: Mat) (QRError || std.mem.Allocator.Error)![2]Mat {
    if (A.rows != A.cols) return QRError.NotSquare;
    const m = A.rows;
    if (m <= 1) return QRError.BadShape;

    var Q = try Mat.initIdentity(alloc, m, m);
    errdefer Q.deinit();

    var q_initialized: usize = 0;
    var Q_vec = try alloc.alloc(Mat, m - 1);
    defer {
        for (0..q_initialized) |i| {
            Q_vec[i].deinit();
        }
        alloc.free(Q_vec);
    }

    for (0..Q_vec.len) |idx| {
        Q_vec[idx] = try Mat.initZero(alloc, m, m);
        q_initialized += 1;
    }

    var R = try Mat.initZero(alloc, m, m);
    errdefer R.deinit();
    var work = try A.clone();
    defer work.deinit();

    for (0..m - 1) |i| {
        var ai = try Vec.initZero(alloc, m, true);
        defer ai.deinit();
        // Set a to be the coloumn vector
        for (i..m) |v_i| {
            ai.setUnsafe(v_i, work.atUnsafe(v_i, i));
        }
        // ||a_i|| * e_i == [zeros_0->i-1, alfa_i, zeros_i+1->m] ^ T
        var ei = try Vec.initZero(alloc, m, true);
        defer ei.deinit();
        ei.setUnsafe(i, 1);

        const alfa: f64 = if (ai.atUnsafe(i) >= 0) -ai.norm() else ai.norm();

        ei.multConst(alfa);

        // u = X - alfa * e_i
        // v = u / ||u||
        // x = a_i
        var u = try Vec.initZero(alloc, m, true);
        defer u.deinit();
        for (0..m) |idx_u| {
            u.setUnsafe(idx_u, ai.atUnsafe(idx_u) - ei.atUnsafe(idx_u));
        }

        const norm_u = u.norm();

        var v = try Vec.initZero(alloc, m, true);
        defer v.deinit();
        for (0..m) |idx_v| {
            v.setUnsafe(idx_v, u.atUnsafe(idx_v) / norm_u);
        }

        // Q_i = I - v_col * v_row
        var I = try Mat.initIdentity(alloc, m, m);
        defer I.deinit();
        // V * V
        var v_v = try vec.vecMult(alloc, v, v);
        defer v_v.deinit();
        v_v.multAll(2);

        try Mat.subInPlace(I, v_v);

        try mat.copyMat(I, Q_vec[i]);
        var work_next = try mat.matMult(alloc, I, work);
        work_next.setUnsafe(0, 0, work_next.atUnsafe(0, 0));
        work.deinit();
        work = work_next;
    }

    // Idea:
    // Q = Q_1^T * ... Q_i^T
    // I * Q_1 = Q_1
    var idx: u16 = 0;
    var I_loop = try Mat.initIdentity(alloc, m, m);
    defer I_loop.deinit();
    while (idx < Q_vec.len) {
        try Q_vec[idx].transposeInPlace();
        var temp = try mat.matMult(alloc, I_loop, Q_vec[idx]);
        defer temp.deinit();
        try mat.copyMat(temp, I_loop);

        idx += 1;
    }
    try mat.copyMat(I_loop, Q);

    // R:
    // R = Q_T * A
    var Q_t = try mat.transpose(Q, alloc);
    defer Q_t.deinit();
    var temp = try mat.matMult(alloc, Q_t, A);
    defer temp.deinit();

    try mat.copyMat(temp, R);

    const retObj = [2]Mat{ Q, R };
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
