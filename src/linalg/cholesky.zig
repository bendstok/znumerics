const std = @import("std");
const mat = @import("../core/mat.zig");
const err_mod = @import("../error.zig");

const Mat = mat.Mat;

pub const CholeskyError = error{
    NotSquare,
    NotPositiveDefinite,
} || err_mod.Common;

/// Returns the lower diagonal of the decomposition.
/// For a real matrix, A = L * L^T.
///
/// Uses the Cholesky–Banachiewicz algorithm.
///
/// Returns a CholeskyError.NotSquare if the matrix is not square.
/// Returns a CholeskyError.NotPositiveDefinite if the matrix is not
/// Hermitian positive-definite (a diagonal pivot <= 0 is hit).
pub fn cholesky(alloc: std.mem.Allocator, matrix: Mat) (CholeskyError || std.mem.Allocator.Error)!Mat {
    if (matrix.cols != matrix.rows) return CholeskyError.NotSquare;
    var i: usize = 0;
    var L = try Mat.initZero(alloc, matrix.rows, matrix.cols);
    errdefer L.deinit();

    while (i < matrix.rows) : (i += 1) {
        var j: usize = 0;
        while (j <= i) : (j += 1) {
            var sum: f64 = 0;

            var k: usize = 0;
            while (k < j) : (k += 1) {
                sum += try L.at(i, k) * try L.at(j, k);
            }

            if (i == j) {
                const d = try matrix.at(i, i) - sum;
                if (d <= 0.0) return CholeskyError.NotPositiveDefinite;
                try L.set(i, j, std.math.sqrt(d));
            } else {
                try L.set(i, j, (1.0 / (try L.at(j, j)) * (try matrix.at(i, j) - sum)));
            }
        }
    }
    return L;
}

test "Cholesky: Test" {
    const alloc = std.testing.allocator;
    var A = try mat.Mat.initZero(alloc, 3, 3);
    defer A.deinit();

    try A.set(0, 0, 4);
    try A.set(0, 1, 12);
    try A.set(0, 2, -16);
    try A.set(1, 0, 12);
    try A.set(1, 1, 37);
    try A.set(1, 2, -43);
    try A.set(2, 0, -16);
    try A.set(2, 1, -43);
    try A.set(2, 2, 98);
    var L = try cholesky(alloc, A);
    defer L.deinit();

    var L_T = try mat.transpose(L, alloc);
    defer L_T.deinit();
    var A2 = try mat.matMult(alloc, L, L_T);
    defer A2.deinit();

    try std.testing.expectApproxEqAbs(A.atUnsafe(0, 0), A2.atUnsafe(0, 0), 1e-8);
    try std.testing.expectApproxEqAbs(A.atUnsafe(0, 1), A2.atUnsafe(0, 1), 1e-8);
    try std.testing.expectApproxEqAbs(A.atUnsafe(0, 2), A2.atUnsafe(0, 2), 1e-8);
    try std.testing.expectApproxEqAbs(A.atUnsafe(1, 0), A2.atUnsafe(1, 0), 1e-8);
    try std.testing.expectApproxEqAbs(A.atUnsafe(1, 1), A2.atUnsafe(1, 1), 1e-8);
    try std.testing.expectApproxEqAbs(A.atUnsafe(1, 2), A2.atUnsafe(1, 2), 1e-8);
    try std.testing.expectApproxEqAbs(A.atUnsafe(2, 0), A2.atUnsafe(2, 0), 1e-8);
    try std.testing.expectApproxEqAbs(A.atUnsafe(2, 1), A2.atUnsafe(2, 1), 1e-8);
    try std.testing.expectApproxEqAbs(A.atUnsafe(2, 2), A2.atUnsafe(2, 2), 1e-8);
}

test "Cholesky: not square -> NotSquare" {
    const alloc = std.testing.allocator;
    var A = try mat.Mat.initZero(alloc, 2, 3);
    defer A.deinit();

    try std.testing.expectError(CholeskyError.NotSquare, cholesky(alloc, A));
}

test "Cholesky: not positive-definite -> NotPositiveDefinite" {
    const alloc = std.testing.allocator;
    var A = try mat.Mat.initZero(alloc, 2, 2);
    defer A.deinit();

    // Symmetric but indefinite: eigenvalues 1 and -1.
    try A.setRow(0, [_]f64{ 0.0, 1.0 });
    try A.setRow(1, [_]f64{ 1.0, 0.0 });

    try std.testing.expectError(CholeskyError.NotPositiveDefinite, cholesky(alloc, A));
}
