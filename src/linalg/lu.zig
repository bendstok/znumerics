const std = @import("std");
const mat = @import("../core/mat.zig");
const vec = @import("../core/vec.zig");
const err_mod = @import("../error.zig");
const sclr = @import("../core/scalar.zig");

const Mat = mat.Mat;

pub const LUError = error{ NotSquare, Singular } || err_mod.Common;

pub fn LUResult(comptime T: type) type {
    return struct {
        const Self = @This();
        lu: mat.Matrix(T), // U on/above the diagonal, L multipliers below it
        perm: []usize, // row i of the factored system is row perm[i] of A
        odd_swaps: bool, // permutation parity, for the determinant sign
        alloc: std.mem.Allocator,

        pub fn deinit(self: *Self) void {
            self.lu.deinit();
            self.alloc.free(self.perm);
            self.* = undefined;
        }

        /// Solves Ax = b using the stored factorization. O(n^2), so the
        /// factorization can be reused for many right-hand sides.
        ///
        /// Returns a LUError.SizeMismatch if b does not match the factored size.
        pub fn solve(self: Self, alloc: std.mem.Allocator, b: vec.Vector(T)) (LUError || std.mem.Allocator.Error)!vec.Vector(T) {
            const n = self.lu.rows;
            if (b.len() != n) return LUError.SizeMismatch;

            var x = try vec.Vector(T).initZero(alloc, n, true);
            errdefer x.deinit();

            // Forward substitution: L y = P b (L has an implicit unit diagonal).
            // y is stored in x.
            for (0..n) |i| {
                var s = b.atUnsafe(self.perm[i]);
                for (0..i) |j| {
                    s = sclr.sub(s, sclr.mul(self.lu.atUnsafe(i, j), x.atUnsafe(j)));
                }
                x.setUnsafe(i, s);
            }

            // Back substitution: U x = y
            var i = n;
            while (i > 0) {
                i -= 1;
                var s = x.atUnsafe(i);
                for (i + 1..n) |j| {
                    s = sclr.sub(s, sclr.mul(self.lu.atUnsafe(i, j), x.atUnsafe(j)));
                }
                x.setUnsafe(i, sclr.div(s, self.lu.atUnsafe(i, i)));
            }
            return x;
        }

        /// Returns det(A) as the product of U's diagonal, negated for an
        /// odd number of row swaps. O(n).
        pub fn det(self: Self) T {
            var d = sclr.one(T);
            for (0..self.lu.rows) |i| {
                d = sclr.mul(d, self.lu.atUnsafe(i, i));
            }
            return if (self.odd_swaps) sclr.neg(d) else d;
        }
    };
}

/// LU-decomposes A with partial pivoting: P * A = L * U.
///
/// L and U are packed into one matrix (L's unit diagonal is implicit).
/// The returned LUResult owns its memory; call .deinit() when done.
///
/// Partial pivoting is used to adequately reduce round-off errors.
///
/// Returns a LUError.NotSquare if A is not square.
/// Returns a LUError.Singular if a pivot column is exactly zero.
pub fn lu(alloc: std.mem.Allocator, A: anytype) (LUError || std.mem.Allocator.Error)!LUResult(mat.ElementOf(@TypeOf(A))) {
    const T = mat.ElementOf(@TypeOf(A));
    if (A.rows != A.cols) return LUError.NotSquare;
    const n = A.rows;

    var f = try A.clone();
    errdefer f.deinit();

    const perm = try alloc.alloc(usize, n);
    errdefer alloc.free(perm);
    for (0..n) |i| perm[i] = i;
    var odd_swaps = false;

    for (0..n) |k| {
        // 1: Find pivot (largest |.| in column k, rows k..n)
        var pivot_row: usize = k;
        var val: sclr.Real(T) = sclr.abs(f.atUnsafe(k, k));
        for (k + 1..n) |i| {
            const a = sclr.abs(f.atUnsafe(i, k));
            if (a > val) {
                val = a;
                pivot_row = i;
            }
        }
        if (val == 0) return LUError.Singular;

        // 2: Pivot
        if (pivot_row != k) {
            try f.swapRow(pivot_row, k);
            std.mem.swap(usize, &perm[k], &perm[pivot_row]);
            odd_swaps = !odd_swaps;
        }

        // 3: Eliminate below the pivot, storing the multipliers in L's slots
        const pivot = f.atUnsafe(k, k);
        for (k + 1..n) |i| {
            const m = sclr.div(f.atUnsafe(i, k), pivot);
            f.setUnsafe(i, k, m);
            for (k + 1..n) |j| {
                f.setUnsafe(i, j, sclr.sub(f.atUnsafe(i, j), sclr.mul(m, f.atUnsafe(k, j))));
            }
        }
    }

    return .{ .lu = f, .perm = perm, .odd_swaps = odd_swaps, .alloc = alloc };
}

/// Solves Ax = b, returns vector x.
///
/// Factorizes and solves in one call. For repeated solves against the
/// same A, call lu() once and reuse LUResult.solve().
pub fn solve(alloc: std.mem.Allocator, A: anytype, b: vec.Vector(mat.ElementOf(@TypeOf(A)))) (LUError || std.mem.Allocator.Error)!vec.Vector(mat.ElementOf(@TypeOf(A))) {
    var f = try lu(alloc, A);
    defer f.deinit();
    return f.solve(alloc, b);
}

test "LU: PA == LU reconstruction" {
    const alloc = std.testing.allocator;

    var A = try Mat.initZero(alloc, 3, 3);
    defer A.deinit();
    try A.setRow(0, [_]f64{ 2.0, 1.0, -1.0 });
    try A.setRow(1, [_]f64{ -3.0, -1.0, 2.0 });
    try A.setRow(2, [_]f64{ -2.0, 1.0, 2.0 });

    var f = try lu(alloc, A);
    defer f.deinit();

    // Unpack L (unit diagonal) and U from the packed factor
    var L = try Mat.initIdentity(alloc, 3, 3);
    defer L.deinit();
    var U = try Mat.initZero(alloc, 3, 3);
    defer U.deinit();
    for (0..3) |i| for (0..3) |j| {
        if (j < i) {
            L.setUnsafe(i, j, f.lu.atUnsafe(i, j));
        } else {
            U.setUnsafe(i, j, f.lu.atUnsafe(i, j));
        }
    };

    var LU_m = try mat.matMult(alloc, L, U);
    defer LU_m.deinit();
    for (0..3) |i| for (0..3) |j| {
        try std.testing.expectApproxEqAbs(A.atUnsafe(f.perm[i], j), LU_m.atUnsafe(i, j), 1e-12);
    };
}

test "LU: solve matches known solution and gaussJordan system" {
    const alloc = std.testing.allocator;

    // Same system as the Gauss-Jordan test: x = (2, 3, -1)
    var A = try Mat.initZero(alloc, 3, 3);
    defer A.deinit();
    try A.setRow(0, [_]f64{ 2.0, 1.0, -1.0 });
    try A.setRow(1, [_]f64{ -3.0, -1.0, 2.0 });
    try A.setRow(2, [_]f64{ -2.0, 1.0, 2.0 });

    var b = try vec.Vec.initZero(alloc, 3, true);
    defer b.deinit();
    b.setAllUnsafe([_]f64{ 8.0, -11.0, -3.0 });

    var x = try solve(alloc, A, b);
    defer x.deinit();
    try std.testing.expectApproxEqAbs(2.0, x.atUnsafe(0), 1e-8);
    try std.testing.expectApproxEqAbs(3.0, x.atUnsafe(1), 1e-8);
    try std.testing.expectApproxEqAbs(-1.0, x.atUnsafe(2), 1e-8);

    // Reuse: same factorization, second right-hand side (b = A * e1 -> x = e1)
    var f = try lu(alloc, A);
    defer f.deinit();
    var b2 = try vec.Vec.initZero(alloc, 3, true);
    defer b2.deinit();
    b2.setAllUnsafe([_]f64{ 2.0, -3.0, -2.0 }); // first column of A
    var x2 = try f.solve(alloc, b2);
    defer x2.deinit();
    try std.testing.expectApproxEqAbs(1.0, x2.atUnsafe(0), 1e-8);
    try std.testing.expectApproxEqAbs(0.0, x2.atUnsafe(1), 1e-8);
    try std.testing.expectApproxEqAbs(0.0, x2.atUnsafe(2), 1e-8);

    // det(A) = -1 for this system
    try std.testing.expectApproxEqAbs(-1.0, f.det(), 1e-8);
}

test "LU: determinant sign follows row swaps" {
    const alloc = std.testing.allocator;

    // [[0, 1], [1, 0]]: pivoting must swap, det = -1
    var A = try Mat.initZero(alloc, 2, 2);
    defer A.deinit();
    try A.set(0, 1, 1.0);
    try A.set(1, 0, 1.0);

    var f = try lu(alloc, A);
    defer f.deinit();
    try std.testing.expectApproxEqAbs(-1.0, f.det(), 1e-12);
}

test "LU: Complex system where pivoting must fire" {
    const alloc = std.testing.allocator;
    const Cx = std.math.Complex(f64);
    const tol: f64 = 1e-8;

    // Same system as the Gauss-Jordan complex test: x = (1, 1)
    // |3i| > |i| in column 0, so partial pivoting swaps the rows.
    var A = try mat.CMat.initZero(alloc, 2, 2);
    defer A.deinit();
    try A.setRow(0, [_]Cx{ Cx.init(0, 1), Cx.init(2, 0) });
    try A.setRow(1, [_]Cx{ Cx.init(0, 3), Cx.init(1, 0) });

    var b = try vec.CVec.initZero(alloc, 2, true);
    defer b.deinit();
    b.setAllUnsafe([_]Cx{ Cx.init(2, 1), Cx.init(1, 3) });

    var x = try solve(alloc, A, b);
    defer x.deinit();
    try std.testing.expect(sclr.approxEq(x.atUnsafe(0), Cx.init(1, 0), tol));
    try std.testing.expect(sclr.approxEq(x.atUnsafe(1), Cx.init(1, 0), tol));

    // det = i*1 - 2*3i = -5i
    var f = try lu(alloc, A);
    defer f.deinit();
    try std.testing.expect(sclr.approxEq(f.det(), Cx.init(0, -5), tol));
}

test "LU: singular matrix -> Singular, non-square -> NotSquare" {
    const alloc = std.testing.allocator;

    var S = try Mat.initZero(alloc, 2, 2);
    defer S.deinit();
    try S.setRow(0, [_]f64{ 1.0, 1.0 });
    try S.setRow(1, [_]f64{ 1.0, 1.0 });
    try std.testing.expectError(LUError.Singular, lu(alloc, S));

    var R = try Mat.initZero(alloc, 2, 3);
    defer R.deinit();
    try std.testing.expectError(LUError.NotSquare, lu(alloc, R));
}

test "LU: OOM does not leak" {
    var A = try Mat.initZero(std.testing.allocator, 3, 3);
    defer A.deinit();
    try A.setRow(0, [_]f64{ 2.0, 1.0, -1.0 });
    try A.setRow(1, [_]f64{ -3.0, -1.0, 2.0 });
    try A.setRow(2, [_]f64{ -2.0, 1.0, 2.0 });

    var b = try vec.Vec.initZero(std.testing.allocator, 3, true);
    defer b.deinit();
    b.setAllUnsafe([_]f64{ 8.0, -11.0, -3.0 });

    for (0..10) |fail_index| {
        var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const alloc = fa.allocator();

        const res = solve(alloc, A, b);
        if (res) |x_val| {
            var x = x_val;
            x.deinit();
        } else |e| {
            try std.testing.expect(e == error.OutOfMemory);
        }
    }
}
