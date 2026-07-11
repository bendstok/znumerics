const std = @import("std");
const mat = @import("../core/mat.zig");
const vec = @import("../core/vec.zig");
const err_mod = @import("../error.zig");
const sclr = @import("../core/scalar.zig");

const Vec = vec.Vec;
const Mat = mat.Mat;

pub const GaussJordanError = error{
    FreeVariable,
    Singular,
} || err_mod.Common;

/// Solves Ax = b, returns vector x.
///
/// Uses the Gauss-Jordan Algorithm to solve the system.
///
/// Partial pivoting is used to adequately reduce round-off errors.
///
/// Returns an GaussJordanError.FreeVariable when free variables have to be used to solve the system.
/// Returns a GaussJordanError.Singular if it hits a divide by zero.
pub fn gaussJordan(alloc: std.mem.Allocator, A: anytype, b: vec.Vector(mat.ElementOf(@TypeOf(A)))) (GaussJordanError || std.mem.Allocator.Error)!vec.Vector(mat.ElementOf(@TypeOf(A))) {
    const T = mat.ElementOf(@TypeOf(A));

    var solved = try vec.Vector(T).initZero(alloc, b.len(), true);
    errdefer solved.deinit();
    var A_mod = try mat.Matrix(T).initZero(alloc, A.rows, A.cols + 1);
    defer A_mod.deinit();

    try mat.copyMat(A, A_mod);

    try A_mod.setCol(A.cols, b.data);

    if (A.rows < A.cols) return GaussJordanError.FreeVariable;
    for (0..A_mod.rows - 1) |c| {
        // 1: Find pivot
        var pivot: usize = c;
        var val: sclr.Real(T) = sclr.abs(try A_mod.at(c, c));
        var j: usize = c;
        while (j < A_mod.rows) : (j += 1) {
            if (sclr.abs(try A_mod.at(j, c)) > val) {
                val = sclr.abs(try A_mod.at(j, c));
                pivot = j;
            }
        }
        // 2: Pivot
        if (pivot != c) {
            try A_mod.swapRow(pivot, c);
        }

        // 3: Row Reduce
        var i: usize = c + 1;
        while (i < A_mod.rows) : (i += 1) {
            const denom = try A_mod.at(c, c);
            if (sclr.eql(denom, sclr.zero(T))) return GaussJordanError.Singular;

            const L = sclr.neg(sclr.div(try A_mod.at(i, c), denom));
            for (c..A_mod.cols) |col| {
                const new_val = sclr.add(try A_mod.at(i, col), sclr.mul(try A_mod.at(c, col), L));
                try A_mod.set(i, col, new_val);
            }
        }
    }

    // Step 2: Backsolve
    var c: usize = A_mod.rows - 1;
    while (true) {
        var i = c;
        while (i > 0) {
            i -= 1;
            const denom = try A_mod.at(c, c);
            const L = sclr.neg(sclr.div(try A_mod.at(i, c), denom));
            for (c..A_mod.cols) |col| {
                const new_val = sclr.add(try A_mod.at(i, col), sclr.mul(try A_mod.at(c, col), L));
                try A_mod.set(i, col, new_val);
            }
        }
        if (c == 0) break;
        c -= 1;
    }

    // Step 3: Solve
    for (0..A_mod.rows) |r| {
        if (sclr.eql(try A_mod.at(r, r), sclr.zero(T))) return GaussJordanError.Singular;
        try solved.set(r, sclr.div(try A_mod.at(r, A_mod.cols - 1), try A_mod.at(r, r)));
    }
    return solved;
}

test "Gauss-Jordan: Test" {
    const alloc = std.testing.allocator;

    var B = try Mat.initZero(alloc, 3, 3);
    defer B.deinit();
    const row0 = [_]f64{ 2.0, 1.0, -1.0 };
    const row1 = [_]f64{ -3.0, -1.0, 2.0 };
    const row2 = [_]f64{ -2.0, 1.0, 2.0 };
    try B.setRow(0, row0);
    try B.setRow(1, row1);
    try B.setRow(2, row2);

    var y = try Vec.initZero(alloc, 3, true);
    defer y.deinit();

    const y_vals = [_]f64{ 8, -11, -3 };
    y.setAllUnsafe(y_vals);

    var output = try gaussJordan(alloc, B, y);
    defer output.deinit();

    try std.testing.expectApproxEqAbs(2.0, output.atUnsafe(0), 1e-8);
    try std.testing.expectApproxEqAbs(3.0, output.atUnsafe(1), 1e-8);
    try std.testing.expectApproxEqAbs(-1.0, output.atUnsafe(2), 1e-8);
}

test "Gauss-Jordan: singular system -> Singular" {
    const alloc = std.testing.allocator;

    var B = try Mat.initZero(alloc, 2, 2);
    defer B.deinit();
    try B.setRow(0, [_]f64{ 1.0, 1.0 });
    try B.setRow(1, [_]f64{ 1.0, 1.0 });

    var y = try Vec.initZero(alloc, 2, true);
    defer y.deinit();
    y.setAllUnsafe([_]f64{ 1.0, 1.0 });

    try std.testing.expectError(GaussJordanError.Singular, gaussJordan(alloc, B, y));
}

test "Gauss-Jordan: underdetermined system -> FreeVariable" {
    const alloc = std.testing.allocator;

    var B = try Mat.initZero(alloc, 2, 3);
    defer B.deinit();

    var y = try Vec.initZero(alloc, 2, true);
    defer y.deinit();

    try std.testing.expectError(GaussJordanError.FreeVariable, gaussJordan(alloc, B, y));
}

test "Gauss-Jordan: Complex system" {
    const alloc = std.testing.allocator;
    const Cx = std.math.Complex(f64);
    const tol: f64 = 1e-8;

    // [[1, i], [i, 1]] * (1, 1)^T = (1+i, 1+i)^T, det = 1 - i^2 = 2
    var B = try mat.CMat.initZero(alloc, 2, 2);
    defer B.deinit();
    try B.setRow(0, [_]Cx{ Cx.init(1, 0), Cx.init(0, 1) });
    try B.setRow(1, [_]Cx{ Cx.init(0, 1), Cx.init(1, 0) });

    var y = try vec.Vector(Cx).initZero(alloc, 2, true);
    defer y.deinit();
    y.setAllUnsafe([_]Cx{ Cx.init(1, 1), Cx.init(1, 1) });

    var output = try gaussJordan(alloc, B, y);
    defer output.deinit();

    try std.testing.expect(sclr.approxEq(output.atUnsafe(0), Cx.init(1, 0), tol));
    try std.testing.expect(sclr.approxEq(output.atUnsafe(1), Cx.init(1, 0), tol));
}

test "Gauss-Jordan: Complex system where pivoting must fire" {
    const alloc = std.testing.allocator;
    const Cx = std.math.Complex(f64);
    const tol: f64 = 1e-8;

    // |3i| > |i| in column 0, so partial pivoting swaps the rows.
    // [[i, 2], [3i, 1]] * (1, 1)^T = (2+i, 1+3i)^T
    var B = try mat.CMat.initZero(alloc, 2, 2);
    defer B.deinit();
    try B.setRow(0, [_]Cx{ Cx.init(0, 1), Cx.init(2, 0) });
    try B.setRow(1, [_]Cx{ Cx.init(0, 3), Cx.init(1, 0) });

    var y = try vec.Vector(Cx).initZero(alloc, 2, true);
    defer y.deinit();
    y.setAllUnsafe([_]Cx{ Cx.init(2, 1), Cx.init(1, 3) });

    var output = try gaussJordan(alloc, B, y);
    defer output.deinit();

    try std.testing.expect(sclr.approxEq(output.atUnsafe(0), Cx.init(1, 0), tol));
    try std.testing.expect(sclr.approxEq(output.atUnsafe(1), Cx.init(1, 0), tol));
}
