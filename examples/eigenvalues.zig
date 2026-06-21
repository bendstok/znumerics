const std = @import("std");
const znum = @import("znumerics");
const Mat = znum.Mat;

fn setMat(A: Mat, rows: anytype) void {
    for (0..A.rows) |i| {
        for (0..A.cols) |j| A.setUnsafe(i, j, rows[i][j]);
    }
}

fn report(label: []const u8, eigs: []const f64, iters: usize) void {
    std.debug.print("{s}: {d:>3} iterations  ->  [", .{ label, iters });
    for (eigs, 0..) |e, i| {
        if (i != 0) std.debug.print(", ", .{});
        std.debug.print("{d:.6}", .{e});
    }
    std.debug.print("]\n", .{});
}

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    std.debug.print("=== Eigenvalues via shifted QR (Wilkinson + deflation) ===\n\n", .{});

    // 2x2 symmetric, QR algorithm directly.
    {
        var A = try Mat.initZero(alloc, 2, 2);
        defer A.deinit();
        setMat(A, [_][2]f64{ .{ 2, 1 }, .{ 1, 2 } });
        var it: usize = 0;
        const eigs = try znum.eigen.qrAlgorithm(alloc, A, 1000, 1e-12, &it);
        defer alloc.free(eigs);
        report("2x2 [[2,1],[1,2]]  (qrAlgorithm)", eigs, it);
    }

    // 4x4 tridiagonal, QR algorithm directly.
    {
        var A = try Mat.initZero(alloc, 4, 4);
        defer A.deinit();
        setMat(A, [_][4]f64{
            .{ 4, 1, 0, 0 },
            .{ 1, 3, 1, 0 },
            .{ 0, 1, 2, 1 },
            .{ 0, 0, 1, 1 },
        });
        var it: usize = 0;
        const eigs = try znum.eigen.qrAlgorithm(alloc, A, 1000, 1e-12, &it);
        defer alloc.free(eigs);
        report("4x4 tridiagonal    (qrAlgorithm)", eigs, it);
    }

    // 3x3 dense symmetric, full pipeline (Arnoldi reduction + QR).
    {
        var A = try Mat.initZero(alloc, 3, 3);
        defer A.deinit();
        setMat(A, [_][3]f64{
            .{ 2, 1, 1 },
            .{ 1, 3, 2 },
            .{ 1, 2, 4 },
        });
        var it: usize = 0;
        const eigs = try znum.eigenvalues(alloc, A, 1000, 1e-12, &it);
        defer alloc.free(eigs);
        report("3x3 dense          (eigenvalues) ", eigs, it);
    }

    // 4x4 tridiagonal, full pipeline.
    {
        var A = try Mat.initZero(alloc, 4, 4);
        defer A.deinit();
        setMat(A, [_][4]f64{
            .{ 4, 1, 0, 0 },
            .{ 1, 3, 1, 0 },
            .{ 0, 1, 2, 1 },
            .{ 0, 0, 1, 1 },
        });
        var it: usize = 0;
        const eigs = try znum.eigenvalues(alloc, A, 1000, 1e-12, &it);
        defer alloc.free(eigs);
        report("4x4 tridiagonal    (eigenvalues) ", eigs, it);
    }
}
