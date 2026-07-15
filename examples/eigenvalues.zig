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

fn reportComplex(label: []const u8, eigs: []const std.math.Complex(f64)) void {
    std.debug.print("{s}: [", .{label});
    for (eigs, 0..) |e, i| {
        if (i != 0) std.debug.print(", ", .{});
        std.debug.print("{d:.4} {s} {d:.4}i", .{ e.re, if (e.im < 0) "-" else "+", @abs(e.im) });
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
        const eigs = try znum.eigen.eigenvaluesArnoldi(alloc, A, 1000, 1e-12, &it);
        defer alloc.free(eigs);
        report("3x3 dense          (eigenvaluesArnoldi) ", eigs, it);
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
        const eigs = try znum.eigen.eigenvaluesArnoldi(alloc, A, 1000, 1e-12, &it);
        defer alloc.free(eigs);
        report("4x4 tridiagonal    (eigenvaluesArnoldi) ", eigs, it);
    }

    // 8x8 dense symmetric: same matrix both ways, to compare iteration counts.
    // qrAlgorithm runs the shifted QR directly on the dense matrix;
    // eigenvaluesArnoldi first reduces it to Hessenberg form via Arnoldi,
    // which the QR iteration then chews through in fewer sweeps.
    {
        std.debug.print("\n8x8 dense symmetric (same matrix, two routes):\n", .{});
        const data = [_][8]f64{
            .{ 0, 3, 4, 1, 6, 5, 3, 4 },
            .{ 3, 6, 3, 4, 2, 2, 1, 4 },
            .{ 4, 3, 6, 2, 2, 6, 3, 4 },
            .{ 1, 4, 2, 4, 3, 6, 5, 0 },
            .{ 6, 2, 2, 3, 6, 0, 4, 3 },
            .{ 5, 2, 6, 6, 0, 0, 1, 4 },
            .{ 3, 1, 3, 5, 4, 1, 6, 3 },
            .{ 4, 4, 4, 0, 3, 4, 3, 0 },
        };

        var A1 = try Mat.initZero(alloc, 8, 8);
        defer A1.deinit();
        setMat(A1, data);
        var it_direct: usize = 0;
        const e_direct = try znum.eigen.qrAlgorithm(alloc, A1, 1000, 1e-12, &it_direct);
        defer alloc.free(e_direct);
        std.mem.sort(f64, e_direct, {}, std.sort.desc(f64));
        report("  qrAlgorithm (direct on dense)    ", e_direct, it_direct);

        var A2 = try Mat.initZero(alloc, 8, 8);
        defer A2.deinit();
        setMat(A2, data);
        var it_pipe: usize = 0;
        const e_pipe = try znum.eigen.eigenvaluesArnoldi(alloc, A2, 1000, 1e-12, &it_pipe);
        defer alloc.free(e_pipe);
        std.mem.sort(f64, e_pipe, {}, std.sort.desc(f64));
        report("  eigenvaluesArnoldi (Arnoldi -> QR)", e_pipe, it_pipe);
    }

    std.debug.print("\n=== Complex eigenvalues (real Schur form) ===\n\n", .{});

    // 2x2 rotation matrix: eigenvalues cos(t) ± i*sin(t).
    {
        const t = std.math.pi / 4.0;
        var A = try Mat.initZero(alloc, 2, 2);
        defer A.deinit();
        setMat(A, [_][2]f64{
            .{ @cos(t), -@sin(t) },
            .{ @sin(t), @cos(t) },
        });
        const eigs = try znum.eigen.qrAlgorithmComplex(alloc, A, 1000, 1e-12, null);
        defer alloc.free(eigs);
        reportComplex("2x2 rotation (pi/4)          (qrAlgorithmComplex)", eigs);
    }

    // Companion matrix of (x-1)(x^2-2x+5): eigenvalues 1, 1 ± 2i.
    {
        var A = try Mat.initZero(alloc, 3, 3);
        defer A.deinit();
        setMat(A, [_][3]f64{
            .{ 3, -7, 5 },
            .{ 1, 0, 0 },
            .{ 0, 1, 0 },
        });
        const eigs = try znum.eigen.qrAlgorithmComplex(alloc, A, 1000, 1e-12, null);
        defer alloc.free(eigs);
        reportComplex("companion of (x-1)(x^2-2x+5) (qrAlgorithmComplex)", eigs);
    }

    std.debug.print("\n=== Eigenvectors via inverse iteration ===\n\n", .{});

    // Same companion matrix: eigenvalues 1 and 1 ± 2i, so one real and two
    // genuinely complex eigenvectors. Each vector is paired with its
    // eigenvalue and verified by the residual ||Av - lambda*v||, which is
    // bounded by the QR tolerance (the eigenvalue error dominates it).
    {
        const Cx = std.math.Complex(f64);

        var A = try Mat.initZero(alloc, 3, 3);
        defer A.deinit();
        setMat(A, [_][3]f64{
            .{ 3, -7, 5 },
            .{ 1, 0, 0 },
            .{ 0, 1, 0 },
        });

        // Same (max_iter, tolerance) as passed to complexEigenvectors below,
        // so the eigenvalues come out in the same order as the vectors.
        const eigs = try znum.eigen.qrAlgorithmComplex(alloc, A, 1000, 1e-12, null);
        defer alloc.free(eigs);

        const eigVecs = try znum.eigen.complexEigenvectors(alloc, A, 1000, 1e-12);
        defer {
            for (eigVecs) |*v| v.deinit();
            alloc.free(eigVecs);
        }

        std.debug.print("companion of (x-1)(x^2-2x+5), unit-norm eigenpairs:\n", .{});
        for (eigs, eigVecs) |lambda, v| {
            std.debug.print("  lambda = {d:7.4} {s} {d:.4}i   v = [", .{
                lambda.re, if (lambda.im < 0) "-" else "+", @abs(lambda.im),
            });
            for (0..v.len()) |i| {
                const vi = v.atUnsafe(i);
                if (i != 0) std.debug.print(", ", .{});
                std.debug.print("{d:.4} {s} {d:.4}i", .{ vi.re, if (vi.im < 0) "-" else "+", @abs(vi.im) });
            }

            var res2: f64 = 0;
            for (0..A.rows) |r| {
                var s = Cx.init(0, 0);
                for (0..A.cols) |c| s = s.add(Cx.init(A.atUnsafe(r, c), 0).mul(v.atUnsafe(c)));
                const diff = s.sub(lambda.mul(v.atUnsafe(r)));
                res2 += diff.re * diff.re + diff.im * diff.im;
            }
            std.debug.print("]   ||Av - lambda*v|| = {e:.2}\n", .{@sqrt(res2)});
        }
    }
}
