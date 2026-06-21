const std = @import("std");
const znum = @import("znumerics");
const Mat = znum.Mat; // This is done for convenience
const MatOp = znum.mat;

fn fillDeterministic(m: *Mat) void {
    // Deterministic, non-trivial values (no RNG needed)
    for (0..m.rows) |i| {
        for (0..m.cols) |j| {
            const v = @as(f64, @floatFromInt((i * 131 + j * 17) % 1000)) * 0.001;
            m.setUnsafe(i, j, v);
        }
    }
}

fn checksum(m: Mat) f64 {
    // Cheap checksum so results get "used"
    var s: f64 = 0.0;
    for (0..m.rows) |i| {
        for (0..m.cols) |j| {
            s += m.atUnsafe(i, j);
        }
    }
    return s;
}

pub fn matAddvsSIMDadd(alloc: std.mem.Allocator, io: std.Io) !void {
    const R: usize = 512;
    const C: usize = 512;

    var A = try Mat.initZero(alloc, R, C);
    defer A.deinit();
    var B = try Mat.initZero(alloc, R, C);
    defer B.deinit();

    fillDeterministic(&A);
    fillDeterministic(&B);

    // Choose iterations so total runtime is measurable.
    const iters: usize = 10;

    // Warmup (helps stabilize clocks/caches/JIT-ish effects)
    {
        var k: usize = 0;
        while (k < 10) : (k += 1) {
            var tmp = try A.add(B);
            std.mem.doNotOptimizeAway(checksum(tmp));
            tmp.deinit();

            var tmp2 = try A.addSIMD(B);
            std.mem.doNotOptimizeAway(checksum(tmp2));
            tmp2.deinit();
        }
    }

    // --- time add() ---
    const timer_add_start = std.Io.Timestamp.now(io, .awake);
    var sum_add: f64 = 0.0;

    {
        var k: usize = 0;
        while (k < iters) : (k += 1) {
            var tmp = try A.add(B);
            sum_add += checksum(tmp);
            tmp.deinit();
        }
    }

    const ns_add: u64 = @intCast(timer_add_start.untilNow(io, .awake).toNanoseconds());
    std.mem.doNotOptimizeAway(sum_add);

    // --- time addSIMD() ---
    const timer_simd_start = std.Io.Timestamp.now(io, .awake);
    var sum_simd: f64 = 0.0;

    {
        var k: usize = 0;
        while (k < iters) : (k += 1) {
            var tmp = try A.addSIMD(B);
            sum_simd += checksum(tmp);
            tmp.deinit();
        }
    }

    const ns_simd: u64 = @intCast(timer_simd_start.untilNow(io, .awake).toNanoseconds());
    std.mem.doNotOptimizeAway(sum_simd);

    // Sanity: results should match (within floating error; here they should be exact)
    try std.testing.expectApproxEqAbs(sum_add, sum_simd, 0.0);

    const add_per_iter = @as(f64, @floatFromInt(ns_add)) / @as(f64, @floatFromInt(iters));
    const simd_per_iter = @as(f64, @floatFromInt(ns_simd)) / @as(f64, @floatFromInt(iters));

    std.debug.print(
        "\n[bench] Mat {d}x{d}, iters={d}\n  add     : {d} ns total, {d} ns/iter\n  addSIMD : {d} ns total, {d} ns/iter\n  speedup : {d}x\n",
        .{
            R,             C,                            iters,
            ns_add,        add_per_iter,                 ns_simd,
            simd_per_iter, add_per_iter / simd_per_iter,
        },
    );
}

pub fn matMulvsSIMDmatMul(alloc: std.mem.Allocator, io: std.Io) !void {

    // Pick square sizes for simplicity
    // (m x n) * (n x p)
    const N: usize = 64;
    const iters: usize = 20;

    var A = try Mat.initZero(alloc, N, N);
    defer A.deinit();
    var B = try Mat.initZero(alloc, N, N);
    defer B.deinit();

    fillDeterministic(&A);
    fillDeterministic(&B);

    // Warmup
    {
        var w: usize = 0;
        while (w < 3) : (w += 1) {
            var C1 = try MatOp.matMult(alloc, A, B);
            std.mem.doNotOptimizeAway(checksum(C1));
            C1.deinit();

            var C2 = try MatOp.matMultSIMD(alloc, A, B);
            std.mem.doNotOptimizeAway(checksum(C2));
            C2.deinit();
        }
    }

    // --- time scalar matmul ---
    const t_scalar_start = std.Io.Timestamp.now(io, .awake);
    var sum_scalar: f64 = 0.0;

    {
        var k: usize = 0;
        while (k < iters) : (k += 1) {
            var C = try MatOp.matMult(alloc, A, B);
            sum_scalar += checksum(C);
            C.deinit();
        }
    }

    const ns_scalar: u64 = @intCast(t_scalar_start.untilNow(io, .awake).toNanoseconds());
    std.mem.doNotOptimizeAway(sum_scalar);

    // --- time SIMD matmul ---
    const t_simd_start = std.Io.Timestamp.now(io, .awake);
    var sum_simd: f64 = 0.0;

    {
        var k: usize = 0;
        while (k < iters) : (k += 1) {
            var C = try MatOp.matMultSIMD(alloc, A, B);
            sum_simd += checksum(C);
            C.deinit();
        }
    }

    const ns_simd: u64 = @intCast(t_simd_start.untilNow(io, .awake).toNanoseconds());
    std.mem.doNotOptimizeAway(sum_simd);

    // Sanity: checksums should match closely.
    // Floating point may differ slightly because SIMD can change associativity.
    // Tolerance here is conservative; adjust if needed.
    try std.testing.expectApproxEqAbs(sum_scalar, sum_simd, 1e-6 * @as(f64, @floatFromInt(iters)) * @as(f64, @floatFromInt(N * N)));

    const scalar_per_iter = @as(f64, @floatFromInt(ns_scalar)) / @as(f64, @floatFromInt(iters));
    const simd_per_iter = @as(f64, @floatFromInt(ns_simd)) / @as(f64, @floatFromInt(iters));

    std.debug.print(
        "\n[bench] matmul {d}x{d} * {d}x{d}, iters={d}\n  matMult      : {d} ns total, {d} ns/iter\n  matMultSIMD  : {d} ns total, {d} ns/iter\n  speedup      : {d}x\n",
        .{
            N,         N,               N,       N,             iters,
            ns_scalar, scalar_per_iter, ns_simd, simd_per_iter, scalar_per_iter / simd_per_iter,
        },
    );
}
