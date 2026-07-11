const std = @import("std");
const mat = @import("../core/mat.zig");
const vec = @import("../core/vec.zig");
const err_mod = @import("../error.zig");
const signal = @import("lti_conversion.zig");

const Vec = vec.Vec;
const Mat = mat.Mat;
const StateSpace = signal.StateSpace;

pub const LsimError = error{ NotDiscrete, NotContinuous } || signal.LTIError || mat.ExpmError || err_mod.Common;

/// Options for the simulation loop.
pub const SimOptions = struct {
    /// If set, every state component and the output are clamped to
    /// [-clamp, clamp] after each step, and NaN is replaced by 0. Keeps
    /// unstable systems finite so results stay plottable instead of
    /// running off to Inf/NaN. null (the default) simulates exactly.
    clamp: ?f64 = null,
};

fn clampFinite(v: f64, lim: f64) f64 {
    if (std.math.isNan(v)) return 0.0;
    return std.math.clamp(v, -lim, lim);
}

/// The result of a simulation. Owns its memory, call .deinit() when done.
pub const LsimResult = struct {
    t: Vec, // Time axis, t[k] = k * dt
    y: Vec, // Output, y[k] = y(t[k])
    x: Mat, // State trajectory. Coloumn k is the state at t[k]

    pub fn deinit(self: *LsimResult) void {
        self.t.deinit();
        self.y.deinit();
        self.x.deinit();
        self.* = undefined;
    }
};

/// Simulates a Discrete StateSpace with a dynamic input:
///   x[k+1] = A x[k] + B u_k,  y[k] = C x[k] + D u_k
///
/// 'u' is called once per step as u(ctx, k, t_k, x) and held over the
/// step (ZOH). 'ctx' carries whatever state the input needs: a
/// controller, an RNG, a slice, or void for a pure signal.
///
/// 'x0' is the initial state. It is cloned, so the caller keeps
/// ownership. null means the zero state.
///
/// NB: the callback sees the state *before* the step, so feedback
/// controllers read the measurement without the D feedthrough
/// (no algebraic loop).
///
/// 'opts' tunes the loop; pass .{} for the exact default behaviour.
/// See SimOptions (per-step clamping for unstable systems).
///
/// Returns a LsimError.NotDiscrete if the system is not discrete.
/// Returns a LsimError.BadShape if n_steps == 0, and a
/// LsimError.SizeMismatch if x0 does not match the state dimension.
pub fn dlsimFn(
    alloc: std.mem.Allocator,
    ss: StateSpace,
    n_steps: usize,
    ctx: anytype,
    comptime u: fn (@TypeOf(ctx), usize, f64, Vec) f64,
    x0: ?Vec,
    opts: SimOptions,
) (LsimError || std.mem.Allocator.Error)!LsimResult {
    if (ss.domain != .Discrete) return LsimError.NotDiscrete;
    if (n_steps == 0) return LsimError.BadShape;
    const n = ss.A.rows;

    var x = if (x0) |x0_| blk: {
        if (x0_.len() != n) return LsimError.SizeMismatch;
        break :blk try x0_.clone();
    } else try Vec.initZero(alloc, n, true);
    defer x.deinit();
    var x_next = try Vec.initZero(alloc, n, true);
    defer x_next.deinit();

    var t = try vec.linspace(alloc, 0.0, ss.dt * @as(f64, @floatFromInt(n_steps - 1)), n_steps, true);
    errdefer t.deinit();
    var y = try Vec.initZero(alloc, n_steps, false);
    errdefer y.deinit();

    var X = try Mat.initZero(alloc, n, n_steps);
    errdefer X.deinit();

    for (0..n_steps) |k| {
        const tk = ss.dt * @as(f64, @floatFromInt(k));
        const uk = u(ctx, k, tk, x);

        // y[k] = C * x + D * u_k
        var yk: f64 = ss.D.atUnsafe(0) * uk;
        for (0..n) |j| yk += ss.C.atUnsafe(j) * x.atUnsafe(j);
        if (opts.clamp) |lim| yk = clampFinite(yk, lim);
        y.setUnsafe(k, yk);
        try X.setCol(k, x.data);

        // x_next = A * x + B * u_k
        for (0..n) |i| {
            var s: f64 = ss.B.atUnsafe(i) * uk;
            for (0..n) |j| s += ss.A.atUnsafe(i, j) * x.atUnsafe(j);
            if (opts.clamp) |lim| s = clampFinite(s, lim);
            x_next.setUnsafe(i, s);
        }
        // Ping-pong the two state buffers, so the loop never allocates
        std.mem.swap(Vec, &x, &x_next);
    }
    return .{ .t = t, .y = y, .x = X };
}

/// Simulates a Discrete StateSpace against a precomputed input signal.
///
/// u[k] is held over step k (ZOH); the simulation runs for u.len steps.
/// See dlsimFn for x0, opts and the errors.
pub fn dlsim(alloc: std.mem.Allocator, ss: StateSpace, u: []const f64, x0: ?Vec, opts: SimOptions) (LsimError || std.mem.Allocator.Error)!LsimResult {
    const S = struct {
        fn at(us: []const f64, k: usize, t: f64, x: Vec) f64 {
            _ = t;
            _ = x;
            return us[k];
        }
    };
    return dlsimFn(alloc, ss, u.len, u, S.at, x0, opts);
}

/// Simulates a Continuous StateSpace with a dynamic input.
///
/// ZOH-discretizes a copy of the system at dt, then runs dlsimFn on it.
/// The caller's StateSpace is left untouched. Exact at the sample
/// instants, since the input is piecewise constant by construction.
///
/// Returns a LsimError.NotContinuous if the system is already discrete.
/// Discretization can also fail (see cont2discrete/expm).
pub fn lsimFn(
    alloc: std.mem.Allocator,
    ss: StateSpace,
    dt: f64,
    n_steps: usize,
    ctx: anytype,
    comptime u: fn (@TypeOf(ctx), usize, f64, Vec) f64,
    x0: ?Vec,
    opts: SimOptions,
) (LsimError || std.mem.Allocator.Error)!LsimResult {
    if (ss.domain != .Continuous) return LsimError.NotContinuous;
    var ssd = try ss.clone();
    defer ssd.deinit();
    try signal.cont2discrete(alloc, &ssd, dt);
    return dlsimFn(alloc, ssd, n_steps, ctx, u, x0, opts);
}

/// Simulates a Continuous StateSpace against a precomputed input signal.
///
/// u[k] is held over step k (ZOH); the simulation runs for u.len steps.
/// For a constant u there is no need to build the slice, see step or
/// lsimFn with a comptime fn returning the constant.
pub fn lsim(
    alloc: std.mem.Allocator,
    ss: StateSpace,
    u: []const f64,
    dt: f64,
    x0: ?Vec,
    opts: SimOptions,
) (LsimError || std.mem.Allocator.Error)!LsimResult {
    const S = struct {
        fn at(us: []const f64, k: usize, t: f64, x: Vec) f64 {
            _ = t;
            _ = x;
            return us[k];
        }
    };
    return lsimFn(alloc, ss, dt, u.len, u, S.at, x0, opts);
}

/// Step response: u = 1 for all t, x0 = 0.
///
/// For a step of a different height, scale the output (the system is
/// linear), or use lsimFn with a comptime fn returning the constant.
pub fn step(alloc: std.mem.Allocator, ss: StateSpace, dt: f64, n_steps: usize) (LsimError || std.mem.Allocator.Error)!LsimResult {
    const S = struct {
        fn one(_: void, _: usize, _: f64, _: Vec) f64 {
            return 1.0;
        }
    };
    return lsimFn(alloc, ss, dt, n_steps, {}, S.one, null, .{});
}

/// Impulse response: y(t) = C * e^(At) * B, computed exactly as the
/// unforced response from x0 = B (no 1/dt delta approximation).
pub fn impulse(alloc: std.mem.Allocator, ss: StateSpace, dt: f64, n_steps: usize) (LsimError || std.mem.Allocator.Error)!LsimResult {
    const S = struct {
        fn zero(_: void, _: usize, _: f64, _: Vec) f64 {
            return 0.0;
        }
    };
    return lsimFn(alloc, ss, dt, n_steps, {}, S.zero, ss.B, .{});
}

// Test plant: x' = -x + u, y = x. Analytic responses are known in closed form.
fn testLag(alloc: std.mem.Allocator) !StateSpace {
    var ss = try StateSpace.initContinuous(alloc, 1);
    errdefer ss.deinit();
    try ss.A.set(0, 0, -1.0);
    try ss.B.set(0, 1.0);
    try ss.C.set(0, 1.0);
    return ss;
}

test "lsim: constant input is exact at sample instants (ZOH)" {
    const alloc = std.testing.allocator;
    var ss = try testLag(alloc);
    defer ss.deinit();

    // u = 2 -> y(t) = 2 * (1 - e^-t). Same system as the RK4 test, but
    // ZOH is exact for constant u, so the tolerance is ~1e6 tighter.
    const dt: f64 = 1e-3;
    const n_steps: usize = 1001; // t = 0 .. 1.0
    const u = [_]f64{2.0} ** n_steps;

    var res = try lsim(alloc, ss, &u, dt, null, .{});
    defer res.deinit();

    try std.testing.expectApproxEqAbs(0.0, res.y.atUnsafe(0), 1e-12);
    try std.testing.expectApproxEqAbs(1.0, res.t.atUnsafe(n_steps - 1), 1e-12);
    const expected = 2.0 * (1.0 - std.math.exp(-1.0));
    try std.testing.expectApproxEqAbs(expected, res.y.atUnsafe(n_steps - 1), 1e-9);

    // The caller's system must be untouched (still continuous)
    try std.testing.expect(ss.domain == .Continuous);
    try std.testing.expectApproxEqAbs(-1.0, ss.A.atUnsafe(0, 0), 1e-12);
}

test "dlsim: discrete system matches hand computation" {
    const alloc = std.testing.allocator;

    // x[k+1] = 0.5 x[k] + u[k], y = x. Impulse in u -> y = 0, 1, 0.5, 0.25
    var ss = try StateSpace.initDiscrete(alloc, 1, 1.0);
    defer ss.deinit();
    try ss.A.set(0, 0, 0.5);
    try ss.B.set(0, 1.0);
    try ss.C.set(0, 1.0);

    const u = [_]f64{ 1.0, 0.0, 0.0, 0.0 };
    var res = try dlsim(alloc, ss, &u, null, .{});
    defer res.deinit();

    try std.testing.expectApproxEqAbs(0.0, res.y.atUnsafe(0), 1e-12);
    try std.testing.expectApproxEqAbs(1.0, res.y.atUnsafe(1), 1e-12);
    try std.testing.expectApproxEqAbs(0.5, res.y.atUnsafe(2), 1e-12);
    try std.testing.expectApproxEqAbs(0.25, res.y.atUnsafe(3), 1e-12);

    // State trajectory: coloumn k is x[k]
    try std.testing.expectApproxEqAbs(1.0, res.x.atUnsafe(0, 1), 1e-12);
}

test "step / impulse: match the analytic responses" {
    const alloc = std.testing.allocator;
    var ss = try testLag(alloc);
    defer ss.deinit();

    const dt: f64 = 1e-3;
    const n_steps: usize = 1001; // t = 0 .. 1.0

    // Step: y(t) = 1 - e^-t
    var st = try step(alloc, ss, dt, n_steps);
    defer st.deinit();
    try std.testing.expectApproxEqAbs(1.0 - std.math.exp(-1.0), st.y.atUnsafe(n_steps - 1), 1e-9);

    // Impulse: y(t) = e^-t, exact from x0 = B
    var im = try impulse(alloc, ss, dt, n_steps);
    defer im.deinit();
    try std.testing.expectApproxEqAbs(1.0, im.y.atUnsafe(0), 1e-12);
    try std.testing.expectApproxEqAbs(std.math.exp(-1.0), im.y.atUnsafe(n_steps - 1), 1e-9);
}

test "lsimFn: closed-loop proportional control settles at kp*r/(1+kp)" {
    const alloc = std.testing.allocator;
    var ss = try testLag(alloc);
    defer ss.deinit();

    // u[k] = kp * (r - y[k]), y = x[0]. Closed loop: x' = -(1+kp) x + kp r,
    // steady state x* = kp * r / (1 + kp) = 0.9. The discrete steady state
    // is the same, so the check can be tight.
    const P = struct {
        kp: f64,
        r: f64,
        fn u(self: @This(), _: usize, _: f64, x: Vec) f64 {
            return self.kp * (self.r - x.atUnsafe(0));
        }
    };

    var res = try lsimFn(alloc, ss, 1e-3, 5000, P{ .kp = 9.0, .r = 1.0 }, P.u, null, .{});
    defer res.deinit();
    try std.testing.expectApproxEqAbs(0.9, res.y.atUnsafe(4999), 1e-6);
}

test "lsim: x0 is used and not consumed" {
    const alloc = std.testing.allocator;
    var ss = try testLag(alloc);
    defer ss.deinit();

    // Unforced decay from x0 = 1: y(t) = e^-t
    var x0 = try Vec.initZero(alloc, 1, true);
    defer x0.deinit();
    x0.setUnsafe(0, 1.0);

    const u = [_]f64{0.0} ** 1001;
    var res = try lsim(alloc, ss, &u, 1e-3, x0, .{});
    defer res.deinit();
    try std.testing.expectApproxEqAbs(std.math.exp(-1.0), res.y.atUnsafe(1000), 1e-9);

    // x0 is cloned internally, so the caller still owns a live copy
    try std.testing.expectApproxEqAbs(1.0, x0.atUnsafe(0), 1e-12);
}

test "lsim / dlsim: domain and shape errors" {
    const alloc = std.testing.allocator;

    var cont = try testLag(alloc);
    defer cont.deinit();
    var disc = try StateSpace.initDiscrete(alloc, 1, 1.0);
    defer disc.deinit();

    const u = [_]f64{1.0};

    // Wrong domain, both directions
    try std.testing.expectError(LsimError.NotDiscrete, dlsim(alloc, cont, &u, null, .{}));
    try std.testing.expectError(LsimError.NotContinuous, lsim(alloc, disc, &u, 1e-3, null, .{}));

    // Empty input
    try std.testing.expectError(LsimError.BadShape, dlsim(alloc, disc, u[0..0], null, .{}));

    // x0 of the wrong size
    var x0 = try Vec.initZero(alloc, 3, true);
    defer x0.deinit();
    try std.testing.expectError(LsimError.SizeMismatch, dlsim(alloc, disc, &u, x0, .{}));
}

test "dlsim: clamp keeps an unstable system finite and bounded" {
    const alloc = std.testing.allocator;

    // x[k+1] = 2 x[k] + u, y = x: doubles every step, so it runs away fast.
    var ss = try StateSpace.initDiscrete(alloc, 1, 1.0);
    defer ss.deinit();
    try ss.A.set(0, 0, 2.0);
    try ss.B.set(0, 1.0);
    try ss.C.set(0, 1.0);

    const u = [_]f64{1.0} ** 64;

    // Unclamped reference: far beyond the clamp bound by the last step.
    var free_run = try dlsim(alloc, ss, &u, null, .{});
    defer free_run.deinit();
    try std.testing.expect(free_run.y.atUnsafe(63) > 1e15);

    // Clamped: every output and state stays within [-100, 100].
    var clamped = try dlsim(alloc, ss, &u, null, .{ .clamp = 100.0 });
    defer clamped.deinit();
    for (0..64) |k| {
        try std.testing.expect(@abs(clamped.y.atUnsafe(k)) <= 100.0);
        try std.testing.expect(@abs(clamped.x.atUnsafe(0, k)) <= 100.0);
    }
    // Saturates at the bound once the growth hits it.
    try std.testing.expectApproxEqAbs(100.0, clamped.y.atUnsafe(63), 1e-12);

    // Before saturation the clamped run matches the exact one.
    try std.testing.expectApproxEqAbs(free_run.y.atUnsafe(3), clamped.y.atUnsafe(3), 1e-12);
}

test "dlsimFn: clamp replaces NaN so the trajectory stays plottable" {
    const alloc = std.testing.allocator;

    var ss = try StateSpace.initDiscrete(alloc, 1, 1.0);
    defer ss.deinit();
    try ss.A.set(0, 0, 0.5);
    try ss.B.set(0, 1.0);
    try ss.C.set(0, 1.0);

    // Input turns NaN after the first step (e.g. a controller divided by 0).
    const S = struct {
        fn u(_: void, k: usize, _: f64, _: Vec) f64 {
            return if (k == 0) 1.0 else std.math.nan(f64);
        }
    };

    // Without clamping the NaN propagates into the state and output.
    var dirty = try dlsimFn(alloc, ss, 4, {}, S.u, null, .{});
    defer dirty.deinit();
    try std.testing.expect(std.math.isNan(dirty.y.atUnsafe(3)));

    // With clamping every sample is finite.
    var clean = try dlsimFn(alloc, ss, 4, {}, S.u, null, .{ .clamp = 1e6 });
    defer clean.deinit();
    for (0..4) |k| {
        try std.testing.expect(std.math.isFinite(clean.y.atUnsafe(k)));
        try std.testing.expect(std.math.isFinite(clean.x.atUnsafe(0, k)));
    }
}

test "lsim: OOM does not leak" {
    var ss = try testLag(std.testing.allocator);
    defer ss.deinit();

    const u = [_]f64{1.0} ** 8;

    for (0..60) |fail_index| {
        var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const alloc = fa.allocator();

        const res = lsim(alloc, ss, &u, 1e-3, null, .{});
        if (res) |r_val| {
            var r = r_val;
            r.deinit();
        } else |e| {
            try std.testing.expect(e == error.OutOfMemory);
        }
    }
}
