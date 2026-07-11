//// Contains the RK solver for ODE's

const std = @import("std");
const StateSpace = @import("../signal/lti_conversion.zig").StateSpace;
// 0
// c2 a21
// c3 a31 a32
// ..
//    b1  b2 ...

// RK4
// 0
// 1/2  1/2
// 1/2  0       1/2
// 1    0       0       1
// X    1/6     1/3     1/3     1/6

// StateSpace:
// x_dot(t) = A * x(t) + B * u(t)
//
// y(t)     = C * x(t) + D * u(t)

/// Linear state-space derivative: dx = A*x + B*u
fn deriv(comptime n: usize, ss: StateSpace, x: [n]f64, u: f64) [n]f64 {
    var dx: [n]f64 = undefined;
    for (0..n) |i| {
        var acc: f64 = 0;
        for (0..n) |j| acc += ss.A.atUnsafe(i, j) * x[j];
        dx[i] = acc + ss.B.atUnsafe(i) * u;
    }
    return dx;
}

/// Helper: x + a*k
fn axpy(comptime n: usize, x: [n]f64, a: f64, k: [n]f64) [n]f64 {
    var out: [n]f64 = undefined;
    for (0..n) |i| out[i] = x[i] + a * k[i];
    return out;
}

/// Helper: x + h * sum(a[j] * ks[j]), the argument to an RK stage.
fn stageArg(comptime n: usize, x: [n]f64, h: f64, ks: []const [n]f64, a: []const f64) [n]f64 {
    var out = x;
    for (ks, a) |k, c| {
        for (0..n) |i| out[i] += h * c * k[i];
    }
    return out;
}

pub fn RK4(comptime n: usize, ss: StateSpace, x: [n]f64, u: f64, dt: f64) [n]f64 {
    const k1 = deriv(n, ss, x, u);
    const k2 = deriv(n, ss, axpy(n, x, dt / 2.0, k1), u);
    const k3 = deriv(n, ss, axpy(n, x, dt / 2.0, k2), u);
    const k4 = deriv(n, ss, axpy(n, x, dt, k3), u);
    var out: [n]f64 = undefined;
    for (0..n) |i| {
        out[i] = x[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    return out;
}

pub const OdeError = error{StepSizeTooSmall};

pub const RKF45Options = struct {
    /// Absolute local error tolerance per step (max-norm of x5 - x4).
    tol: f64 = 1e-8,
    /// Initial step size guess.
    h0: f64 = 1e-3,
    /// Abort with StepSizeTooSmall if the controller drives h below this.
    h_min: f64 = 1e-12,
    /// Upper bound on step size. 0 means "no bound beyond tf - t0".
    h_max: f64 = 0,
};

/// Runge-Kutta-Fehlberg 4(5): adaptive-step integration of the linear
/// state-space system x' = A*x + B*u (u held constant) from t0 to tf.
///
/// Each step computes six stages, forms both a 4th- and a 5th-order
/// solution from them, and uses |x5 - x4| as a free local-error estimate.
/// Steps whose error exceeds `tol` are rejected and retried with a
/// smaller h; accepted steps propagate the 5th-order solution.
///
/// Fehlberg tableau:
///   0
///   1/4     1/4
///   3/8     3/32       9/32
///   12/13   1932/2197  -7200/2197  7296/2197
///   1       439/216    -8          3680/513    -845/4104
///   1/2     -8/27      2           -3544/2565  1859/4104   -11/40
///   b5:     16/135     0           6656/12825  28561/56430 -9/50   2/55
///   b4:     25/216     0           1408/2565   2197/4104   -1/5    0
pub fn RKF45(
    comptime n: usize,
    ss: StateSpace,
    x0: [n]f64,
    u: f64,
    t0: f64,
    tf: f64,
    opts: RKF45Options,
) OdeError![n]f64 {
    // Stage coefficients a_ij (see tableau above; row 2..6).
    const a2 = [_]f64{1.0 / 4.0};
    const a3 = [_]f64{ 3.0 / 32.0, 9.0 / 32.0 };
    const a4 = [_]f64{ 1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0 };
    const a5 = [_]f64{ 439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0 };
    const a6 = [_]f64{ -8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0 };
    // 5th-order weights (k2 has weight 0 in both).
    const b = [_]f64{ 16.0 / 135.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0 };
    // 4th-order weights (k6 also has weight 0).
    const d = [_]f64{ 25.0 / 216.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0 };

    const h_max = if (opts.h_max > 0) opts.h_max else tf - t0;
    var h = @min(opts.h0, h_max);
    var t = t0;
    var x = x0;

    while (t < tf) {
        if (h < opts.h_min) return OdeError.StepSizeTooSmall;
        var last = false;
        if (t + h >= tf) {
            h = tf - t;
            last = true;
        }

        const k1 = deriv(n, ss, x, u);
        const k2 = deriv(n, ss, stageArg(n, x, h, &.{k1}, &a2), u);
        const k3 = deriv(n, ss, stageArg(n, x, h, &.{ k1, k2 }, &a3), u);
        const k4 = deriv(n, ss, stageArg(n, x, h, &.{ k1, k2, k3 }, &a4), u);
        const k5 = deriv(n, ss, stageArg(n, x, h, &.{ k1, k2, k3, k4 }, &a5), u);
        const k6 = deriv(n, ss, stageArg(n, x, h, &.{ k1, k2, k3, k4, k5 }, &a6), u);

        // 5th-order solution and max-norm of the embedded error estimate.
        var x5: [n]f64 = undefined;
        var err: f64 = 0;
        for (0..n) |i| {
            x5[i] = x[i] + h * (b[0] * k1[i] + b[1] * k3[i] + b[2] * k4[i] + b[3] * k5[i] + b[4] * k6[i]);
            const x4 = x[i] + h * (d[0] * k1[i] + d[1] * k3[i] + d[2] * k4[i] + d[3] * k5[i]);
            err = @max(err, @abs(x5[i] - x4));
        }

        if (err <= opts.tol) {
            x = x5;
            if (last) break;
            t += h;
        }

        // Step-size controller: safety factor 0.9, growth clamped to [0.1, 4].
        const s: f64 = if (err == 0) 4.0 else 0.9 * std.math.pow(f64, opts.tol / err, 0.2);
        h = @min(h * std.math.clamp(s, 0.1, 4.0), h_max);
    }
    return x;
}

test "RK4: scalar linear system matches analytic solution" {
    const alloc = std.testing.allocator;

    // System:  x' = -x + u,  with u = 2 (constant), x(0) = 0
    // Analytic solution: x(t) = 2 * (1 - e^{-t})
    var ss = try StateSpace.initContinuous(alloc, 1);
    defer ss.deinit();
    try ss.A.set(0, 0, -1.0); // A = [-1]
    try ss.B.set(0, 1.0); //     B = [ 1]

    const u: f64 = 2.0;
    const dt: f64 = 1e-3;
    var x = [_]f64{0.0};

    // Integrate from t = 0 to t = 1 (1000 steps of dt).
    var step: usize = 0;
    while (step < 1000) : (step += 1) {
        x = RK4(1, ss, x, u, dt);
    }

    const expected = 2.0 * (1.0 - std.math.exp(-1.0)); // x(1) ≈ 1.2642411
    try std.testing.expectApproxEqAbs(expected, x[0], 1e-6);
}

test "RKF45: scalar linear system matches analytic solution" {
    const alloc = std.testing.allocator;

    // Same system:  x' = -x + u,  u = 2,  x(0) = 0
    var ss = try StateSpace.initContinuous(alloc, 1);
    defer ss.deinit();
    try ss.A.set(0, 0, -1.0);
    try ss.B.set(0, 1.0);

    const x = try RKF45(1, ss, .{0.0}, 2.0, 0.0, 1.0, .{ .tol = 1e-10 });

    const expected = 2.0 * (1.0 - std.math.exp(-1.0));
    try std.testing.expectApproxEqAbs(expected, x[0], 1e-8);
}

test "RKF45: 2D oscillator matches analytic solution" {
    const alloc = std.testing.allocator;

    // Harmonic oscillator:  x'' = -x  =>  x1' = x2, x2' = -x1
    // x(0) = [1, 0]  =>  x(t) = [cos t, -sin t]
    var ss = try StateSpace.initContinuous(alloc, 2);
    defer ss.deinit();
    try ss.A.set(0, 1, 1.0);
    try ss.A.set(1, 0, -1.0);

    const x = try RKF45(2, ss, .{ 1.0, 0.0 }, 0.0, 0.0, 2.0 * std.math.pi, .{ .tol = 1e-10 });

    try std.testing.expectApproxEqAbs(1.0, x[0], 1e-7);
    try std.testing.expectApproxEqAbs(0.0, x[1], 1e-7);
}
