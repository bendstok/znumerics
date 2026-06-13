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

/// Helper
fn axpy(comptime n: usize, x: [n]f64, a: f64, k: [n]f64) [n]f64 {
    var out: [n]f64 = undefined;
    for (0..n) |i| out[i] = x[i] + a * k[i];
    return out;
}

pub fn RK4(comptime n: usize, ss: StateSpace, x: [n]f64, u: f64, dt: f64) [n]f64 {
    const f = struct {
        fn deriv(s: StateSpace, xx: [n]f64, uu: f64) [n]f64 {
            var dx: [n]f64 = undefined;
            for (0..n) |i| {
                var acc: f64 = 0;
                for (0..n) |j| acc += s.A.atUnsafe(i, j) * xx[j];
                dx[i] = acc + s.B.atUnsafe(i) * uu;
            }
            return dx;
        }
    }.deriv;

    const k1 = f(ss, x, u);
    const k2 = f(ss, axpy(n, x, dt / 2.0, k1), u);
    const k3 = f(ss, axpy(n, x, dt / 2.0, k2), u);
    const k4 = f(ss, axpy(n, x, dt, k3), u);
    var out: [n]f64 = undefined;
    for (0..n) |i| {
        out[i] = x[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    return out;
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
