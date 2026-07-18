//! By convention, root.zig is the root source file for a Zig library
const std = @import("std");

// Namespaces
pub const errors = @import("error.zig");
pub const scalar = @import("core/scalar.zig");
pub const vec = @import("core/vec.zig");
pub const mat = @import("core/mat.zig");

pub const cholesky = @import("linalg/cholesky.zig");
pub const gaussJordan = @import("linalg/gaussjordan.zig");
pub const LU = @import("linalg/lu.zig");
pub const QR = @import("linalg/qrdecomposition.zig");
pub const signal = @import("signal/lti_conversion.zig");
pub const lsim = @import("signal/lsim.zig");
pub const control = @import("control/pid.zig");
pub const ode = @import("ode/runge_kutta.zig");
pub const eigen = @import("linalg/eigen.zig");
pub const SVD = @import("linalg/svd.zig");

// Convenience re-export
pub const Vector = vec.Vector;
pub const Vec = vec.Vec;
pub const CVec = vec.CVec;
pub const Matrix = mat.Matrix;
pub const Mat = mat.Mat;
pub const CMat = mat.CMat;
pub const StateSpace = signal.StateSpace;
pub const PID = control.PID_DEO_Sim;
pub const RK4 = ode.RK4;
pub const RKF45 = ode.RKF45;
pub const eigenvalues = eigen.eigenvalues;
pub const eigenvaluesComplex = eigen.eigenvaluesComplex;
pub const roots = eigen.roots;

test {
    _ = @import("core/scalar.zig");
    _ = @import("core/vec.zig");
    _ = @import("core/mat.zig");
    _ = @import("linalg/cholesky.zig");
    _ = @import("linalg/gaussjordan.zig");
    _ = @import("linalg/lu.zig");
    _ = @import("signal/lti_conversion.zig");
    _ = @import("signal/lsim.zig");
    _ = @import("linalg/qrdecomposition.zig");
    _ = @import("ode/runge_kutta.zig");
    _ = @import("linalg/eigen.zig");
    _ = @import("linalg/svd.zig");
    _ = @import("proptest.zig");
}
