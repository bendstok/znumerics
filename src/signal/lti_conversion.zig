const std = @import("std");
const mat = @import("../core/mat.zig");
const vec = @import("../core/vec.zig");
const err_mod = @import("../error.zig");

const Vec = vec.Vec;
const Mat = mat.Mat;

pub const LTIError = error{
    DenominatorLeadingZero,
    AlreadyDiscrete,
    AlreadyContinuous,
} || err_mod.Common;

pub const LTIDomain = enum { Continuous, Discrete };

/// Representation of the State-space:
///
/// x_dot(t) = A * x(t) + B * u(t) + w(t)
///
/// y(t)     = C * x(t) + D * u(t) + v(t)
///
/// Note that:
///
/// v & w, the zero-mean white noise sources are presumed 0.
///
/// Only A is of Matrix form.
///
/// D is a vector of size 1.
pub const StateSpace = struct {
    A: Mat,
    B: Vec,
    C: Vec,
    D: Vec,
    alloc: std.mem.Allocator,
    domain: LTIDomain,
    dt: f64,

    /// Inits all the members to zero, size n.
    ///
    /// Note that dt = 0.0
    pub fn initContinous(alloc: std.mem.Allocator, n: usize) !StateSpace {
        var A = try Mat.initZero(alloc, n, n);
        errdefer A.deinit();
        var B = try Vec.initZero(alloc, n, true);
        errdefer B.deinit();
        var C = try Vec.initZero(alloc, n, false);
        errdefer C.deinit();
        var D = try Vec.initZero(alloc, 1, false);
        errdefer D.deinit();

        // TODO: Find a way around this.
        A.multAll(1);
        B.multConst(1);
        C.multConst(1);
        D.multConst(1);
        return .{ .A = A, .B = B, .C = C, .D = D, .alloc = alloc, .domain = LTIDomain.Continuous, .dt = 0.0 };
    }

    pub fn initDiscrete(alloc: std.mem.Allocator, n: usize, dt: f64) !StateSpace {
        var A = try Mat.initZero(alloc, n, n);
        errdefer A.deinit();
        var B = try Vec.initZero(alloc, n, true);
        errdefer B.deinit();
        var C = try Vec.initZero(alloc, n, false);
        errdefer C.deinit();
        var D = try Vec.initZero(alloc, 1, false);
        errdefer D.deinit();

        // TODO: Find a way around this.
        A.multAll(1);
        B.multConst(1);
        C.multConst(1);
        D.multConst(1);
        return .{ .A = A, .B = B, .C = C, .D = D, .alloc = alloc, .domain = LTIDomain.Discrete, .dt = dt };
    }

    pub fn deinit(self: *StateSpace) void {
        self.A.deinit();
        self.B.deinit();
        self.C.deinit();
        self.D.deinit();
        self.* = undefined;
    }
};

/// Unified transfer function representation for both
/// continuous (s) and discrete (z^-1)
///
/// The coefficients are stored in descending powers:
///
/// - Continuous: s^n ... s^0
/// - Discrete: z^-0, z^-1 ... z^-n
///
/// dt when TransferFunction is of type 'Continuous' is set to 0.0
pub const TransferFunction = struct {
    num: []f64,
    den: []f64,
    domain: LTIDomain,
    dt: f64,
    alloc: std.mem.Allocator,

    pub fn initZero(alloc: std.mem.Allocator, size: usize, domain_in: LTIDomain, dt_in: f64) !TransferFunction {
        if (size == 0) return LTIError.BadShape;
        const num = try alloc.alloc(f64, size);
        errdefer alloc.free(num);
        const den = try alloc.alloc(f64, size);
        errdefer alloc.free(den);
        @memset(num, 0.0);
        @memset(den, 0.0);
        return .{ .num = num, .den = den, .domain = domain_in, .dt = dt_in, .alloc = alloc };
    }

    pub fn initDiscrete(alloc: std.mem.Allocator, num_in: []const f64, den_in: []const f64, dt: f64) !TransferFunction {
        if (num_in.len == 0 or den_in.len == 0) return LTIError.BadShape;
        const num = try alloc.alloc(f64, num_in.len);
        errdefer alloc.free(num);
        const den = try alloc.alloc(f64, den_in.len);
        errdefer alloc.free(den);
        @memcpy(num, num_in);
        @memcpy(den, den_in);
        return .{ .num = num, .den = den, .domain = LTIDomain.Discrete, .dt = dt, .alloc = alloc };
    }

    pub fn initContinuous(alloc: std.mem.Allocator, num_in: []const f64, den_in: []const f64) !TransferFunction {
        if (num_in.len == 0 or den_in.len == 0) return LTIError.BadShape;
        const num = try alloc.alloc(f64, num_in.len);
        errdefer alloc.free(num);
        const den = try alloc.alloc(f64, den_in.len);
        errdefer alloc.free(den);
        @memcpy(num, num_in);
        @memcpy(den, den_in);
        return .{ .num = num, .den = den, .domain = LTIDomain.Continuous, .dt = 0.0, .alloc = alloc };
    }

    pub fn deinit(self: *TransferFunction) void {
        self.alloc.free(self.num);
        self.alloc.free(self.den);
        self.* = undefined;
    }

    pub fn setNum(self: TransferFunction, idx: usize, val: f64) !void {
        if (idx >= self.num.len) return LTIError.IndexOutOfBounds;
        self.num[idx] = val;
    }

    pub fn setDen(self: TransferFunction, idx: usize, val: f64) !void {
        if (idx >= self.den.len) return LTIError.IndexOutOfBounds;
        self.den[idx] = val;
    }

    pub fn padNumToDen(self: *TransferFunction) !void {
        if (self.num.len >= self.den.len) return;
        const pad = self.den.len - self.num.len;
        var buf = try self.alloc.alloc(f64, self.den.len);
        @memset(buf[0..pad], 0.0);
        @memcpy(buf[pad..], self.num);
        self.alloc.free(self.num);
        self.num = buf;
    }

    pub fn toStateSpace(self: TransferFunction) !StateSpace {
        switch (self.domain) {
            .Continuous => {
                return try tf2ss(self.alloc, self.num, self.den);
            },
            .Discrete => {
                // TODO: Implement this.
                @compileError(".toStateSpace| Not implemented for Discrete Transferfunctions \n");
            },
            else => unreachable,
        }
    }
};

/// Transfer function to state-space representation.
/// Equivelant to scipy's tf2ss.
///
///
/// num / den are the coefficients of the numerator / denominator
/// polynomials, in descending degree. The denominator MUST be
/// at least as long as the numerator.
///
/// Example:
///
/// (1+2.0s) / (1+2.0s+ 3.0s^2) gives:
///
/// num =  [2.0, 1.0] & den = [3.0, 2.0, 1.0]
///
/// Returns a StateSpace representing the transfer function.
pub fn tf2ss(alloc: std.mem.Allocator, num: []const f64, den: []const f64) !StateSpace {
    if (num.len > den.len) return LTIError.BadShape;
    const k = den.len;
    if (den[0] == 0.0) return LTIError.DenominatorLeadingZero;

    // We need to pad numerator if its shorter than the numerator
    var numP: []const f64 = num; // Copy the numerator to numP

    var numPBuff: []f64 = &[_]f64{};
    if (num.len < den.len) {
        numPBuff = alloc.alloc(f64, den.len) catch |e| {
            return e;
        };

        const pad = den.len - num.len;
        // We pad with zeros
        for (0..pad) |i| {
            numPBuff[i] = 0.0;
        }
        for (0..num.len) |i| {
            numPBuff[pad + i] = num[i];
        }
        numP = numPBuff;
    }
    defer alloc.free(numPBuff); // TODO: FIX THIS
    // We need to keep the buffer alive for the function lifetime,
    // Although we SHOULD only need it for the simple loop.

    const n = k - 1;
    var SS = try StateSpace.initContinous(alloc, n);
    const den0 = den[0];

    const has_D = (numP.len == den.len);
    const Dval = if (has_D) numP[0] / den0 else 0.0;
    // Can use unsafe since we know D's length is 1.
    SS.D.setUnsafe(0, Dval);

    if (n == 1) {
        // first order
        try SS.A.set(0, 0, -den[1] / den[0]);
        try SS.B.set(0, 1.0);
        try SS.C.set(0, (numP[1] / den0) - (Dval * (den[1] / den0)));
    } else {
        // Companion form.
        for (0..n - 1) |i| {
            try SS.A.set(i + 1, i, 1.0);
        }
        for (0..n) |j| {
            try SS.A.set(0, j, -den[j + 1] / den[0]);
            const cj = (numP[j + 1] / den0) - (Dval * (den[j + 1] / den0));
            try SS.C.set(j, cj);
        }
        try SS.B.set(0, 1.0);
    }

    return SS;
}

/// Converts the inputted StateSpace to a discrete StateSpace using ZOH.
///
/// Based upon SciPy's function with the same name.
///
/// Needs an allocater to build and find the discrete representation of A & B.
pub fn cont2discrete(
    alloc: std.mem.Allocator,
    SS: *StateSpace,
    dt: f64,
) !void {
    const n = SS.A.rows;
    const m = SS.A.rows + 1;
    if (SS.domain == LTIDomain.Discrete) return LTIError.AlreadyDiscrete;

    var em = try Mat.initZero(alloc, m, m);
    defer em.deinit();

    // Build em = [A*dt, B*dt; 0.0]
    for (0..n) |i| {
        for (0..n) |j| {
            try em.set(i, j, try SS.A.get(i, j) * dt);
        }
        // B
        try em.set(i, n, try SS.B.get(i) * dt);
    }

    var expEm = try mat.expm(alloc, em);
    defer expEm.deinit();

    // Extract Discrete A, B
    for (0..n) |i| {
        for (0..n) |j| {
            try SS.A.set(i, j, try expEm.get(i, j));
        }
        try SS.B.set(i, try expEm.get(i, n));
    }
    SS.dt = dt;
    SS.domain = LTIDomain.Discrete;
}

/// Convert discrete‐time state‐space (A,B,C,D) into a SISO TF in z⁻¹ form:
///   H(z) = (num[0] + num[1]·z⁻¹ + … + num[n]·z⁻ⁿ)
///   (den[0] + den[1]·z⁻¹ + … + den[n]·z⁻ⁿ)
///
/// Returns a discrete TransferFunction.
pub fn ss2tf(
    alloc: std.mem.Allocator,
    SS: StateSpace,
    dt: f64,
) !TransferFunction {
    const n = SS.A.rows;

    const polyA = try alloc.alloc(f64, n + 1);
    defer alloc.free(polyA);
    const polyM = try alloc.alloc(f64, n + 1);
    defer alloc.free(polyM);
    var M = try Mat.initZero(alloc, n, n);
    defer M.deinit();
    // 1) Denominator = characteristic polynomial of A
    try mat.charPoly(alloc, SS.A, polyA);

    // 2) Build M = A - B·Cᵀ (outer product)
    // TODO: Make this cleaner by utilizing / making
    // new tools in mat.zig
    for (0..M.rows) |i| {
        for (0..M.cols) |j| {
            M.setUnsafe(i, j, SS.A.atUnsafe(i, j) - SS.B.atUnsafe(i) * SS.C.atUnsafe(j));
        }
    }
    // 3) Numerator base = charPolyFL(M)
    try mat.charPoly(alloc, M, polyM);

    // 4) Final TF = polyM + (D[0]-1)*polyA  over  polyA
    var TF = try TransferFunction.initZero(alloc, n + 1, LTIDomain.Discrete, dt);
    const scale = SS.D.atUnsafe(0) - 1.0;
    for (0..n + 1) |i| {
        try TF.setNum(i, polyM[i] + scale * polyA[i]);
        try TF.setDen(i, polyA[i]);
    }
    return TF;
}

pub fn cont2discrete_tf(alloc: std.mem.Allocator, TF: *TransferFunction, dt: f64) !void {
    const n = TF.den.len - 1;
    if (n == 0 or TF.num.len > TF.den.len) {
        return LTIError.BadShape;
    }
    // Pad numerator
    try TF.padNumToDen();

    // Turn into State-Space
    var SS = try tf2ss(alloc, TF.num, TF.den);
    defer SS.deinit();

    // Compute ZOH conversion
    try cont2discrete(alloc, &SS, dt);

    // Back to TF via ss2tf
    const TFd = try ss2tf(alloc, SS, dt);
    errdefer TFd.deinit();

    // Swap ownership
    TF.deinit();
    TF.* = TFd;
}

test "tf2ss: failure when numerator is longer than denominator" {
    const alloc = std.testing.allocator;

    const num: []const f64 = &[_]f64{ 1, 3, 4 };
    const den: []const f64 = &[_]f64{ 1, 5 };
    const res = tf2ss(alloc, num, den);
    try std.testing.expectError(LTIError.BadShape, res);
}

test "tf2ss: failure when den[0] == 0.0" {
    const alloc = std.testing.allocator;

    const num: []const f64 = &[_]f64{ 1, 2 };
    const den: []const f64 = &[_]f64{ 0, 1, 5 };
    const res = tf2ss(alloc, num, den);
    try std.testing.expectError(LTIError.DenominatorLeadingZero, res);
}

test "tf2ss: Padding when num is shorter than den" {
    const alloc = std.testing.allocator;

    // Equivalent to num_padded = [0, 1, 3], den = [1, 5, 6]
    const num: []const f64 = &[_]f64{ 1, 3 };
    const den: []const f64 = &[_]f64{ 1, 5, 6 };
    var SS = try tf2ss(alloc, num, den);
    defer SS.deinit();

    // A should be [[-5, -6], [1, 0]]
    try std.testing.expectEqual(@as(usize, 2), SS.A.rows);
    try std.testing.expectEqual(@as(usize, 2), SS.A.cols);
    try std.testing.expectApproxEqRel(@as(f64, -5.0), try SS.A.get(0, 0), 1e-12);
    try std.testing.expectApproxEqRel(@as(f64, -6.0), try SS.A.get(0, 1), 1e-12);
    try std.testing.expectApproxEqRel(@as(f64, 1.0), try SS.A.get(1, 0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), try SS.A.get(1, 1), 1e-12);

    // B should be [1, 0]^T
    try std.testing.expectEqual(@as(usize, 2), SS.B.len());
    try std.testing.expectApproxEqRel(@as(f64, 1.0), try SS.B.get(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), try SS.B.get(1), 1e-12);

    // C should be [1, 3]
    try std.testing.expectEqual(@as(usize, 2), SS.C.len());
    try std.testing.expectApproxEqRel(@as(f64, 1.0), try SS.C.get(0), 1e-12);
    try std.testing.expectApproxEqRel(@as(f64, 3.0), try SS.C.get(1), 1e-12);

    // D should be 0
    try std.testing.expectEqual(@as(usize, 1), SS.D.len());
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), try SS.D.get(0), 1e-12);
}

test "tf2ss: first-order system (n==1) with nonzero D" {
    const alloc = std.testing.allocator;

    const num: []const f64 = &[_]f64{ 1.0, 1.0 };
    const den: []const f64 = &[_]f64{ 2.0, 3.0 };
    var SS = try tf2ss(alloc, num, den);
    defer SS.deinit();

    // A = [-den[1]/den[0]] = [-1.5]
    try std.testing.expectEqual(@as(usize, 1), SS.A.rows);
    try std.testing.expectEqual(@as(usize, 1), SS.A.cols);
    try std.testing.expectApproxEqRel(@as(f64, -1.5), try SS.A.get(0, 0), 1e-12);

    // B = [1]
    try std.testing.expectEqual(@as(usize, 1), SS.B.len());
    try std.testing.expectApproxEqRel(@as(f64, 1.0), try SS.B.get(0), 1e-12);

    // C = num[1]/den0 - D * den[1]/den0 = (1/2) - (0.5 * 3/2) = 0.5 - 0.75 = -0.25
    try std.testing.expectEqual(@as(usize, 1), SS.C.len());
    try std.testing.expectApproxEqRel(@as(f64, -0.25), try SS.C.get(0), 1e-12);

    // D = 0.5
    try std.testing.expectEqual(@as(usize, 1), SS.D.len());
    try std.testing.expectApproxEqRel(@as(f64, 0.5), try SS.D.get(0), 1e-12);
}

test "tf2ss: higher-order system without D (num shorter), correct C with subtraction term" {
    const alloc = std.testing.allocator;

    // num = [2] => padded to [0, 0, 2]; den = [2, 4, 3]
    const num: []const f64 = &[_]f64{2.0};
    const den: []const f64 = &[_]f64{ 2.0, 4.0, 3.0 };
    var SS = try tf2ss(alloc, num, den);
    defer SS.deinit();

    // A first row = -den[1..]/den0 = [-2, -1.5]; subdiagonal ones
    try std.testing.expectApproxEqRel(@as(f64, -2.0), try SS.A.get(0, 0), 1e-12);
    try std.testing.expectApproxEqRel(@as(f64, -1.5), try SS.A.get(0, 1), 1e-12);
    try std.testing.expectApproxEqRel(@as(f64, 1.0), try SS.A.get(1, 0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), try SS.A.get(1, 1), 1e-12);

    // B top = 1, bottom = 0
    try std.testing.expectApproxEqRel(@as(f64, 1.0), try SS.B.get(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), try SS.B.get(1), 1e-12);

    // D = 0 since num is shorter than den
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), try SS.D.get(0), 1e-12);

    // C = num_p[1..]/den0 - D * den[1..]/den0 = [0/2, 2/2] - 0 * [...] = [0, 1]
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), try SS.C.get(0), 1e-12);
    try std.testing.expectApproxEqRel(@as(f64, 1.0), try SS.C.get(1), 1e-12);
}

test "cont2discrete: dt = 0 yields identity A and zero B, C/D copied" {
    const alloc = std.testing.allocator;

    // Build a simple 2x2 continuous SS
    var SS = try StateSpace.initContinous(alloc, 2);
    defer SS.deinit();

    // A = [[-5, -6], [1, 0]], B = [1, 0], C = [1, 3], D = [0]
    try SS.A.set(0, 0, -5.0);
    try SS.A.set(0, 1, -6.0);
    try SS.A.set(1, 0, 1.0);
    try SS.A.set(1, 1, 0.0);
    try SS.B.set(0, 1.0);
    try SS.B.set(1, 0.0);
    try SS.C.set(0, 1.0);
    try SS.C.set(1, 3.0);
    try SS.D.set(0, 0.0);

    try cont2discrete(alloc, &SS, 0.0);

    // A_d should be identity
    try std.testing.expectApproxEqRel(@as(f64, 1.0), try SS.A.get(0, 0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), try SS.A.get(0, 1), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), try SS.A.get(1, 0), 1e-12);
    try std.testing.expectApproxEqRel(@as(f64, 1.0), try SS.A.get(1, 1), 1e-12);

    // B_d should be zeros
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), try SS.B.get(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), try SS.B.get(1), 1e-12);

    // C/D copied
    try std.testing.expectApproxEqRel(@as(f64, 1.0), try SS.C.get(0), 1e-12);
    try std.testing.expectApproxEqRel(@as(f64, 3.0), try SS.C.get(1), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), try SS.D.get(0), 1e-12);
}

test "cont2discrete: 1st-order known result via expm block extraction" {
    const alloc = std.testing.allocator;

    // Continuous: x' = a x + b u, y = c x + d u
    // a = -2, b = 3, c = 4, d = 0, dt = 0.1
    var SS = try StateSpace.initContinous(alloc, 1);
    defer SS.deinit();

    try SS.A.set(0, 0, -2.0);
    try SS.B.set(0, 3.0);
    try SS.C.set(0, 4.0);
    try SS.D.set(0, 0.0);

    const dt = 0.1;
    try cont2discrete(alloc, &SS, dt);

    // A_d = exp(a*dt)
    const A_d = std.math.exp(-2.0 * dt);
    try std.testing.expectApproxEqRel(A_d, try SS.A.get(0, 0), 1e-12);

    // B_d = ∫_0^dt exp(a τ) b dτ = b * (1 - exp(a*dt)) / (-a)
    const B_d = 3.0 * (1.0 - A_d) / 2.0;
    try std.testing.expectApproxEqRel(B_d, try SS.B.get(0), 1e-12);

    // C/D unchanged
    try std.testing.expectApproxEqRel(@as(f64, 4.0), try SS.C.get(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), try SS.D.get(0), 1e-12);
}

test "cont2discrete_tf: failure when numerator is longer than denominator" {
    const alloc = std.testing.allocator;

    // num len > den len => BadShape
    var TF = try TransferFunction.initContinuous(alloc, &[_]f64{ 1, 2, 3 }, &[_]f64{ 1, 4 });
    defer TF.deinit();

    const res = cont2discrete_tf(alloc, &TF, 0.1);
    try std.testing.expectError(LTIError.BadShape, res);
}

test "cont2discrete_tf: failure when denominator length is 1 (n == 0)" {
    const alloc = std.testing.allocator;

    // den.len == 1 => n == 0 => BadShape
    var TF = try TransferFunction.initContinuous(alloc, &[_]f64{1}, &[_]f64{2});
    defer TF.deinit();

    const res = cont2discrete_tf(alloc, &TF, 0.1);
    try std.testing.expectError(LTIError.BadShape, res);
}

test "ss2tf: success path via cont2discrete_tf matches manual ss2tf(cont2discrete(tf2ss()))" {
    const alloc = std.testing.allocator;

    // Continuous TF: (2) / (1 + 3 s) => num=[2], den=[1,3]
    // After padding num => [0, 2] to match den length.
    var TF = try TransferFunction.initContinuous(alloc, &[_]f64{2.0}, &[_]f64{ 1.0, 3.0 });
    defer TF.deinit();

    // Use the API under test
    const dt = 0.1;
    try cont2discrete_tf(alloc, &TF, dt);

    // Compute reference via manual route:
    // tf2ss -> cont2discrete -> ss2tf
    var SS = try tf2ss(alloc, TF.num, TF.den);
    defer SS.deinit();

    // However TF is now discrete; for a clean manual reference re-build continuous SS:
    var SS_cont = try tf2ss(alloc, &[_]f64{2.0}, &[_]f64{ 1.0, 3.0 });
    defer SS_cont.deinit();

    try cont2discrete(alloc, &SS_cont, dt);
    var TF_ref = try ss2tf(alloc, SS_cont, dt);
    defer TF_ref.deinit();

    // Compare lengths and coefficients
    try std.testing.expectEqual(TF_ref.num.len, TF.num.len);
    try std.testing.expectEqual(TF_ref.den.len, TF.den.len);

    for (0..TF.num.len) |i| {
        try std.testing.expectApproxEqRel(TF_ref.num[i], TF.num[i], 1e-10);
        try std.testing.expectApproxEqRel(TF_ref.den[i], TF.den[i], 1e-10);
    }
    try std.testing.expectEqual(@as(LTIDomain, LTIDomain.Discrete), TF.domain);
    try std.testing.expectApproxEqRel(dt, TF.dt, 1e-12);
}

test "ss2tf: OOM at various allocation points does not leak" {
    // Try a bunch of failure points so we hit different partial-construction states.
    // Each run uses a fresh failing allocator with its own leak tracking.
    for (0..30) |fail_index| {
        var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const alloc = fa.allocator();

        var SS = StateSpace.initDiscrete(alloc, 1, 0.1) catch |e| {
            try std.testing.expect(e == error.OutOfMemory);
            // If this fails, there was a leak in initDiscrete itself.
            continue;
        };
        defer SS.deinit();

        // nontrivial
        try SS.A.set(0, 0, 0.9);
        try SS.B.set(0, 1.0);
        try SS.C.set(0, 2.0);
        try SS.D.set(0, 0.0);

        const res = ss2tf(alloc, SS, 0.1);
        if (res) |tf_val| {
            // Success path also must not leak
            var tf = tf_val;
            tf.deinit();
        } else |e| {
            try std.testing.expect(e == error.OutOfMemory);
        }
    }
}

test "tf2ss: OOM at various allocation points does not leak" {
    // num shorter -> forces padding alloc inside tf2ss
    const num: []const f64 = &[_]f64{ 1.0, 3.0 };
    const den: []const f64 = &[_]f64{ 1.0, 5.0, 6.0 };

    for (0..80) |fail_index| {
        var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const alloc = fa.allocator();

        const res = tf2ss(alloc, num, den);
        if (res) |ss_val| {
            var ss = ss_val;
            ss.deinit();
        } else |e| {
            try std.testing.expect(e == error.OutOfMemory or e == LTIError.BadShape or e == LTIError.DenominatorLeadingZero);
        }
    }
}

test "cont2discrete: OOM at various allocation points does not leak" {
    for (0..200) |fail_index| {
        var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const alloc = fa.allocator();

        var SS = StateSpace.initContinous(alloc, 2) catch |e| {
            try std.testing.expect(e == error.OutOfMemory);
            continue;
        };
        defer SS.deinit();

        // Nontrivial A,B to exercise expm
        try SS.A.set(0, 0, -2.0);
        try SS.A.set(0, 1, 1.0);
        try SS.A.set(1, 0, -0.5);
        try SS.A.set(1, 1, -3.0);
        try SS.B.set(0, 1.0);
        try SS.B.set(1, 0.25);

        const r = cont2discrete(alloc, &SS, 0.1);
        if (r) |_| {} else |e| {
            try std.testing.expect(e == error.OutOfMemory);
        }
    }
}

test "cont2discrete_tf: OOM at various allocation points does not leak" {
    // TF that triggers padding + nontrivial SS conversion
    const num0: []const f64 = &[_]f64{2.0};
    const den0: []const f64 = &[_]f64{ 1.0, 3.0, 2.0 };

    for (0..300) |fail_index| {
        var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const alloc = fa.allocator();

        // Construct TF under failing allocator too (important: constructor OOM should be clean)
        var TF = TransferFunction.initContinuous(alloc, num0, den0) catch |e| {
            try std.testing.expect(e == error.OutOfMemory);
            continue;
        };
        defer TF.deinit();

        const r = cont2discrete_tf(alloc, &TF, 0.1);
        if (r) |_| {} else |e| {
            // cont2discrete_tf also can throw BadShape if inputs invalid;
            // for this input it should only OOM.
            try std.testing.expect(e == error.OutOfMemory);
        }
    }
}

test "TransferFunction.padNumToDen: OOM does not leak, success pads correctly" {
    for (0..50) |fail_index| {
        var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const alloc = fa.allocator();

        var TF = TransferFunction.initDiscrete(alloc, &[_]f64{5.0}, &[_]f64{ 1.0, 2.0, 3.0 }, 0.1) catch |e| {
            try std.testing.expect(e == error.OutOfMemory);
            continue;
        };
        defer TF.deinit();

        const r = TF.padNumToDen();
        if (r) |_| {
            // On success, num must be padded to length 3: [0,0,5]
            try std.testing.expectEqual(@as(usize, 3), TF.num.len);
            try std.testing.expectApproxEqAbs(@as(f64, 0.0), TF.num[0], 0.0);
            try std.testing.expectApproxEqAbs(@as(f64, 0.0), TF.num[1], 0.0);
            try std.testing.expectApproxEqAbs(@as(f64, 5.0), TF.num[2], 0.0);
        } else |e| {
            try std.testing.expect(e == error.OutOfMemory);
        }
    }
}
