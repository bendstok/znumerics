//! Comptime scalar-ops layer.
//!
//! Generic code (Matrix(T), Vector(T), linalg) should go through these
//! helpers instead of using operators/builtins directly, so the same
//! algorithm works for floats and std.math.Complex(...).
//!
//! Conventions:
//! - add/sub/mul/div/neg work for ints, floats and Complex.
//! - abs/sqrt/approxEq require a float or Complex type (compile error on ints).
//! - abs returns the underlying real type (magnitude for Complex).

const std = @import("std");

/// True if T is std.math.Complex(...) or structurally compatible:
/// a struct with .re/.im fields and the usual method set).
pub fn isComplex(comptime T: type) bool {
    return @typeInfo(T) == .@"struct" and @hasField(T, "re") and @hasField(T, "im");
}

/// The real type underlying T: f64 for Complex(f64), T itself otherwise.
pub fn Real(comptime T: type) type {
    return if (isComplex(T)) @FieldType(T, "re") else T;
}

fn assertFloatOrComplex(comptime T: type, comptime who: []const u8) void {
    if (comptime !(isComplex(T) or @typeInfo(T) == .float)) {
        @compileError("SCALAR| " ++ who ++ ": expected a float or Complex type, got: " ++ @typeName(T));
    }
}

// ---------- constants / construction ----------

pub fn zero(comptime T: type) T {
    return if (comptime isComplex(T)) T.init(0, 0) else 0;
}

pub fn one(comptime T: type) T {
    return if (comptime isComplex(T)) T.init(1, 0) else 1;
}

/// Wrap a real value as T (identity for real T). Handy for literals:
/// `scalar.fromReal(T, 2.0)`.
pub fn fromReal(comptime T: type, x: Real(T)) T {
    return if (comptime isComplex(T)) T.init(x, 0) else x;
}

// ---------- arithmetic ----------

pub fn add(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return if (comptime isComplex(@TypeOf(a))) a.add(b) else a + b;
}

pub fn sub(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return if (comptime isComplex(@TypeOf(a))) a.sub(b) else a - b;
}

pub fn mul(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return if (comptime isComplex(@TypeOf(a))) a.mul(b) else a * b;
}

pub fn div(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return if (comptime isComplex(@TypeOf(a))) a.div(b) else a / b;
}

pub fn neg(a: anytype) @TypeOf(a) {
    return if (comptime isComplex(@TypeOf(a))) a.neg() else -a;
}

/// Complex conjugate; identity for real types.
pub fn conj(a: anytype) @TypeOf(a) {
    return if (comptime isComplex(@TypeOf(a))) a.conjugate() else a;
}

// ---------- magnitude & roots ----------

/// |a|. Magnitude for Complex; @abs for floats. Always returns the real type,
/// so results can be compared with `<` for pivoting/tolerance checks.
pub fn abs(a: anytype) Real(@TypeOf(a)) {
    const T = @TypeOf(a);
    comptime assertFloatOrComplex(T, "abs");
    return if (comptime isComplex(T)) a.magnitude() else @abs(a);
}

/// Principal square root.
pub fn sqrt(a: anytype) @TypeOf(a) {
    const T = @TypeOf(a);
    comptime assertFloatOrComplex(T, "sqrt");
    return if (comptime isComplex(T)) std.math.complex.sqrt(a) else @sqrt(a);
}

/// |x|^2 without the sqrt: x*x for reals, re^2 + im^2 for Complex.
pub fn absSq(x: anytype) Real(@TypeOf(x)) {
    const T = @TypeOf(x);
    return if (comptime isComplex(T))
        x.re * x.re + x.im * x.im
    else
        x * x;
}

// ---------- comparison ----------

/// For Complex, takes the higher magnitude. Else uses @max()
pub fn max(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    if (comptime isComplex(@TypeOf(a))) {
        if (a.magnitude() >= b.magnitude()) return a else return b;
    } else {
        return @max(a, b);
    }
}

/// Comparison: a >= b.
///
/// For complex types, compares the magnitude.
pub fn geq(a: anytype, b: @TypeOf(a)) bool {
    return if (comptime isComplex(@TypeOf(a))) a.magnitude() >= b.magnitude() else a >= b;
}

/// Comparison: a <= b.
///
/// For complex types, compares the magnitude.
pub fn leq(a: anytype, b: @TypeOf(a)) bool {
    return if (comptime isComplex(@TypeOf(a))) a.magnitude() <= b.magnitude() else a <= b;
}

/// Comparison: a > b.
///
/// For complex types, compares the magnitude.
pub fn greaterThan(a: anytype, b: @TypeOf(a)) bool {
    return if (comptime isComplex(@TypeOf(a))) a.magnitude() > b.magnitude() else a > b;
}

/// Comparison: a < b.
///
/// For complex types, compares the magnitude.
pub fn lessThan(a: anytype, b: @TypeOf(a)) bool {
    return if (comptime isComplex(@TypeOf(a))) a.magnitude() < b.magnitude() else a < b;
}

/// Exact equality (== on floats; both components on Complex).
pub fn eql(a: anytype, b: @TypeOf(a)) bool {
    return if (comptime isComplex(@TypeOf(a))) a.re == b.re and a.im == b.im else a == b;
}

/// |a - b| <= tol. Tolerance is always in the real type.
pub fn approxEq(a: anytype, b: @TypeOf(a), tol: Real(@TypeOf(a))) bool {
    comptime assertFloatOrComplex(@TypeOf(a), "approxEq");
    return abs(sub(a, b)) <= tol;
}

/// True if |a| <= tol. Common deflation/pivot check.
pub fn isZeroApprox(a: anytype, tol: Real(@TypeOf(a))) bool {
    comptime assertFloatOrComplex(@TypeOf(a), "isZeroApprox");
    return abs(a) <= tol;
}

// ---------- tests ----------

const C = std.math.Complex(f64);
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

test "real scalar ops" {
    try expectEqual(@as(f64, 0), zero(f64));
    try expectEqual(@as(f64, 1), one(f64));
    try expectEqual(@as(f64, 2.5), fromReal(f64, 2.5));

    try expectEqual(@as(f64, 5), add(@as(f64, 2), 3));
    try expectEqual(@as(f64, -1), sub(@as(f64, 2), 3));
    try expectEqual(@as(f64, 6), mul(@as(f64, 2), 3));
    try expectEqual(@as(f64, 2), div(@as(f64, 6), 3));
    try expectEqual(@as(f64, -2), neg(@as(f64, 2)));
    try expectEqual(@as(f64, 2), conj(@as(f64, 2)));

    try expectEqual(@as(f64, 3), abs(@as(f64, -3)));
    try expectEqual(@as(f64, 3), sqrt(@as(f64, 9)));

    try expect(eql(@as(f64, 2), 2));
    try expect(approxEq(@as(f64, 1.0), 1.0 + 1e-13, 1e-12));
    try expect(!approxEq(@as(f64, 1.0), 1.1, 1e-12));
    try expect(isZeroApprox(@as(f64, 1e-14), 1e-12));
}

test "complex scalar ops" {
    try expect(eql(zero(C), C.init(0, 0)));
    try expect(eql(one(C), C.init(1, 0)));
    try expect(eql(fromReal(C, 2.5), C.init(2.5, 0)));

    const a = C.init(1, 2);
    const b = C.init(3, -1);
    try expect(eql(add(a, b), C.init(4, 1)));
    try expect(eql(sub(a, b), C.init(-2, 3)));
    try expect(eql(mul(a, b), C.init(5, 5))); // (1+2i)(3-i) = 5+5i
    try expect(approxEq(div(mul(a, b), b), a, 1e-12));
    try expect(eql(neg(a), C.init(-1, -2)));
    try expect(eql(conj(a), C.init(1, -2)));

    try std.testing.expectApproxEqAbs(@as(f64, 5), abs(C.init(3, 4)), 1e-12);
    try expect(approxEq(mul(sqrt(a), sqrt(a)), a, 1e-12));

    try expect(isZeroApprox(C.init(1e-14, -1e-14), 1e-12));
}

test "arithmetic also works for ints" {
    try expectEqual(@as(i32, 5), add(@as(i32, 2), 3));
    try expectEqual(@as(i32, 6), mul(@as(i32, 2), 3));
    try expectEqual(@as(i32, -2), neg(@as(i32, 2)));
    try expectEqual(@as(i32, 0), zero(i32));
}

test "Real type resolution" {
    try expect(Real(f64) == f64);
    try expect(Real(f32) == f32);
    try expect(Real(C) == f64);
    try expect(isComplex(C));
    try expect(!isComplex(f64));
}
