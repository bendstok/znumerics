const std = @import("std");
const mat = @import("mat.zig");
const err_mod = @import("../error.zig");
const sclr = @import("scalar.zig");

pub const VecError = err_mod.Common;

/// The Vector type constructor.
/// Note that all vectors are presumed row vectors.
///
/// Stored on the heap.
pub fn Vector(comptime T: type) type {
    return struct {
        const Self = @This();
        pub const element = T;

        alloc: std.mem.Allocator,
        data: []T,
        colvec: bool,

        pub fn init(alloc: std.mem.Allocator, size: usize, colvec: bool) (VecError || std.mem.Allocator.Error)!Self {
            if (size == 0) {
                return VecError.Empty;
            }
            const data = try alloc.alloc(T, size);
            errdefer alloc.free(data);

            return .{ .alloc = alloc, .data = data, .colvec = colvec };
        }

        pub fn initZero(alloc: std.mem.Allocator, size: usize, colvec: bool) (VecError || std.mem.Allocator.Error)!Self {
            if (size == 0) {
                return VecError.Empty;
            }
            const data = try alloc.alloc(T, size);
            errdefer alloc.free(data);

            @memset(data, sclr.zero(T));
            return .{ .alloc = alloc, .data = data, .colvec = colvec };
        }

        pub fn initOnes(alloc: std.mem.Allocator, size: usize, colvec: bool) (VecError || std.mem.Allocator.Error)!Self {
            if (size == 0) {
                return VecError.Empty;
            }
            const data = try alloc.alloc(T, size);
            errdefer alloc.free(data);

            @memset(data, sclr.one(T));
            return .{ .alloc = alloc, .data = data, .colvec = colvec };
        }

        pub fn initRandom(
            alloc: std.mem.Allocator,
            size: usize,
            colvec: bool,
            seed: u64,
            min: T,
            max: T,
        ) (VecError || std.mem.Allocator.Error)!Self {
            if (size == 0) return VecError.Empty;

            const data = try alloc.alloc(T, size);
            errdefer alloc.free(data);

            var prng: std.Random.DefaultPrng = .init(seed);
            const rand = prng.random();

            for (data) |*v| {
                v.* = switch (@typeInfo(T)) {
                    .float => (min + (max - min) * rand.float(T)),
                    .int => rand.intRangeAtMost(T, min, max),
                    else => blk: {
                        if (comptime sclr.isComplex(T)) {
                            const F = sclr.Real(T); // Underlying type
                            break :blk T.init(
                                min.re + (max.re - min.re) * rand.float(F),
                                min.im + (max.im - min.im) * rand.float(F),
                            );
                        }
                        @compileError("initRandom: unsupported type " ++ @typeName(T));
                    },
                };
            }
            return .{ .alloc = alloc, .data = data, .colvec = colvec };
        }

        pub fn deinit(self: *Self) void {
            self.alloc.free(self.data);
            self.* = undefined;
        }

        /// Returns the length of items in the Vector.
        ///
        /// See .norm() for euclidean length.
        pub fn len(self: Self) usize {
            return self.data.len;
        }

        /// Returns the Euclidean norm of the vector.
        pub fn norm(self: Self) sclr.Real(T) {
            var sum: sclr.Real(T) = 0;
            for (self.data) |d| {
                sum += sclr.absSq(d);
            }
            return std.math.sqrt(sum);
        }

        /// Normalizes the vector using the Euclidean norm.
        ///
        /// NB! Does not check for a zero norm, dividing by zero.
        pub fn normalize(self: Self) void {
            const n = sclr.fromReal(T, self.norm());
            for (0..self.len()) |i| {
                var d = self.atUnsafe(i);
                d = sclr.div(d, n);
                self.setUnsafe(i, d);
            }
        }

        /// Returns the value currently stored at idx.
        pub fn atUnsafe(self: Self, idx: usize) T {
            return self.data[idx];
        }

        /// Returns the value currently stored at idx.
        ///
        /// Does bounds checks, returns a VecError.IndexOutOfBounds on failure.
        pub fn at(self: Self, idx: usize) VecError!T {
            if (idx >= self.len()) return VecError.IndexOutOfBounds;
            return self.data[idx];
        }

        /// Synonymous with .at()
        pub fn get(self: Self, idx: usize) VecError!T {
            return (try at(self, idx));
        }

        /// Set one individual index to a value.
        pub fn setUnsafe(self: Self, idx: usize, val: T) void {
            self.data[idx] = val;
        }

        /// Set one individual index to a value.
        ///
        /// Does bounds checks. Returns a VecError.IndexOutOfBounds on failure.
        pub fn set(self: Self, idx: usize, val: T) VecError!void {
            if (idx >= self.len()) return VecError.IndexOutOfBounds;
            self.data[idx] = val;
        }

        /// Transposes the vector, currently toggles a flag.
        pub fn transpose(self: *Self) void {
            self.colvec = !self.colvec;
        }

        /// Multiplies all the vector indicies by a constant.
        pub fn multConstUnsafe(self: Self, val: T) void {
            for (0..self.len()) |idx| {
                self.setUnsafe(idx, (sclr.mul(self.atUnsafe(idx), val)));
            }
        }

        /// Multiplies all the vector indicies by a constant.
        pub fn multConst(self: Self, val: T) void {
            for (0..self.len()) |idx| {
                // Can use unsafe since we explicitly traverse the vector.
                self.setUnsafe(idx, sclr.mul(self.atUnsafe(idx), val));
            }
        }

        /// Sets all the values to a constant.
        ///
        /// NB! Only checks surface type, and does no size checking. See .setAll().
        ///
        /// cons must be of type: T, comptime_float, comptime_int, [_]T or []T.
        pub fn setAllUnsafe(self: Self, cons: anytype) void {
            const T_new = @TypeOf(cons);

            switch (@typeInfo(T_new)) {
                .array => |arr| {
                    if (arr.child != T) @compileError("VEC| setAllUnsafe .array: got wrong type inside array, got: " ++ @typeName(T_new) ++ ". Expected: " ++ @typeName(T) ++ "\n");
                    @memcpy(self.data, cons[0..]);
                },
                .pointer => |p| {
                    if (p.child != T) @compileError("VEC| setAllUnsafe .pointer: wrong type of children, got: " ++ @typeName(p.child) ++ ". Expected: " ++ @typeName(T) ++ "\n");
                    if (p.size != .slice) @compileError("VEC| setAllUnsafe .pointer: Pointer must be of a Zig slice. \n");
                    @memcpy(self.data, cons[0..]);
                },
                else => {
                    // Scalar: accept T itself, plus comptime literals that coerce to T.
                    if (T_new != T and T_new != comptime_float and T_new != comptime_int) {
                        @compileError("VEC| setAllUnsafe: types do not match, got: " ++ @typeName(T_new) ++ ". Expected: " ++ @typeName(T) ++ "\n");
                    }
                    @memset(self.data, @as(T, cons));
                },
            }
        }

        /// Sets all the values to a constant.
        ///
        /// Checks lengths of input and types.
        /// Throws a compileError on type failure. Returns a
        /// VecError.SizeMismatch on size failure.
        ///
        /// cons must be of type: T, comptime_float, comptime_int, [_]T or []T.
        pub fn setAll(self: Self, cons: anytype) VecError!void {
            const T_new = @TypeOf(cons);

            switch (@typeInfo(T_new)) {
                .array => |arr| {
                    if (arr.child != T) @compileError("VEC| setAll .array: got wrong type inside array, got: " ++ @typeName(T_new) ++ ". Expected: " ++ @typeName(T) ++ "\n");
                    if (cons.len != self.data.len) return VecError.SizeMismatch;
                    @memcpy(self.data, cons[0..]);
                },
                .pointer => |p| {
                    if (p.child != T) @compileError("VEC| setAll .pointer: got wrong type inside array, got: " ++ @typeName(T_new) ++ ". Expected: " ++ @typeName(T) ++ "\n");
                    if (p.size != .slice) @compileError("VEC| setAll .pointer: Pointer must be of a Zig slice .\n");
                    if (cons.len != self.data.len) return VecError.SizeMismatch;
                    @memcpy(self.data, cons[0..]);
                },
                else => {
                    // Scalar: accept T itself, plus comptime literals that coerce to T.
                    if (T_new != T and T_new != comptime_float and T_new != comptime_int) {
                        @compileError("VEC| setAll: types do not match, got: " ++ @typeName(T_new) ++ ". Expected: " ++ @typeName(T) ++ "\n");
                    }
                    @memset(self.data, @as(T, cons));
                },
            }
        }

        /// Prints a view of the vector to std.debug.print
        ///
        /// Can fail because of bufPrint.
        pub fn printVec(self: Self) error{NoSpaceLeft}!void {
            for (0..self.len()) |idx| {
                //std.debug.print("c={}", .{c});
                // In-bounds by loop construction.
                const val: T = self.atUnsafe(idx);
                var tmp: [64]u8 = undefined;
                const s = if (comptime sclr.isComplex(T))
                    try std.fmt.bufPrint(&tmp, "{d:.3}{c}{d:.3}i", .{
                        val.re,
                        @as(u8, if (val.im < 0) '-' else '+'),
                        @abs(val.im),
                    })
                else
                    try std.fmt.bufPrint(&tmp, "{:.3}", .{val});
                std.debug.print("{s: >10} ", .{s});
                if (self.colvec) std.debug.print("\n", .{});
            }
            std.debug.print("\n", .{});
        }

        /// Resizes the vector. When expanding, fills the new space with 'fill'.
        ///
        /// Returns an error.OutOfMemory or VecError.Empty (if new_len == 0) on failure.
        pub fn resize(self: *Self, new_len: usize, fill: T) (VecError || std.mem.Allocator.Error)!void {
            if (new_len == 0) return VecError.Empty;
            if (new_len == self.data.len) return;
            const old_len = self.len();
            self.data = try self.alloc.realloc(self.data, new_len);

            if (new_len > old_len) {
                @memset(self.data[old_len..], fill);
            }
        }

        /// Add in place
        ///
        /// X = X + Y
        ///
        /// Only takes vectors of same size.
        pub fn addInPlace(self: Self, ToAdd: Self) VecError!void {
            if (self.len() != ToAdd.len()) return VecError.SizeMismatch;
            for (0..self.len()) |idx| {
                self.setUnsafe(idx, sclr.add(self.atUnsafe(idx), ToAdd.atUnsafe(idx)));
            }
        }

        /// Subtract in place
        ///
        /// X = X - Y
        ///
        /// Only takes vectors of same size.
        pub fn subInPlace(self: Self, toSub: Self) VecError!void {
            if (self.len() != toSub.len()) return VecError.SizeMismatch;
            for (0..self.len()) |idx| {
                self.setUnsafe(idx, sclr.sub(self.atUnsafe(idx), toSub.atUnsafe(idx)));
            }
        }

        /// Returns a vector which is a deep copy of itself
        pub fn clone(self: Self) std.mem.Allocator.Error!Self {
            const data = try self.alloc.dupe(T, self.data);
            return .{ .alloc = self.alloc, .data = data, .colvec = self.colvec };
        }
    };
}
// Backwards compatibility.
pub const Vec = Vector(f64);
pub const CVec = Vector(std.math.Complex(f64));
pub const Vec_32 = Vector(f32);
pub const CVec_32 = Vector(std.math.Complex(f32));

/// Comptime: Check whether V is a Vector(...).
pub fn isVector(comptime V: type) bool {
    return @typeInfo(V) == .@"struct" and @hasDecl(V, "element") and V == Vector(V.element);
}

/// Comptime: asserts V is a Vector(...) and returns its element type.
/// Use at the top of free functions taking `anytype`.
pub fn ElementOf(comptime V: type) type {
    if (!isVector(V)) @compileError("VEC| expected a Vector(T), got: " ++ @typeName(V));
    return V.element;
}

/// Multiplies two vectors, where the left one has to be a coloumn vector.
///
/// Note that: a^T * b == b^T * a. (All vectors unless otherwise stated, are row vectors).
///
/// Returns the matrix.
pub fn vecMult(alloc: std.mem.Allocator, left: anytype, right: @TypeOf(left)) (VecError || std.mem.Allocator.Error)!mat.Matrix(ElementOf(@TypeOf(left))) {
    const T = ElementOf(@TypeOf(left));

    if (!left.colvec) return VecError.BadShape;
    if (left.len() != right.len()) return VecError.SizeMismatch;
    var returnMat = try mat.Matrix(T).initZero(alloc, left.len(), left.len());
    errdefer returnMat.deinit();

    // This will always be a square matrix so which vec decides the iterator is arbitrary
    // Its set as this for ease of thinking
    for (0..left.len()) |r| {
        for (0..right.len()) |c| {
            try returnMat.set(r, c, sclr.mul(left.atUnsafe(r), right.atUnsafe(c)));
        }
    }
    return returnMat;
}

/// Computes the 3D cross product vector from two base vectors.
///
/// Returns the produced vector.
pub fn crossProd3d(alloc: std.mem.Allocator, left: anytype, right: @TypeOf(left)) (VecError || std.mem.Allocator.Error)!Vector(ElementOf(@TypeOf(left))) {
    const T = ElementOf(@TypeOf(left));

    if (left.len() != 3 or right.len() != 3) return VecError.BadShape;
    // Can use unsafe since we explicitly know the length.
    const s1 = sclr.sub(sclr.mul(left.atUnsafe(1), right.atUnsafe(2)), sclr.mul(left.atUnsafe(2), right.atUnsafe(1)));
    const s2 = sclr.sub(sclr.mul(left.atUnsafe(2), right.atUnsafe(0)), sclr.mul(left.atUnsafe(0), right.atUnsafe(2)));
    const s3 = sclr.sub(sclr.mul(left.atUnsafe(0), right.atUnsafe(1)), sclr.mul(left.atUnsafe(1), right.atUnsafe(0)));
    var retVec = try Vector(T).init(alloc, 3, false);
    retVec.setUnsafe(0, s1);
    retVec.setUnsafe(1, s2);
    retVec.setUnsafe(2, s3);
    return retVec;
}

/// Similar to np.linspace
///
/// Returns a Vec that contains all the steps from 'start' to 'end'.
///
/// Always includes start value. If 'includeEndPoint' is true,
/// the last value will be 'end'. Otherwise it the range will be
/// [start,end).
///
/// Does not support:
/// - start == end
/// - steps == 0
pub fn linspace(alloc: std.mem.Allocator, start: f64, end: f64, steps: usize, includeEndPoint: bool) (VecError || std.mem.Allocator.Error)!Vector(f64) {
    if (steps == 0) return VecError.BadShape;

    var retVec = try Vector(f64).initZero(alloc, steps, false);
    errdefer retVec.deinit();
    if (steps == 1) {
        retVec.setUnsafe(0, start);
        return retVec;
    }

    var denom: f64 = 0.0;
    if (includeEndPoint) {
        denom = @as(f64, @floatFromInt(steps - 1));
    } else {
        denom = @as(f64, @floatFromInt(steps));
    }

    const stepVal = (end - start) / denom;

    for (0..steps) |s| {
        retVec.setUnsafe(s, start + @as(f64, @floatFromInt(s)) * stepVal);
    }

    return retVec;
}

/// Computes the dot product between two vectors. Ignores row/column
/// orientation, both are treated as plain sequences of elements.
///
/// For complex vectors the left side is conjugated (inner product),
/// so dot(x, x) == norm(x)^2.
///
/// Returns the value.
pub fn dot(left: anytype, right: @TypeOf(left)) VecError!ElementOf(@TypeOf(left)) {
    const T = ElementOf(@TypeOf(left));

    if (left.len() != right.len()) return VecError.SizeMismatch;
    var res = sclr.zero(T);
    for (0..left.len()) |i| {
        // Can use unsafe here because we know we are within bounds
        res = sclr.add(res, sclr.mul(sclr.conj(left.atUnsafe(i)), right.atUnsafe(i)));
    }
    return res;
}

test "Vector: all decls compile for f32/f64/Complex" {
    std.testing.refAllDecls(Vector(f32));
    std.testing.refAllDecls(Vector(f64));
    std.testing.refAllDecls(Vector(std.math.Complex(f64)));
    std.testing.refAllDecls(Vector(std.math.Complex(f32)));
}

test "Vector: Test" {
    const alloc = std.testing.allocator;
    var e0 = try Vec.initZero(alloc, 3, false);
    defer e0.deinit();
    e0.setUnsafe(0, 1);
    const v = e0.atUnsafe(0);
    try std.testing.expect(v == 1);

    var e1 = try Vec.initZero(alloc, 3, false);
    defer e1.deinit();
    try e1.set(1, 1);
    const b = try dot(e0, e1);
    try std.testing.expect(b == 0);

    var e2 = try crossProd3d(alloc, e0, e1);
    defer e2.deinit();
    const z = try e2.at(2);
    try std.testing.expect(z == 1);

    var e3 = try Vec.initZero(alloc, 2, false);
    defer e3.deinit();
    try e3.setAll(1.0);
    try std.testing.expect(e3.atUnsafe(0) == 1.0);

    const float: f64 = 2.0;
    try e3.setAll(float);
    try std.testing.expect(e3.atUnsafe(0) == 2.0);

    const floatSliceCompTime = [_]f64{ 3.0, 4.0 };
    try e3.setAll(floatSliceCompTime);
    try std.testing.expect(e3.atUnsafe(0) == 3.0);

    const floatSlice: []f64 = try alloc.alloc(f64, 2);
    defer alloc.free(floatSlice);
    floatSlice[0] = 4.0;
    floatSlice[1] = 5.0;
    try e3.setAll(floatSlice);
    try std.testing.expect(e3.atUnsafe(0) == 4.0);
}

test "Vec.init / initZero: OOM does not leak" {
    // Each constructor does one alloc; fail_index 0 should trigger OOM.
    for (0..10) |fail_index| {
        var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const alloc = fa.allocator();

        const r1 = Vec.init(alloc, 8, false);
        if (r1) |v1_val| {
            var v1 = v1_val;
            v1.deinit();
        } else |e| {
            try std.testing.expect(e == error.OutOfMemory);
        }

        const r2 = Vec.initZero(alloc, 8, false);
        if (r2) |v2_val| {
            var v2 = v2_val;
            v2.deinit();
        } else |e| {
            try std.testing.expect(e == error.OutOfMemory);
        }
    }
}

test "Vec.resize: OOM does not leak and leaves vector unchanged on failure" {
    // We need a non-failing allocator for initial allocation, then a failing allocator
    // for the resize attempt. The easiest way is to allocate the Vec with the failing allocator too,
    // but choose a fail_index that allows init and then fails on realloc.
    for (0..10) |fail_index| {
        var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const alloc = fa.allocator();

        var v = Vec.initZero(alloc, 4, false) catch |e| {
            try std.testing.expect(e == error.OutOfMemory);
            continue;
        };
        defer v.deinit();

        v.setUnsafe(0, 11.0);
        v.setUnsafe(1, 22.0);
        v.setUnsafe(2, 33.0);
        v.setUnsafe(3, 44.0);

        const old_ptr = @intFromPtr(v.data.ptr);
        const old_len = v.len();

        const r = v.resize(1000, 9.0);
        if (r) |_| {
            // Success: length increased, old values preserved, tail filled
            try std.testing.expect(v.len() == 1000);
            try std.testing.expect(v.atUnsafe(0) == 11.0);
            try std.testing.expect(v.atUnsafe(3) == 44.0);
            try std.testing.expect(v.atUnsafe(old_len) == 9.0);
        } else |e| {
            try std.testing.expect(e == error.OutOfMemory);
            // Must be unchanged
            try std.testing.expect(v.len() == old_len);
            try std.testing.expect(@intFromPtr(v.data.ptr) == old_ptr);
            try std.testing.expect(v.atUnsafe(0) == 11.0);
            try std.testing.expect(v.atUnsafe(3) == 44.0);
        }
    }
}

test "crossProd3d: OOM does not leak" {
    for (0..10) |fail_index| {
        var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const alloc = fa.allocator();

        var a = Vec.initZero(alloc, 3, false) catch |e| {
            try std.testing.expect(e == error.OutOfMemory);
            continue;
        };
        defer a.deinit();

        var b = Vec.initZero(alloc, 3, false) catch |e| {
            try std.testing.expect(e == error.OutOfMemory);
            continue;
        };
        defer b.deinit();

        a.setUnsafe(0, 1);
        a.setUnsafe(1, 0);
        a.setUnsafe(2, 0);
        b.setUnsafe(0, 0);
        b.setUnsafe(1, 1);
        b.setUnsafe(2, 0);

        const res = crossProd3d(alloc, a, b);
        if (res) |v_val| {
            var v = v_val;
            v.deinit();
        } else |e| {
            // either OOM or BadShape
            try std.testing.expect(e == error.OutOfMemory or e == VecError.BadShape);
        }
    }
}

test "vecMult: OOM at various allocation points does not leak" {
    for (0..120) |fail_index| {
        var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const alloc = fa.allocator();

        var left = Vec.initZero(alloc, 3, true) catch |e| {
            try std.testing.expect(e == error.OutOfMemory);
            continue;
        };
        defer left.deinit();

        var right = Vec.initZero(alloc, 3, false) catch |e| {
            try std.testing.expect(e == error.OutOfMemory);
            continue;
        };
        defer right.deinit();

        const res = vecMult(alloc, left, right);
        if (res) |m_val| {
            var m = m_val;
            m.deinit();
        } else |e| {
            // vecMult can also fail with BadShape/SizeMismatch, but here it should mostly be OOM.
            try std.testing.expect(e == error.OutOfMemory or e == VecError.BadShape or e == VecError.SizeMismatch);
        }
    }
}

test "linspace: Test OOM and proper behaviour" {
    for (0..50) |fail_index| {
        var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const alloc = fa.allocator();

        const ls = linspace(alloc, 0.0, 10.0, 10, false);
        if (ls) |lin| {
            var k = lin;
            defer k.deinit();
            try std.testing.expect(lin.len() == 10);
            try std.testing.expect(lin.atUnsafe(0) == 0.0);
        } else |e| {
            try std.testing.expect(e == error.OutOfMemory);
        }
        const ls2 = linspace(alloc, 0.0, 10.0, 10, true);
        if (ls2) |lin2| {
            var k = lin2;
            defer k.deinit();
            try std.testing.expect(lin2.len() == 10);
            try std.testing.expect(lin2.atUnsafe(0) == 0.0);
            try std.testing.expect(lin2.atUnsafe(lin2.len() - 1) == 10.0);
        } else |e| {
            try std.testing.expect(e == error.OutOfMemory);
        }
    }
}

test "Vector: f32 struct ops and dot" {
    const alloc = std.testing.allocator;
    const V = Vector(f32);
    const tol_32: f32 = 1e-6;

    var a = try V.initZero(alloc, 3, false);
    defer a.deinit();
    try a.setAll([_]f32{ 1.0, 2.0, 2.0 });
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), a.norm(), tol_32);

    var b = try V.initZero(alloc, 3, false);
    defer b.deinit();
    try b.setAll(1); // comptime_int, coerces to f32
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), try dot(a, b), tol_32);

    try a.addInPlace(b);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), a.atUnsafe(0), tol_32);
    try a.subInPlace(b);
    a.multConst(2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), a.atUnsafe(0), tol_32);

    a.normalize();
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), a.norm(), tol_32);
}

test "Vector: Complex(f64) norm, normalize, setAll and dot conjugation" {
    const alloc = std.testing.allocator;
    const C = std.math.Complex(f64);
    const CV = Vector(C);
    const tol_c: f64 = 1e-12;

    var x = try CV.initZero(alloc, 2, false);
    defer x.deinit();
    x.setUnsafe(0, C.init(3.0, 4.0)); // |3+4i| = 5
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), x.norm(), tol_c);

    // dot(x, x) == norm(x)^2, and must be real
    const dxx = try dot(x, x);
    try std.testing.expectApproxEqAbs(@as(f64, 25.0), dxx.re, tol_c);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), dxx.im, tol_c);

    // Conjugation is on the left side: dot(x, y) == conj(dot(y, x))
    var y = try CV.initZero(alloc, 2, false);
    defer y.deinit();
    y.setUnsafe(0, C.init(1.0, -2.0));
    y.setUnsafe(1, C.init(2.0, 1.0));
    const dxy = try dot(x, y);
    const dyx = try dot(y, x);
    try std.testing.expect(sclr.approxEq(dxy, C.init(-5.0, -10.0), tol_c)); // conj(3+4i)*(1-2i)
    try std.testing.expect(sclr.approxEq(dxy, sclr.conj(dyx), tol_c));

    // setAll with a scalar of struct type T
    try y.setAll(C.init(1.0, 1.0));
    try std.testing.expect(sclr.eql(y.atUnsafe(1), C.init(1.0, 1.0)));

    x.normalize();
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), x.norm(), tol_c);
}

test "crossProd3d: signs for f64 and Complex" {
    const alloc = std.testing.allocator;
    const tol_c: f64 = 1e-12;

    var ex = try Vec.initZero(alloc, 3, false);
    defer ex.deinit();
    ex.setUnsafe(0, 1);
    var ey = try Vec.initZero(alloc, 3, false);
    defer ey.deinit();
    ey.setUnsafe(1, 1);

    // y-hat cross x-hat == -z-hat (catches wrong sign convention,
    // unlike x-hat cross y-hat where the negative terms are all zero)
    var c1 = try crossProd3d(alloc, ey, ex);
    defer c1.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), c1.atUnsafe(2), tol_c);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), c1.atUnsafe(0), tol_c);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), c1.atUnsafe(1), tol_c);

    // Complex: (0, i, 0) cross (0, 0, i) == (i*i, 0, 0) == (-1, 0, 0)
    const C = std.math.Complex(f64);
    const CV = Vector(C);
    var u = try CV.initZero(alloc, 3, false);
    defer u.deinit();
    u.setUnsafe(1, C.init(0.0, 1.0));
    var w = try CV.initZero(alloc, 3, false);
    defer w.deinit();
    w.setUnsafe(2, C.init(0.0, 1.0));
    var cw = try crossProd3d(alloc, u, w);
    defer cw.deinit();
    try std.testing.expect(sclr.approxEq(cw.atUnsafe(0), C.init(-1.0, 0.0), tol_c));
    try std.testing.expect(sclr.approxEq(cw.atUnsafe(1), sclr.zero(C), tol_c));
    try std.testing.expect(sclr.approxEq(cw.atUnsafe(2), sclr.zero(C), tol_c));
}

test "vecMult: outer product values for f64 and Complex" {
    const alloc = std.testing.allocator;
    const tol_c: f64 = 1e-12;

    var l = try Vec.initZero(alloc, 2, true);
    defer l.deinit();
    try l.setAll([_]f64{ 1.0, 2.0 });
    var r = try Vec.initZero(alloc, 2, false);
    defer r.deinit();
    try r.setAll([_]f64{ 3.0, 4.0 });

    var M = try vecMult(alloc, l, r);
    defer M.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), M.atUnsafe(0, 0), tol_c);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), M.atUnsafe(0, 1), tol_c);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), M.atUnsafe(1, 0), tol_c);
    try std.testing.expectApproxEqAbs(@as(f64, 8.0), M.atUnsafe(1, 1), tol_c);

    // Complex: no conjugation, M = l * r^T
    const C = std.math.Complex(f64);
    const CV = Vector(C);
    var lc = try CV.initZero(alloc, 2, true);
    defer lc.deinit();
    lc.setUnsafe(0, C.init(0.0, 1.0)); // i
    lc.setUnsafe(1, C.init(1.0, 0.0));
    var rc = try CV.initZero(alloc, 2, false);
    defer rc.deinit();
    rc.setUnsafe(0, C.init(0.0, 1.0)); // i
    rc.setUnsafe(1, C.init(2.0, 0.0));

    var MC = try vecMult(alloc, lc, rc);
    defer MC.deinit();
    try std.testing.expect(sclr.approxEq(MC.atUnsafe(0, 0), C.init(-1.0, 0.0), tol_c)); // i*i
    try std.testing.expect(sclr.approxEq(MC.atUnsafe(0, 1), C.init(0.0, 2.0), tol_c)); // i*2
    try std.testing.expect(sclr.approxEq(MC.atUnsafe(1, 0), C.init(0.0, 1.0), tol_c)); // 1*i
    try std.testing.expect(sclr.approxEq(MC.atUnsafe(1, 1), C.init(2.0, 0.0), tol_c)); // 1*2
}
