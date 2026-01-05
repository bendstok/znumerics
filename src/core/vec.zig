const std = @import("std");
const mat = @import("mat.zig");
const err_mod = @import("../error.zig");

pub const VecError = error{} || err_mod.Common;

/// The vector type.
/// Note that all vectors are presumed row vectors.
///
/// Stored on the heap.
pub const Vec = struct {
    alloc: std.mem.Allocator,
    data: []f64,
    colvec: bool,

    pub fn init(alloc: std.mem.Allocator, size: usize, colvec: bool) !Vec {
        if (size == 0) {
            return VecError.Empty;
        }
        const data = try alloc.alloc(f64, size);
        errdefer alloc.free(data);

        return .{ .alloc = alloc, .data = @constCast(data), .colvec = colvec };
    }

    pub fn initZero(alloc: std.mem.Allocator, size: usize, colvec: bool) !Vec {
        if (size == 0) {
            return VecError.Empty;
        }
        const data = try alloc.alloc(f64, size);
        errdefer alloc.free(data);

        @memset(data, 0);
        return .{ .alloc = alloc, .data = data, .colvec = colvec };
    }

    pub fn deinit(self: *Vec) void {
        self.alloc.free(self.data);
        self.* = undefined;
    }

    /// Returns the length of items in the Vector.
    ///
    /// See .norm() for euclidean length.
    pub fn len(self: Vec) usize {
        return self.data.len;
    }

    /// Returns the Euclidean norm of the vector.
    pub fn norm(self: Vec) f64 {
        var sum: f64 = 0.0;
        for (self.data) |d| {
            sum += std.math.pow(f64, d, 2);
        }
        sum = std.math.sqrt(sum);
        return sum;
    }

    /// Returns the value currently stored at idx.
    pub fn atUnsafe(self: Vec, idx: usize) f64 {
        return self.data[idx];
    }

    /// Returns the value currently stored at idx.
    ///
    /// Does bounds checks, returns a VecError.IndexOutOfBounds on failure.
    pub fn at(self: Vec, idx: usize) !f64 {
        if (idx >= self.len()) return VecError.IndexOutOfBounds;
        return self.data[idx];
    }

    /// Synonymys with .atSafe()
    pub fn get(self: Vec, idx: usize) !f64 {
        return (try at(self, idx));
    }

    /// Set one individual index to a value.
    pub fn setUnsafe(self: Vec, idx: usize, val: f64) void {
        self.data[idx] = val;
    }

    /// Set one individual index to a value.
    ///
    /// Does bounds checks. Returns a VecError.IndexOutOfBounds on failure.
    pub fn set(self: Vec, idx: usize, val: f64) !void {
        if (idx >= self.len()) return VecError.IndexOutOfBounds;
        self.data[idx] = val;
    }

    /// Transposes the vector, currently sets a flag.
    pub fn transpose(self: Vec) void {
        self.colvec = true;
    }

    /// Multiplies all the vector indicies by a constant.
    pub fn multConstUnsafe(self: Vec, val: f64) void {
        for (0..self.len()) |idx| {
            self.setUnsafe(idx, (self.atUnsafe(idx) * val));
        }
    }

    /// Multiplies all the vector indicies by a constant.
    pub fn multConst(self: Vec, val: f64) void {
        for (0..self.len()) |idx| {
            // Can use unsafe since we explicitly traverse the vector.
            self.setUnsafe(idx, self.atUnsafe(idx) * val);
        }
    }

    /// Sets all the values to a constant.
    ///
    /// NB! Only checks surface type, and does no size checking. See .setAll().
    ///
    /// cons must be of type: f64, comptime_float, [_]f64 or []f64.
    pub fn setAllUnsafe(self: Vec, cons: anytype) void {
        const T = @TypeOf(cons);

        switch (@typeInfo(T)) {
            .float => {
                @memset(self.data, cons);
            },
            .array => {
                @memcpy(self.data, cons[0..]);
            },
            .pointer => {
                @memcpy(self.data, cons[0..]);
            },
            .comptime_float => {
                @memset(self.data, cons);
            },
            else => {
                @compileError("VEC| setAll: Expected f64, comptime_float, []f64 or [_]f64, got " ++ @typeName(T));
            },
        }
    }

    /// Sets all the values to a constant.
    ///
    /// Checks lengths of input and types.
    /// Throws a compileError on type failure. Returns a
    /// VecError.SizeMismatch on size failure.
    ///
    /// cons must be of type: f64, comptime_float, [_]f64 or []f64.
    pub fn setAll(self: Vec, cons: anytype) !void {
        const T = @TypeOf(cons);

        switch (@typeInfo(T)) {
            .float => {
                if (T != f64) @compileError("VEC| setAll .float: only f64 is supported, got: " ++ @typeName(T) ++ "\n");
                @memset(self.data, cons);
            },
            .array => |arr| {
                if (arr.child != f64) @compileError("VEC| setAll .array: only []f64 supported, got: " ++ @typeName(T) ++ "\n");
                if (cons.len != self.data.len) return VecError.SizeMismatch;
                @memcpy(self.data, cons[0..]);
            },
            .pointer => |p| {
                if (p.child != f64) @compileError("VEC| setAll .pointer: Only f64 children are allowed, got: " ++ @typeName(p.child) ++ "\n");
                if (p.size != .slice) @compileError("VEC| setAll .pointer: Pointer must be of a Zig slice .\n");
                if (cons.len != self.data.len) return VecError.SizeMismatch;
                @memcpy(self.data, cons[0..]);
            },
            .comptime_float => {
                @memset(self.data, cons);
            },
            else => {
                @compileError("VEC| setAll: Expected f64, comptime_float, []f64 or [_]f64, got " ++ @typeName(T));
            },
        }
    }

    pub fn printVec(self: Vec) !void {
        for (0..self.len()) |idx| {
            //std.debug.print("c={}", .{c});
            const val: f64 = try self.at(idx);
            var tmp: [64]u8 = undefined;
            const s = try std.fmt.bufPrint(&tmp, "{:.3}", .{val});
            std.debug.print("{s: >10} ", .{s});
            if (self.colvec) std.debug.print("\n", .{});
        }
        std.debug.print("\n", .{});
    }

    /// Expands the vector, and fills the new space with 'fill'.
    ///
    /// Returns an error.OutOfMemory or VecError.Empty (if new_len == 0) on failure.
    pub fn resize(self: *Vec, new_len: usize, fill: f64) !void {
        if (new_len == 0) return VecError.Empty;
        if (new_len == self.data.len) return;
        const old_len = self.len();
        self.data = try self.alloc.realloc(self.data, new_len);
        errdefer self.deinit();

        if (new_len > old_len) {
            @memset(self.data[old_len..], fill);
        }
    }

    /// Add in place
    ///
    /// X = X + Y
    ///
    /// Only takes vectors of same size.
    pub fn addInPlace(self: Vec, ToAdd: Vec) !void {
        if (self.len() != ToAdd.len()) return VecError.SizeMismatch;
        for (0..self.len()) |idx| {
            self.setUnsafe(idx, self.atUnsafe(idx) + ToAdd.at(idx));
        }
    }

    /// Subtract in place
    ///
    /// X = X - Y
    ///
    /// Only takes vectors of same size.
    pub fn subInPlace(self: Vec, toSub: Vec) !void {
        if (self.len() != toSub.len()) return VecError.SizeMismatch;
        for (0..self.len()) |idx| {
            self.setUnsafe(idx, self.atUnsafe(idx) - toSub.at(idx));
        }
    }
};

/// Multiplies two vectors, where the left one has to be a coloumn vector.
///
/// Note that: a^T * b == b^T * a. (All vectors unless otherwise stated, are row vectors).
///
/// Returns the matrix.
pub fn vecMult(alloc: std.mem.Allocator, left: Vec, right: Vec) !mat.Mat {
    if (!left.colvec) return VecError.BadShape;
    if (left.len() != right.len()) return VecError.SizeMismatch;
    var returnMat = try mat.Mat.initZero(alloc, left.len(), left.len());
    errdefer returnMat.deinit();

    // This will always be a square matrix so which vec decides the iterator is arbitrary
    // Its set as this for ease of thinking
    for (0..left.len()) |r| {
        for (0..right.len()) |c| {
            try returnMat.set(r, c, (try left.at(r)) * (try right.at(c)));
        }
    }
    return returnMat;
}

/// Computes the 3D cross product vector from two base vectors.
///
/// Returns the produced vector.
pub fn crossProd3d(alloc: std.mem.Allocator, left: Vec, right: Vec) !Vec {
    if (left.len() != 3 or right.len() != 3) return VecError.BadShape;
    // Can use unsafe since we explicitly know the length.
    const s1 = (left.atUnsafe(1)) * (right.atUnsafe(2)) + (left.atUnsafe(2)) * (right.atUnsafe(1));
    const s2 = (left.atUnsafe(2)) * (right.atUnsafe(0)) + (left.atUnsafe(0)) * (right.atUnsafe(2));
    const s3 = (left.atUnsafe(0)) * (right.atUnsafe(1)) + (left.atUnsafe(1)) * (right.atUnsafe(0));
    var retVec = try Vec.init(alloc, 3, false);
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
pub fn linspace(alloc: std.mem.Allocator, start: f64, end: f64, steps: usize, includeEndPoint: bool) !Vec {
    if (steps == 0) return VecError.BadShape;

    var retVec = try Vec.initZero(alloc, steps, false);
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

/// Computes the dot product between two vectors. These have to both row vectors.
///
/// Returns the value.
pub fn dot(left: Vec, right: Vec) !f64 {
    if (left.len() != right.len()) return VecError.SizeMismatch;
    if (left.colvec and right.colvec) return VecError.BadShape;
    var res: f64 = 0;
    for (0..left.len()) |i| {
        // Can use unsafe here because we know we are within bounds
        res += (left.atUnsafe(i)) * (right.atUnsafe(i));
    }
    return res;
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
