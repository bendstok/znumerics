const std = @import("std");
const vec = @import("vec.zig");
const gj = @import("../linalg/gaussjordan.zig");
const err_mod = @import("../error.zig");
const sclr = @import("scalar.zig");

const Vec = vec.Vec;

pub const MatError = err_mod.Common;

pub const InverseError = error{
    Singular,
} || err_mod.Common;

/// Everything expm() can fail with: its own shape checks plus the
/// Gauss-Jordan solve of the Pade system and allocation.
pub const ExpmError = MatError || gj.GaussJordanError || std.mem.Allocator.Error;

/// The Matrix type constructor.
/// The data is stored as one flat slice, row-major.
///
/// Stored on the heap.
pub fn Matrix(comptime T: type) type {
    return struct {
        const Self = @This();
        pub const element = T;

        alloc: std.mem.Allocator,
        rows: usize,
        cols: usize,
        data: []T,

        pub fn init(alloc: std.mem.Allocator, rows: usize, cols: usize) (MatError || std.mem.Allocator.Error)!Self {
            if (rows == 0 or cols == 0) return MatError.Empty;
            const data = try alloc.alloc(T, rows * cols);
            errdefer alloc.free(data);

            return .{ .alloc = alloc, .rows = rows, .cols = cols, .data = data };
        }

        pub fn initZero(alloc: std.mem.Allocator, rows: usize, cols: usize) (MatError || std.mem.Allocator.Error)!Self {
            if (rows == 0 or cols == 0) return MatError.Empty;
            const data = try alloc.alloc(T, rows * cols);
            errdefer alloc.free(data);

            @memset(data, sclr.zero(T));
            return .{ .alloc = alloc, .rows = rows, .cols = cols, .data = data };
        }

        pub fn initIdentity(alloc: std.mem.Allocator, rows: usize, cols: usize) (MatError || std.mem.Allocator.Error)!Self {
            if (rows == 0 or cols == 0) return MatError.Empty;
            const data = try alloc.alloc(T, rows * cols);
            errdefer alloc.free(data);
            @memset(data, sclr.zero(T));

            const i: usize = @min(rows, cols);
            for (0..i) |eq| {
                data[eq * cols + eq] = sclr.one(T);
            }

            return .{ .alloc = alloc, .rows = rows, .cols = cols, .data = data };
        }

        pub fn deinit(self: *Self) void {
            self.alloc.free(self.data);
            self.* = undefined;
        }

        fn idx(self: Self, r: usize, c: usize) usize {
            return (r * self.cols + c);
        }

        fn get_row(self: Self, r: usize) []T {
            return (self.data[r * self.cols ..][0..self.cols]);
        }

        /// Returns the value at the corresponding location.
        ///
        /// Does bounds checking. Returns a MatError.IndexOutOfBounds on failure.
        pub fn at(self: Self, row: usize, col: usize) MatError!T {
            try boundsCheck(self, row, col);
            return self.data[idx(self, row, col)];
        }

        /// Synonymys with .at().
        pub fn get(self: Self, row: usize, col: usize) MatError!T {
            return (try at(self, row, col));
        }

        /// Unsafe version of .at()
        ///
        /// No Boundary checks.
        pub fn atUnsafe(self: Self, row: usize, col: usize) T {
            return self.data[idx(self, row, col)];
        }

        /// Sets the location to the value.
        ///
        /// Does bounds checking. Returns a MatError.IndexOutOfBounds on failure.
        pub fn set(self: Self, row: usize, col: usize, val: T) MatError!void {
            try boundsCheck(self, row, col);
            self.data[idx(self, row, col)] = val;
        }

        /// Sets the location to the value.
        ///
        /// Does no bounds checking, unsafe. See .set().
        pub fn setUnsafe(self: Self, row: usize, col: usize, val: T) void {
            self.data[idx(self, row, col)] = val;
        }

        /// Sets all the values in the matrix.
        pub fn setAll(self: Self, val: T) void {
            @memset(self.data, val);
        }

        // TODO: We probably need more advanced type matching for comptime floats?
        /// Sets all values in a row to 'new_values'.
        ///
        /// 'new_values' must be of type: f64, comptime_float, [_]f64 or []f64.
        /// Returns MatError.IndexOutOfBounds on a bad row and
        /// MatError.SizeMismatch on a length mismatch.
        pub fn setRow(self: Self, row: usize, new_values: anytype) MatError!void {
            if (row >= self.rows) return MatError.IndexOutOfBounds;
            const dst = self.get_row(row);
            const T_new = @TypeOf(new_values);

            switch (@typeInfo(T_new)) {
                .array => |arr| {
                    if (arr.child != T) @compileError("MAT| setRow .array: got wrong type inside array, got: " ++ @typeName(T_new) ++ ". Expected: " ++ @typeName(T) ++ "\n");
                    if (new_values.len != self.cols) return MatError.SizeMismatch;
                    @memcpy(dst, new_values[0..]);
                },
                .pointer => |p| {
                    if (p.child != T) @compileError("MAT| setRow .pointer: wrong type of children, got: " ++ @typeName(p.child) ++ ". Expected: " ++ @typeName(T) ++ "\n");
                    if (p.size != .slice) @compileError("MAT| setRow .pointer: Pointer must be of a Zig slice. \n");
                    if (new_values.len != self.cols) return MatError.SizeMismatch;
                    @memcpy(dst, new_values);
                },
                else => {
                    if (T_new != T) {
                        @compileError("MAT| setRow: types do not match, got: " ++ @typeName(T_new) ++ ". Expected: " ++ @typeName(T) ++ "\n");
                    } else {
                        @memset(dst, new_values);
                    }
                },
            }
        }

        /// Returns the row as a Vector(T). No bounds checks on inputted row.
        ///
        /// Deep copies.
        pub fn getRow(self: Self, row: usize, alloc: std.mem.Allocator) (vec.VecError || std.mem.Allocator.Error)!vec.Vector(T) {
            var ret_vec = try vec.Vector(T).initZero(alloc, self.cols, false);
            errdefer ret_vec.deinit();
            for (0..self.cols) |j| {
                ret_vec.setUnsafe(j, self.atUnsafe(row, j));
            }
            return ret_vec;
        }

        /// Returns the col as a Vector(T). No bounds checks on inputted col.
        ///
        /// Deep copies.
        pub fn getCol(self: Self, col: usize, alloc: std.mem.Allocator) (vec.VecError || std.mem.Allocator.Error)!vec.Vector(T) {
            var ret_vec = try vec.Vector(T).initZero(alloc, self.rows, true);
            errdefer ret_vec.deinit();
            for (0..self.rows) |i| {
                ret_vec.setUnsafe(i, self.atUnsafe(i, col));
            }
            return ret_vec;
        }
        // TODO: We probably need more advanced type matching for comptime floats?
        /// Sets all values in a coloumn to the values in 'new_values'.
        ///
        /// Checks underlying type of pointers and arrays and does bound checks.
        /// Returns a MatError on failure.
        ///
        /// 'cons' must be of same type as underlying data.
        pub fn setCol(self: Self, col: usize, new_values: anytype) MatError!void {
            const T_new = @TypeOf(new_values);

            switch (@typeInfo(T_new)) {
                .array => |arr| {
                    if (arr.child != T) @compileError("MAT| setCol .array: children do match type with data type, got: " ++ @typeName(T_new) ++ ". Expected: " ++ @typeName(T) ++ "\n");
                    if (new_values.len != self.rows) return MatError.SizeMismatch;
                    for (0..self.rows) |r| {
                        try self.set(r, col, new_values[r]);
                    }
                },
                .pointer => |p| {
                    if (p.child != T) @compileError("MAT| setCol .pointer: children do match type with data type, got: " ++ @typeName(p.child) ++ ". Expected: " ++ @typeName(T) ++ "\n");
                    if (p.size != .slice) @compileError("MAT| setCol .pointer: Pointer must be of a Zig slice. \n");
                    if (new_values.len != self.rows) return MatError.SizeMismatch;
                    for (0..self.rows) |r| {
                        try self.set(r, col, new_values[r]);
                    }
                },
                else => {
                    if (T_new != T) {
                        @compileError("MAT| setCol: type gotten does not match matrix type, got: " ++ @typeName(T_new) ++ ". Expected: " ++ @typeName(T) ++ "\n");
                    } else {
                        for (0..self.rows) |r| {
                            try self.set(r, col, new_values);
                        }
                    }
                },
            }
        }

        /// Prints a view of the matrix to std.debug.print
        ///
        /// Can fail because of bufPrint.
        pub fn printMat(self: Self) error{NoSpaceLeft}!void {
            for (0..self.rows) |r| {
                //std.debug.print("r={}: ", .{r});
                for (0..self.cols) |c| {
                    //std.debug.print("c={}", .{c});
                    // In-bounds by loop construction.
                    const val: T = self.atUnsafe(r, c);
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
                }
                std.debug.print("\n", .{});
            }
        }

        /// Transposes the matrix in place.
        ///
        /// Only works if the matrix is square.
        ///
        /// Returns a MatError.BadShape on failure.
        // TODO: Update this, we can now expand the matrix easily.
        pub fn transposeInPlace(self: Self) MatError!void {
            if (self.rows != self.cols) return MatError.BadShape;

            for (0..self.rows) |r| {
                for (r + 1..self.cols) |c| {
                    const s1 = try self.at(r, c);
                    const s2 = try self.at(c, r);
                    try self.set(r, c, s2);
                    try self.set(c, r, s1);
                }
            }
        }

        /// Expands a matrix to the specified rows and cols.
        ///
        /// Returns an MatError.Empty if the new rows and cols are 0.
        ///
        /// To expand the matrix a new matrix is made, and takes the spot of the old one.
        /// This is to prevent complex situations in case of OOM error.
        pub fn expand(self: *Self, rows: usize, cols: usize, fill: T) (MatError || std.mem.Allocator.Error)!void {
            if (rows == 0 or cols == 0) return MatError.Empty;

            if (self.rows >= rows or self.cols >= cols) {
                std.log.warn("Attempted shrinking when using expand() \n", .{});
                return;
            }

            const old_rows = self.rows;
            const old_cols = self.cols;

            const new_rows: usize = @max(old_rows, rows);
            const new_cols: usize = @max(old_cols, cols);

            // Build new matrix
            var newMat = try Matrix(T).initZero(self.alloc, new_rows, new_cols);
            errdefer newMat.deinit();

            // Copy old contents
            for (0..self.rows) |r| {
                for (0..self.cols) |c| {
                    newMat.setUnsafe(r, c, self.atUnsafe(r, c));
                }
            }
            // Fancy way of saying if  fill != zero
            if (!sclr.eql(fill, sclr.zero(T))) {
                for (self.rows..new_rows) |r| {
                    for (0..new_cols) |c| {
                        newMat.setUnsafe(r, c, fill);
                    }
                }
                // fill new cols in old rows
                for (0..self.rows) |r| {
                    for (self.cols..new_cols) |c| {
                        newMat.setUnsafe(r, c, fill);
                    }
                }
            }

            // Swap
            self.deinit();
            self.* = newMat;
        }

        // TODO: Check that mult type matches data type
        /// Multiplies all values in a row by 'mult'
        pub fn multRow(self: Self, row: usize, mult: T) MatError!void {
            try boundsCheck(self, row, 0);
            for (0..self.cols) |j| {
                self.setUnsafe(row, j, sclr.mul(mult, self.atUnsafe(row, j)));
            }
        }

        // TODO: Type check
        /// Multiplies all values in the matrix by 'mult'
        pub fn multAll(self: Self, mult: T) void {
            for (self.data) |*v| {
                v.* = sclr.mul(v.*, mult);
            }
        }

        /// Swaps two rows.
        ///
        /// Does boundary checks. Returns an MatError.IndexOutOfBounds on failure.
        pub fn swapRow(self: Self, row1: usize, row2: usize) MatError!void {
            if (row1 >= self.rows or row2 >= self.rows) return MatError.IndexOutOfBounds;
            const r1 = self.get_row(row1);
            const r2 = self.get_row(row2);
            for (r1, r2) |*a, *b| {
                std.mem.swap(T, a, b);
            }
        }

        /// Returns the Norm_1 (max absolute coloumn sum) of the matrix
        pub fn norm1(self: Self) sclr.Real(T) {
            var max_sum: sclr.Real(T) = 0;
            for (0..self.cols) |c| {
                var col_sum: sclr.Real(T) = 0;
                for (0..self.rows) |r| {
                    // In-bounds by loop construction.
                    col_sum += sclr.abs(self.atUnsafe(r, c));
                }
                if (col_sum > max_sum) max_sum = col_sum;
            }
            return max_sum;
        }

        /// Returns a matrix which is a deep copy of itself
        pub fn clone(self: Self) std.mem.Allocator.Error!Matrix(T) {
            const data = try self.alloc.dupe(T, self.data);
            return .{ .alloc = self.alloc, .rows = self.rows, .cols = self.cols, .data = data };
        }

        // TODO: Type check
        /// C = A + B, written into caller-provided 'out'.
        pub fn addInto(self: Self, toAdd: Matrix(T), out: Matrix(T)) MatError!void {
            if (self.rows != toAdd.rows or self.cols != toAdd.cols) return MatError.SizeMismatch;
            if (out.rows != self.rows or out.cols != self.cols) return MatError.SizeMismatch;
            for (out.data, self.data, toAdd.data) |*o, a, b| o.* = sclr.add(a, b);
        }

        // TODO: Type check
        /// Adds the two matrices index by index
        /// and returns the result. The Matrices must be same size.
        ///
        /// C = A + B, where A -> self and B -> toAdd.
        ///
        /// See .addInPlace() for no return. See .addInto() for providing the result matrix.
        pub fn add(self: Self, toAdd: Matrix(T)) (MatError || std.mem.Allocator.Error)!Matrix(T) {
            const r = self.rows;
            const c = self.cols;
            if (r != toAdd.rows or c != toAdd.cols) return MatError.SizeMismatch;
            const retMat = try Matrix(T).init(self.alloc, r, c);

            for (retMat.data, self.data, toAdd.data) |*o, a, b| {
                o.* = sclr.add(a, b);
            }
            return retMat;
        }

        // TODO: Type check
        /// Uses SIMD to add the matrices together, and returns the matrix. Same as .add().
        ///
        /// Tries to use AVX-512 SIMD, but the compiler will fallback to the greatest
        /// available SIMD instruction available.
        ///
        /// With contiguous storage the whole matrix is one flat slice, so the
        /// adds are direct 8-lane vector loads/stores over the full data,
        /// not per-row gathers. The tail is done in non-SIMD.
        ///
        /// Falls back to normal .add() for Complex Matrices.
        pub fn addSIMD(self: Self, toAdd: Matrix(T)) (MatError || std.mem.Allocator.Error)!Matrix(T) {
            if (self.rows != toAdd.rows or self.cols != toAdd.cols) return MatError.SizeMismatch;
            if (comptime sclr.isComplex(T)) return self.add(toAdd); // scalar fallback for complex
            const vec_len: usize = std.simd.suggestVectorLength(T) orelse return self.add(toAdd);

            const retMat = try Matrix(T).init(self.alloc, self.rows, self.cols);
            const n = self.data.len;
            const simd_n = (n / vec_len) * vec_len;

            var i: usize = 0;
            while (i < simd_n) : (i += vec_len) {
                const a: @Vector(vec_len, T) = self.data[i..][0..vec_len].*;
                const b: @Vector(vec_len, T) = toAdd.data[i..][0..vec_len].*;
                retMat.data[i..][0..vec_len].* = sclr.add(a, b);
            }
            // Tail
            while (i < n) : (i += 1) {
                retMat.data[i] = sclr.add(self.data[i], toAdd.data[i]);
            }
            return retMat;
        }

        // TODO: Type check
        /// Adds 'toAdd' to self.
        ///
        /// A = A + B, where A -> self and B -> toAdd.
        ///
        /// See .add() to return a matrix with the result,
        /// leaving A & B unchanged.
        pub fn addInPlace(self: Self, toAdd: Matrix(T)) MatError!void {
            const r = self.rows;
            const c = self.cols;
            if (r != toAdd.rows or c != toAdd.cols) return MatError.SizeMismatch;
            for (self.data, toAdd.data) |*a, b| {
                a.* = sclr.add(a.*, b.*);
            }
        }

        // TODO: Type check
        /// Subtracts the two matrices index by index
        /// and returns the result. The Matrices must be same size.
        ///
        /// C = A - B, where A -> self and B is toSub.
        ///
        /// See .subInPlace() for no return.
        pub fn sub(self: Self, toSub: Matrix(T)) (MatError || std.mem.Allocator.Error)!Matrix(T) {
            const r = self.rows;
            const c = self.cols;
            if (r != toSub.rows or c != toSub.cols) return MatError.SizeMismatch;
            const retMat = try Matrix(T).init(self.alloc, r, c);

            for (retMat.data, self.data, toSub.data) |*o, a, b| {
                o.* = sclr.sub(a, b);
            }
            return retMat;
        }

        // TODO: Type check
        /// Subtracts self by toSub.
        ///
        /// C = A - B, where A -> self and B is toSub.
        ///
        /// See .sub() for returning the result,
        /// leaving A & B unchanged.
        pub fn subInPlace(self: Self, toSub: Matrix(T)) MatError!void {
            const r = self.rows;
            const c = self.cols;
            if (r != toSub.rows or c != toSub.cols) return MatError.SizeMismatch;
            for (self.data, toSub.data) |*a, b| {
                a.* = sclr.sub(a.*, b);
            }
        }

        // TODO: Type check
        /// Returns the trace of the matrix
        ///
        /// Must be square
        pub fn trace(self: Self) MatError!T {
            if (self.rows != self.cols) return MatError.BadShape;
            var tr: T = sclr.zero(T);
            for (0..self.rows) |i| {
                tr = sclr.add(tr, self.atUnsafe(i, i));
            }
            return tr;
        }

        /// Check bounds. Throws an IndexOutOfBounds on failure.
        fn boundsCheck(self: Self, row: usize, col: usize) MatError!void {
            if (row >= self.rows) {
                std.log.warn("Tried to access row {}, but the valid range is [0,{}] \n", .{ row, self.rows - 1 });
                return MatError.IndexOutOfBounds;
            }
            if (col >= self.cols) {
                std.log.warn("Tried to access coloumn: {}, but the valid range is [0,{}] \n", .{ col, self.cols - 1 });
                return MatError.IndexOutOfBounds;
            }
            return;
        }
    };
}

/// Comptime: true if M is a Matrix(...) instantiation.
pub fn isMatrix(comptime M: type) bool {
    if (@typeInfo(M) != .@"struct") return false;
    if (!@hasDecl(M, "element")) return false;
    return M == Matrix(M.element);
}

/// Comptime: asserts M is a Matrix(...) and returns its element type.
/// Use at the top of free functions taking `anytype`.
pub fn ElementOf(comptime M: type) type {
    if (!isMatrix(M)) @compileError("MAT| expected a Matrix(T), got: " ++ @typeName(M));
    return M.element;
}

/// Comptime: Check whether two matrices have the same element type.
fn sameElement(comptime M1: type, comptime M2: type) bool {
    return ElementOf(M1) == ElementOf(M2);
}

// Backward Compatibility
pub const Mat = Matrix(f64);
pub const CMat = Matrix(std.math.Complex(f64));
pub const Mat_32 = Matrix(f32);
pub const CMat_32 = Matrix(std.math.Complex(f32));

/// Deep copies the values from the 'from' matrix to the 'to' matrix
///
/// The recipient matrix must be >= the 'from' matrix.
pub fn copyMat(from: anytype, to: @TypeOf(from)) MatError!void {
    _ = ElementOf(@TypeOf(from)); // assert 'from' is a Matrix(T)

    if (to.cols < from.cols or to.rows < from.rows) return MatError.SizeMismatch;
    for (0..from.rows) |r| {
        @memcpy(to.get_row(r)[0..from.cols], from.get_row(r));
    }
}

/// Returns a new matrix that is the transpose of 'self'
///
/// Works for any matrix.
pub fn transpose(self: anytype, alloc: std.mem.Allocator) (MatError || std.mem.Allocator.Error)!@TypeOf(self) {
    const M = @TypeOf(self);
    _ = ElementOf(M); // assert its a Matrix(T)

    var retMat = try M.initZero(alloc, self.cols, self.rows);
    errdefer retMat.deinit();

    for (0..self.rows) |r| {
        for (0..self.cols) |c| {
            try retMat.set(c, r, try self.at(r, c));
        }
    }
    return retMat;
}

/// Tries to inverse the matrix. Only accepts square matrices.
///
/// Uses the fact reduced row echelon of [A | I] = [I | A^(-1)].
///
/// Returns the inverse of the matrix.
///
/// Returns an InverseError.BadShape if A is not square.
/// Returns an InverseError.Singular if it reaches a divide by zero
/// when row reducing or backsolving.
pub fn inverse(alloc: std.mem.Allocator, A: anytype) (InverseError || std.mem.Allocator.Error)!@TypeOf(A) {
    if (A.rows != A.cols) return InverseError.BadShape;
    const M = @TypeOf(A);
    const T = ElementOf(M); // assert its a Matrix(T)

    var mat_mod = try M.initZero(alloc, A.rows, A.cols * 2);
    defer mat_mod.deinit();
    var ret_mat = try M.initZero(alloc, A.rows, A.cols);
    errdefer ret_mat.deinit();

    try copyMat(A, mat_mod);
    for (A.cols..A.cols * 2) |c| {
        for (0..A.rows) |r| {
            if (c - A.cols != r) continue;
            try mat_mod.set(r, c, sclr.one(T));
        }
    }
    // Reduced row echelon with partial pivoting
    for (0..mat_mod.rows - 1) |c| {
        // Bring the row with the largest |value| in column c up to row c.
        // Improves numerical stability and handles zeros on the diagonal.
        var pivot_row = c;
        var pivot_mag = sclr.abs(try mat_mod.at(c, c));
        for (c + 1..mat_mod.rows) |r| {
            const mag = sclr.abs(try mat_mod.at(r, c));
            if (mag > pivot_mag) {
                pivot_mag = mag;
                pivot_row = r;
            }
        }
        if (pivot_mag == 0) return InverseError.Singular;
        if (pivot_row != c) try mat_mod.swapRow(c, pivot_row);

        // Row Reduce
        const denom = try mat_mod.at(c, c);
        var i: usize = c + 1;
        while (i < mat_mod.rows) : (i += 1) {
            const L = sclr.neg(sclr.div(try mat_mod.at(i, c), denom));
            for (c..mat_mod.cols) |col| {
                const new_val = sclr.add(try mat_mod.at(i, col), sclr.mul(L, try mat_mod.at(c, col)));
                try mat_mod.set(i, col, new_val);
            }
        }
    }

    // Backsolve
    var c: usize = mat_mod.rows - 1;
    while (true) {
        var i = c;
        while (i > 0) {
            i -= 1;
            const denom = try mat_mod.at(c, c);
            if (sclr.eql(denom, sclr.zero(T))) return InverseError.Singular;

            const L = sclr.neg(sclr.div(try mat_mod.at(i, c), denom));
            for (c..mat_mod.cols) |col| {
                const new_val = sclr.add(try mat_mod.at(i, col), sclr.mul(try mat_mod.at(c, col), L));
                try mat_mod.set(i, col, new_val);
            }
        }
        if (c == 0) break;
        c -= 1;
    }

    // Make left side the identity matrix
    for (0..mat_mod.rows) |r| {
        const diag = try mat_mod.at(r, r);
        if (sclr.eql(diag, sclr.zero(T))) return InverseError.Singular;
        try mat_mod.multRow(r, sclr.div(sclr.one(T), diag));
    }

    // Copy Right side into ret_mat
    for (0..mat_mod.rows) |rw| {
        for (A.cols..A.cols * 2) |cl| {
            const ret_mat_cidx = cl - A.cols;
            try ret_mat.set(rw, ret_mat_cidx, try mat_mod.at(rw, cl));
        }
    }
    return ret_mat;
}

/// Multiplies the two matrices by each other.
///
/// Returns an MatError.Sizemismatch on failure.
pub fn matMult(alloc: std.mem.Allocator, left: anytype, right: @TypeOf(left)) (MatError || std.mem.Allocator.Error)!@TypeOf(left) {
    if (left.cols != right.rows) return MatError.SizeMismatch;
    const M = @TypeOf(left);
    _ = ElementOf(M); // assert: is a Matrix(T)

    const retMat = try M.initZero(alloc, left.rows, right.cols);

    // i-k-j loop order: all three row slices are traversed
    // sequentially, which is cache friendly and lets the
    // compiler auto-vectorize the inner loop.
    for (0..left.rows) |i| {
        const left_row = left.get_row(i);
        const out_row = retMat.get_row(i);
        for (0..left.cols) |k| {
            const a = left_row[k];
            const right_row = right.get_row(k);
            for (out_row, right_row) |*o, b| {
                o.* = sclr.add(o.*, sclr.mul(a, b));
            }
        }
    }
    return retMat;
}

/// A (mxn) * x (n) -> out (mx1)
pub fn matVec(alloc: std.mem.Allocator, A: anytype, x: vec.Vector(ElementOf(@TypeOf(A)))) (MatError || std.mem.Allocator.Error)!vec.Vector(ElementOf(@TypeOf(A))) {
    if (A.cols != x.len()) return MatError.SizeMismatch;
    const T = ElementOf(@TypeOf(A));

    var out = try vec.Vector(T).initZero(alloc, A.rows, true);
    for (0..A.rows) |i| {
        var s = sclr.zero(T);
        for (0..A.cols) |j| s = sclr.add(s, sclr.mul(A.atUnsafe(i, j), x.atUnsafe(j)));
        out.setUnsafe(i, s);
    }
    return out;
}

/// Multiplies the two matrices by each other using explicit SIMD.
///
/// Transposes 'right' so both operands of each dot product are
/// contiguous rows, then accumulates in 8-lane vectors with direct
/// slice loads. The tail is done in non-SIMD.
///
/// Complex matrices have no SIMD path and silently fall back to matMult().
///
/// Returns an MatError.SizeMismatch on failure.
pub fn matMultSIMD(alloc: std.mem.Allocator, left: anytype, right: @TypeOf(left)) (MatError || std.mem.Allocator.Error)!@TypeOf(left) {
    if (left.cols != right.rows) return MatError.SizeMismatch;
    const M = @TypeOf(left);
    const T = ElementOf(M);

    // Complex has no SIMD path; falls back to the scalar multiply (see doc comment).
    if (comptime sclr.isComplex(T)) {
        return matMult(alloc, left, right);
    }
    var retMat = try M.init(alloc, left.rows, right.cols);
    // In case of error, we need to deinit the matrix
    errdefer retMat.deinit();

    const vec_len = comptime std.simd.suggestVectorLength(T) orelse 8;

    // Coloumns become rows so we can access them contiguously
    var rightT = try transpose(right, alloc);
    defer rightT.deinit();

    const n = left.cols;
    const simd_n = (n / vec_len) * vec_len;

    for (0..left.rows) |r| {
        const left_row = left.get_row(r);
        const out_row = retMat.get_row(r);
        for (0..right.cols) |c| {
            const rightT_row = rightT.get_row(c);
            var accumulator: @Vector(vec_len, T) = @splat(sclr.zero(T));

            var k: usize = 0;
            while (k < simd_n) : (k += vec_len) {
                const a: @Vector(vec_len, T) = left_row[k..][0..vec_len].*;
                const b: @Vector(vec_len, T) = rightT_row[k..][0..vec_len].*;
                // Guaranteed this works since not Complex
                accumulator += a * b;
            }
            // Horizontal sum of SIMD accumulator
            var sum: T = @reduce(.Add, accumulator);
            // Tail
            while (k < n) : (k += 1) {
                sum += left_row[k] * rightT_row[k];
            }

            out_row[c] = sum;
        }
    }
    return retMat;
}

/// Compute the matrix exponential exp(A) for a square matrix A.
///
/// Uses the scaling and squaring method with a Pade approximant of
/// degree 13, following SciPy. This reduces numerical error,
/// by not solving via inverses or other methods.
///
/// Returns exp(A).
///
/// On Failure: returns an ExpmError (shape errors, a singular Pade
/// system, or OutOfMemory).
pub fn expm(alloc: std.mem.Allocator, A: anytype) ExpmError!@TypeOf(A) {
    const M = @TypeOf(A);
    const T = ElementOf(M);
    // The Pade solve goes through Vec + gaussJordan.

    // We  use an Arena allocator since these live short and
    // Die together.
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // Taken from scipy.
    if (A.rows != A.cols) return MatError.BadShape;
    const THETA_13: f64 = 5.371920351148152;
    const norm = A.norm1();

    const n = A.rows; // Square matrix
    var s: u32 = 0;
    if (norm > THETA_13) {
        s = @as(u32, @intFromFloat(@ceil(std.math.log2(norm / THETA_13))));
    }

    const scale = std.math.pow(f64, 2.0, @as(f64, @floatFromInt(s)));
    var As = try A.clone();
    defer As.deinit();

    As.multAll(sclr.fromReal(T, @floatCast(1.0 / scale)));

    // A2, A4, A6

    var A2 = try matMult(arena, As, As);
    defer A2.deinit();
    var A4 = try matMult(arena, A2, A2);
    defer A4.deinit();
    var A6 = try matMult(arena, A4, A2);
    defer A6.deinit();

    var U = try M.initZero(arena, n, n);
    defer U.deinit();
    var V = try M.initZero(arena, n, n);
    defer V.deinit();

    try pade13(arena, As, A2, A4, A6, U, V);

    // (V - U) * R = (V + U) | Q * R = P
    var P = try V.add(U);
    defer P.deinit();
    var Q = try V.sub(U);
    defer Q.deinit();
    // Solve Q * R = P

    var R = try solvePadeSystem(alloc, Q, P);

    // Repeated squaring of R
    var k: u32 = 0;
    while (k < s) : (k += 1) {
        const R2 = try matMult(arena, R, R);
        R.deinit();
        R = R2;
    }
    defer R.deinit();
    const result_data = try alloc.dupe(T, R.data);

    return .{ .alloc = alloc, .rows = R.rows, .cols = R.cols, .data = result_data };
}

/// Compute the characteristic polynomial of an n×n matrix A:
///   det(z·I - A) = z^n + coeffs[1]·z^(n-1) + … + coeffs[n]
/// and return it in “z⁻¹ form” as coeffs[0..n]:
///   [1, a₁, a₂, …, aₙ]
///
/// - coeffs: output slice of length ≥ n+1
pub fn charPoly(
    alloc: std.mem.Allocator,
    A: anytype,
    coeffs: []ElementOf(@TypeOf(A)),
) (MatError || std.mem.Allocator.Error)!void {
    const M = @TypeOf(A);
    const T = ElementOf(M);
    if (A.rows != A.cols) return MatError.BadShape;
    if (coeffs.len < A.rows + 1) return MatError.SizeMismatch;
    const n = A.rows;
    var Bprev = try M.initIdentity(alloc, n, n);
    defer Bprev.deinit();

    coeffs[0] = sclr.one(T);

    // Iteratively compute bₖ and update B_prev
    for (1..n + 1) |k| {
        // AB = A * Bprev
        var AB = matMult(alloc, A, Bprev) catch |e| {
            return e;
        };
        defer AB.deinit();

        const tr = try AB.trace();

        // bₖ = –(1/k)*trace
        const bk = sclr.neg(sclr.div(tr, sclr.fromReal(T, @floatFromInt(k))));
        coeffs[k] = bk;

        // Bprev = AB + bk·I

        // copy AB back into Bprev
        try copyMat(AB, Bprev);
        // Add bk to diagonal (square by the check above)
        for (0..Bprev.rows) |i| {
            Bprev.setUnsafe(i, i, sclr.add(bk, Bprev.atUnsafe(i, i)));
        }
    }
}

/// Solves the Pade linear system Q * R = P for R.
///
/// Where Q = (V - U) and P = (V + U).
///
/// The inverse of R is not computed explicitly. The system is solved
/// via Gauss-Jordan for numerical stability, as explicit inverses CAN
/// introduce large errors.
fn solvePadeSystem(alloc: std.mem.Allocator, Q: anytype, P: @TypeOf(Q)) !Matrix(ElementOf(@TypeOf(Q))) {
    const T = ElementOf(@TypeOf(Q));
    const n = Q.rows; // square

    var rhs = try vec.Vector(T).initZero(alloc, n, true);
    defer rhs.deinit();
    var retMat = try Matrix(T).initZero(alloc, n, n);
    errdefer retMat.deinit();

    // Build RHS
    for (0..n) |col| {
        for (0..n) |i| {
            try rhs.set(i, try P.at(i, col));
        }
        // Solve Q * x = RHS
        var res = try gj.gaussJordan(alloc, Q, rhs);
        try retMat.setCol(col, res.data);
        res.deinit();
    }

    return retMat;
}

/// Computes the Pade [13/13] numerator 'U' and denominator 'V'.
///
/// 'As' is the scaled version of 'A'.
fn pade13(alloc: std.mem.Allocator, As: anytype, A2: @TypeOf(As), A4: @TypeOf(As), A6: @TypeOf(As), U: @TypeOf(As), V: @TypeOf(As)) !void {
    const T = ElementOf(@TypeOf(As));

    var A8 = try matMult(alloc, A4, A4);
    defer A8.deinit();
    var A10 = try matMult(alloc, A2, A8);
    defer A10.deinit();
    var A12 = try matMult(alloc, A2, A10);
    defer A12.deinit();

    const PADE_13_COEFF: [14]f64 = [_]f64{ 64764752532480000, 32382376266240000, 7771770303897600, 1187353796428800, 129060195264000, 10559470521600, 670442572800, 33522128640, 1323241920, 40840800, 960960, 16380, 182, 1 };
    // Coefficients converted to T at comptime (@floatCast narrows for f32).
    const b = comptime blk: {
        var out: [14]T = undefined;
        for (PADE_13_COEFF, 0..) |coeff, k| out[k] = sclr.fromReal(T, @floatCast(coeff));
        break :blk out;
    };
    const n = As.rows;

    var temp = try Matrix(T).initZero(alloc, n, n);
    defer temp.deinit();

    for (0..n) |i| for (0..n) |j| {
        // V gets the even coefficients, temp (-> U) the odd ones:
        // valV = b0*I + b2*A2 + b4*A4 + ... + b12*A12
        // valT = b1*I + b3*A2 + b5*A4 + ... + b13*A12
        var valV = if (i == j) b[0] else sclr.zero(T);
        var valT = if (i == j) b[1] else sclr.zero(T);
        inline for (.{ A2, A4, A6, A8, A10, A12 }, 0..) |Ak, k| {
            const a = try Ak.at(i, j);
            valV = sclr.add(valV, sclr.mul(b[2 * k + 2], a));
            valT = sclr.add(valT, sclr.mul(b[2 * k + 3], a));
        }
        try V.set(i, j, valV);
        try temp.set(i, j, valT);
    };
    var temp2 = try matMult(alloc, As, temp);
    defer temp2.deinit();
    try copyMat(temp2, U);
}

// TODO: Improve this function. Make general.
/// Recursively determines the determinant for larger than 3x3 matrices.
///
/// WARNING: Because of recursive nature and therefore allocation / deallocation,
/// use CAUTION when used on large matrices! This will be a factor for 5x5 matrices or larger,
/// because 1 iteration over a 5x5 matrix will spawn 1 4x4 child,
/// which will in turn spawn 4 3x3 children. The lowest children
/// are only alloced one at a time.
///
/// The Matrix must be square.
pub fn determinant(alloc: std.mem.Allocator, mat: anytype) (MatError || std.mem.Allocator.Error)!ElementOf(@TypeOf(mat)) {
    const T = ElementOf(@TypeOf(mat));
    if (mat.rows != mat.cols) return MatError.BadShape;
    const n = mat.rows;
    if (n == 0) return sclr.one(T);
    if (n == 1) return try mat.at(0, 0);
    if (n == 2) {
        const a = try mat.at(0, 0);
        const d = try mat.at(1, 1);

        const b = try mat.at(0, 1);
        const c = try mat.at(1, 0);
        // a*d - b*c
        return sclr.sub(sclr.mul(a, d), sclr.mul(b, c));
    }
    if (n == 3) {
        const a = try mat.at(0, 0);
        const e = try mat.at(1, 1);
        const i = try mat.at(2, 2);

        const b = try mat.at(0, 1);
        const f = try mat.at(1, 2);
        const g = try mat.at(2, 0);

        const c = try mat.at(0, 2);
        const d = try mat.at(1, 0);
        const h = try mat.at(2, 1);
        // aei + bfg + cdh - ceg - bdi - afh
        const pos = sclr.add(sclr.add(sclr.mul(sclr.mul(a, e), i), sclr.mul(sclr.mul(b, f), g)), sclr.mul(sclr.mul(c, d), h));
        const negs = sclr.add(sclr.add(sclr.mul(sclr.mul(c, e), g), sclr.mul(sclr.mul(b, d), i)), sclr.mul(sclr.mul(a, f), h));
        return sclr.sub(pos, negs);
    }
    const expand_row: usize = 0;

    var sum = sclr.zero(T);

    for (0..n) |col| {
        const a = try mat.at(expand_row, col);
        if (sclr.isZeroApprox(a, 1e-8)) continue;
        // sign = (-1)^(row + col)
        const sign: T = if (((expand_row + col) & 1) == 0) sclr.one(T) else sclr.neg(sclr.one(T));
        var minor = try makeMinor(alloc, mat, expand_row, col);

        const det_minor = try determinant(alloc, minor);

        minor.deinit(); // no longer needed.
        sum = sclr.add(sum, sclr.mul(sclr.mul(sign, a), det_minor));
    }
    return sum;
}

/// Makes a copy of the matrix, but skips 1 row and 1 coloumn.
fn makeMinor(alloc: std.mem.Allocator, mat: anytype, skip_row: usize, skip_col: usize) !@TypeOf(mat) {
    const n = mat.rows;
    var m = try @TypeOf(mat).initZero(alloc, n - 1, n - 1);

    var rr: usize = 0;
    for (0..n) |r| {
        if (r == skip_row) continue;

        var cc: usize = 0;
        for (0..n) |c| {
            if (c == skip_col) continue;

            try m.set(rr, cc, try mat.at(r, c));

            cc += 1;
        }
        rr += 1;
    }
    return m;
}

/// Determines if a matrix is lower triangular.
///
/// 'tolerance' is the strict limit
/// a value can be before being considered non-zero.
///
/// I.e 'tolerance = 1e-8' -> Everything below in absolute
/// value considered zero.
///
/// If the absolute difference between the matrix value
/// and the tolerance is less than 1e-10, writes a console
/// warning.
pub fn isLowerTriangular(A: anytype, tolerance: sclr.Real(ElementOf(@TypeOf(A)))) bool {
    for (0..A.rows) |r| {
        for (r + 1..A.cols) |c| {
            // In-bounds by loop construction.
            if (sclr.abs(A.atUnsafe(r, c)) > tolerance) {
                if (@abs(sclr.abs(A.atUnsafe(r, c)) - tolerance) <= 1e-10) {
                    std.log.warn("isLowerTriangular| Difference between 'tolerance' and matrix values is <= 1e-10. \n", .{});
                }
                return false;
            }
        }
    }
    return true;
}

/// Determines if a matrix is upper triangular.
///
/// 'tolerance' is the strict limit
/// a value can be before being considered non-zero.
///
/// I.e 'tolerance = 1e-8' -> Everything below in absolute
/// value considered zero.
///
/// If the absolute difference between the matrix value
/// and the tolerance is less than 1e-10, writes a console
/// warning.
pub fn isUpperTriangular(A: anytype, tolerance: sclr.Real(ElementOf(@TypeOf(A)))) bool {
    for (0..A.rows) |r| {
        for (0..r) |c| {
            // In-bounds by loop construction.
            if (sclr.abs(A.atUnsafe(r, c)) > tolerance) {
                if (@abs(sclr.abs(A.atUnsafe(r, c)) - tolerance) <= 1e-10) {
                    std.log.warn("isUpperTriangular| Difference between 'tolerance' and matrix values is <= 1e-10. \n", .{});
                }
                return false;
            }
        }
    }
    return true;
}

test "Matrix(Complex): smoke test of generic paths" {
    const alloc = std.testing.allocator;
    const Cx = std.math.Complex(f64);

    // Struct methods
    var A = try CMat.initIdentity(alloc, 2, 2);
    defer A.deinit();
    try A.set(0, 0, Cx.init(0, 1)); // i
    try A.set(1, 1, Cx.init(2, 0));

    var B = try A.add(A);
    defer B.deinit();
    try std.testing.expect(sclr.eql(try B.at(0, 0), Cx.init(0, 2)));

    const tr = try A.trace();
    try std.testing.expect(sclr.approxEq(tr, Cx.init(2, 1), 1e-12));
    try std.testing.expectApproxEqAbs(@as(f64, 2), A.norm1(), 1e-12);

    // matMult: A * A = [[i^2, 0], [0, 4]] = [[-1, 0], [0, 4]]
    var AA = try matMult(alloc, A, A);
    defer AA.deinit();
    try std.testing.expect(sclr.approxEq(try AA.at(0, 0), Cx.init(-1, 0), 1e-12));
    try std.testing.expect(sclr.approxEq(try AA.at(1, 1), Cx.init(4, 0), 1e-12));

    // matMultSIMD falls back to matMult for Complex
    var AA2 = try matMultSIMD(alloc, A, A);
    defer AA2.deinit();
    try std.testing.expect(sclr.approxEq(try AA2.at(0, 0), Cx.init(-1, 0), 1e-12));

    // inverse: A * A^-1 = I
    var Ainv = try inverse(alloc, A);
    defer Ainv.deinit();
    var ident = try matMult(alloc, A, Ainv);
    defer ident.deinit();
    try std.testing.expect(sclr.approxEq(try ident.at(0, 0), Cx.init(1, 0), 1e-12));
    try std.testing.expect(sclr.approxEq(try ident.at(0, 1), Cx.init(0, 0), 1e-12));
    try std.testing.expect(sclr.approxEq(try ident.at(1, 1), Cx.init(1, 0), 1e-12));

    // determinant: det(A) = i * 2 = 2i
    const det = try determinant(alloc, A);
    try std.testing.expect(sclr.approxEq(det, Cx.init(0, 2), 1e-12));

    // charPoly: det(zI - A) = z^2 - (2+i)z + 2i -> coeffs [1, -(2+i), 2i]
    var coeffs: [3]Cx = undefined;
    try charPoly(alloc, A, &coeffs);
    try std.testing.expect(sclr.approxEq(coeffs[1], Cx.init(-2, -1), 1e-12));
    try std.testing.expect(sclr.approxEq(coeffs[2], Cx.init(0, 2), 1e-12));

    // transpose + triangular checks
    var At = try transpose(A, alloc);
    defer At.deinit();
    try std.testing.expect(isUpperTriangular(At, 1e-12));
    try std.testing.expect(isLowerTriangular(At, 1e-12));

    // 4x4 determinant exercises makeMinor recursion
    var D = try CMat.initIdentity(alloc, 4, 4);
    defer D.deinit();
    try D.set(3, 3, Cx.init(0, 1));
    const det4 = try determinant(alloc, D);
    try std.testing.expect(sclr.approxEq(det4, Cx.init(0, 1), 1e-12));

    // getRow / getCol return Vector(Cx). A = [[i, 0], [0, 2]]
    var r0 = try A.getRow(0, alloc);
    defer r0.deinit();
    try std.testing.expect(!r0.colvec);
    try std.testing.expect(sclr.approxEq(r0.atUnsafe(0), Cx.init(0, 1), 1e-12));
    try std.testing.expect(sclr.approxEq(r0.atUnsafe(1), Cx.init(0, 0), 1e-12));

    var c1 = try A.getCol(1, alloc);
    defer c1.deinit();
    try std.testing.expect(c1.colvec);
    try std.testing.expect(sclr.approxEq(c1.atUnsafe(0), Cx.init(0, 0), 1e-12));
    try std.testing.expect(sclr.approxEq(c1.atUnsafe(1), Cx.init(2, 0), 1e-12));
}

test "Matrix(f32): getRow / getCol" {
    const alloc = std.testing.allocator;
    var A = try Matrix(f32).initIdentity(alloc, 2, 2);
    defer A.deinit();

    var r = try A.getRow(1, alloc);
    defer r.deinit();
    try std.testing.expect(!r.colvec);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), r.atUnsafe(0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), r.atUnsafe(1), 1e-6);

    var c = try A.getCol(0, alloc);
    defer c.deinit();
    try std.testing.expect(c.colvec);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), c.atUnsafe(0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), c.atUnsafe(1), 1e-6);
}

test "Matrix: Test" {
    const alloc = std.testing.allocator;
    var m1 = try Mat.initZero(alloc, 2, 2);
    defer m1.deinit();
    try m1.set(0, 0, 1);
    const z = try m1.at(0, 0);
    try std.testing.expect(z == 1);

    var m2 = try Mat.initZero(alloc, 2, 2);
    defer m2.deinit();
    try m2.set(0, 0, 1);

    var m3 = try matMult(alloc, m1, m2);
    defer m3.deinit();
    const a1 = try m3.at(0, 0);
    try std.testing.expect(a1 == 1);

    try m3.expand(3, 3, 0.0);
    try std.testing.expectError(MatError.Empty, m3.expand(0, 0, 0));

    var m4 = try Mat.initZero(alloc, 3, 3);
    defer m4.deinit();
    const float: f64 = 2.0;
    const floatSliceCompTime = [_]f64{ 3.0, 4.0, 5.0 };
    const floatSlice: []f64 = try alloc.alloc(f64, 3);
    defer alloc.free(floatSlice);
    floatSlice[0] = 5.0;
    floatSlice[1] = 6.0;
    floatSlice[2] = 7.0;
    try m4.setRow(0, float);
    try m4.setRow(1, floatSliceCompTime);
    try m4.setRow(2, floatSlice);

    m4.multAll(2.0);

    try m4.swapRow(0, 1);

    // Testing: Inverse
    var m5 = try Mat.initZero(alloc, 3, 3);
    defer m5.deinit();
    try m5.setRow(0, [_]f64{ 2.0, -1.0, 0.0 });
    try m5.setRow(1, [_]f64{ -1.0, 2.0, -1.0 });
    try m5.setRow(2, [_]f64{ 0.0, -1.0, 2.0 });
    var m5Inv = try inverse(alloc, m5);
    defer m5Inv.deinit();
    var id_m5 = try matMult(alloc, m5, m5Inv);
    defer id_m5.deinit();

    try std.testing.expectApproxEqAbs(1.0, try id_m5.at(0, 0), 1e-8);
    try std.testing.expectApproxEqAbs(0.0, try id_m5.at(0, 1), 1e-8);
    try std.testing.expectApproxEqAbs(0.0, try id_m5.at(0, 2), 1e-8);
    try std.testing.expectApproxEqAbs(0.0, try id_m5.at(1, 0), 1e-8);
    try std.testing.expectApproxEqAbs(1.0, try id_m5.at(1, 1), 1e-8);
    try std.testing.expectApproxEqAbs(0.0, try id_m5.at(1, 2), 1e-8);
    try std.testing.expectApproxEqAbs(0.0, try id_m5.at(2, 0), 1e-8);
    try std.testing.expectApproxEqAbs(0.0, try id_m5.at(2, 1), 1e-8);
    try std.testing.expectApproxEqAbs(1.0, try id_m5.at(2, 2), 1e-8);

    var det_m5 = try determinant(alloc, m5);
    try std.testing.expectApproxEqAbs(4.0, det_m5, 1e-8);

    try m5.expand(m5.rows + 1, m5.rows + 1, 0.0);
    det_m5 = try determinant(alloc, m5);
    try std.testing.expectApproxEqAbs(0.0, det_m5, 1e-8);

    try m5.set(3, 3, 2);
    det_m5 = try determinant(alloc, m5);
    try std.testing.expectApproxEqAbs(8.0, det_m5, 1e-8);

    var det_m1 = try determinant(alloc, m1);
    try std.testing.expectApproxEqAbs(0.0, det_m1, 1e-8);

    try m1.set(1, 1, 1.0);
    det_m1 = try determinant(alloc, m1);
    try std.testing.expectApproxEqAbs(1.0, det_m1, 1e-8);

    // Exmp

    var A = try Mat.initZero(alloc, 2, 2);
    defer A.deinit();

    try A.set(0, 1, 1.0);

    var Aexpm = try expm(alloc, A);
    defer Aexpm.deinit();
    try std.testing.expectApproxEqAbs(1.0, try Aexpm.at(0, 0), 1e-8);
    try std.testing.expectApproxEqAbs(1.0, try Aexpm.at(0, 1), 1e-8);
    try std.testing.expectApproxEqAbs(0.0, try Aexpm.at(1, 0), 1e-8);
    try std.testing.expectApproxEqAbs(1.0, try Aexpm.at(1, 1), 1e-8);

    var A_skew = try Mat.initZero(alloc, 4, 4);
    try A_skew.set(0, 1, -1.0);
    try A_skew.set(1, 0, 1.0);
    try A_skew.set(2, 3, -3.0);
    try A_skew.set(3, 2, 3.0);
    defer A_skew.deinit();

    var A_skewExpm = try expm(alloc, A_skew);
    defer A_skewExpm.deinit();
    try std.testing.expectApproxEqAbs(@cos(1.0), try A_skewExpm.at(0, 0), 1e-8);
    try std.testing.expectApproxEqAbs(-@sin(1.0), try A_skewExpm.at(0, 1), 1e-8);
    try std.testing.expectApproxEqAbs(@sin(1.0), try A_skewExpm.at(1, 0), 1e-8);
    try std.testing.expectApproxEqAbs(@cos(1.0), try A_skewExpm.at(1, 1), 1e-8);
    try std.testing.expectApproxEqAbs(@cos(3.0), try A_skewExpm.at(2, 2), 1e-8);
    try std.testing.expectApproxEqAbs(-@sin(3.0), try A_skewExpm.at(2, 3), 1e-8);
    try std.testing.expectApproxEqAbs(@sin(3.0), try A_skewExpm.at(3, 2), 1e-8);
    try std.testing.expectApproxEqAbs(@cos(3.0), try A_skewExpm.at(3, 3), 1e-8);
}

test "expm: Complex diagonal (Euler), scaling+squaring path" {
    const alloc = std.testing.allocator;
    const Cx = std.math.Complex(f64);
    const tol: f64 = 1e-8;

    // A = diag(i*pi, 6i) -> expm(A) = diag(exp(i*pi), exp(6i)) = diag(-1, cos6 + i*sin6).
    // norm1 = 6 > THETA_13, so the scaling + repeated squaring branch fires.
    var A = try CMat.initZero(alloc, 2, 2);
    defer A.deinit();
    try A.set(0, 0, Cx.init(0.0, std.math.pi));
    try A.set(1, 1, Cx.init(0.0, 6.0));

    var eA = try expm(alloc, A);
    defer eA.deinit();

    try std.testing.expect(sclr.approxEq(try eA.at(0, 0), Cx.init(-1.0, 0.0), tol));
    try std.testing.expect(sclr.approxEq(try eA.at(1, 1), Cx.init(@cos(6.0), @sin(6.0)), tol));
    try std.testing.expect(sclr.approxEq(try eA.at(0, 1), Cx.init(0.0, 0.0), tol));
    try std.testing.expect(sclr.approxEq(try eA.at(1, 0), Cx.init(0.0, 0.0), tol));
}

test "expm: f64 rotation with scaling+squaring (norm > THETA_13)" {
    const alloc = std.testing.allocator;
    const tol: f64 = 1e-8;
    const theta: f64 = 7.0; // norm1 = 7 > THETA_13

    var A = try Mat.initZero(alloc, 2, 2);
    defer A.deinit();
    try A.set(0, 1, -theta);
    try A.set(1, 0, theta);

    var eA = try expm(alloc, A);
    defer eA.deinit();

    try std.testing.expectApproxEqAbs(@cos(theta), try eA.at(0, 0), tol);
    try std.testing.expectApproxEqAbs(-@sin(theta), try eA.at(0, 1), tol);
    try std.testing.expectApproxEqAbs(@sin(theta), try eA.at(1, 0), tol);
    try std.testing.expectApproxEqAbs(@cos(theta), try eA.at(1, 1), tol);
}

test "expm: f32 rotation" {
    const alloc = std.testing.allocator;
    const tol: f32 = 1e-4; // Pade coefficients are rounded in f32

    var A = try Matrix(f32).initZero(alloc, 2, 2);
    defer A.deinit();
    try A.set(0, 1, -1.0);
    try A.set(1, 0, 1.0);

    var eA = try expm(alloc, A);
    defer eA.deinit();

    try std.testing.expectApproxEqAbs(@cos(@as(f32, 1.0)), try eA.at(0, 0), tol);
    try std.testing.expectApproxEqAbs(-@sin(@as(f32, 1.0)), try eA.at(0, 1), tol);
    try std.testing.expectApproxEqAbs(@sin(@as(f32, 1.0)), try eA.at(1, 0), tol);
    try std.testing.expectApproxEqAbs(@cos(@as(f32, 1.0)), try eA.at(1, 1), tol);
}

test "inverse: singular matrix -> Singular, non-square -> BadShape" {
    const alloc = std.testing.allocator;

    var S = try Mat.initZero(alloc, 2, 2);
    defer S.deinit();
    try S.setRow(0, [_]f64{ 1.0, 2.0 });
    try S.setRow(1, [_]f64{ 2.0, 4.0 });
    try std.testing.expectError(InverseError.Singular, inverse(alloc, S));

    var R = try Mat.initZero(alloc, 2, 3);
    defer R.deinit();
    try std.testing.expectError(InverseError.BadShape, inverse(alloc, R));
}

test "inverse: zero on diagonal is handled by pivoting" {
    const alloc = std.testing.allocator;

    // Permutation matrix: invertible, but has a 0 pivot without row swaps.
    var P = try Mat.initZero(alloc, 2, 2);
    defer P.deinit();
    try P.setRow(0, [_]f64{ 0.0, 1.0 });
    try P.setRow(1, [_]f64{ 1.0, 0.0 });

    var Pinv = try inverse(alloc, P);
    defer Pinv.deinit();

    // P is its own inverse.
    for (0..2) |r| {
        for (0..2) |c| {
            try std.testing.expectApproxEqAbs(try P.at(r, c), try Pinv.at(r, c), 1e-12);
        }
    }

    // 1x1 zero matrix: normalization guard must catch it (no pivot loop runs).
    var Z = try Mat.initZero(alloc, 1, 1);
    defer Z.deinit();
    try std.testing.expectError(InverseError.Singular, inverse(alloc, Z));
}

test "Mat.init / initZero / initIdentity: OOM does not leak" {
    for (0..80) |fail_index| {
        var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const alloc = fa.allocator();

        const r1 = Mat.init(alloc, 4, 4);
        if (r1) |m_val| {
            var m = m_val;
            m.deinit();
        } else |e| {
            try std.testing.expect(e == error.OutOfMemory);
        }

        const r2 = Mat.initZero(alloc, 4, 4);
        if (r2) |m_val| {
            var m = m_val;
            m.deinit();
        } else |e| {
            try std.testing.expect(e == error.OutOfMemory);
        }

        const r3 = Mat.initIdentity(alloc, 4, 4);
        if (r3) |m_val| {
            var m = m_val;
            m.deinit();
        } else |e| {
            try std.testing.expect(e == error.OutOfMemory);
        }
    }
}

test "transpose: OOM at various allocation points does not leak" {
    for (0..120) |fail_index| {
        var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const alloc = fa.allocator();

        var A = Mat.initZero(alloc, 3, 5) catch |e| {
            try std.testing.expect(e == error.OutOfMemory);
            continue;
        };
        defer A.deinit();

        const res = transpose(A, alloc);
        if (res) |t_val| {
            var t = t_val;
            t.deinit();
        } else |e| {
            try std.testing.expect(e == error.OutOfMemory);
        }
    }
}

test "matMult: OOM at various allocation points does not leak" {
    for (0..200) |fail_index| {
        var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const alloc = fa.allocator();

        var A = Mat.initZero(alloc, 4, 4) catch |e| {
            try std.testing.expect(e == error.OutOfMemory);
            continue;
        };
        defer A.deinit();

        var B = Mat.initZero(alloc, 4, 4) catch |e| {
            try std.testing.expect(e == error.OutOfMemory);
            continue;
        };
        defer B.deinit();

        const res = matMult(alloc, A, B);
        if (res) |m_val| {
            var m = m_val;
            m.deinit();
        } else |e| {
            try std.testing.expect(e == error.OutOfMemory or e == MatError.SizeMismatch);
        }
    }
}

test "inverse: OOM at various allocation points does not leak" {
    for (0..400) |fail_index| {
        var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const alloc = fa.allocator();

        var A = Mat.initZero(alloc, 3, 3) catch |e| {
            try std.testing.expect(e == error.OutOfMemory);
            continue;
        };
        defer A.deinit();

        // Make it invertible-ish
        try A.setRow(0, [_]f64{ 2.0, -1.0, 0.0 });
        try A.setRow(1, [_]f64{ -1.0, 2.0, -1.0 });
        try A.setRow(2, [_]f64{ 0.0, -1.0, 2.0 });

        const res = inverse(alloc, A);
        if (res) |inv_val| {
            var inv = inv_val;
            inv.deinit();
        } else |e| {
            try std.testing.expect(e == error.OutOfMemory or e == MatError.BadShape);
        }
    }
}

test "charPoly: OOM at various allocation points does not leak" {
    for (0..400) |fail_index| {
        var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const alloc = fa.allocator();

        var A = Mat.initZero(alloc, 3, 3) catch |e| {
            try std.testing.expect(e == error.OutOfMemory);
            continue;
        };
        defer A.deinit();

        // something non-trivial
        try A.set(0, 0, 1);
        try A.set(0, 1, 2);
        try A.set(0, 2, 3);
        try A.set(1, 0, 0);
        try A.set(1, 1, 4);
        try A.set(1, 2, 5);
        try A.set(2, 0, 0);
        try A.set(2, 1, 0);
        try A.set(2, 2, 6);

        const coeffs = alloc.alloc(f64, A.rows + 1) catch |e| {
            try std.testing.expect(e == error.OutOfMemory);
            continue;
        };
        defer alloc.free(coeffs);

        const r = charPoly(alloc, A, coeffs);
        if (r) |_| {} else |e| {
            try std.testing.expect(e == error.OutOfMemory or e == MatError.BadShape or e == MatError.SizeMismatch);
        }
    }
}

test "determinant (n>3): OOM at various allocation points does not leak" {
    for (0..500) |fail_index| {
        var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const alloc = fa.allocator();

        var A = Mat.initZero(alloc, 4, 4) catch |e| {
            try std.testing.expect(e == error.OutOfMemory);
            continue;
        };
        defer A.deinit();

        // Simple upper-tri-ish so determinant isn't too wild
        try A.set(0, 0, 2);
        try A.set(1, 1, 3);
        try A.set(2, 2, 4);
        try A.set(3, 3, 5);
        try A.set(0, 1, 1);
        try A.set(1, 2, 1);
        try A.set(2, 3, 1);

        const r = determinant(alloc, A);
        if (r) |_| {} else |e| {
            try std.testing.expect(e == error.OutOfMemory or e == MatError.BadShape);
        }
    }
}

test "Mat.expand: OOM at various allocation points does not leak" {
    for (0..400) |fail_index| {
        var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const alloc = fa.allocator();

        var A = Mat.initZero(alloc, 2, 2) catch |e| {
            try std.testing.expect(e == error.OutOfMemory);
            continue;
        };
        defer A.deinit();

        // Give some random values
        try A.set(0, 0, 1.1);
        try A.set(1, 1, 2.1);

        const r = A.expand(10, 10, 0.0);
        if (r) |_| {
            try std.testing.expect(A.rows == 10);
            try std.testing.expect(A.cols == 10);
        } else |e| {
            try std.testing.expect(e == error.OutOfMemory or e == MatError.Empty);
            // A might be partially expanded but we only care about no leak
        }
    }
}

test "expm: OOM at various allocation points does not leak" {
    for (0..1200) |fail_index| {
        var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const alloc = fa.allocator();

        var A = Mat.initZero(alloc, 2, 2) catch |e| {
            try std.testing.expect(e == error.OutOfMemory);
            continue;
        };
        defer A.deinit();

        // Simple nilpotent-ish matrix
        try A.set(0, 1, 1.0);

        const res = expm(alloc, A);
        if (res) |E_val| {
            var E = E_val;
            E.deinit();
        } else |e| {
            try std.testing.expect(e == error.OutOfMemory or e == MatError.BadShape or e == MatError.SizeMismatch);
        }
    }
}
