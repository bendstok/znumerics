const std = @import("std");
const vec = @import("vec.zig");
const gj = @import("../linalg/gaussjordan.zig");
const err_mod = @import("../error.zig");

const Vec = vec.Vec;

pub const MatError = error{
    Empty,
} || err_mod.Common;

pub const InverseError = error{
    Singular,
} || err_mod.Common;

/// The Matrix type.
/// The data is stored as row vectors.
///
/// Stored on the heap.
pub const Mat = struct {
    alloc: std.mem.Allocator,
    rows: usize,
    cols: usize,
    data: []Vec,

    pub fn init(alloc: std.mem.Allocator, rows: usize, cols: usize) !Mat {
        if (rows == 0 or cols == 0) return MatError.Empty;
        const data = try alloc.alloc(Vec, rows);

        var inited: usize = 0;
        // We only free data we actually got to
        // initialize
        errdefer {
            for (0..inited) |r| data[r].deinit();
            alloc.free(data);
        }

        for (0..data.len) |r| {
            data[r] = try Vec.init(alloc, cols, false);
            inited += 1;
        }
        return .{ .alloc = alloc, .rows = rows, .cols = cols, .data = @constCast(data) };
    }

    pub fn initZero(alloc: std.mem.Allocator, rows: usize, cols: usize) !Mat {
        if (rows == 0 or cols == 0) return MatError.Empty;
        const data = try alloc.alloc(Vec, rows);

        var inited: usize = 0;
        // We only free data we actually got to
        // initialize
        errdefer {
            for (0..inited) |r| data[r].deinit();
            alloc.free(data);
        }
        for (0..data.len) |r| {
            data[r] = try Vec.initZero(alloc, cols, false);
            inited += 1;
        }
        return .{ .alloc = alloc, .rows = rows, .cols = cols, .data = @constCast(data) };
    }

    pub fn initIdentity(alloc: std.mem.Allocator, rows: usize, cols: usize) !Mat {
        if (rows == 0 or cols == 0) return MatError.Empty;
        const data = try alloc.alloc(Vec, rows);
        var inited: usize = 0;
        // We only free data we actually got to
        // initialize
        errdefer {
            for (0..inited) |r| data[r].deinit();
            alloc.free(data);
        }
        for (0..data.len) |r| {
            data[r] = try Vec.initZero(alloc, cols, false);
            inited += 1;
        }
        // Make it an identity matrix
        for (0..data.len) |i| {
            for (0..data[i].len()) |j| {
                // We can use unsafe here because we will never go outside bounds
                if (i == j) data[i].setUnsafe(j, 1.0);
            }
        }
        return .{ .alloc = alloc, .rows = rows, .cols = cols, .data = @constCast(data) };
    }

    pub fn deinit(self: *Mat) void {
        for (0..self.rows) |r| {
            if (self.data[r].len() != 0) {
                self.data[r].deinit();
            }
        }
        self.alloc.free(self.data);
        self.* = undefined;
    }

    /// Returns the value at the corresponding location.
    ///
    /// Does bounds checking. Returns a VecError/MatError.IndexOutOfBounds on failure.
    pub fn at(self: Mat, row: usize, col: usize) !f64 {
        try boundsCheck(self, row, col);
        return try self.data[row].at(col);
    }

    /// Synonymys with .at().
    pub fn get(self: Mat, row: usize, col: usize) !f64 {
        return (try at(self, row, col));
    }

    /// Unsafe version of .at()
    ///
    /// No Boundary checks.
    pub fn atUnsafe(self: Mat, row: usize, col: usize) f64 {
        return self.data[row].atUnsafe(col);
    }

    /// Sets the location to the value.
    ///
    /// Does bounds checking. Returns a VecError/MatError.IndexOutOfBounds on failure.
    pub fn set(self: Mat, row: usize, col: usize, val: f64) !void {
        try boundsCheck(self, row, col);
        try self.data[row].set(col, val);
    }

    /// Sets the location to the value.
    ///
    /// Does no bounds checking, unsafe. See .set().
    pub fn setUnsafe(self: Mat, row: usize, col: usize, val: f64) void {
        self.data[row].setUnsafe(col, val);
    }

    /// Sets all the values in the matrix.
    pub fn setAll(self: Mat, val: f64) void {
        for (0..self.rows) |r| {
            self.data[r].setAll(val);
        }
    }

    /// Does length checks on 'new_values' before inserting.
    ///
    /// Returns a VecError.SizeMismatch on failure.
    pub fn setRow(self: Mat, row: usize, new_values: anytype) !void {
        try self.data[row].setAll(new_values);
    }

    /// Sets all values in a coloumn to the values in 'new_values'.
    ///
    /// Checks underlying type of pointers and arrays and does bound checks.
    /// Returns either a VecError or a MatError on failure.
    ///
    /// 'cons' must be of type: f64, comptime_float, [_]f64 or []f64.
    pub fn setCol(self: Mat, col: usize, new_values: anytype) !void {
        const T = @TypeOf(new_values);

        switch (@typeInfo(T)) {
            .float => {
                if (T != f64) @compileError("MAT| setCol .float: only f64 is supported, got: " ++ @typeName(T) ++ "\n");
                for (0..self.rows) |r| {
                    try self.setSafe(r, col, new_values);
                }
            },
            .array => |arr| {
                if (arr.child != f64) @compileError("MAT| setCol .array: only []f64 supported, got: " ++ @typeName(T) ++ "\n");
                if (new_values.len != self.rows) return MatError.SizeMismatch;
                for (0..self.rows) |r| {
                    for (0..new_values.len) |i| {
                        try self.setSafe(r, col, new_values[i]);
                    }
                }
            },
            .pointer => |p| {
                if (p.child != f64) @compileError("MAT| setCol .pointer: only f64 children are allowed, got: " ++ @typeName(p.child) ++ "\n");
                if (p.size != .slice) @compileError("MAT| setCol .pointer: Pointer must be of a Zig slice. \n");
                if (new_values.len != self.rows) return MatError.SizeMismatch;
                for (0..self.rows) |r| {
                    try self.set(r, col, new_values[r]);
                }
            },
            .comptime_float => {
                for (0..self.rows) |r| {
                    try self.set(r, col, new_values);
                }
            },
            else => {
                @compileError("MAT| setCol: Expected f64, comptime_float, []f64 or [_]f64, got: " ++ @typeName(T) ++ "\n");
            },
        }
    }

    /// Prints a view of the matrix to std.debug.print
    ///
    /// Can fail because of bufPrint.
    pub fn printMat(self: Mat) !void {
        for (0..self.rows) |r| {
            //std.debug.print("r={}: ", .{r});
            for (0..self.cols) |c| {
                //std.debug.print("c={}", .{c});
                const val: f64 = try self.at(r, c);
                var tmp: [64]u8 = undefined;
                const s = try std.fmt.bufPrint(&tmp, "{:.3}", .{val});
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
    pub fn transposeInPlace(self: Mat) (MatError || err_mod.Common)!void {
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
    pub fn expand(self: *Mat, rows: usize, cols: usize, fill: f64) !void {
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
        var newMat = try Mat.initZero(self.alloc, new_rows, new_cols);
        errdefer newMat.deinit();

        // Copy old contents
        for (0..self.rows) |r| {
            for (0..self.cols) |c| {
                newMat.setUnsafe(r, c, self.atUnsafe(r, c));
            }
        }

        if (fill != 0) {
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

    /// Multiplies all values in a row by 'mult'
    pub fn multRow(self: Mat, row: usize, mult: f64) !void {
        try boundsCheck(self, row, 0);
        self.data[row].multConst(mult);
    }

    /// Multiplies all values in the matrix by 'mult'
    pub fn multAll(self: Mat, mult: f64) void {
        for (0..self.rows) |r| {
            // We can use unsafe here since we know
            // we wont go out of bounds.
            self.data[r].multConstUnsafe(mult);
        }
    }

    /// Swaps two rows.
    ///
    /// Does boundary checks. Returns an MatError.IndexOutOfBounds on failure.
    pub fn swapRow(self: Mat, row1: usize, row2: usize) !void {
        if (row1 >= self.rows or row2 >= self.rows) return MatError.IndexOutOfBounds;
        const temp = self.data[row1];
        self.data[row1] = self.data[row2];
        self.data[row2] = temp;
    }

    /// Returns the Norm_1 (max coloumn sum) of the matrix
    pub fn norm1(self: Mat) !f64 {
        var max_sum: f64 = 0.0;
        for (0..self.cols) |c| {
            var col_sum: f64 = 0.0;
            for (0..self.rows) |r| {
                col_sum += try self.at(r, c);
            }
            if (col_sum > max_sum) max_sum = col_sum;
        }
        return max_sum;
    }

    /// Returns a matrix which is a deep copy of itself
    pub fn clone(self: Mat) !Mat {
        var retMat = try Mat.initZero(self.alloc, self.rows, self.cols);
        try copyMat(self, retMat);
        // TODO: Below is a dirty 'hack' to satisfy
        // retMat being a 'var', find a fix. @constCast ?
        try retMat.set(0, 0, try retMat.at(0, 0));
        return retMat;
    }

    /// Adds the two matrices index by index
    /// and returns the result. The Matrices must be same size.
    ///
    /// C = A + B, where A -> self and B -> toAdd.
    ///
    /// See .addInPlace() for no return.
    pub fn add(self: Mat, toAdd: Mat) !Mat {
        const r = self.rows;
        const c = self.cols;
        if (r != toAdd.rows or c != toAdd.cols) return MatError.SizeMismatch;
        var retMat = try Mat.initZero(self.alloc, r, c);
        for (0..r) |i| {
            for (0..c) |j| {
                try retMat.set(i, j, try self.at(i, j) + try toAdd.at(i, j));
            }
        }
        return retMat;
    }

    /// Adds 'toAdd' to self.
    ///
    /// A = A + B, where A -> self and B -> toAdd.
    ///
    /// See .add() to return a matrix with the result,
    /// leaving A & B unchanged.
    pub fn addInPlace(self: Mat, toAdd: Mat) !void {
        const r = self.rows;
        const c = self.cols;
        if (r != toAdd.rows or c != toAdd.cols) return MatError.SizeMismatch;
        for (0..r) |i| {
            for (0..c) |j| {
                try self.set(i, j, try self.at(i, j) + try toAdd.at(i, j));
            }
        }
    }

    /// Subtracts the two matrices index by index
    /// and returns the result. The Matrices must be same size.
    ///
    /// C = A - B, where A -> self and B is toSub.
    ///
    /// See .subInPlace() for no return.
    pub fn sub(self: Mat, toSub: Mat) !Mat {
        const r = self.rows;
        const c = self.cols;
        if (r != toSub.rows or c != toSub.cols) return MatError.SizeMismatch;
        var retMat = try Mat.initZero(self.alloc, r, c);
        for (0..r) |i| {
            for (0..c) |j| {
                try retMat.set(i, j, try self.at(i, j) - try toSub.at(i, j));
            }
        }
        return retMat;
    }

    /// Subtracts self by toSub.
    ///
    /// C = A - B, where A -> self and B is toSub.
    ///
    /// See .sub() for returning the result,
    /// leaving A & B unchanged.
    pub fn subInPlace(self: Mat, toSub: Mat) !void {
        const r = self.rows;
        const c = self.cols;
        if (r != toSub.rows or c != toSub.cols) return MatError.SizeMismatch;
        for (0..r) |i| {
            for (0..c) |j| {
                try self.set(i, j, try self.at(i, j) - try toSub.at(i, j));
            }
        }
    }

    /// Returns the trace of the matrix
    ///
    /// Must be square
    pub fn trace(self: Mat) !f64 {
        if (self.rows != self.cols) return MatError.BadShape;
        var tr: f64 = 0.0;
        for (0..self.rows) |i| {
            for (0..self.cols) |j| {
                if (i == j) tr += self.atUnsafe(i, j);
            }
        }
        return tr;
    }

    /// Check bounds. Throws an IndexOutOfBounds on failure.
    fn boundsCheck(self: Mat, row: usize, col: usize) MatError!void {
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

/// Deep copies the values from the 'from' matrix to the 'to' matrix
///
/// The recipient matrix must be >= the 'from' matrix.
pub fn copyMat(from: Mat, to: Mat) !void {
    if (to.cols < from.cols or to.rows < from.rows) return MatError.SizeMismatch;
    for (0..from.rows) |r| {
        for (0..from.cols) |c| {
            try to.set(r, c, try from.at(r, c));
        }
    }
}

/// Returns a new matrix that is the transpose of 'self'
///
/// Works for any matrix.
pub fn transpose(self: Mat, alloc: std.mem.Allocator) !Mat {
    var retMat = try Mat.initZero(alloc, self.cols, self.rows);
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
/// Might fail if it reaches a divide by zero when row reducing or backsolving.
/// Returns the matrix when this is hit, and writes a warning to console.
// TODO: Should we error here? ^^^
pub fn inverse(alloc: std.mem.Allocator, A: Mat) !Mat {
    var mat_mod = try Mat.initZero(alloc, A.rows, A.cols * 2);
    defer mat_mod.deinit();
    var ret_mat = try Mat.initZero(alloc, A.rows, A.cols);

    try copyMat(A, mat_mod);
    for (A.cols..A.cols * 2) |c| {
        for (0..A.rows) |r| {
            if (c - A.cols != r) continue;
            try mat_mod.set(r, c, 1.0);
        }
    }
    // Reduced row echelon, no pivoting
    for (0..mat_mod.rows - 1) |c| {
        // Row Reduce
        var i: usize = c + 1;
        while (i < mat_mod.rows) : (i += 1) {
            const denom = try mat_mod.at(c, c);
            if (denom == 0.00) {
                std.log.warn("Divide by zero found during row reduction in inverse().\nReturning empty matrix.\n", .{});
                return ret_mat;
            }

            const L = -(try mat_mod.at(i, c) / denom);
            for (c..mat_mod.cols) |col| {
                const new_val = try mat_mod.at(i, col) + try mat_mod.at(c, col) * L;
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
            if (denom == 0.00) {
                std.log.warn("Divide by zero found during backsolving in inverse().\nReturning empty matrix.\n", .{});
                return ret_mat;
            }

            const L = -(try mat_mod.at(i, c) / denom);
            for (c..mat_mod.cols) |col| {
                const new_val = try mat_mod.at(i, col) + try mat_mod.at(c, col) * L;
                try mat_mod.set(i, col, new_val);
            }
        }
        if (c == 0) break;
        c -= 1;
    }

    // Make left side the identity matrix
    for (0..mat_mod.rows) |r| {
        try mat_mod.multRow(r, 1 / (try mat_mod.at(r, r)));
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
pub fn matMult(alloc: std.mem.Allocator, left: Mat, right: Mat) !Mat {
    if (left.rows != right.cols) return MatError.SizeMismatch;
    var retMat: Mat = try Mat.initZero(alloc, left.rows, right.cols);
    // In case of error, we need to deinit the matrix
    errdefer retMat.deinit();

    for (0..left.rows) |r| {
        for (0..right.cols) |c| {
            // We now have a loop over all the values in the retMat
            // We work our way backwards.
            var val: f64 = 0;
            for (0..left.cols) |k| {
                val += (try left.at(r, k)) * (try right.at(k, c));
            }
            try retMat.set(r, c, val);
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
/// On Failure: returns either a MatError, VecError or OutOfMemory error.
pub fn expm(alloc: std.mem.Allocator, A: Mat) !Mat {
    // Taken from scipy.
    if (A.rows != A.cols) return MatError.BadShape;
    const THETA_13: f64 = 5.371920351148152;
    const norm = try A.norm1();

    const n = A.rows; // Square matrix
    var s: u32 = 0;
    if (norm > THETA_13) {
        s = @as(u32, @intFromFloat(@ceil(std.math.log2(norm / THETA_13))));
    }

    const scale = std.math.pow(f64, 2.0, @as(f64, @floatFromInt(s)));
    var As = try A.clone();
    defer As.deinit();

    As.multAll(1.0 / scale);

    // A2, A4, A6

    var A2 = try matMult(alloc, As, As);
    defer A2.deinit();
    var A4 = try matMult(alloc, A2, A2);
    defer A4.deinit();
    var A6 = try matMult(alloc, A4, A2);
    defer A6.deinit();

    var U = try Mat.initZero(alloc, n, n);
    defer U.deinit();
    var V = try Mat.initZero(alloc, n, n);
    defer V.deinit();

    try pade13(alloc, As, A2, A4, A6, U, V);

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
        const R2 = try matMult(alloc, R, R);
        R.deinit();
        R = R2;
    }

    return R;
}

/// Compute the characteristic polynomial of an n×n matrix A:
///   det(z·I - A) = z^n + coeffs[1]·z^(n-1) + … + coeffs[n]
/// and return it in “z⁻¹ form” as coeffs[0..n]:
///   [1, a₁, a₂, …, aₙ]
///
/// - coeffs: output slice of length ≥ n+1
pub fn charPoly(
    alloc: std.mem.Allocator,
    A: Mat,
    coeffs: []f64,
) !void {
    if (A.rows != A.cols) return MatError.BadShape;
    if (coeffs.len < A.rows + 1) return MatError.SizeMismatch;
    const n = A.rows;
    var Bprev = try Mat.initIdentity(alloc, n, n);
    defer Bprev.deinit();

    coeffs[0] = 1.0;

    // Iteratively compute bₖ and update B_prev
    for (1..n + 1) |k| {
        // AB = A * Bprev
        var AB = matMult(alloc, A, Bprev) catch |e| {
            return e;
        };
        defer AB.deinit();

        const tr = try AB.trace();

        // bₖ = –(1/k)*trace
        const bk = -tr / @as(f64, @floatFromInt(k));
        coeffs[k] = bk;

        // Bprev = AB + bk·I

        // copy AB back into Bprev
        try copyMat(AB, Bprev);
        // Add bk to diagonal
        for (0..Bprev.rows) |i| {
            for (0..Bprev.cols) |j| {
                if (i == j) Bprev.setUnsafe(i, j, bk + Bprev.atUnsafe(i, j));
            }
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
fn solvePadeSystem(alloc: std.mem.Allocator, Q: Mat, P: Mat) !Mat {
    const n = Q.rows; // square

    var rhs = try Vec.initZero(alloc, n, true);
    defer rhs.deinit();
    var retMat = try Mat.initZero(alloc, n, n);
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
fn pade13(alloc: std.mem.Allocator, As: Mat, A2: Mat, A4: Mat, A6: Mat, U: Mat, V: Mat) !void {
    var A8 = try matMult(alloc, A4, A4);
    defer A8.deinit();
    var A10 = try matMult(alloc, A2, A8);
    defer A10.deinit();
    var A12 = try matMult(alloc, A2, A10);
    defer A12.deinit();

    const PADE_13_COEFF: [14]f64 = [_]f64{ 64764752532480000, 32382376266240000, 7771770303897600, 1187353796428800, 129060195264000, 10559470521600, 670442572800, 33522128640, 1323241920, 40840800, 960960, 16380, 182, 1 };
    const b0 = PADE_13_COEFF[0];
    const b1 = PADE_13_COEFF[1];
    const b2 = PADE_13_COEFF[2];
    const b3 = PADE_13_COEFF[3];
    const b4 = PADE_13_COEFF[4];
    const b5 = PADE_13_COEFF[5];
    const b6 = PADE_13_COEFF[6];
    const b7 = PADE_13_COEFF[7];
    const b8 = PADE_13_COEFF[8];
    const b9 = PADE_13_COEFF[9];
    const b10 = PADE_13_COEFF[10];
    const b11 = PADE_13_COEFF[11];
    const b12 = PADE_13_COEFF[12];
    const b13 = PADE_13_COEFF[13];
    const n = As.rows;

    var temp = try Mat.initZero(alloc, n, n);
    defer temp.deinit();

    for (0..n) |i| for (0..n) |j| {
        // TODO: Make this nicer!
        const valV = (if (i == j) b0 else 0.0) + b2 * try A2.at(i, j) + b4 * try A4.at(i, j) + b6 * try A6.at(i, j) + b8 * try A8.at(i, j) + b10 * try A10.at(i, j) + b12 * try A12.at(i, j);
        const valT = (if (i == j) b1 else 0.0) + b3 * try A2.at(i, j) + b5 * try A4.at(i, j) + b7 * try A6.at(i, j) + b9 * try A8.at(i, j) + b11 * try A10.at(i, j) + b13 * try A12.at(i, j);
        try V.set(i, j, valV);
        try temp.set(i, j, valT);
    };
    var temp2 = try matMult(alloc, As, temp);
    defer temp2.deinit();
    try copyMat(temp2, U);
}

// TODO: Implement these below, maybe these should
// be under /linalg?

/// Does a best effort diagonalization of matrix A.
///
/// Returns a [3]Mat, which contains PDP^(-1) which equals A.
//pub fn diagonalizeMat(alloc: std.mem.Allocator, A: Mat) ![3]Mat {}

//pub fn eigenValues(alloc: std.mem.Allocator, mat: Mat) ![]f64 {}

//pub fn eigenVectors(alloc: std.mem.Allocator, eigVals: []f64) !Mat {}

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
pub fn determinant(alloc: std.mem.Allocator, mat: Mat) !f64 {
    if (mat.rows != mat.cols) return MatError.BadShape;
    const n = mat.rows;
    if (n == 0) return 1.0;
    if (n == 1) return try mat.at(0, 0);
    if (n == 2) {
        const a = try mat.at(0, 0);
        const d = try mat.at(1, 1);

        const b = try mat.at(0, 1);
        const c = try mat.at(1, 0);
        return (a * d - b * c);
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
        return (a * e * i + b * f * g + c * d * h - c * e * g - b * d * i - a * f * h);
    }
    const expand_row: usize = 0;

    var sum: f64 = 0.0;

    for (0..n) |col| {
        const a = try mat.at(expand_row, col);
        if (std.math.approxEqAbs(f64, a, 0.0, 1e-8)) continue;
        // sign = (-1)^(row + col)
        const sign: f64 = if (((expand_row + col) & 1) == 0) 1.0 else -1.0;
        var minor = try makeMinor(alloc, mat, expand_row, col);

        const det_minor = try determinant(alloc, minor);

        minor.deinit(); // no longer needed.
        sum += sign * a * det_minor;
    }
    return sum;
}

/// Makes a copy of the matrix, but skips 1 row and 1 coloumn.
fn makeMinor(alloc: std.mem.Allocator, mat: Mat, skip_row: usize, skip_col: usize) !Mat {
    const n = mat.rows;
    var m = try Mat.initZero(alloc, n - 1, n - 1);

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
pub fn isLowerTriangular(A: Mat, tolerance: f64) bool {
    for (0..A.rows) |r| {
        for (r + 1..A.cols) |c| {
            if (@abs(try A.at(r, c)) > tolerance) {
                if (@abs(@abs(try A.at(r, c)) - tolerance) <= 1e-10) {
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
pub fn isUpperTriangular(A: Mat, tolerance: f64) bool {
    for (0..A.rows) |r| {
        for (0..r) |c| {
            if (@abs(try A.at(r, c)) > tolerance) {
                if (@abs(@abs(try A.at(r, c)) - tolerance) <= 1e-10) {
                    std.log.warn("isUpperTriangular| Difference between 'tolerance' and matrix values is <= 1e-10. \n", .{});
                }
                return false;
            }
        }
    }
    return true;
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
