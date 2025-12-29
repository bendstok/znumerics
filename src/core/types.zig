const std = @import("std");
const mat = @import("../core/mat.zig");
const vec = @import("../core/vec.zig");

const Vec = vec.Vec;
const Mat = mat.Mat;

/// The Complex type, meaning a + i*b
pub const Complex = struct {
    a: f64,
    b: f64,
};

