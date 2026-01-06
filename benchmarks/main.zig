const std = @import("std");
const znum = @import("znumerics");
const matbench = @import("mat.zig");
const Mat = znum.Mat; // This is done for convenience
const MatOp = znum.mat;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    try matbench.matAddvsSIMDadd(alloc);
    try matbench.matMulvsSIMDmatMul(alloc);
}
