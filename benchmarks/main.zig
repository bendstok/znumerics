const std = @import("std");
const znum = @import("znumerics");
const matbench = @import("mat.zig");
const eigbench = @import("eigen.zig");
const Mat = znum.Mat; // This is done for convenience
const MatOp = znum.mat;

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    try matbench.matAddvsSIMDadd(alloc, io);
    try matbench.matMulvsSIMDmatMul(alloc, io);
    try eigbench.qrDirectVsPipeline(alloc, io);
}
