const std = @import("std");
const znum = @import("znumerics");
const Mat = znum.Mat; // This is done for convenience

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    var m1 = try Mat.initZero(alloc, 2, 2);
    defer m1.deinit();
    std.debug.print("Matrix 1 initZero: \n", .{});
    try m1.printMat();

    try m1.set(0, 0, 1);
    std.debug.print("Matrix 1 after set(0,0,1) \n", .{});
    try m1.printMat();

    var m2 = try Mat.initZero(alloc, 2, 2);
    defer m2.deinit();
    try m2.set(0, 0, 1);
    std.debug.print("Matrix 2: \n", .{});
    try m2.printMat();

    var m3 = try znum.mat.matMult(alloc, m1, m2);
    defer m3.deinit();
    std.debug.print("Matrix 3 = matMult(Matrix 1, Matrix 2): \n", .{});
    try m3.printMat();

    // Expands the matrix to 3x3 and fills the new
    // spots with 0.0
    try m3.expand(3, 3, 0.0);
    std.debug.print("Matrix 3 Expanded: \n", .{});
    try m3.printMat();

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
    std.debug.print("Matrix 4: \n", .{});
    try m4.printMat();

    m4.multAll(2.0);
    std.debug.print("2 * Matrix 4: \n", .{});
    try m4.printMat();
    try m4.swapRow(0, 1);
    std.debug.print("Matrix 4, rows 0 and 1 swapped: \n", .{});
    try m4.printMat();

    var m5 = try Mat.initZero(alloc, 3, 3);
    defer m5.deinit();
    try m5.setRow(0, [_]f64{ 2.0, -1.0, 0.0 });
    try m5.setRow(1, [_]f64{ -1.0, 2.0, -1.0 });
    try m5.setRow(2, [_]f64{ 0.0, -1.0, 2.0 });

    std.debug.print("Matrix 5: \n", .{});
    try m5.printMat();

    var m5Inv = try znum.mat.inverse(alloc, m5);
    defer m5Inv.deinit();

    std.debug.print("Matrix 5 Inverse: \n", .{});
    try m5Inv.printMat();

    var id_m5 = try znum.mat.matMult(alloc, m5, m5Inv);
    defer id_m5.deinit();
    std.debug.print("Matrix 5 * Matrix 5 Inverse: \n", .{});
    try id_m5.printMat();
}
