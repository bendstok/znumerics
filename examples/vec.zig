const std = @import("std");
const znum = @import("znumerics");
const Vec = znum.Vec; // This is done for convenience

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    // (alloc, start, end, steps, include endpoint)
    var v1 = try znum.vec.linspace(alloc, 0.0, 1.0, 5, true);
    defer v1.deinit();
    std.debug.print("Vector 1 linspace(0, 1, 5): \n", .{});
    try v1.printVec();

    const d = try znum.vec.dot(v1, v1);
    std.debug.print("dot(Vector 1, Vector 1): {d:.3} \n", .{d});
    std.debug.print("Vector 1 norm: {d:.3} \n", .{v1.norm()});

    v1.normalize();
    std.debug.print("Vector 1 normalized (norm = {d:.3}): \n", .{v1.norm()});
    try v1.printVec();

    var v2 = try Vec.initZero(alloc, 3, false);
    defer v2.deinit();

    const float: f64 = 2.0;
    try v2.setAll(float);
    std.debug.print("Vector 2 after setAll(2.0): \n", .{});
    try v2.printVec();

    try v2.setAll([_]f64{ 1.0, 0.0, 0.0 });
    std.debug.print("Vector 2 after setAll([1, 0, 0]): \n", .{});
    try v2.printVec();

    var v3 = try Vec.initZero(alloc, 3, false);
    defer v3.deinit();
    try v3.set(1, 1.0);

    // x-hat cross y-hat = z-hat
    var v4 = try znum.vec.crossProd3d(alloc, v2, v3);
    defer v4.deinit();
    std.debug.print("Vector 4 = crossProd3d(Vector 2, Vector 3): \n", .{});
    try v4.printVec();

    // Outer product: left has to be a coloumn vector
    var v5 = try Vec.initZero(alloc, 2, true);
    defer v5.deinit();
    try v5.setAll([_]f64{ 1.0, 2.0 });
    var v6 = try Vec.initZero(alloc, 2, false);
    defer v6.deinit();
    try v6.setAll([_]f64{ 3.0, 4.0 });

    var m1 = try znum.vec.vecMult(alloc, v5, v6);
    defer m1.deinit();
    std.debug.print("Matrix 1 = vecMult(Vector 5, Vector 6) (outer product): \n", .{});
    try m1.printMat();

    // Vectors are generic over the element type.
    // Vec is Vector(f64), CVec is Vector(Complex(f64)).
    const CVec = znum.CVec;
    const Cx = std.math.Complex(f64);

    var v7 = try CVec.initZero(alloc, 2, false);
    defer v7.deinit();
    try v7.setAll([_]Cx{ Cx.init(3.0, 4.0), Cx.init(0.0, 1.0) });
    std.debug.print("Vector 7 (complex): \n", .{});
    try v7.printVec();

    // norm() is always real
    std.debug.print("Vector 7 norm: {d:.3} \n", .{v7.norm()});

    // dot conjugates the left side, so dot(x, x) == norm(x)^2
    const d7 = try znum.vec.dot(v7, v7);
    const im_sign: u8 = if (d7.im < 0) '-' else '+';
    std.debug.print("dot(Vector 7, Vector 7): {d:.3}{c}{d:.3}i \n", .{ d7.re, im_sign, @abs(d7.im) });
}
