const std = @import("std");
const znum = @import("znumerics");

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    // Plant: x' = -x + u, y = x
    var ss = try znum.StateSpace.initContinuous(alloc, 1);
    defer ss.deinit();
    try ss.A.set(0, 0, -1.0);
    try ss.B.set(0, 1.0);
    try ss.C.set(0, 1.0);

    // Step response: y(t) = 1 - e^-t
    var st = try znum.lsim.step(alloc, ss, 0.1, 11);
    defer st.deinit();
    std.debug.print("Step response of x' = -x + u, t = 0..1: \n", .{});
    try st.t.printVec();
    try st.y.printVec();

    // Impulse response: y(t) = e^-t
    var im = try znum.lsim.impulse(
        alloc,
        ss,
        0.1,
        11,
    );
    defer im.deinit();
    std.debug.print("Impulse response: \n", .{});
    try im.y.printVec();

    // Open loop with an input signal: u = 2 -> y(t) = 2 * (1 - e^-t)
    const u = [_]f64{2.0} ** 11;
    var ol = try znum.lsim.lsim(alloc, ss, &u, 0.1, null, .{});
    defer ol.deinit();
    std.debug.print("Response to u = 2: \n", .{});
    try ol.y.printVec();

    // Closed loop with the PID controller. The ctx + comptime fn pattern
    // lets the input depend on the state; the ctx is a *pointer* here,
    // so the controller keeps its integral state between steps.
    const dt: f64 = 1e-3;
    // (K_p, T_i, T_d, dt, clamp_min, clamp_max)
    var pid = znum.PID.init(9.0, 0.5, 0.0, dt, -100.0, 100.0);
    const C = struct {
        fn w(p: *znum.PID, k: usize, t: f64, x: znum.Vec) f64 {
            _ = k;
            _ = t;
            return p.compute(1.0, x.atUnsafe(0)); // ref = 1, y = x[0]
        }
    };
    var cl = try znum.lsim.lsimFn(alloc, ss, dt, 5000, &pid, C.w, null, .{});
    defer cl.deinit();
    std.debug.print("Closed loop with the PID (K_p = 9, T_i = 0.5), ref = 1: \n", .{});
    std.debug.print("y settles at {d:.4} (integral action removes the P-only offset of 0.9) \n", .{cl.y.atUnsafe(4999)});
}
