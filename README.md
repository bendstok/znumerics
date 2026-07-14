Hobby project for numerical computing in Zig

## Requirements
Only tested for Zig 0.16

## Use as a dependency
```sh
zig fetch --save git+https://github.com/bendstok/znumerics
```
```zig
// build.zig
const znumerics = b.dependency("znumerics", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("znumerics", znumerics.module("znumerics"));
```

## Test
```sh
zig build test
``` 

## Benchmarks for SIMD
```sh
zig build benchmark -Doptimize=ReleaseFast
```

Compared against Python on the same machine (`python benchmarks/bench.py`), ns/iter:

| Benchmark | znumerics | NumPy | Pure Python |
|---|---|---|---|
| add 512x512 | 579,750 | 682,320 | 10,966,380 |
| matmul 64x64 | 23,500 | 27,985 | 9,800,710 |
| eigenvalues 20x20 | 15,084 | 14,264 | - |
| complex eigenvalues 20x20 | 3,664 | 11,768 | - |

NB: addSIMD is approx equal to add (the plain loop auto-vectorizes),
matMultSIMD is approx 2.8x faster than matMult.

NB: complex eigenvalues are approx 3x faster than numpy here, but this is with
matrices that are small. Expect that the gap shrinks with size.

## Run example
```sh
zig build examples
```

### Matrices
```zig
const znum = @import("znumerics");
const Mat = znum.Mat; // Mat is Matrix(f64)

var A = try Mat.initZero(alloc, 2, 2);
defer A.deinit();
try A.setRow(0, [_]f64{ 2.0, 1.0 });
try A.setRow(1, [_]f64{ 1.0, 2.0 });

var B = try A.add(A); // also: sub, addInPlace, subInPlace, addSIMD
defer B.deinit();

var C = try znum.mat.matMult(alloc, A, B); // also: matMultSIMD, matVec
defer C.deinit();

var Ainv = try znum.mat.inverse(alloc, A);
defer Ainv.deinit();

var eA = try znum.mat.expm(alloc, A); // matrix exponential
defer eA.deinit();

// Random matrix: (alloc, rows, cols, seed, min, max). Same seed, same
// matrix. Floats are drawn from [min, max), ints from [min, max] inclusive.
// For complex, re and im are drawn independently within their own bounds.
var R = try Mat.initRandom(alloc, 3, 3, 42, -1.0, 1.0);
defer R.deinit();

// also: initIdentity, transpose, determinant, trace, charPoly, norm1,
// expand, swapRow, getRow/getCol/setCol, isUpperTriangular/isLowerTriangular
```

Matrices are generic over the element type via `znum.Matrix(T)`.
Aliases: `Mat` (f64), `CMat` (Complex(f64)), and `znum.mat.Mat_32` / `CMat_32` for f32.

```zig
const CMat = znum.CMat;
const Cx = std.math.Complex(f64);

var Z = try CMat.initIdentity(alloc, 2, 2);
defer Z.deinit();
try Z.set(0, 0, Cx.init(0.0, 1.0)); // i

// Same functions as for f64
var Zinv = try znum.mat.inverse(alloc, Z);
defer Zinv.deinit();
const det = try znum.mat.determinant(alloc, Z); // det = i * 1 = i
```

NB: matMultSIMD falls back to matMult for complex matrices.

### Vectors
```zig
// (alloc, start, end, steps, include endpoint)
var v = try znum.vec.linspace(alloc, 0.0, 1.0, 5, true);
defer v.deinit();

const d = try znum.vec.dot(v, v);
const n = v.norm();

// also: crossProd3d, vecMult (outer product), normalize, resize,
// setAll, addInPlace/subInPlace
```

Vectors are generic over the element type via `znum.Vector(T)`.
Aliases: `Vec` (f64), `CVec` (Complex(f64)), and `znum.vec.Vec_32` / `CVec_32` for f32.

```zig
const CVec = znum.CVec;

var w = try CVec.initZero(alloc, 2, false);
defer w.deinit();
try w.setAll([_]Cx{ Cx.init(3.0, 4.0), Cx.init(0.0, 0.0) });

// Same functions as for f64. norm() is always real
const nw = w.norm(); // |3+4i| = 5.0

// dot conjugates the left side, so dot(x, x) == norm(x)^2
const dw = try znum.vec.dot(w, w); // 25 + 0i
```

NB: dot ignores row/column orientation.
vecMult does not conjugate (a * b^T, like np.outer).

### Eigenvalues
```zig
// Real spectrum (e.g. symmetric matrices)
// (alloc, A, max QR sweeps, deflation tolerance, optional out: sweep count)
var it: usize = 0;
const eigs = try znum.eigenvalues(alloc, A, 1000, 1e-12, &it);
defer alloc.free(eigs);

// Complex spectrum: returns []std.math.Complex(f64). Same inputs, pass
// null to skip the sweep count.
const ceigs = try znum.eigenvaluesComplex(alloc, A, 1000, 1e-12, null);
defer alloc.free(ceigs);

// Also accepts complex matrices (CMat / CMat_32): the QR iteration then
// runs in complex arithmetic and eigenvalues do NOT come in conjugate
// pairs. Returns []Complex(f64) for CMat, []Complex(f32) for CMat_32.
const zeigs = try znum.eigenvaluesComplex(alloc, Z, 1000, 1e-12, null);
defer alloc.free(zeigs);

// Polynomial roots (descending coefficients, matching charPoly):
// x^2 - 3x + 2 -> roots 1 and 2, returned as []Complex(f64)
const rts = try znum.roots(alloc, &[_]f64{ 1.0, -3.0, 2.0 }, 1000, 1e-12);
defer alloc.free(rts);

// also: znum.eigen.qrAlgorithm / qrAlgorithmComplex (shifted QR with
// balancing + Householder Hessenberg reduction),
// znum.eigen.eigenvaluesArnoldi / eigenvaluesComplexArnoldi (Arnoldi
// reduction pipeline, for the sparse/partial-spectrum niche),
// znum.eigen.arnoldi_iteration (Krylov basis + Hessenberg projection)
```

### Solve Ax = b
```zig
var x = try znum.LU.solve(alloc, A, b);
defer x.deinit();

// LU: factor once, then each solve is O(n^2)
var f = try znum.LU.lu(alloc, A);
defer f.deinit();
var x2 = try f.solve(alloc, b);
defer x2.deinit();
const det = f.det(); // det(A) from the factorization, O(n)

// also: znum.gaussJordan.gaussJordan, znum.cholesky.cholesky,
// znum.QR.qrDecomposition
```

NB: lu and gaussJordan pivot on |.|, so complex systems work as-is.
qrDecomposition uses Householder reflections (I - 2vv^H); Q is unitary
and R = Q^H * A for complex matrices.

### Simulate LTI systems (lsim)
```zig
// x' = -x + u, y = x
var ss = try znum.StateSpace.initContinuous(alloc, 1);
defer ss.deinit();
try ss.A.set(0, 0, -1.0);
try ss.B.set(0, 1.0);
try ss.C.set(0, 1.0);

// Open loop: (alloc, system, input signal, dt, optional x0, options)
const u = [_]f64{2.0} ** 1001;
var res = try znum.lsim.lsim(alloc, ss, &u, 1e-3, null, .{});
defer res.deinit();
// res.t, res.y, res.x (state trajectory, coloumn k is x[k])

// Closed loop: u[k] = kp * (r - y[k]) via a ctx + comptime fn.
// The ctx carries the controller state; here a stateless P-controller.
const P = struct {
    kp: f64,
    r: f64,
    fn w(self: @This(), k: usize, t: f64, x: znum.Vec) f64 {
        _ = k;
        _ = t;
        return self.kp * (self.r - x.atUnsafe(0));
    }
};
var cl = try znum.lsim.lsimFn(alloc, ss, 1e-3, 5000, P{ .kp = 9.0, .r = 1.0 }, P.w, null, .{});
defer cl.deinit();
// y settles at kp * r / (1 + kp) = 0.9

// For unstable systems, .{ .clamp = 1e3 } bounds every state/output
// sample to [-1e3, 1e3] (NaN -> 0) so the result stays plottable.

// also: dlsim/dlsimFn (discrete systems), step, impulse
```

NB: the input is held constant over each step (ZOH), which is exact at
the sample instants for piecewise-constant u. The callback sees the
state *before* the step, so feedback has no algebraic loop.

NB: for a constant u there is no need to build the input slice:
step covers u = 1 (scale the output for other heights, the system is
linear), and lsimFn with a comptime fn returning the constant covers
the rest without the allocation.

### ODE (RK4, RKF45)
```zig
// x' = -x + u, u = 2.0, dt = 1e-3
var ss = try znum.StateSpace.initContinuous(alloc, 1);
defer ss.deinit();
try ss.A.set(0, 0, -1.0);
try ss.B.set(0, 1.0);

var x = [_]f64{0.0};
// One step: (state dim (comptime), system, state, input u, dt)
x = znum.RK4(1, ss, x, 2.0, 1e-3);

// RKF45: adaptive step size, integrates all the way from t0 to tf.
// (state dim (comptime), system, x0, input u, t0, tf, options)
const xf = try znum.RKF45(1, ss, .{0.0}, 2.0, 0.0, 1.0, .{});

// Options (all have defaults): tol, h0, h_min, h_max
const xf2 = try znum.RKF45(1, ss, .{0.0}, 2.0, 0.0, 1.0, .{ .tol = 1e-10 });

// also: znum.PID, znum.signal (tf2ss, ss2tf, cont2discrete,
// StateSpace.fromSlices, dcgain)
```

NB: RKF45 takes 4th- and 5th-order solutions from the same six stages and
uses their difference as the local error estimate. Steps above tol are
rejected and retried smaller; returns StepSizeTooSmall if h underflows h_min.

