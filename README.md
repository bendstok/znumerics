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
| matmul 64x64 | 35,020 | 27,985 | 9,800,710 |
| eigenvalues 20x20 | 15,084 | 14,264 | - |
| complex eigenvalues 20x20 | 3,664 | 11,768 | - |

NB: addSIMD is approx equal to add (the plain loop auto-vectorizes),
matMultSIMD is approx 1.8x faster than matMult.

NB: complex eigenvalues are approx 3x faster than numpy here, but this is with
matrices that are small. Expect that the gap shrinks with size.

## Run example
```sh
zig build examples
```

### Matrices
```zig
const znum = @import("znumerics");
const Mat = znum.Mat;

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

// also: initIdentity, transpose, determinant, trace, charPoly, norm1,
// expand, swapRow, getRow/getCol/setCol, isUpperTriangular/isLowerTriangular
```

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

// also: znum.eigen.qrAlgorithm / qrAlgorithmComplex (shifted QR directly on the dense matrix),
// znum.eigen.arnoldi_iteration (Krylov basis + Hessenberg projection)
```

### Solve Ax = b
```zig
var x = try znum.gaussJordan.gaussJordan(alloc, A, b);
defer x.deinit();

// also: znum.cholesky.cholesky, znum.QR.qrDecomposition
```

### ODE (RK4)
```zig
// x' = -x + u, u = 2.0, dt = 1e-3
var ss = try znum.StateSpace.initContinuous(alloc, 1);
defer ss.deinit();
try ss.A.set(0, 0, -1.0);
try ss.B.set(0, 1.0);

var x = [_]f64{0.0};
// One step: (state dim (comptime), system, state, input u, dt)
x = znum.RK4(1, ss, x, 2.0, 1e-3);

// also: znum.PID, znum.signal (tf2ss, ss2tf, cont2discrete)
```

