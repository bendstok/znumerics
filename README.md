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

## Run example
```sh
zig build examples
```

