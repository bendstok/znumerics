Hobby project for numerical computing in Zig

## Requirements
Zig >= 0.15.2

## Use as a dependency
```sh
zig fetch --save git+https://github.com/<you>/znumerics
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
zig build benchmark
```

## Run example
```sh
zig build examples
```

