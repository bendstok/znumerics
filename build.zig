const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("znumerics", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Tests
    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    const run_mod_tests = b.addRunArtifact(mod_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);

    // Examples
    const example_step = b.step("examples", "Run examples");
    const mat_example = b.addExecutable(.{
        .name = "matrix",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/mat.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Benchmarks for SIMD
    const benchmark_step = b.step("benchmark", "Run SIMD benchmarks");
    const benchmark_example = b.addExecutable(.{
        .name = "benchmark",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    benchmark_example.root_module.addImport("znumerics", mod);
    const run_benchmark_example = b.addRunArtifact(benchmark_example);
    benchmark_step.dependOn(&run_benchmark_example.step);

    mat_example.root_module.addImport("znumerics", mod);
    const run_mat_example = b.addRunArtifact(mat_example);
    example_step.dependOn(&run_mat_example.step);

    const eig_example = b.addExecutable(.{
        .name = "eigenvalues",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/eigenvalues.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    eig_example.root_module.addImport("znumerics", mod);
    const run_eig_example = b.addRunArtifact(eig_example);
    example_step.dependOn(&run_eig_example.step);

    const vec_example = b.addExecutable(.{
        .name = "vector",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/vec.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    vec_example.root_module.addImport("znumerics", mod);
    const run_vec_example = b.addRunArtifact(vec_example);
    example_step.dependOn(&run_vec_example.step);

    const lsim_example = b.addExecutable(.{
        .name = "lsim",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/lsim.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    lsim_example.root_module.addImport("znumerics", mod);
    const run_lsim_example = b.addRunArtifact(lsim_example);
    example_step.dependOn(&run_lsim_example.step);
}
