# Eigen Universal Integration

This is a header only library that overrides some of Eigen's internal routines to use the fused dot product in case they are called with universal's posits.

It applies to small matrix-matrix products (lazyproduct), large matrix-matrix products (gemm), matrix-vector products of any size (gemv), dot products of runtime vectors (inner) and dot products of compile time vectors (dot).

## Dependencies

This library depends on the Eigen and Universal libraries.

## Installation & Usage

As this is simply a CMake library, just intall it with cmake and add it to your CMake project includes.