# Eigen Universal Integration

This is a header only library that overrides some of Eigen's internal routines to use the fused dot product in case they are called with universal's posits.

It applies to small matrix-matrix products (lazyproduct), large matrix-matrix products (gemm), matrix-vector products of any size (gemv), dot products of runtime vectors (inner) and dot products of compile time vectors (dot).

## Dependencies

This library depends on the Eigen and Universal libraries.

## Installation & Usage

As this is simply a CMake library, just intall it with cmake and add it to your CMake project includes.

## Patched Eigen

This repository contains a patch for Eigen's BDCSVD, which stalls when using posits. If you plan on using the BDCSVD with posits, you should patch your installation of Eigen with that.

## Patched Universal

This repository contains a patch to use the Universal library on aarch64 architectures.

# Flake

This repository provides a flake for easy use in a Nix environment. This flake provides 3 main outputs:
 - `universal`: a build of the universal library with the above patch applied for aarch64 platforms
 - `eigen`: a build of eigen with the SVD patch applied.
 - `EigenUniversalIntegration`: a build of the library provided by this repository