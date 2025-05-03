# Persistent Laplacians
The paper [Persistent Laplacians: properties, algorithms and implications](https://arxiv.org/pdf/2012.02808) suggests an algorithm to compute the persistent Laplacian of a filtration in Theorem 5.1. This repository implements the suggested algorithm in Rust and exposes Python bindings. Sparse matrix computations are used in the implementation of the algorithm.

## Persistent homology 

To evaluate correctness, the nullity of the computed persistent laplacians is computed using SVD on dense matrices. This is the rank of the persistent homology, so it can be cross checked with available implementations (tests use [`gudhi`](https://github.com/GUDHI)). 

## Compilation, tests, benchmarking 
Run `maturin develop --release` to build in release mode. You can run `cargo bench` to see benchmarks for an example set of points using the Alpha complex from `gudhi`. For now you'll need to do that in a pip env named `venv` due to hardcoded config. Run `pytest -v python/tests/persistent_homology.py; cargo test` to run tests. 
