# Persistent Laplacians
The paper [Persistent Laplacians: properties, algorithms and implications](https://arxiv.org/pdf/2012.02808) suggests an algorithm to compute the persistent Laplacian of a filtration in Theorem 5.1.
Here is some code towards an implementation of this algorithm.

Run `maturin develop` to build. You can run `cargo bench` to see benchmarks for an example set of points using the Alpha complex from `gudhi`.  