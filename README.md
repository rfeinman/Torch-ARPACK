# Torch ARPACK

Torch ARPACK is a PyTorch c++ extension library for large-scale eigenvalue problems. It is inspired heavily by the [ARPACK](https://www.caam.rice.edu/software/ARPACK/) Fortran library and its corresponding [SciPy wrappers](https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html). The motivation is to identify one or a handful of desired eigenpairs for a very large linear system. The library uses restarted Arnoldi & Lanczos algorithms.

### Motivation
When working with large linear systems, it is often useful to quickly access one or a handful of eigenpairs without computing the complete spectrum. SciPy's `sparse.linalg` module offers a number of high-performance, numpy-based eigensolvers for problems of this kind. In contrast, PyTorch offers only the `lobpcg` function, which is very slow and underperforms complete decomposition in all of my benchmarks thus far (not to mention its severe underperformance vs. SciPy's equivalent lopcg). The `nn.utils.spectral_norm` tool computes a single eigenpair using the power iteration method; however, power iteration has poor convergence properties for ill-conditioned matrices, and in the special case of symmetric/hermitian matrices, its convergence speed can be easily increased with little trade-off.


