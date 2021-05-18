# Torch ARPACK

Torch ARPACK is a PyTorch C++ extension for solving large-scale eigenvalue problems. It is inspired heavily by the [ARPACK](https://www.caam.rice.edu/software/ARPACK/) Fortran library and its corresponding [SciPy wrappers](https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html). The motivation is to identify one or a handful of desired eigenpairs for a very large linear system using restarted Arnoldi & Lanczos algorithms.

### Motivation
When working with large linear systems, it is useful to access one or a handful of eigenpairs without computing the complete spectrum. SciPy's `sparse.linalg` module offers a number of high-performance, numpy-based eigensolvers for problems of this kind. In contrast, PyTorch offers only the `lobpcg` function, which is very slow: in my runtime benchmarks thus far, it underperforms full decomposition in all cases, and it severely underperforms SciPy's equivalent [lopcg](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lobpcg.html#scipy.sparse.linalg.lobpcg). The `nn.utils.spectral_norm` tool computes a single eigenpair using the power iteration method; however, power iteration has poor convergence properties for ill-conditioned matrices, and in the special case of symmetric/hermitian matrices, its convergence speed can be easily increased with little trade-off.

The motivation of Torch ARPACK is to augment the PyTorch library with fast, compiled eigensolvers based on restarted Arnoldi methods. The current version includes a single function `eigsh` which is limited to symmetric matrices. At each step, the algorithm performs a Lanczos factorization and calls LAPACK routines *?stebz* and *?stein* to compute a partial Schur decomposition of the resulting tridiagonal system. There are no MAGMA analogues to *?stebz* and *?stein* that I'm aware of; therefore, the current implementation is limited to CPU.


