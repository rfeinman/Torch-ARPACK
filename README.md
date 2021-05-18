# Torch ARPACK

Torch ARPACK is a PyTorch C++ extension for solving large-scale eigenvalue problems. It is inspired heavily by the [ARPACK](https://www.caam.rice.edu/software/ARPACK/) Fortran library and corresponding [SciPy wrappers](https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html). The motivation is to identify one or a handful of desired eigenpairs for a very large linear system using restarted Arnoldi & Lanczos algorithms.

### Motivation
When working with large linear systems, it is useful to access one or a handful of eigenpairs without computing the complete spectrum. SciPy's `sparse.linalg` module offers a number of high-performance, numpy-based eigensolvers for problems of this kind. In contrast, PyTorch offers only the `lobpcg` function, which is very slow: in my runtime benchmarks thus far, it underperforms full decomposition in all cases, and it severely underperforms SciPy's equivalent [lopcg](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lobpcg.html#scipy.sparse.linalg.lobpcg). The `nn.utils.spectral_norm` tool computes a single eigenpair using power iteration; however, power iteration has poor convergence properties for many applications, and in the special case of symmetric/hermitian matrices, convergence speed can be easily increased with little trade-off using alternative Krylov methods.

The motivation of Torch ARPACK is to augment the PyTorch library with fast, compiled eigensolvers based on restarted Arnoldi methods. The current version includes a single function `eigsh` which is limited to symmetric matrices. At each step, the algorithm performs a Lanczos factorization and calls LAPACK routines *?stebz* and *?stein* to compute a partial Schur decomposition of the resulting tridiagonal system. There are no MAGMA analogues to *?stebz* and *?stein* that I'm aware of; therefore, the current implementation is limited to CPU.

### Appendix A: CPU / CUDA partial eigensolver index (SPARSE)

Here I keep track of the available CPU/CUDA backend implementations for partial eigenvalue problems with large, sparse linear systems. The most comprehensive existing solution is the Intel MKL extremal eigensolver (CPU only). For an overview of the MKL module, see [here](https://software.intel.com/content/www/us/en/develop/articles/intel-mkl-support-for-largestsmallest-eigenvalue-and-sparse-svd-problem.html). For a reference on the solver parameters, see [here](https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/extended-eigensolver-routines/extended-eigensolver-interfaces-for-extremal-eigenvalues-singular-values/extended-eigensolver-input-parameters-for-extremal-eigenvalue-problem.html).

#### MKL (CPU)
  
- Routines (`range = 'I'`)
    - [mkl_sparse_?_ev](https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/extended-eigensolver-routines/extended-eigensolver-interfaces-for-extremal-eigenvalues-singular-values/extended-eigensolver-interfaces-to-find-largest-smallest-eigenvalues/mkl-sparse-ev.html) (standard; symmetric) - Solve for extremal eigenvalues using (1) Krylov-Schur or (2) FEAST-based subspace projection.
    - [mkl_sparse_?_gv](https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/extended-eigensolver-routines/extended-eigensolver-interfaces-for-extremal-eigenvalues-singular-values/extended-eigensolver-interfaces-to-find-largest-smallest-eigenvalues/mkl-sparse-gv.html) (generalized; symmetric) - Solve for extremal eigenvalues using (1) Krylov-Schur, or (2) FEAST-based subspace projection. 
  
- Routines (`range = 'V'`)
  - TODO
    
#### MAGMA (CUDA)

- Routines (`range = 'I'`)
    - [magma_?lobpcg](http://icl.cs.utk.edu/projectsfiles/magma/doxygen/group__magmasparse__ssyev.html) (standard; symmetric positive-definite) - Solve for extremal eigenvalues via LOBPCG.
    
#### CuSOLVER (CUDA)

- Routines (`range = 'V'`)
    - [cusolverSp\<t\>csreigvsi](https://docs.nvidia.com/cuda/cusolver/index.html#cusolver-lt-t-gt-csreigsi) (standard; general) - Given an initial eigenvalue estimate, solve for the nearest eigenpair using inverse iteration.
    - [cusolverSp\<t\>csreigs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolver-lt-t-gt-csreigs) (standard; general) - Compute the number of eigenpairs that lie within the interval \[low, high\].


### Appendix B: CPU / CUDA partial eigensolver index (DENSE)

In addition to sparse eigensolvers—which use fast subspace methods (e.g. Lanczos, Arnoldi, supspace iteration)—there are a handful of partial eigensolvers that operate on complete bases. Like the standard "full" eigensolvers (e.g. *?geev* and *?syev*), these algorithms begin with a complete Hessenberg/tridiagonal factorization. However, in the second phase—whereas "full" solvers perform a complete Schur decomposition—partial solvers find a subset of eigenpairs using various techniques that reduce the computational complexity.

#### MKL (CPU)

- Routines (`range = 'I', 'V'`)
  - ?syevr - Symmetric standard
  - ?syevx - Symmetric standard
  - ?sygvx - Symmetric generalized
  
#### MAGMA (CUDA)

- Routines

#### CuSOLVER (CUDA)

- Routines (`range = 'I', 'V'`)
  - [cusolverDnXsyevdx](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDnXsyevdx) - Symmetric standard
  - (legacy) [cusolverDn\<t\>sygvdx](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-sygvdx) - Symmetric generalized


