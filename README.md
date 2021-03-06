# Torch ARPACK

Torch ARPACK is a PyTorch C++ extension for solving large-scale eigenvalue problems. It is inspired by the [ARPACK](https://www.caam.rice.edu/software/ARPACK/) Fortran library and corresponding [SciPy wrappers](https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html). The package uses efficient Arnoldi & Lanczos algorithms to identify one or a few* desired eigenpairs of a matrix.

__Author__: Reuben Feinman

*At the moment, only a single eigenpair is supported. Extensions for multiple pairs are in the works.

### Motivation
When working with large linear systems, it's often useful to access one or a handful of eigenpairs without computing the complete spectrum. SciPy's `sparse.linalg` module offers a number of high-performance, numpy-based eigensolvers for problems of this kind. In contrast, PyTorch offers only the `lobpcg` function, which is very slow: in my runtime benchmarks thus far, it underperforms full decomposition in all cases, and it severely underperforms SciPy's equivalent [lopcg](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lobpcg.html#scipy.sparse.linalg.lobpcg). The `nn.utils.spectral_norm` tool computes a single eigenpair using power iteration; however, power iteration has poor convergence properties for many applications, and in the special case of symmetric/hermitian matrices, convergence speed can be easily increased with little trade-off using alternative Krylov methods.

The motivation of Torch ARPACK is to augment the PyTorch library with fast, compiled eigensolvers based on restarted Arnoldi & Lanczos methods. The current library includes a single tool `eigsh` which is applicable to symmetric matrices. At each step, the algorithm performs an m-step Lanczos factorization and calls LAPACK routines *?stebz* and *?stein* to compute a partial Schur decomposition of the resulting tridiagonal system. There are no MAGMA analogues to *?stebz* and *?stein* that I'm aware of; therefore, only CPU is currently supported. I've also included a wrapper for an equivalent function from the Intel Math Kernel Library (MKL).

### API

There are two functions currently implemented: `arpack.eigsh` and `argpack.eigsh_mkl`. Each of these can be used to quickly compute the largest or smallest (algebraic) eigenpair of a symmetric matrix. The differences are explained below.

1. `arpack.eigsh` - This function represents my own custom implementation of the explicitly restarted Lanczos method. It is very fast and should be the default for most cases. Note, however, that the algorithm has not yet been rigorously tested.
2. `argpack.eigsh_mkl` - This function wraps the [Intel MKL implementation](https://software.intel.com/content/www/us/en/develop/articles/intel-mkl-support-for-largestsmallest-eigenvalue-and-sparse-svd-problem.html) of the Krylov-Schur algorithm for extremal eigenvalue problems. From a quick read of the Krylov-Schur paper, the algorithm appears very similar to my own (but MKL is not open-source, so who knows). In early benchmarks the implementation is roughly 2x slower than mine. Being an Intel-backed product, however, it is likely more reliable at the current time.

For an example of how to use these two functions, see `scripts/test.py`. To benchmark the runtime performance of these functions against alternatives from native PyTorch, see `scripts/benchmark`.

### Appendix: CPU / CUDA partial eigensolver index

Here I keep track of the available CPU & CUDA libraries for solving partial eigenvalue problems. Using LAPACK terminology, the specifier `range = 'I'` refers to the problem of locating a subset of eigenpairs by index (e.g. find the largest 5 pairs) and `range = 'V'` locating by value (e.g. find all eigenpairs with values in range [low, high]). All tools currently implemented in Torch ARPACK concern the former (index).

#### Approximate solvers

Approximate algorithms for computing select eigenpairs rely on iterative subspace routines such as Arnoldi, Lanczos, and subspace iteration. These algorithms use only matrix-vector products and have a smaller memory footprint compared to exact routines. For matrices stored in *sparse* formats (e.g. coo, csr), approximate methods are the only applicable routines. For matrices stored in *dense* format, approximate methods still provide a significant speed-up over exact solvers in many cases. In general, these are the best options.

 The most comprehensive existing solution is the Intel MKL extremal eigensolver (CPU only). For an overview of the MKL module, see [here](https://software.intel.com/content/www/us/en/develop/articles/intel-mkl-support-for-largestsmallest-eigenvalue-and-sparse-svd-problem.html). For a reference on the solver parameters, see [here](https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/extended-eigensolver-routines/extended-eigensolver-interfaces-for-extremal-eigenvalues-singular-values/extended-eigensolver-input-parameters-for-extremal-eigenvalue-problem.html).

- MKL
  
  - Routines (`range = 'I'`)
      - [mkl_sparse_?_ev](https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/extended-eigensolver-routines/extended-eigensolver-interfaces-for-extremal-eigenvalues-singular-values/extended-eigensolver-interfaces-to-find-largest-smallest-eigenvalues/mkl-sparse-ev.html) (standard; symmetric) - Solve for extremal eigenvalues using (1) Krylov-Schur or (2) FEAST-based subspace projection.
      - [mkl_sparse_?_gv](https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/extended-eigensolver-routines/extended-eigensolver-interfaces-for-extremal-eigenvalues-singular-values/extended-eigensolver-interfaces-to-find-largest-smallest-eigenvalues/mkl-sparse-gv.html) (generalized; symmetric) - Solve for extremal eigenvalues using (1) Krylov-Schur, or (2) FEAST-based subspace projection. 
    
  - Routines (`range = 'V'`)
    - [feast](https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/extended-eigensolver-routines/the-feast-algorithm.html) - TODO
    
- MAGMA

  - Routines (`range = 'I'`)
      - [magma_?lobpcg](http://icl.cs.utk.edu/projectsfiles/magma/doxygen/group__magmasparse__ssyev.html) (standard; symmetric positive-definite) - Solve for extremal eigenvalues via LOBPCG.
    
- CuSOLVER

  - Routines (`range = 'V'`)
      - [cusolverSp\<t\>csreigvsi](https://docs.nvidia.com/cuda/cusolver/index.html#cusolver-lt-t-gt-csreigsi) (standard; general) - Given an initial eigenvalue estimate, solve for the nearest eigenpair using inverse iteration.
      - [cusolverSp\<t\>csreigs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolver-lt-t-gt-csreigs) (standard; general) - Compute the number of eigenpairs that lie within the interval \[low, high\].


#### Exact solvers

In addition to approximate eigensolvers???which solve for eigenpairs in an evolving subspace???there are a handful of *exact* partial eigensolvers that operate on complete bases. Like full-spectrum eigensolvers (e.g. *?geev* and *?syev*), these algorithms begin with a complete Hessenberg/tridiagonal factorization. However, in the second phase???whereas "full" solvers perform a complete Schur decomposition???partial solvers find a subset of eigenpairs using various techniques that reduce the computational complexity.

- MKL

  - Routines (`range = 'I', 'V'`)
    - [?syevx](https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/lapack-least-squares-and-eigenvalue-problem-routines/lapack-least-squares-and-eigenvalue-problem-driver-routines/symmetric-eigenvalue-problems-lapack-driver-routines/syevx.html) (standard; symmetric) - Tridiagonal reduction --> bisection --> inverse iteration --> tridiagonal orthogonal transform.
    - [?sygvx](https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/lapack-least-squares-and-eigenvalue-problem-routines/lapack-least-squares-and-eigenvalue-problem-driver-routines/generalized-symmetric-definite-eigenvalue-problems-lapack-driver-routines/sygvx.html#sygvx) (generalized; symmetric) - See above.
  
- MAGMA

  - Routines (`range = 'I', 'V'`)
    - [magma_?syevx](http://icl.cs.utk.edu/projectsfiles/magma/doxygen/group__magma__heevx.html) (standard; symmetric) - QR iteration algorithm.
    - [magma_?sygvx](http://icl.cs.utk.edu/projectsfiles/magma/doxygen/group__magma__hegvx.html)  (generalized; symmetric) - See above.
    - [magma_?syevdx](http://icl.cs.utk.edu/projectsfiles/magma/doxygen/group__magma__heevdx.html) (standard; symmetric) - Divide-and-conquer algorithm.
    - [magma_?sygvdx](http://icl.cs.utk.edu/projectsfiles/magma/doxygen/group__magma__hegvdx.html) (generalized; symmetric) - See above.

- CuSOLVER

  - Routines (`range = 'I', 'V'`)
    - [cusolverDnXsyevdx](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDnXsyevdx) (standard; symmetric)
    - [cusolverDn\<t\>sygvdx](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-sygvdx) (generalized; symmetric)


