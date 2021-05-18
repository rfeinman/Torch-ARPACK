//
// Created by Reuben Feinman on 5/17/21.
//
#include <torch/extension.h>
#include <c10/util/irange.h>

#include <mkl_spblas.h>
#include <mkl_solvers_ee.h>

#include <vector>
#include <cmath>


namespace arpack {  // begin arpack namespace

    template<class scalar_t>
    void mkl_sparse_create_coo(
            sparse_matrix_t *A, sparse_index_base_t indexing, int rows, int cols,
            int nnz, int *row_indx, int *col_indx, scalar_t *values, sparse_status_t *status);

    template<>
    void mkl_sparse_create_coo<double>(
            sparse_matrix_t *A, const sparse_index_base_t indexing, const int rows, const int cols,
            const int nnz, int *row_indx, int *col_indx, double *values, sparse_status_t *status) {
        *status = mkl_sparse_d_create_coo(A, indexing, rows, cols, nnz, row_indx, col_indx, values);
    }

    template<>
    void mkl_sparse_create_coo<float>(
            sparse_matrix_t *A, const sparse_index_base_t indexing, const int rows, const int cols,
            const int nnz, int *row_indx, int *col_indx, float *values, sparse_status_t *status) {
        *status = mkl_sparse_s_create_coo(A, indexing, rows, cols, nnz, row_indx, col_indx, values);
    }

    template<class scalar_t>
    void mkl_sparse_ev(
            char *which, int *pm, sparse_matrix_t A, struct matrix_descr descrA, int k0,
            int *k, scalar_t *E, scalar_t *X, scalar_t *res, sparse_status_t *status);

    template<>
    void mkl_sparse_ev<double>(
            char *which, int *pm, sparse_matrix_t A, struct matrix_descr descrA, int k0,
            int *k, double *E, double *X, double *res, sparse_status_t *status) {
        *status = mkl_sparse_d_ev(which, pm, A, descrA, k0, k, E, X, res);
    }

    template<>
    void mkl_sparse_ev<float>(
            char *which, int *pm, sparse_matrix_t A, struct matrix_descr descrA, int k0,
            int *k, float *E, float *X, float *res, sparse_status_t *status) {
        *status = mkl_sparse_s_ev(which, pm, A, descrA, k0, k, E, X, res);
    }

    namespace {  // begin anonymous namespace

        template<typename scalar_t>
        void apply_eigsh_mkl(
                const at::Tensor& A,
                at::Tensor& val,
                at::Tensor& vec,
                const bool largest,
                const int m,
                const int max_iter,
                const int tol_dps) {

            const int n = A.size(0);
            char which = largest ? 'L' : 'S';
            int k;
            int k0 = 1;  // number of extremal eigenpairs to find (locate top-k0)
            scalar_t res[1];

            // ~~~~~ Set algorithm parameters ~~~~~~
            int pm[128];
            sparse_status_t status = mkl_sparse_ee_init(pm);
            TORCH_INTERNAL_ASSERT(status == 0);

            pm[1] = tol_dps + 1;          // degrees of precision in convergence tolerance (tol = 10^{-k+1})
            pm[2] = 1;                    // use Krylov-Schur method
            pm[3] = std::min<int>(m, n);  // number of Lanczos vectors
            pm[4] = max_iter;             // maximum number of iterations
            // pm[5] = 0;                 // ?power of Chebychev expansion for approximate spectral projector
            pm[6] = 1;                    // whether to compute eigenvectors & eigenvalues (1) or eigenvalues only (0)
            pm[7] = 0;                    // whether to use absolute (1) or relative (0) residual for convergence checks
            pm[8] = 0;                    // whether to use exact residual norm (1) or fast approximation (0)

            // ~~~~~ Convert dense matrix to sparse CSR format ~~~~~
            sparse_matrix_t csrA, cooA;
            at::Tensor rows = at::arange(n, at::kInt).view({n, 1}).repeat({1, n}).flatten();
            at::Tensor cols = at::arange(n, at::kInt).view({1, n}).repeat({n, 1}).flatten();
            mkl_sparse_create_coo(&cooA, SPARSE_INDEX_BASE_ZERO, n, n, n * n, rows.data_ptr<int>(),
                                  cols.data_ptr<int>(), A.data_ptr<scalar_t>(), &status);
            TORCH_INTERNAL_ASSERT(status == 0);
            status = mkl_sparse_convert_csr(cooA, SPARSE_OPERATION_NON_TRANSPOSE, &csrA);
            TORCH_INTERNAL_ASSERT(status == 0);

            struct matrix_descr descrA;
            descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

            // ~~~~~ Execute eigensolver ~~~~~
            mkl_sparse_ev<scalar_t>(&which, pm, csrA, descrA, k0, &k, val.data_ptr<scalar_t>(),
                                    vec.data_ptr<scalar_t>(), res, &status);
            TORCH_CHECK(status == 0, "non-zero sparse status");

        }

        void eigsh_mkl_kernel(
                const at::Tensor& A,
                at::Tensor& val,
                at::Tensor& vec,
                const bool largest,
                const int m,
                const int max_iter,
                const int tol_dps) {
            AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "eigsh_mkl_cpu", [&] {
                apply_eigsh_mkl<scalar_t>(A, val, vec, largest, m, max_iter, tol_dps);
            });
        }

    }  // end anonymous namespace

    /*
      Solve for the extremal eigenpair of a symmetric matrix using MKL's
      Krylov-Schur eigensolver

      Parameters:
      * 'A' - symmetric coefficient matrix
      * 'largest' - whether to search for largest or smallest eigenpair
      * 'm' - number of Lanczos vectors to use
      * 'max_iter' - maximum number of iterations allowed
      * 'tol_dps' - degrees of precision in convergence tolerance on the norm ||A * x - lambda * x||_2
      Returns:
      * 'val' - solution eigenvalue buffer
      * 'vec' - solution eigenvector buffer
    */
    std::tuple<at::Tensor, at::Tensor> eigsh_mkl(
            const at::Tensor& A,
            const bool largest = true,
            const int m = 20,
            const int max_iter = 10000,
            const int tol_dps = 5) {

        TORCH_CHECK(!A.is_cuda(), "A should be a CPU tensor.");
        TORCH_CHECK(A.dim() == 2, "A should have 2 dimensions");
        TORCH_CHECK(A.size(0) == A.size(1), "A should be square");

        at::Tensor val = at::empty({1}, A.options());
        at::Tensor vec = at::empty({A.size(0)}, A.options());

        eigsh_mkl_kernel(A, val, vec, largest, m, max_iter, tol_dps);

        return std::make_tuple(val, vec);
    }

}  // end arpack namespace
