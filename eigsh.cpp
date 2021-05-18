//
// Created by Reuben Feinman on 5/17/21.
//
#include <torch/extension.h>
#include <c10/util/irange.h>

#include <vector>
#include <cmath>


extern "C" void dstebz_(char *range, char *order, int *n, double *vl, double *vu, int *il, int *iu, double *abstol,
                        double *d, double *e, int *m, int *nsplit, double *w, int *iblock, int *isplit, double *work,
                        int *iwork, int *info);
extern "C" void sstebz_(char *range, char *order, int *n, float *vl, float *vu, int *il, int *iu, float *abstol,
                        float *d, float *e, int *m, int *nsplit, float *w, int *iblock, int *isplit, float *work,
                        int *iwork, int *info);

extern "C" void dstein_(int *n, double *d, double *e, int *m, double *w, int *iblock, int *isplit, double *z, int *ldz,
                        double *work, int *iwork, int *ifail, int *info);
extern "C" void sstein_(int *n, float *d, float *e, int *m, float *w, int *iblock, int *isplit, float *z, int *ldz,
                        float *work, int *iwork, int *ifail, int *info);


namespace arpack {  // begin arpack namespace

    template<class scalar_t>
    void lapackStebz(
            char range, char order, int n, scalar_t vl, scalar_t vu, int il, int iu,
            scalar_t abstol, scalar_t *d, scalar_t *e, int *m, int *nsplit, scalar_t *w,
            int *iblock, int *isplit, scalar_t *work, int *iwork, int *info);

    template<>
    void lapackStebz<double>(
            char range, char order, int n, double vl, double vu, int il, int iu,
            double abstol, double *d, double *e, int *m, int *nsplit, double *w,
            int *iblock, int *isplit, double *work, int *iwork, int *info) {
        dstebz_(&range, &order, &n, &vl, &vu, &il, &iu, &abstol, d, e, m, nsplit,
                w, iblock, isplit, work, iwork, info);
    }

    template<>
    void lapackStebz<float>(
            char range, char order, int n, float vl, float vu, int il, int iu,
            float abstol, float *d, float *e, int *m, int *nsplit, float *w,
            int *iblock, int *isplit, float *work, int *iwork, int *info) {
        sstebz_(&range, &order, &n, &vl, &vu, &il, &iu, &abstol, d, e, m, nsplit,
                w, iblock, isplit, work, iwork, info);
    }

    template<class scalar_t>
    void lapackStein(
            int n, scalar_t *d, scalar_t *e, int m, scalar_t *w, int *iblock, int *isplit, scalar_t *z,
            int ldz, scalar_t *work, int *iwork, int *ifail, int *info);

    template<>
    void lapackStein<double>(
            int n, double *d, double *e, int m, double *w, int *iblock, int *isplit, double *z,
            int ldz, double *work, int *iwork, int *ifail, int *info) {
        dstein_(&n, d, e, &m, w, iblock, isplit, z, &ldz, work, iwork, ifail, info);
    }

    template<>
    void lapackStein<float>(
            int n, float *d, float *e, int m, float *w, int *iblock, int *isplit, float *z,
            int ldz, float *work, int *iwork, int *ifail, int *info) {
        sstein_(&n, d, e, &m, w, iblock, isplit, z, &ldz, work, iwork, ifail, info);
    }

    namespace {  // begin anonymous namespace

        template<typename scalar_t>
        static void apply_eigsh(
                const at::Tensor& A,
                at::Tensor& val,
                at::Tensor& vec,
                int & n_iter,
                const bool largest,
                const int _m,
                const int max_iter,
                const double _tol) {

            scalar_t tol = (scalar_t) _tol;
            scalar_t lanczos_tol = 1.0e-7;
            int n = A.size(0);
            int m = std::min<int>(_m, n);

            // stebz input/output
            char range = 'I';
            char order = 'B';  // todo: use order = 'E'?
            int mk = m;
            scalar_t vl = 0;
            scalar_t vu = 1;
            int il, iu;
            if (largest) {
                il = m;
                iu = m;
            } else {
                il = 1;
                iu = 1;
            }
            scalar_t stebz_tol = 0;
            int nev, nsplit, info;
            at::Tensor w = at::empty({m}, A.options());
            at::Tensor iblock = at::empty({m}, A.options().dtype(at::kInt));
            at::Tensor isplit = at::empty({m}, A.options().dtype(at::kInt));

            // stein input/output
            at::Tensor z = at::empty({1, m}, A.options()).transpose(0, 1);  // column-major order
            int ldz = std::max<int>(1, m);
            at::Tensor ifail = at::empty({std::max<int>(1, m)}, A.options().dtype(at::kInt));

            // work buffers (used by stebz & stein)
            at::Tensor work = at::empty({std::max<int>(1, 5 * m)}, A.options());
            at::Tensor iwork = at::empty({std::max<int>(1, 3 * m)}, A.options().dtype(at::kInt));

            // lanczos buffers
            at::Tensor V = at::zeros({m, n}, A.options());
            at::Tensor a = at::zeros({m}, A.options());
            at::Tensor b = at::zeros({m}, A.options());
            at::Tensor r = at::zeros({n}, A.options());

            // accessors
            auto z_access = z.accessor<scalar_t, 2>();
            auto a_access = a.accessor<scalar_t, 1>();
            auto b_access = b.accessor<scalar_t, 1>();
            auto w_access = w.accessor<scalar_t, 1>();

            // data pointers
            auto a_data = a.data_ptr<scalar_t>();
            auto b_data = b.data_ptr<scalar_t>();
            auto w_data = w.data_ptr<scalar_t>();
            auto z_data = z.data_ptr<scalar_t>();
            auto work_data = work.data_ptr<scalar_t>();
            auto iwork_data = iwork.data_ptr<int>();
            auto iblock_data = iblock.data_ptr<int>();
            auto ifail_data = ifail.data_ptr<int>();
            auto isplit_data = isplit.data_ptr<int>();

            // temporary tensors/scalars
            at::Tensor a_i, b_i;
            scalar_t residual;

            // initialize vec with random normal
            vec.normal_();

            // Perform restarted Lanczos iterations
            n_iter = 0;
            while (n_iter < max_iter) {
                n_iter++;

                mk = m;  // Number of lanczos steps at this iteration (in case we exit early)

                // Perform m-step Lanczos factorization starting at V[0]
                vec.div_(at::norm(vec));
                for (const auto i : c10::irange(m)) {
                    V.index_put_({i}, vec);
                    at::matmul_out(/*out=*/r, A, vec);
                    if (i > 0) {
                        r.sub_(V.index({i-1}), /*alpha=*/b_access[i-1]);
                    }
                    // a.index_put_({i}, at::dot(vec, r));
                    a_i = a[i];
                    at::dot_out(/*out=*/a_i, vec, r);
                    r.sub_(vec, /*alpha=*/a_access[i]);
                    // b.index_put_({i}, at::norm(r));
                    b_i = b[i];
                    at::norm_out(/*out=*/b_i, r, /*p=*/2, /*dim=*/0, /*keepdim=*/false);
                    if (std::isless(b_access[i], lanczos_tol)) {
                        mk = i + 1;  // early exit
                        break;
                    }
                    at::div_out(/*out=*/vec, r, b_i);
                }

                if (mk == 1) {
                    // Lanczos exited after 1 step. We will exit now.
                    val[0] = a_access[0];
                    break;
                } else if (mk != m) {
                    // Lanczos exited early. We will compute val/vec and then exit.
                    if (largest) {
                        il = mk;
                        iu = mk;
                    }
                }

                // Compute the largest/smallest eigval of T
                lapackStebz<scalar_t>(
                        range, order, mk, vl, vu, il, iu, stebz_tol, a_data, b_data,
                        &nev, &nsplit, w_data, iblock_data, isplit_data, work_data,
                        iwork_data, &info);
                TORCH_INTERNAL_ASSERT(info == 0);

                // Compute the corresponding eigvec
                lapackStein<scalar_t>(
                        mk, a_data, b_data, nev, w_data, iblock_data, isplit_data,
                        z_data, ldz, work_data, iwork_data, ifail_data, &info);
                TORCH_INTERNAL_ASSERT(info == 0);

                // store eigval result
                val[0] = w_access[0];

                // compute projection of eigvec back to R^n
                at::matmul_out(/*out=*/vec, z.index({"...", 0}), V);

                // terminate if Lanczos exited early (invariant subspace was reached)
                if (mk != m) {
                    break;
                }

                // terminate if convergence tolerance is reached
                residual = b_access[mk-1] * std::fabs(z_access[mk-1][0]);
                if (std::isless(residual, tol)) {
                    break;
                }
            }
        }

        static void eigsh_kernel(
                const at::Tensor& A,
                at::Tensor& val,
                at::Tensor& vec,
                int & n_iter,
                const bool largest,
                const int m,
                const int max_iter,
                const double tol) {
            AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "eigsh_cpu", [&] {
                apply_eigsh<scalar_t>(A, val, vec, n_iter, largest, m, max_iter, tol);
            });
        }

    } // end anonymous namespace


    /*
      Solve for the extremal eigenpair of a symmetric matrix using restarted Lanczos.

      Parameters:
      * 'A' - symmetric coefficient matrix
      * 'largest' - whether to search for largest or smallest eigenpair
      * 'm' - number of Lanczos vectors to use
      * 'max_iter' - maximum number of iterations allowed
      * 'tol' - convergence tolerance on the norm ||A * x - lambda * x||_2
      Returns:
      * 'val' - solution eigenvalue buffer
      * 'vec' - solution eigenvector buffer
      * 'n_iter' - number of iterations that were run
    */
    std::tuple<at::Tensor, at::Tensor, int> eigsh(
            const at::Tensor& A,
            const bool largest = true,
            const int m = 20,
            const int max_iter = 1000,
            const double tol = 1.0e-4) {

        TORCH_CHECK(!A.is_cuda(), "A should be a CPU tensor.");
        TORCH_CHECK(A.dim() == 2, "A should have 2 dimensions");
        TORCH_CHECK(A.size(0) == A.size(1), "A should be square");

        int n_iter;
        at::Tensor val = at::empty({1}, A.options());
        at::Tensor vec = at::empty({A.size(0)}, A.options());

        eigsh_kernel(A, val, vec, n_iter, largest, m, max_iter, tol);

        return std::make_tuple(val, vec, n_iter);
    }

}  // end eigen namespace
