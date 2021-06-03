"""
Naming convention for matrix variables:
    a_<module>
- module: {pt, sp} whether the matrix is a PyTorch or SciPy object

Sparse matrices can be converted to dense as follows:
    (PyTorch) a.to_dense()
    (SciPy) a.toarray()

"""
import argparse
import numpy as np
from scipy import sparse
import torch
import torch.utils.benchmark as benchmark

parser = argparse.ArgumentParser()
parser.add_argument('--precision', type=str, choices=['single', 'double'], default='double')
parser.add_argument('--format', type=str, choices=['coo', 'csr'], default='csr')
args = parser.parse_args()

# if CSR format is requested, check torch version
if args.format == 'csr':
    if 'sparse_csr_tensor' not in torch.__dict__:
        raise RuntimeError('Cannot use CSR format with this pytorch distribution.')
    import platform
    if 'macOS' in platform.platform():
        raise RuntimeError('CSR matvec is not supported on macOS.')

# if MKL is available, we use int32 CSR indices to enable fast MKL-based matvec
# (ignored when format='coo')
USE_INT32 = torch.backends.mkl.is_available()

# set float type for the experiment.
# NOTE: all numpy arrays in this script originate from torch, so this
# will set the float type for scipy too.
torch.set_default_dtype(torch.float64 if args.precision == 'double' else torch.float32)


# ==== Helper utilities ====

def _torch_from_scipy(a):
    """convert a scipy sparse matrix to a torch sparse matrix"""
    if sparse.isspmatrix_csr(a):
        return torch.sparse_csr_tensor(
            crow_indices=a.indptr.astype(np.int32 if USE_INT32 else np.int64).copy(),
            col_indices=a.indices.astype(np.int32 if USE_INT32 else np.int64).copy(),
            values=a.data.copy(),
            size=a.shape
        )
    elif sparse.isspmatrix_coo(a):
        return torch.sparse_coo_tensor(
            indices=(a.row.copy(), a.col.copy()),
            values=a.data.copy(),
            size=a.shape
        )
    else:
        raise ValueError('invalid input type {} for _torch_from_scipy'.format(type(a)))


def _sample_symmetric(dim, mean=0, std=10, **kwargs):
    """generate a random symmetric matrix"""
    # sample random orthogonal matrix (uniformly)
    Q, R = torch.linalg.qr(torch.randn(dim, dim, **kwargs))
    Q.mul_(R.diagonal().sign().view(1, -1))

    # sample eigenvalues from normal distribution
    e = torch.empty(dim, **kwargs).normal_(mean, std)

    return Q @ torch.diag(e) @ Q.T


# ==== Matrix generators ====

def gen_diag(dim):
    """generate sparse diagonal matrix"""
    diag = torch.randn(dim)
    a_sp = sparse.diags(diag.numpy(), format=args.format)
    a_pt = _torch_from_scipy(a_sp)
    return a_pt, a_sp


def gen_block_diag(dim, num_blocks=2):
    """generate sparse block diagonal matrix"""
    blocks = [_sample_symmetric(dim // num_blocks) for _ in range(num_blocks)]
    a_sp = sparse.block_diag([b.numpy() for b in blocks], format=args.format)
    a_pt = _torch_from_scipy(a_sp)
    return a_pt, a_sp


# ==== Experiment code ====

def run_lobpcg_comparison(label, generator, generator_settings, k=5, largest=True, tol=1e-5):
    label = '{} {} (k={}, largest={})'.format(args.format.upper(), label, k, largest)

    results = []
    for kwargs in generator_settings:
        # generate input matrix
        a_pt, a_sp = generator(**kwargs)

        # use same initial eigenvectors for both scipy and pytorch
        x_pt = torch.randn(a_pt.size(0), k)
        x_sp = x_pt.numpy()

        description = '{:.4e}'.format(a_pt.size(0))

        t1 = benchmark.Timer(
            stmt="torch.lobpcg(a, X=x, largest=largest, tol=tol)",
            setup="import torch",
            globals=dict(a=a_pt, x=x_pt, largest=largest, tol=tol),
            num_threads=torch.get_num_threads(),
            label=label,
            sub_label='torch_lobpcg',
            description=description,
        )

        t2 = benchmark.Timer(
            stmt="lobpcg(a, X=x, largest=largest, tol=tol)",
            setup="from scipy.sparse.linalg import lobpcg",
            globals=dict(a=a_sp, x=x_sp, largest=largest, tol=tol),
            num_threads=torch.get_num_threads(),
            label=label,
            sub_label='scipy_lobpcg',
            description=description,
        )

        results.append(t1.blocked_autorange(min_run_time=1.))
        results.append(t2.blocked_autorange(min_run_time=1.))

    compare = benchmark.Compare(results)
    compare.print()


if __name__ == '__main__':
    print()

    run_lobpcg_comparison(
        label='diag',
        generator=gen_diag,
        generator_settings=[
            {'dim': 2**10},
            {'dim': 2**14},
            {'dim': 2**18},
        ]
    )
    print()

    # in this example we always use a block size of 256
    run_lobpcg_comparison(
        label='block_diag',
        generator=gen_block_diag,
        generator_settings=[
            {'dim': 2**10, 'num_blocks': 2**2},
            {'dim': 2**14, 'num_blocks': 2**6},
            {'dim': 2**18, 'num_blocks': 2**10},
        ]
    )
    print()

