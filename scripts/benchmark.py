import argparse
import torch
import torch.utils.benchmark as benchmark
import arpack
try:
    from scipy.sparse import linalg as splinalg
    has_scipy = True
except:
    has_scipy = False

parser = argparse.ArgumentParser()
parser.add_argument('--precision', type=str, choices=['single', 'double'], default='single')
parser.add_argument('--seed', type=int, default=391)
args = parser.parse_args()

DTYPE = torch.float32 if args.precision == 'single' else torch.float64
torch.manual_seed(args.seed)


def eigen_solve(A, mode):
    """solve for eigenpairs using a specified method"""
    if mode == 'arpack_eigsh':
        return arpack.eigsh(A, largest=True, tol=1e-4)
    elif mode == 'arpack_eigsh_mkl':
        return arpack.eigsh_mkl(A, largest=True, tol_dps=4)
    elif mode == 'torch_eigh':
        return torch.linalg.eigh(A)
    elif mode == 'torch_lobpcg':
        return torch.lobpcg(A, k=1, largest=True, tol=1e-4)
    elif mode == 'scipy_eigsh':
        # For some reason scipy's eigsh requires slightly smaller tolerance
        # (1e-5 vs 1e-4) to reach equiavelent accuracy
        return splinalg.eigsh(A.numpy(), k=1, which="LA", tol=1e-5)
    elif mode == 'scipy_lobpcg':
        X = A.new_empty(A.size(0), 1).normal_()
        return splinalg.lobpcg(A.numpy(), X.numpy(), largest=True, tol=1e-4)
    else:
        raise ValueError


def sample_symmetric(dim, mean=0, std=10, **kwargs):
    """generate a random symmetric matrix with a particular eigenvalue distribution"""
    Q, R = torch.linalg.qr(torch.randn(dim, dim, **kwargs))
    Q.mul_(R.diagonal().sign().view(1, -1))
    e = torch.empty(dim, **kwargs).normal_(mean, std)
    A = Q @ torch.diag(e) @ Q.T
    return A


num_threads = torch.get_num_threads()
modes = ['torch_eigh', 'torch_lobpcg', 'arpack_eigsh', 'arpack_eigsh_mkl']
if has_scipy:
    modes += ['scipy_eigsh', 'scipy_lobpcg']
results = []
for dim in [256, 512, 1024, 2048]:
    A = sample_symmetric(dim, dtype=DTYPE)
    for mode in modes:
        res = benchmark.Timer(
            stmt="eigen_solve(A, mode)",
            setup="from __main__ import eigen_solve",
            globals=dict(A=A, mode=mode),
            num_threads=num_threads,
            label='eigensolvers',
            sub_label=mode,
            description='%d' % dim,
        ).blocked_autorange(min_run_time=1.)
        results.append(res)

compare = benchmark.Compare(results)
compare.print()
