import torch
from torch.autograd import Function
from torch import Tensor
from typing import Tuple, Optional

from . import _C


class Eigsh(Function):

    @staticmethod
    def forward(ctx, *args):
        e, v, n_iter = _C.eigsh(*args)
        return e, v, n_iter

    @staticmethod
    def backward(ctx, *args):
        raise NotImplementedError('backward is not yet implemented for `eigsh`.')


class EigshMKL(Function):

    @staticmethod
    def forward(ctx, *args):
        e, v = _C.eigsh_mkl(*args)
        return e, v

    @staticmethod
    def backward(ctx, *args):
        raise NotImplementedError('backward is not yet implemented for `eigsh_mkl`.')


def eigsh(A, largest=True, m=20, max_iter=10000, tol=1e-5):
    e, v, n_iter = Eigsh.apply(A, largest, m, max_iter, tol)
    return e, v, n_iter


def eigsh_mkl(A, largest=True, m=20, max_iter=10000, tol_dps=5):
    e, v = EigshMKL.apply(A, largest, m, max_iter, tol_dps)
    return e, v


def eigsh_py(A,
             largest: bool = True,
             m: int = 20,
             max_iter: int = 10000,
             tol: float = 1e-5,
             seed: Optional[int] = None
             ) -> Tuple[Tensor, Tensor, int]:
    """pure-python variant of eigsh"""

    if seed is not None:
        torch.manual_seed(seed)

    n, _ = A.shape
    m = min(n, m)
    dtype = A.dtype
    device = A.device

    # lanczos buffers
    V = torch.zeros(m, n, dtype=dtype, device=device)
    T = torch.zeros(m, m, dtype=dtype, device=device)
    beta = torch.tensor(0., dtype=dtype, device=device)
    r = torch.empty(n, dtype=dtype, device=device)

    # eigh buffers
    eigvals = torch.zeros(m, dtype=dtype, device=device)
    eigvecs = torch.zeros(m, m, dtype=dtype, device=device).t()

    # ritz buffers
    w = torch.zeros(1, dtype=dtype, device=device)
    z = torch.zeros(m, dtype=dtype, device=device)

    # begin restarted Lanczos iterations
    n_iter = 0
    V[0].normal_()
    while n_iter <= max_iter:
        n_iter += 1

        # Perform m-step Lanczos factorization starting at V[0]
        V[0].div_(torch.linalg.norm(V[0]))
        mk = m
        for i in range(m):
            torch.matmul(A, V[i], out=r)
            if i > 0:
                r.sub_(beta * V[i-1])
            T[i, i] = torch.dot(V[i], r)
            r.sub_(T[i, i] * V[i])
            beta = torch.linalg.norm(r)
            if beta < 1e-10:
                mk = i + 1
                break
            if i+1 < m:
                T[i, i+1] = beta
                T[i+1, i] = beta
                torch.div(r, beta, out=V[i+1])

        if mk == 1:
            # Lanczos factorization exited after 1 step.
            # We have quick access to eigval/vec so we will terminate now.
            w[0] = T[0, 0]
            break
        elif mk != m:
            # lanczos factorization exited early.
            # We will compute eigval/vec and exit.
            V = V[:mk]
            T = T[:mk, :mk].contiguous()
            eigvals = eigvals[:mk]
            eigvecs = eigvecs[:mk, :mk].contiguous().t()
            z = z[:mk]

        # compute largest/smallest eigenpair of T
        if torch.jit.is_scripting():
            torch.linalg.eigh(T, eigvals=eigvals, eigvecs=eigvecs)
        else:
            torch.linalg.eigh(T, out=(eigvals, eigvecs))

        # locate eigval and compute projection
        if largest:
            w[0] = eigvals[-1]
            z[:] = eigvecs[:, -1]
        else:
            w[0] = eigvals[0]
            z[:] = eigvecs[:, 0]

        # compute projection of eigvec back to R^n
        V[0] = torch.matmul(z, V)

        if mk != m or beta * z[-1].abs() < tol:
            break

    return w, V[0], n_iter
