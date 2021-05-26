import torch
from torch.autograd import Function

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
