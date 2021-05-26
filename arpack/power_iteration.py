import warnings
import torch
import torch.nn.functional as F


def power_iteration(A, largest=True, tol=1e-6, max_iter=10000, eps=1e-12, check_freq=20):
    """An implementation of Power Iteration for symmetric matrices.

    This version of power iteration is designed for benchmarking against `eigsh`.
    It can be used to find the largest/smallest eigenpair of a symmetric matrix.

    .. note::
        The behavior of power iteration mirrors `eigsh` only for positive
        definite matrices. For indefinite matrices, power iteration with
        `largest=True` locates the eigenpair largest in __magnitude__, and
        the behavior with `largest=False` is undefined.

    """
    v = F.normalize(A.new_empty(A.size(0)).normal_(), dim=0, eps=eps)
    n_iter = 0
    while n_iter < max_iter:
        n_iter += 1
        u = F.normalize(torch.mv(A, v), dim=0, eps=eps)
        if n_iter % check_freq == 0:
            # check convergence
            if torch.abs(1 - torch.abs(torch.dot(v, u))) < tol:
                v = u
                break
        v = u
    else:
        warnings.warn('power iteration did not converge')

    sigma = torch.dot(v, torch.mv(A, v))

    if largest:
        return sigma, v, n_iter

    A = A.clone()
    A.diagonal().sub_(sigma)
    sigma_, v, n_iter_ = power_iteration(
        A, largest=True, tol=tol, max_iter=max_iter,
        eps=eps, check_freq=check_freq)
    sigma += sigma_
    n_iter += n_iter_

    return sigma, v, n_iter
