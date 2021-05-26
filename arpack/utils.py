import torch

__all__ = ['sample_symmetric']


def sample_symmetric(dim, mean=0, std=10, **kwargs):
    """generate a random symmetric matrix with a particular eigenvalue
    distribution"""
    # sample random orthogonal matrix (uniformly)
    Q, R = torch.linalg.qr(torch.randn(dim, dim, **kwargs))
    Q.mul_(R.diagonal().sign().view(1, -1))

    # sample eigenvalues from normal distribution
    e = torch.empty(dim, **kwargs).normal_(mean, std)

    return Q @ torch.diag(e) @ Q.T
