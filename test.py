import argparse
import torch
import arpack

parser = argparse.ArgumentParser()
parser.add_argument('--mkl', dest='mkl', action='store_true')
args = parser.parse_args()

torch.set_printoptions(sci_mode=False, precision=5)
torch.manual_seed(129)

A = torch.randn(50, 50)
A = A.T @ A
A.diagonal().add_(1e-2)

w, v = torch.linalg.eigh(A)

if args.mkl:
    eigval, eigvec = arpack.eigsh_mkl(A, tol_dps=4)
else:
    eigval, eigvec, n_iter = arpack.eigsh(A, tol=1e-4)

print()
print('eigval true: %0.5f - est.: %0.5f' % (w[-1], eigval))
print()
print('eigvec true: {}'.format(v[:5, -1]))
print('eigvec est.: {}'.format(eigvec[:5]))
print('eigvec true/est dot: %0.5f' % torch.dot(eigvec, v[:, -1]))
print()
