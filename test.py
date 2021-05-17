import torch
import arpack

torch.set_printoptions(sci_mode=False, precision=4)
torch.manual_seed(129)

A = torch.randn(50, 50)
A = A.T @ A

w, v = torch.linalg.eigh(A)

eigval, eigvec, n_iter = arpack.eigsh(A)

print()
print('eigval true: %0.4f - est.: %0.4f' % (w[-1], eigval))
print()
print('eigvec true: {}'.format(v[:5, -1]))
print('eigvec est.: {}'.format(eigvec[:5]))
print('eigvec true/est dot: %0.4f' % torch.dot(eigvec, v[:, -1]))
print()
