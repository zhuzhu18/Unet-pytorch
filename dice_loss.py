import torch
from torch.autograd import Function

class Dicecoeff(Function):
    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 1e-4
        self.inter = torch.dot(input.contiguous().view(-1), target.contiguous().view(-1))
        self.union = torch.sum(input) + torch.sum(target)

        t = (2*self.inter.float()+eps) / self.union.float()
        return t

    def backward(self, gradoutput):
        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = gradoutput * 2 * (target*self.union-self.inter)\
                        / (self.union*self.uion)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

def dice_coeff(input, target, cuda_device):
    if input.is_cuda:
        s = torch.FloatTensor(1).zero_().to(cuda_device)
    else:
        s = torch.FloatTensor(1).zero_()
    for i, c in enumerate(zip(input, target)):
        s += Dicecoeff().forward(c[0], c[1])
    return s / (i+1)
