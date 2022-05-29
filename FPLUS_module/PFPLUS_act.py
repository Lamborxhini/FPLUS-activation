# Author: Boxi D. from SEIT of SYSU, in Canton, China

import torch
import torch.nn as nn
from torch.autograd import Function

class PFPLUSFunction(Function):

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(ctx: torch.tensor, inputs: torch.tensor, lambdas: torch.tensor, mus: torch.tensor) -> torch.tensor:
        ones = torch.ones_like(inputs)
        ctx.save_for_backward(inputs, lambdas, mus, ones)  # param ctx is a context object that can be used to stash information for backward
        outputs = torch.where(inputs < 0, torch.div(lambdas*inputs, (ones - mus*inputs)), lambdas*inputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.tensor) -> torch.tensor:
        inputs, lambdas, mus, ones, = ctx.saved_tensors  # unpack save_tensors and retrieve the info using in the forward pass
        zeros = torch.zeros_like(inputs)
        grad_pfplus = torch.where(inputs < 0, torch.div(lambdas, torch.square(ones - mus * inputs)), lambdas)
        grad_lambda = torch.where(inputs < 0, torch.div(inputs, (ones - mus * inputs)), inputs)
        grad_mu = torch.where(inputs < 0, torch.div((lambdas * inputs * inputs), torch.square(ones - mus * inputs)), zeros)
        return torch.mul(grad_output, grad_pfplus), torch.mul(grad_output, grad_lambda), torch.mul(grad_output, grad_mu)



class PFPLUS(nn.Module):

    def __init__(self, in_channels=None) -> None:
        super().__init__()
        if in_channels:
            self.lambda_factor = torch.nn.Parameter(torch.FloatTensor(1, in_channels, 1, 1), requires_grad=True)
            self.mu_factor = torch.nn.Parameter(torch.FloatTensor(1, in_channels, 1, 1), requires_grad=True)
        else:
            self.lambda_factor = torch.nn.Parameter(torch.FloatTensor(1, 1, 1, 1), requires_grad=True)
            self.mu_factor = torch.nn.Parameter(torch.FloatTensor(1, 1, 1, 1), requires_grad=True)

        '''initialization of lambda and mu parameters'''
        nn.init.constant_(tensor=self.lambda_factor, val=1.0)
        nn.init.constant_(tensor=self.mu_factor, val=1.0)
        self.forward_pass = PFPLUSFunction.apply  # alias the .apply function of the AF class proposed to activate the operation more simply and assign it to a variable

    def forward(self, inputs: torch.tensor) -> torch.tensor:   # abstract the process of forward pass by calling new variable defined above to give a template for invoking in form.
        lambdas = self.lambda_factor
        mus = self.mu_factor
        # print("\namplitude factor lambda:", lambdas)
        # print("scaling factor mu:", mus)
        output = self.forward_pass(inputs, lambdas, mus)  # as we create a new variable above to represent the calling process of the AF proposed, we can now directly invoke this variable with input parameters to start forward propagation
        return output



if __name__ == "__main__":
    input_ = (torch.randn(1, 3, 32, 32, dtype=torch.double, requires_grad=True))
    test = torch.autograd.gradcheck(PFPLUS(), input_, eps=1e-6, atol=1e-4)
    print("gradient computed in the backward-pass check:", test)
