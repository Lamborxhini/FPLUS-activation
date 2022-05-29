# Author: Boxi D. from SEIT of SYSU, in Canton, China
# ~~~~~~~~~~~~~~~~~~~~~~~ How to design a custom activation function in Pytorch? ~~~~~~~~~~~~~~~~~~~~~~
# official tutorial URL: https://pytorch.org/docs/1.2.0/notes/extending.html
#                        https://pytorch.org/docs/1.2.0/autograd.html#function
# Chinese tutorial URL: https://www.pytorch123.com/ThirdSection/LearningPyTorch/#32-pytorch

"""
torch.autograd.Function RECORDS operation history and DEFINES formulas for differentiating ops.
    Every operation performed on Tensors creates a new function object, which performs the computation, and records what happened.
The history is retained in the form of a DAG of functions, with edges denoting data dependencies (input <- output).
    Then, when backward is CALLED, the graph is PROCESSED in the topological ordering,
by CALLING backward() methods of each Function object, and PASSING returned gradients ON TO next Function.
"""

import torch
from torch.autograd import Function
import torch.nn as nn

'''
In essence, every original autograd operation is implemented on two functions running on the tensor, 
which are forward method and backward method.
Forward function receives input tensor to computes the output tensor produced by the forward pass from the input tensor, while
backward function receives gradient of the loss relative to the output tensor, so as to compute another gradient of the loss
relative to the input tensor. Eventually according to the chain rule, we multiply two gradients together to attain a final gradient.

In Pytorch, we can easily define a subclass by extending torch.autograd.Function as well as implement forward & backward function, 
so that to define a custom autograd and afterwards we can use this new autograd operator.
Of course we can call this operation by constructing an instance and sending in a tensor containing input data,
like calling a function, so that the new autograd operation can be realized.
'''

'''
-------------------------------------------------------------------------------------------------------------
1.First we gotta customize a new activation function class and have it subclassed to torch.autograd.Function 
-------------------------------------------------------------------------------------------------------------
'''


class FPLUSFunction(Function):  # both forward & backward methods need to be overridden if subclass inherits from torch.autograd.Function
    lambda_factor = 1.0  # controls the slope of the line in positive part and saturation degree of the curve in negative part
    mu_factor = 1.0  # controls the attenuation speed of the curve in negative part

    # In the forward pass,we receive a context object 'ctx' and a tensor 'inputs' containing input data,
    # and we gotta return another tensor 'output' containing output data after propagation forward.
    # Besides we can use context object 'ctx' to cache related data in the forward pass, so that they can be directly used again in the backward pass later
    @staticmethod
    def forward(ctx, inputs: torch.tensor) -> torch.tensor:
        ones = torch.ones_like(inputs)
        ctx.save_for_backward(inputs, ones)  # param ctx is a context object that can be used to stash information for backward computation
        outputs = torch.where(inputs < 0, torch.div(FPLUSFunction.lambda_factor * inputs,
                                                    (ones - FPLUSFunction.mu_factor * inputs)),
                              FPLUSFunction.lambda_factor * inputs)
        return outputs

    # forward() method performs the forward pass operation.
    # This function is to be overridden by all subclasses.
    # It must accept a context ctx as the first argument, followed by any number of arguments (tensors or other types).
    # The context can be used to store tensors that can be then retrieved during the backward pass.

    '''
    According to the chain rule for back propagation, d(loss)/d(input) = d(loss)/d(output) * d(output)/d(input),
    in which grad_output is d(loss)/d(output), and d(output)/d(input) means grad_fplus
    '''

    # In the backward pass, we receive a context object 'ctx' and a tensor 'grad_output' which contains gradient of the loss relative to the output
    # We can retrieve data cached before from the context object 'ctx'
    # In addition, we gotta compute and return corresponding gradient of the output relative to the input in the forward pass,
    # i.e. the derivative of the activation function proposed at the input 'inputs' retrieved from 'ctx', a.k.a. grad_fplus
    @staticmethod
    def backward(ctx, grad_output: torch.tensor) -> torch.tensor:
        inputs, ones, = ctx.saved_tensors  # unpack save_tensors and retrieve the info using in the forward pass
        grad_fplus = torch.where(inputs < 0, torch.div(FPLUSFunction.lambda_factor * ones,
                                                       torch.square(ones - FPLUSFunction.mu_factor * inputs)),
                                 FPLUSFunction.lambda_factor * ones)
        return torch.mul(grad_output, grad_fplus)
    # backward() method defines a formula for differentiating the operation.
    # This function is to be overridden by all subclasses.
    #
    # It must accept a context ctx as the first argument, followed by as many outputs did forward() return,
    # and it should return as many tensors, as there were inputs to forward().
    # Each argument is the gradient w.r.t the given output,
    # and each returned value should be the gradient w.r.t. the corresponding input.
    #
    # The context can be used to retrieve tensors saved during the forward pass.
    # It also has an attribute ctx.needs_input_grad as a tuple of booleans representing whether each input needs gradient.
    # E.g., backward() will have ctx.needs_input_grad[0] = True if the first input to forward() needs gradient computed w.r.t. the output.


'''
-----------------------------------------------------------------------------------------------------------------------
2.Second we gotta encapsulate the new activation function class proposed above into a similar form of nn.Module type
-----------------------------------------------------------------------------------------------------------------------
'''


class FPLUS(nn.Module):  # only forward method needs to be overridden if subclass inherits from torch.nn.Module, just because gradient computation for backward has been automatically operated by invoking a series of functions in torch.nn.Functional inherently.
    def __init__(self) -> None:
        super(FPLUS, self).__init__()
        self.forward_pass = FPLUSFunction.apply  # alias the .apply function of the AF class proposed to activate the operation more simply and assign it to a variable

    def forward(self, inputs: torch.tensor) -> torch.tensor:  # abstract the process of forward pass by calling new variable defined above to give a template for invoking in form.
        output = self.forward_pass(inputs)  # as we create a new variable above to represent the calling process of the AF proposed, we can now directly invoke this variable with input parameters to start forward propagation
        return output


'''
------------------------------------------------------------------------------------------------------------------------------
3.We probably want to check if the backward method we implement actually computes the derivative of our function.
It is possible by comparing with numerical approximations using small finite differences.
More details in https://pytorch.org/docs/1.2.0/autograd.html#grad-check
------------------------------------------------------------------------------------------------------------------------------
'''

if __name__ == "__main__":
    input_ = (torch.randn(20, 20, dtype=torch.double, requires_grad=True))  # The default values are designed for input of double precision
    test = torch.autograd.gradcheck(FPLUS(), input_, eps=1e-6, atol=1e-4)
    print("gradient computed in the backward-pass check:", test)
    # gradcheck() accept a function as the first parameter which takes tensor inputs and returns a tensor or a tuple of tensors as output
    # gradcheck() takes a tensor or a tuple of tensors as second parameter, check if your gradient evaluated with these tensors are close enough to
    # the numerical approximations and return True if they all verified this condition.
    # eps means perturbation for finite differences like negligible error, atol means absolute tolerance.
