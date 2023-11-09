import numpy as np

class NesterovSGD:
    """Stochastic Gradient Descent with Nesterov Accelerated Gradient.
    Parameters
    ----------
    learning_rate : float
    Learning rate, checks how much the weights are updated at each iteration.
    Default is 0.01.
    n_iter : int
    Number of iterations. Default is 10000.
    """
    def __init__(self, params,learning_rate=0.01,momentum=0.9):
        self.params = params
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = [np.zeros(param.shape) for param in params]
    def step(self,grads):
        """Update the parameters with the given gradients.
        Parameters
        ----------
        grads : list
        List of gradients for each layer.
        """
        for i in range(len(self.params)):
            old_velocity = self.velocity[i]
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grads[i]
            self.params[i] += -self.momentum * old_velocity + (1 + self.momentum) * self.velocity[i]
            
        


## Sanity check
import numpy as np
import torch
import torch.optim as torch_optim

# Define your model parameters
params = [np.random.rand(3, 3), np.random.rand(3)]
torch_params = [torch.tensor(param, requires_grad=True) for param in params]

# Create the SGD optimizers
custom_optimizer = NesterovSGD(params, learning_rate=0.01, momentum=0.9)
torch_optimizer = torch_optim.SGD(torch_params, lr=0.01, momentum=0.9,nesterov=True)

# Compute gradients (Typically done using your model and loss function, but here we'll just make some up)
grads = [np.random.rand(3, 3), np.random.rand(3)]
torch_grads = [torch.tensor(grad) for grad in grads]

# Parameter update using custom SGD
custom_optimizer.step(grads)

# Parameter update using torch SGD
torch_optimizer.zero_grad()
for i in range(len(torch_params)):
    torch_params[i].backward(torch_grads[i])
torch_optimizer.step()


print(torch_params)
print(params)