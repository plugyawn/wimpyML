import numpy as np

class Adagrad:
    """Adaptive Gradient Algorithm.
    Parameters
    ----------
    learning_rate : float
    Learning rate, checks how much the weights are updated at each iteration.
    Default is 0.01.
    n_iter : int
    epsilon: float, default=1e-8, a small positive constant which ensures that the denominator is never zero.
    Number of iterations. Default is 10000.
    """
    def __init__(self,params,learning_rate=0.01,epsilon=1e-8):
        self.params = params
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.cache = [np.zeros(param.shape) for param in params]
    def step(self,grads):
        """Update the parameters with the given gradients.
        Parameters
        ----------
        grads : list
        List of gradients for each layer.
        """
        for i in range(len(self.params)):
            self.cache[i] += grads[i]**2
            self.params[i] -= self.learning_rate * grads[i] / (np.sqrt(self.cache[i]) + self.epsilon)
    
## Sanity check
import numpy as np
import torch
import torch.optim as torch_optim

# Define your model parameters
params = [np.random.rand(3, 3), np.random.rand(3)]
torch_params = [torch.tensor(param, requires_grad=True) for param in params]

# Create the Adagrad optimizers
custom_optimizer = Adagrad(params, learning_rate=0.01)
torch_optimizer = torch_optim.Adagrad(torch_params, lr=0.01)

# Compute gradients (Typically done using your model and loss function, but here we'll just make some up)
grads = [np.random.rand(3, 3), np.random.rand(3)]
torch_grads = [torch.tensor(grad) for grad in grads]

# Parameter update using custom Adagrad
custom_optimizer.step(grads)

# Parameter update using torch Adagrad
torch_optimizer.zero_grad()
for i in range(len(torch_params)):
    torch_params[i].backward(torch_grads[i])
torch_optimizer.step()


print(torch_params)
print(params)
