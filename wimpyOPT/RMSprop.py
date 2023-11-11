import numpy as np

class RMSprop:
    """RMSprop optimiser.
    Parameters:
    -----------
    learning_rate: float
        The step length that will be taken when following the negative gradient during
        training.
    decay: float
        The decay rate for the moving average.
    epsilon: float
        A small value used to prevent division by zero.
    """

    def __init__(self, params, learning_rate=0.01, decay=0., epsilon=1e-7):
        self.params = params
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.exp_avg = [np.zeros(param.shape) for param in params]

    def step(self,grads):
        """Update the parameters with the given gradients.
        Parameters
        ----------
        grads : list
        List of gradients for each layer.
        """
        for i in range(len(self.params)):
            self.exp_avg[i] = self.decay * self.exp_avg[i] + (1 - self.decay) * np.square(grads[i])
            self.params[i] -= self.learning_rate * grads[i] / (np.sqrt(self.exp_avg[i]) + self.epsilon)


# ## Sanity check
# import numpy as np
# import torch
# import torch.optim as torch_optim

# # Define your model parameters
# params = [np.random.rand(3, 3), np.random.rand(3)]
# torch_params = [torch.tensor(param, requires_grad=True) for param in params]

# # Create the RMSprop optimizers
# custom_optimizer = RMSprop(params, learning_rate=0.01, decay=0.9)
# torch_optimizer = torch_optim.RMSprop(torch_params, lr=0.01 , alpha=0.9, eps=1e-07 )

# # Compute gradients (Typically done using your model and loss function, but here we'll just make some up)
# grads = [np.random.rand(3, 3), np.random.rand(3)]
# torch_grads = [torch.tensor(grad) for grad in grads]

# # Parameter update using custom RMSprop
# custom_optimizer.step(grads)

# # Parameter update using torch RMSprop
# torch_optimizer.zero_grad()
# for i in range(len(torch_params)):
#     torch_params[i].backward(torch_grads[i])
# torch_optimizer.step()


# print(torch_params)
# print(params)
