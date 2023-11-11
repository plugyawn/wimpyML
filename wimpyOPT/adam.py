import numpy as np


class Adam():
    """Adam optimizer.
    ----------
    Parameters:
    learning_rate: float
    Learning rate, checks how much the weights are updated at each iteration.
    default is 0.001.
    beta_1: float
    Exponential decay rate for the first moment estimates.
    default is 0.9.
    beta_2: float
    Exponential decay rate for the second moment estimates.
    default is 0.999.
    epsilon: float
    A small constant for numerical stability.
    default is 1e-7.
    """

    def __init__(self, params, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.params = params
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = [np.zeros(param.shape) for param in params]
        self.v = [np.zeros(param.shape) for param in params]

    def step(self, grads):
        """Update parameters.
        ----------
        Parameters:
        grads: numpy.ndarray
        Gradients of the parameters.
        """
        for i in range(len(self.params)):
            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * grads[i]
            self.v[i] = self.beta_2 * self.v[i] + \
                (1 - self.beta_2) * grads[i]**2
            m_hat = self.m[i] / (1 - self.beta_1)
            v_hat = self.v[i] / (1 - self.beta_2)
            self.params[i] -= self.learning_rate * \
                m_hat / (np.sqrt(v_hat) + self.epsilon)

# # ## Sanity check
# import numpy as np
# import torch
# import torch.optim as torch_optim

# # Define your model parameters
# params = [np.random.rand(3, 3), np.random.rand(3)]
# torch_params = [torch.tensor(param, requires_grad=True) for param in params]

# # Create the Adam optimizers
# custom_optimizer = Adam(params, learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
# torch_optimizer = torch_optim.Adam(torch_params, lr=0.01, betas=(0.9, 0.999), eps=1e-7)

# # Compute gradients (Typically done using your model and loss function, but here we'll just make some up)
# grads = [np.random.rand(3, 3), np.random.rand(3)]
# torch_grads = [torch.tensor(grad) for grad in grads]

# # Parameter update using custom Adam
# custom_optimizer.step(grads)

# # Parameter update using torch Adam
# torch_optimizer.zero_grad()
# for i in range(len(torch_params)):
#     torch_params[i].backward(torch_grads[i])
# torch_optimizer.step()


# print(torch_params)
# print(params)
