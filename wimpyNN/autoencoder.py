import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import dataloader
import matplotlib.pyplot as plt
import matplotlib as mpl

import time
import copy

from tqdm import tqdm

# Hyper Parameters
batch_size = 256
learning_rate = 0.001
num_epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epoch_loss = 0.0


# SECTION 1: Dataloader

# NOTE Here, MNIST is downloaded as image and target pairs. The transform.ToTensor() converts the image to a tensor of size 28x28, where each element is a pixel value in the range [0, 1]. The target_transform is set to None, which means that the target is not transformed. The download parameter is set to True, which means that the dataset is downloaded if it is not already present in the data directory. The data directory is set to "./data/mnist/".

mnist_train = dsets.MNIST("datasets/", train=True,
                          transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test = dsets.MNIST("datasets/", train=False,
                         transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [
                                                       50000, 10000])

# Now, we set the dataloader.
# NOTE The dataloader is a Python iterator that provides all the functions of an iterator. And it also provides the functions of a Python iterable, which means that you can use it in a for loop. Also note that the dataloader is multi-process data loading, which means that the data will be loaded in parallel using subprocesses. This will make the data loading faster.

# num_workers: how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)
# batch_size: how many samples per batch to load (default: 1). This is the number of images that will be passed to the network at a time, not the total number of batches.

dataloaders = {}
dataloaders["train"] = torch.utils.data.DataLoader(
    mnist_train, batch_size=batch_size, shuffle=True, num_workers=2)
dataloaders["val"] = torch.utils.data.DataLoader(
    mnist_val, batch_size=batch_size, shuffle=False, num_workers=2)
dataloaders["test"] = torch.utils.data.DataLoader(
    mnist_test, batch_size=batch_size, shuffle=False, num_workers=2)

# SECTION 2: Model Architecture

# NOTE Here, we build our own autoencoder model; we will use the nn.Sequential() function to build the model. The nn.Sequential() function takes a list of modules as input and returns a sequential container. The modules will be added to the container in the order they are passed in the constructor.

# NOTE In this case, we build a simple architecture with 28*28 = 784 input nodes >> 100 >> 30 >> 100 >> 784 output nodes. Thus, 30 becomes the bottleneck of the autoencoder, which means that the input is compressed to 30 nodes, and then decompressed to 784 nodes. I get a 30-Dimensional latent space, that is non-euclidean in nature.


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # NOTE What super() does is that it returns a temporary object of the superclass that allows you to access methods of the superclass that have been overridden in a method of the subclass. In this case, the superclass is nn.Module, and the subclass is Autoencoder. If we had not used super(), we would have had to write nn.Module.__init__(self) instead. We pass Autoencoder as the first argument to super() because we want to invoke the constructor of the superclass that corresponds to Autoencoder, which is nn.Module. The second argument is self, which is the instance of the Autoencoder class.
        # NOTE Internally, what super() does is that it calls the __init__() method of nn.Module, which is the superclass of Autoencoder. When we pass self as the second argument, it binds the instance of Autoencoder to the self parameter of the __init__() method of nn.Module. Thus, the __init__() method of nn.Module is called with the instance of Autoencoder as the self parameter.

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 100),
            nn.ReLU(True),
            nn.Linear(100, 30),
            nn.ReLU(True)
        )

        # NOTE We pass RELU as the activation function because it is a non-linear activation function, and it is also computationally efficient. If we had not used RELU, we would have had to use a sigmoid function, which is computationally expensive. If we had not used an activation function at all, and just used a linear function, then the network would have been equivalent to a linear regression model.

        self.decoder = nn.Sequential(
            nn.Linear(30, 100),
            nn.ReLU(True),
            nn.Linear(100, 28*28),
            nn.ReLU(True)
        )

    def forward(self, x):
        """
        Forward pass of the autoencoder.

        The Forward pass takes a batch of images as input, and returns the reconstructed images as output.
        The input is passed through the encoder, and then the output of the encoder is passed through the decoder.
        We sometimes also write a backward() function, which is the backward pass of the autoencoder.
        In this case, we do not need to write the backward() function because the backward pass is automatically defined by PyTorch -- we just need to write the forward() function. In some cases, we may need to write the backward() function ourselves because the backward pass is not automatically defined by PyTorch.
        Pytorch doesn't know how to compute gradients for our custom modules, so it needs to know how to do it.
        """

        batch_size = x.size(0)  # size of x is (batch_size, 1, 28, 28)
        x = x.view(batch_size, -1)  # size of x is (batch_size, 784)

        # We reshape the tensor to (batch_size, 784) because the encoder takes a tensor of size (batch_size, 784) as input, and not a grid of pixels. That seems counter-intuitive, but that's how it is.

        # NOTE The view() function is used to reshape the tensor. In this case, we reshape the tensor to (batch_size, 784). The -1 means that the size of the second dimension is recombined from the other dimensions.

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # We now have an x that is of size (batch_size, 784). We need to reshape it to (batch_size, 1, 28, 28) so that it can be passed to the loss function.

        # size of x is (batch_size, 1, 28, 28)
        out = decoded.view(batch_size, 1, 28, 28)

        # We return the reconstructed images, by viewing the tensor as a grid of pixels.

        return out, encoded


# SECTION 3: Loss function, and Optimizer

model = Autoencoder().to(device)
loss_function = nn.MSELoss()

# In autoencoders, we use something called reconstruction loss, which is the mean squared error between the input and the output. We use the nn.MSELoss() function to compute the mean squared error. We could also use the nn.BCELoss() function to compute the binary cross-entropy loss, but we use the nn.MSELoss() function because it is more numerically stable. The BCELoss() is different from MSELoss() because it is used for binary classification, and not for regression.

optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

# Lot to decode here. We use the SGD optimizer, which is the Stochastic Gradient Descent optimizer. We pass the parameters of the model to the optimizer, and we also pass the learning rate, momentum, and weight decay. The momentum is used to accelerate SGD in the relevant direction and dampens oscillations. The weight decay is a regularization term that is added to the loss function. It is used to prevent overfitting. The weight decay is also known as L2 regularization.

# How weight decay works is that it adds a term to the loss function that is proportional to the square of the magnitude of the weights. Thus, the loss function is penalized for having large weights. This is done to prevent overfitting. Numerically, the weight decay term is added to the loss function as follows: loss = loss + weight_decay * sum(w^2) for all weights w.

# For L1 regularization, we use the L1Loss() function, which is also known as the L1 norm. The L1 norm is the sum of the absolute values of the weights. Thus, the loss function is penalized for having large weights. This is done to prevent overfitting. Numerically, the L1 regularization term is added to the loss function as follows: loss = loss + weight_decay * sum(abs(w)) for all weights w.

# SECTION 4: Training the model


def train(model: nn.Module, train_loader: dataloader, optimizer: torch.optim.Optimizer, loss_function: nn.Module, epoch: int):
    """
    Trains the model for one epoch.

    Parameters
    ----------
    model: nn.Module
        The model to train.
    train_loader: torch.DataLoader
        The training data loader.
    optimizer: torch.optim.Optimizer
        The optimizer to use.
    loss_function: nn.Module
        The loss function to use.
    epoch: int
        The epoch number.
    """

    train_loss_history = []
    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    epoch_loss = 0

    # We initialize the best loss to infinity, so that the first loss that we get is always less than the best loss.
    # What deepcopy does is that it creates a copy of the model's state dictionary, and stores it in best_model_wts.

    for epoch in range(num_epochs):
        """
        The training loop.

        We iterate over the training data loader, and pass the images to the model. We then compute the loss, and backpropagate the loss to compute the gradients. We then use the optimizer to update the weights of the model."""

        # We separate each epoch into a training phase and a validation phase. The training phase is where we train the model, and the validation phase is where we evaluate the model.

        for phase in ["train", "val"]:
            """
                PyTorch has a concept of phases. The phases are train and eval.
                The train phase is where we train the model, and the eval phase is where we evaluate the model.
                We use the train() function to set the model to the train phase, and the eval() function to set the model to the eval phase.
                In the train phase, the model is in training mode, and in the eval phase, the model is in evaluation mode. This needs to be done because some layers behave differently in training mode and in evaluation mode. For example, the Dropout layer behaves differently in training mode and in evaluation mode. In training mode, the Dropout layer randomly drops some neurons, and in evaluation mode, the Dropout layer does not drop any neurons.
            """

            if phase == "train":
                model.train()
                phase_text = "Training"
            elif phase == "val":
                model.eval()
                phase_text = "Validate"

            # Iterate over the data.

            for images, labels in tqdm(dataloaders[phase], desc=f"Epoch {epoch + 1}/{num_epochs} ({phase_text}) || Loss {epoch_loss:.4f}"):

                images = images.to(device)
                labels = labels.to(device)

                # We zero the gradients.
                optimizer.zero_grad()
                # What this does is that it sets the gradients of all the weights to zero. This is done because PyTorch accumulates the gradients on subsequent backward passes. This is convenient because we don't want to manually set the gradients to zero at every step.
                # NOTE: This is only done during training, because the model is not evaluated during training.
                # A backward pass is where the gradients are computed. The gradients are computed by backpropagating the loss function. The gradients are stored in the .grad attribute of the weights. The optimizer uses the gradients to update the weights. The optimizer does not compute the gradients. One can access the weights with the .weight attribute of the model, and their grads with the .grad attribute of the weights, and their data with the .data attribute of the weights.

                # Forward pass.
                with torch.set_grad_enabled(phase == "train"):
                    # NOTE: What this does is that it sets the .grad_enabled attribute of the model to True if the phase is train, and False if the phase is eval. This is done because we don't want to compute the gradients during evaluation, because we are not training the model during evaluation. This is done for memory efficiency. The "with" statement is used to set the .grad_enabled attribute of the model to True, and then set it back to False after the "with" statement is executed.

                    outputs, encoded = model(images)
                    loss = loss_function(outputs, images)

                    # Backward pass.
                    if phase == "train":
                        loss.backward()
                        # What this does is that it computes the gradients of the loss function with respect to the weights. The gradients are stored in the .grad attribute of the weights.
                        optimizer.step()
                        # What this does is that it updates the weights of the model using the gradients that were computed in the previous step. Does gradient descent using the gradients that were computed in the previous step.

                # Statistics.
                if phase == "train":
                    train_loss_history.append(loss.item())
                elif phase == "val":
                    val_loss_history.append(loss.item())

        epoch_loss = loss.item()

        if phase == "train":
            train_loss_history.append(epoch_loss)

        elif phase == "val":
            val_loss_history.append(epoch_loss)

        if phase == "val" and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print("Loss: {:.4f}".format(epoch_loss), end="\r")

    print("Training complete with best validation loss {}".format(best_loss))

    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history


# SECTION 5: Testing the model

best_model, train_history, validation_history = train(
    model, dataloaders, optimizer, loss_function, num_epochs)

# save the model
torch.save(best_model.state_dict(), "autoencoder_MNIST.pth")
