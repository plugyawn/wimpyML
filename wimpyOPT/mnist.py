import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
import time

# Define a simple deep neural network
class DeepModel(nn.Module):
    def __init__(self):
        super(DeepModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Custom RMSprop optimizer
class RMSprop:
    def __init__(self, params, learning_rate=0.01, decay=0.9, epsilon=1e-7):
        self.params = list(params)  # Convert generator to list
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.exp_avg = [torch.zeros_like(param) for param in self.params]

    def step(self, grads):
        for i in range(len(self.params)):
            self.exp_avg[i] = self.decay * self.exp_avg[i] + (1 - self.decay) * grads[i] ** 2
            self.params[i].data -= self.learning_rate * grads[i] / (torch.sqrt(self.exp_avg[i]) + self.epsilon)

# Load MNIST data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform),
                                           batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, download=True, transform=transform),
                                          batch_size=64, shuffle=False)

# Instantiate the model and the optimizer
model_custom = DeepModel()
model_torch = DeepModel()
criterion = nn.CrossEntropyLoss()
custom_optimizer = RMSprop(params=model_custom.parameters(), learning_rate=0.001, decay=0.9, epsilon=1e-7)
torch_optimizer = optim.RMSprop(params=model_torch.parameters(), lr=0.001, alpha=0.9, eps=1e-7)

# Speed test parameters
iterations = 5 # Adjust the number of iterations as needed

# Speed test for custom optimizer
start_time = time.time()
for epoch in range(iterations):
    model_custom.train()
    for data, target in train_loader:
        model_custom.zero_grad()
        output = model_custom(data.view(-1, 28 * 28))
        loss = criterion(output, target)
        loss.backward()

        grads = [param.grad for param in model_custom.parameters()]
        custom_optimizer.step(grads)
end_time = time.time()
custom_time = end_time - start_time

# Speed test for torch.optim.RMSprop
start_time = time.time()
for epoch in range(iterations):
    model_torch.train()
    for data, target in train_loader:
        model_torch.zero_grad()
        output = model_torch(data.view(-1, 28 * 28))
        loss = criterion(output, target)
        loss.backward()
        torch_optimizer.step()
end_time = time.time()
torch_time = end_time - start_time

# Evaluation on the test set
model_custom.eval()
model_torch.eval()

with torch.no_grad():
    custom_predictions = []
    torch_predictions = []
    labels = []

    for data, target in test_loader:
        output_custom = model_custom(data.view(-1, 28 * 28))
        output_torch = model_torch(data.view(-1, 28 * 28))

        _, pred_custom = torch.max(output_custom, 1)
        _, pred_torch = torch.max(output_torch, 1)

        custom_predictions.extend(pred_custom.numpy())
        torch_predictions.extend(pred_torch.numpy())
        labels.extend(target.numpy())

# Compare the accuracy
accuracy_custom = accuracy_score(labels, custom_predictions)
accuracy_torch = accuracy_score(labels, torch_predictions)

print(f"Accuracy with Custom RMSprop: {accuracy_custom:.4f}")
print(f"Accuracy with torch.optim.RMSprop: {accuracy_torch:.4f}")
print(f"Custom RMSprop Training Time for {iterations} iterations: {custom_time:.2f} seconds")
print(f"torch.optim.RMSprop Training Time for {iterations} iterations: {torch_time:.2f} seconds")
