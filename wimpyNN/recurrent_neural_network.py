import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pad_sequence
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import dataloader
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib as mpl

import time
import copy
import os
from collections import Counter

from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1
input_size = 28
sequence_length = 200
num_layers = 2
hidden_size = 28
output_size = 28
num_directions = 1

# The input_size is the number of features in the input x. For a paragraph, the input_size is the number of words in the vocabulary. For an image, the input_size is the number of pixels in the image. For a time series, the input_size is the number of features in the time series.

# NOTE: The sequence length is the number of time steps in the input sequence. For a paragraph, the sequence length is the number of words in the paragraph. For an image, the sequence length is the number of pixels in the image. For a time series, the sequence length is the number of time steps in the time series. This is the number of times the RNN cell is unrolled.

# The num_layers describes the number of layers in the RNN. For example, if num_layers = 2, the RNN will have two layers. The first layer is the input layer and the second layer is the hidden layer. The hidden layer is the output of the first layer and the input of the second layer. The output of the second layer is the output of the RNN. This goes inside the RNN cell, whereas the sequence_length and input_size go outside the RNN cell.

# NOTE: The hidden_size is the number of features in the hidden state h. For a paragraph, the hidden_size is the number of words in the vocabulary. For an image, the hidden_size is the number of pixels in the image. For a time series, the hidden_size is the number of features in the time series. The input_size and hidden_size may be different from each other when the input and output are different types of data. In this case, they are not.

# NOTE For example,
if True == False:
    x = torch.randn(batch_size, sequence_length, input_size).to(device)
    h0 = torch.randn(num_layers * num_directions,
                    batch_size, hidden_size).to(device)

    recnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, bidirectional=False, batch_first=True).to(device)

    output, hidden = recnn(x, h0)

# Bidirectional means that the RNN is unrolled in both directions. For example, if the sequence is [1, 2, 3, 4, 5], the bidirectional RNN will unroll the sequence in both directions. The first direction is [1, 2, 3, 4, 5] and the second direction is [5, 4, 3, 2, 1]. The output of the bidirectional RNN will be the concatenation of the outputs of the two directions. The output of the bidirectional RNN will be [1, 2, 3, 4, 5, 5, 4, 3, 2, 1]. The output_size of the bidirectional RNN is the sum of the output_size of the two directions. The output_size of the bidirectional RNN is 10.

# This is useful as it allows the RNN to learn the sequence in both directions. For example, if the sequence is [1, 2, 3, 4, 5], the bidirectional RNN will learn that 1 is the first element in the sequence and 5 is the last element in the sequence. The bidirectional RNN will also learn that 5 is the first element in the sequence and 1 is the last element in the sequence. This is useful when the sequence is a sentence. The bidirectional RNN will learn that the first word in the sentence is the subject and the last word in the sentence is the object. The bidirectional RNN will also learn that the last word in the sentence is the subject and the first word in the sentence is the object.

print("Preliminary tests passed. Now we will initialize and train the RNN.")
# The RNN outputs two things: the output of the RNN itself, and the hidden layer. The output of the RNN is just the output of the last layer of the unrolled RNN.

#################### SECTION 1: MODEL ARCHITECTURE ####################

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, bidirectional) -> None:
        super().__init__()
        # Not passing anything into the super class because the super class is nn.Module. In Autoencoder we passed, but here we don't because we are not using the super class's __init__ method. We are just using the super class's methods.

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        self.input_to_hidden = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.input_to_output = nn.Linear(self.input_size + self.hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_param, hidden):
        combined_input = torch.cat((input_param, hidden), 0)
        hidden = self.input_to_hidden(combined_input)
        output = self.input_to_output(combined_input)
        output = self.sigmoid(output)

        return output, hidden

    def init_hidden(self, length):
        return torch.zeros(self.hidden_size).to(device)



# The RNN outputs two things: the output of the RNN itself, and the hidden layer. The output of the RNN is just the output of the last layer of the unrolled RNN, passed through a fully-connected layer. We pass it through a fully-connected layer because the output of the RNN is a sequence of vectors. We want to convert the sequence of vectors into a sequence of scalars.

################ SECTION 2: PREPROCESSING THE DATA ################

punctuation = """!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\t"""

with open("./datasets/sentiment_analysis/reviews.txt", 'r') as f:
    reviews = f.read()
with open("./datasets/sentiment_analysis/labels.txt", 'r') as f:
    labels = f.read()

reviews = reviews.lower()
all_text = ''.join([c for c in reviews if c not in punctuation]) # Newlines stay intact. This is because we want to split by new line.

reviews = all_text.splitlines() # Split by new line. This is because each review is on a new line.
labels = labels.splitlines() # Split by new line. This is because each label is on a new line.
# reviews = reviews[:100]
# NOTE: Here, we define our vocabulary.

all_text = ' '.join(reviews)
all_words = set(all_text.split()) # Split by space. This is because each word is separated by a space.
vocabulary_size = len(all_words)
print(vocabulary_size, " is vocabulary size") # ! REMOVE
word_collection = Counter(all_text.split())

# NOTE: We tokenize the reviews. We tokenize the reviews because we want to convert the reviews into a sequence of integers. Each integer represents a word in the vocabulary. We tokenize the reviews because we want to convert the reviews into a sequence of integers. Each integer represents a word in the vocabulary.

# Sort the words by frequency. 
sorted_words = word_collection.most_common(vocabulary_size)
# Returns a list of tuples. The first element of the tuple is the word and the second element of the tuple is the frequency of the word.

vocabulary_to_int = {w: i+1 for i, (w,_) in enumerate(sorted_words)}
# Returns a dictionary. The keys are the words and the values are the indices. The indices start at 1 because 0 is reserved for padding. Notice that i is the index, w is the word, _ is the frequency. 

encoded_label = [1 if label == 'positive' else 0 for label in labels]
encoded_label = torch.tensor(encoded_label).to(device)

reviews_int = []

for review in tqdm(reviews):
    words = review.split()
    one_hot_review = []
    for word in (words):
        word_int = vocabulary_to_int[word]
        one_hot = []
        for _ in range(len(sorted_words)):
            if sorted_words[_][0] == word:
                one_hot.append(1)
            else:
                one_hot.append(0)
        one_hot_review.append(one_hot)
    reviews_int.append(one_hot_review)
print(torch.Tensor(reviews_int[0]).size())


example_reviews = []
for i in reviews_int:
    example_reviews.append(torch.Tensor(i).to(device))

padded_reviews = torch.zeros(len(reviews_int), sequence_length).to(device)
example_padded = pad_sequence(example_reviews, batch_first = True)

padded_length = example_padded.size()[1]

split_fraction = 0.8

split_fraction = 0.8
train_x = example_padded[0:int(split_fraction * len(example_padded))]
train_y = encoded_label[0:int(split_fraction * len(example_padded))]

test_x = example_padded[int(split_fraction * len(example_padded)):]
test_y = encoded_label[int(split_fraction * len(example_padded)):]

train_data = TensorDataset((
    train_x),(train_y))
test_data = TensorDataset((test_x), (test_y))

batch_size = 1
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

vocab_size = (vocabulary_size) + 1  # Q) Why do we add + 1?
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 1

sentiment_net = RNN(input_size = len(sorted_words), hidden_size= hidden_dim, output_size = 1, num_layers = 1, bidirectional= 2).to(device)

lr = 0.005
n_epochs = 20
counter = 0

criterion = nn.BCELoss()
opt = torch.optim.Adam(sentiment_net.parameters(), lr=lr)
h0 = torch.randn(num_layers * num_directions,
                    batch_size, hidden_size).to(device)

check_labels = torch.Tensor([]).to(device)
outputs = torch.Tensor([]).to(device)


for epoch in range(n_epochs + 1):
    first = True
    loss = 0
    acc = 0
    total = 0
    for current_input, label in train_loader:
        hidden = sentiment_net.init_hidden(padded_length)
        current_input = current_input[0]
        for _ in (range(padded_length)):
            # size of current_input[_] is (3861) 
            output, hidden = sentiment_net(torch.Tensor(current_input[_]), hidden)
        
        pred = (output.squeeze() > 0.5).float()
        acc += torch.mean((pred == label).float())
        total += 1

        label = torch.ones(output.squeeze().size()) if label == "positive" else torch.zeros(output.squeeze().size())
        label = label.to(device)
        # torch
        # print(output.squeeze(), label, " is __")
        if first:
            outputs = output
            check_labels = label
            first = False
            print(outputs, check_labels, " is __")

        outputs = torch.cat((outputs, output), 0)
        check_labels = torch.cat((check_labels, label), 0)
    loss = criterion(outputs.squeeze(), check_labels)
    # print(loss)

    opt.zero_grad()
    loss.backward()
    opt.step()

    # sentiment_net.eval()
    # test_losses = []
    # test_acc = []

    # for inputs, labels in test_loader:
    #     output = sentiment_net(inputs)
    #     test_loss = criterion(output.squeeze(), label)
    #     test_losses.append(test_loss.item())

    #     pred = (output.squeeze() > 0.5).float()
    #     acc = torch.mean((pred == labels).float())
    #     test_acc.append(acc.item())

    # sentiment_net.train()
    print("Epoch: {}/{} ".format(epoch, n_epochs),
            "Step: {} ".format(counter),
            "Loss: {:.6f} ".format(loss.item()),
            "Acc: {:.6f} ".format(acc.item()/total),
            # "Test Loss: {:.6f} ".format(np.mean(test_losses)),
            # "Test Acc: {:.6f} ".format(np.mean(test_acc))
            )

