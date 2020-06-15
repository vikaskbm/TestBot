import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import stem, tokenize, bag_of_words
from model import ChatNeural

# strings that are not words but puntuations
ignore_words = ['?', '!', ',', '.']

# tags list contains all the tags in our intents.json
tags = []

# all_strings contain all the words that are in our json file
all_strings = []

# ls is  alist of tuples containing pattern and corrosponding tag
ls = []

# x_train and y_train holds the data which serves as training data for the model
x_train = []
y_train = []


# Preparing data for training
with open('intents.json', 'r') as f:
    intents = json.load(f)

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_strings.extend(w)
        # add to ls pair
        ls.append((w, tag))

all_strings = [stem(w) for w in all_strings if w not in ignore_words]
all_strings = sorted(set(all_strings))
tags = sorted(set(tags))

for (pattern_sentence, tag) in ls:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_strings)
    x_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)


# TRAINING THE MODEL
# Defining Hyperparameters for deep learning
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)

# extending Dataset class for our data
class ChatBotDataSet(Dataset):

    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples




dataset = ChatBotDataSet()
train_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cpu')

model = ChatNeural(input_size, hidden_size, output_size)

# Loss and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_dl:
        words = words.to(device)
        labels = labels.to(device)

        # getting outputs, forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        # labels is of int64 datatype
        loss = criterion(outputs, labels.long())

        # backwards and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')


data = {
    'model_state': model.state_dict(),
    'input_size':input_size,
    'output_size': output_size,
    'hidden_size': hidden_size,
    'all_strings': all_strings,
    'tags': tags,
}

FILE = 'data.pth'
torch.save(data, FILE)
print(f'training complete and file saved to {FILE}')