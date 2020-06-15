import random
import json
import torch
from model import ChatNeural
from nltk_utils import bag_of_words, tokenize

device = torch.device('cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = 'data.pth'
data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_strings = data['all_strings']
tags = data['tags']
model_state = data['model_state']

model = ChatNeural(input_size, hidden_size, output_size)
model.load_state_dict(model_state)

model.eval()


bot_name = 'Test'
print('You there, wanna chat??')

while True:
    sentence = input('You: ')
    if sentence=='quit':
        break

    sentence = tokenize(sentence)
    
    x = bag_of_words(sentence, all_strings)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)

    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
    	print(f"{bot_name}: Sorry, I do not understand.")