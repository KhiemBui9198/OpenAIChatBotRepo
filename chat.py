import random
import json
import openai
import torch
from IPython.display import Markdown
import config
import pinecone
import question
from IPython.display import Markdown
import os

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"


# initialize openai API key
openai.api_key = os.environ["OPENAI_API_KEY"]

embed_model = config.embed_model


messages = [{"role": "system", "content": "You are a financial experts that specializes in real estate investment and negotiation"}]


def get_response(msg):
    res = question.question(msg)
    markdown_str = res.data
    print(markdown_str)
    return markdown_str


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)

