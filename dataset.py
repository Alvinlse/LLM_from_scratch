# From here we are going to sampling the data using sliding window method, which let the model predict the next token based on the previous tokens.
import torch
from torch.utils.data import  dataloader
import tiktoken
from preprocess import embedding

class TextDataset:
    def __init__(self, tokens, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(tokens) - max_length, stride):
            input_ids = tokens[i: i+max_length]
            target_ids = tokens[i+1: i+max_length+1]
            self.input_ids.append(input_ids)
            self.target_ids.append(target_ids)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx]), torch.tensor(self.target_ids[idx])

with open("the_verdict.txt", 'r', encoding='utf-8') as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding('gpt2')
tokens = tokenizer.encode(raw_text)
dataset = TextDataset(tokens, max_length= 4 , stride = 4)
dataloader = dataloader.DataLoader(dataset, batch_size = 8, shuffle=True, drop_last= True, num_workers=0)
for input_ids, target_ids in dataloader:
    print("input_ids: ", input_ids)
    print("target_ids: ", target_ids)
    break

# convert token IDs into embedded vectors for model to learn.
embedding_dim =256
vocab_size = tokenizer.n_vocab
embedded_token = embedding(input_ids, vocab_size, embedding_dim)
print(embedded_token.shape)

# embedding position : words meaning can be vary due to the way it stay in the sentence, by embedding it position we can help the model learn better.

embedding_position = torch.nn.Embedding(vocab_size, embedding_dim)(torch.arange(input_ids.shape[1]))
print(embedding_position.shape)

# embedding_input : the final input for model to flexibly learn the meaning of words and its relation in the sentences.
embedding_input = embedded_token + embedding_position
print(embedding_input.shape) 



