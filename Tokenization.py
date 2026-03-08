# Tokenization : converting text into a higher dimensional vector space, where each word is represented as a vector of numbers. 
# This process is crucial for NLP , since it allows machine to understand human language .

import re
import tiktoken
# check if the text can be read properly and no words is missing.
with open("the_verdict.txt", 'r', encoding = 'utf-8') as f:
    raw_text = f.read()

# print(raw_text)
# print(len(raw_text))

# Spliting the text into words(tokens)

words = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
words = [ item for item in words if item.strip()]
#print(words)

# convert token into token ID

# sorting words
vocab = sorted(set(words))
vocab = {word: idx for idx, word in enumerate(vocab)}
vocab["<|endoftext|>"] = len(vocab)
vocab["<|unk|>"] = len(vocab)
vocab_size = len(vocab)
print(vocab_size)

# simple way to encode and decode text using vocab that extracted from the dataset(text).
class SimpleTokenizer:
    def __init__(self, vocab):
        self.int_to_str = {i:s for s,i in vocab.items() }
        self.str_to_int = vocab
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item for item in preprocessed if item.strip()]
        return [self.str_to_int.get(token, self.str_to_int["<|unk|>"]) for token in preprocessed]

    def decode(self, token_ids):
        return ' '.join([self.int_to_str[i] for i in token_ids])

tokenizer = SimpleTokenizer(vocab)

text = " Hello world"
tokens = tokenizer.encode(text)
re_text = tokenizer.decode(tokens)
#print(tokens)
#print(re_text)  
# small dataset , it's easy to encounter the unknown words, which make the sentence reconstruction impossible for this method.
# that's why byte pair encoding is introduced, which break down the unknown words into smaller subwords, and then encode them into token IDs.
# this method have been used by many famous LLM like GPT-2 BERT, etc.

# In this project , I decided to use the trained tokenizer same as GPT-2 called tiktokenizer.
tokenizer = tiktoken.get_encoding('gpt2')

#testing
text = "Hello world ."
tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"}) # allow the special token to let the tokenizer recognize ' <|endoftext|>' which is really important for model to learn.
#print(tokens) # [15496, 995]
re_text = tokenizer.decode(tokens)
#print(re_text) # Hello world

tokens = tokenizer.encode(raw_text)
tokens_size = len(tokens)
#print(tokens)

# Now we have finished the tokenization process.


