# embedding the tokens ID into a vector( 3D ) for model to learn.
import torch

def embedding( tokens, vocab_size, embedding_dim):
    torch.manual_seed = 42
    embedding_matrix = torch.nn.Embedding(vocab_size, embedding_dim)
    return embedding_matrix(tokens)

