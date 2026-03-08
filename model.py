import torch
import torch.nn as nn
import torch.nn.functional as F


# coding all attension mechanism in the transformer architecture, 

input = torch.randn(2,6,3)
 # this input correspond to a str : your journey starts with one step.

class SelfAttention(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SelfAttention, self).__init__()
        self.query_weight = nn.Linear(dim_in, dim_out, bias=False) 
        self.key_weight   = nn.Linear(dim_in, dim_out, bias=False)  
        self.value_weight = nn.Linear(dim_in, dim_out, bias=False) 
    
    def forward(self, input):
        query_vector = self.query_weight(input)
        key_vector   = self.key_weight(input)
        value_vector = self.value_weight(input)
        att_w = query_vector @ key_vector.transpose(-2,-1)
        att_w = torch.softmax( att_w/key_vector.shape[-1]**0.5, dim =-1)
        context_vector = att_w @ value_vector

        return context_vector

# build a causal attention model class using dropout
# dropout : randomly mask some of the attention weight to prevent overfitting and improve generalization.

class CausalAttention(nn.Module):
    def __init__(self, dim_in, dim_out, dropout = 0.1, context_length = 512):
        super(CausalAttention, self).__init__()
        self.query_weight = nn.Linear(dim_in, dim_out, bias = False)
        self.query_key = nn.Linear(dim_in, dim_out, bias = False)
        self.query_value = nn.Linear(dim_in, dim_out, bias = False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal = 1).bool())

    def forward(self, input):
        batch, num_tokens, dim_in = input.shape
        query_vector = self.query_weight(input)
        key_vector   = self.query_key(input)
        value_vector = self.query_value(input)

        att_score = query_vector @ key_vector.transpose(-2,-1)
        att_score = att_score.masked_fill(self.mask[:num_tokens, :num_tokens] , float('-inf'))
        att_w = torch.softmax(att_score / key_vector.shape[-1]**0.5, dim=-1)
        att_w = self.dropout(att_w)
        context_vector = att_w @ value_vector

        return context_vector

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, dropout = 0.1, context_length = 512):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim_in// num_heads
        self.query_weight = nn.Linear(dim_in, dim_in, bias = False)
        self.query_key = nn.Linear(dim_in, dim_in, bias = False)
        self.query_value = nn.Linear(dim_in, dim_in, bias = False)
        self.w_o = nn.Linear(dim_in, dim_in)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal = 1).bool())

    def forward(self, input):
        batch, num_tokens, dim_in = input.shape
        query_vector = self.query_weight(input).view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1,2)
        key_vector   = self.query_key(input).view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1,2)
        value_vector = self.query_value(input).view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1,2)

        att_score = query_vector @ key_vector.transpose(-2,-1)
        att_score = att_score.masked_fill(self.mask[:num_tokens, :num_tokens] , float('-inf'))
        att_w = torch.softmax(att_score / key_vector.shape[-1]**0.5, dim=-1)
        att_w = self.dropout(att_w)
        context_vector = (att_w @ value_vector).transpose(1,2).contiguous().view(batch, num_tokens, -1)

        return self.w_o(context_vector)

class TransformerBlock(nn.Module):
    def __init__(self, dim_in, dim_out,num_heads, dropout=0.1 , context_length= 512 ):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(dim_in)
        self.attn = MultiHeadAttention(dim_in, dim_out, num_heads, dropout, context_length )
        self.ln2 = nn.LayerNorm(dim_in)
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, 4 * dim_in),
            nn.GELU(),
            nn.Linear(4*dim_in, dim_in),
            nn.Dropout(dropout)
        )
    
    def forward(self,x):
        x = x+self.attn(self.ln1(x))
        x = x+self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, dim_in, num_heads, num_layers, context_length, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim_in)
        self.pos_emb   = nn.Embedding(context_length, dim_in)
        self.blocks    = nn.Sequential(*[
            TransformerBlock(dim_in, dim_in, num_heads, dropout, context_length)
            for _ in range(num_layers)
        ])
        self.ln_f  = nn.LayerNorm(dim_in)
        self.head  = nn.Linear(dim_in, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)

        x = self.token_emb(idx) + self.pos_emb(pos)  # (B, T, d_model)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)                        # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss





        
    



    
