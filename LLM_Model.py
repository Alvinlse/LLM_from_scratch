import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SelfAttention, self).__init__()
        self.query_weight = nn.Linear(dim_in, dim_out, bias=False) 
        self.key_weight   = nn.Linear(dim_in, dim_out, bias=False)  
        self.value_weight = nn.Linear(dim_in, dim_out, bias=False) # randonly initialize weights for query , key and value
    
    def forward(self, input):
        query_weight = self.query_weight
        key_weight = self.key_weight
        value_weight = self.value_weight

        query_vector = input @ query_weight  
        key_vector   = input @ key_weight
        value_vector = input @ value_weight    

        att_w = query_vector @ key_vector.T  # define attention weight for each token 
        att_w = torch.softmax( att_w/key_vector.shape[1]**0.5, dim =-1) # normalize 

        context_vector = att_w @ value_vector  # define context vector foor better understanding of words meaning and relation in sentence.

        return context_vector

torch.manual_seed(123)
