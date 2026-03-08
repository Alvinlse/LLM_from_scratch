import torch
import torch.nn as nn


# coding all attension mechanism in the transformer architecture, 

input = torch.tensor([[0.43,0.15,0.89],
                      [0.55,0.87,0.66],
                      [0.57,0.85,0.64],
                      [0.22,0.58, 0.33],
                      [0.77,0.25,0.10],
                      [0.05,0.80,0.55]]
) # this input correspond to a str : your journey starts with one step.

# define the attentions weight
att_w = [[torch.dot(input[i], input[j]) for i in range(input.shape[0])] for j in range(input.shape[0])]

# normalize attention weights
att_w = [torch.softmax(torch.tensor(att_w[i]),  dim =0) for i in range(input.shape[0])]

# context vector 
# metohd 1
#context_vector = []

#for i in range(input.shape[0]):
#    c = torch.zeros(input.shape[1])
#    for j in range(input.shape[0]):
#        c += att_w[i][j] * input[j]
#    context_vector.append(c)
#print(context_vector)

# method 2
context_vector = [att_w[i] @ input for i in range(input.shape[0])]
for i in context_vector:
    print(i)

# trainable self attention

query = input
dim_in = input.shape[1] 
dim_out = 2
# weight matrix
torch.manual_seed(123)
query_weight = torch.nn.Parameter(torch.rand(dim_in, dim_out)) 
key_weight = torch.nn.Parameter(torch.rand(dim_in, dim_out)) 
value_weight = torch.nn.Parameter(torch.rand(dim_in, dim_out))  

query_vector = query @ query_weight
print(query_vector.shape)
key_vector = input @ key_weight
print(key_vector.shape)
value_vector = input @ value_weight
print(value_vector.shape)
att_w = query_vector @ key_vector.T
att_w = torch.softmax( att_w/key_vector.shape[1]**0.5, dim =-1)
context_vector = att_w @ value_vector
print(context_vector)

# summarize all the above calculation into a class

class SelfAttention(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SelfAttention, self).__init__()
        query_weight = nn.Linear(dim_in, dim_out, bias=False) 
        key_weight   = nn.Linear(dim_in, dim_out, bias=False)  
        value_weight = nn.Linear(dim_in, dim_out, bias=False) 
    
    def forward(self, input):
        query_vector = input @ query_weight
        key_vector   = input @ key_weight
        value_vector = input @ value_weight
        att_w = query_vector @ key_vector.T
        att_w = torch.softmax( att_w/key_vector.shape[1]**0.5, dim =-1)
        context_vector = att_w @ value_vector

        return context_vector

# masking the attention weight for model to predict .

