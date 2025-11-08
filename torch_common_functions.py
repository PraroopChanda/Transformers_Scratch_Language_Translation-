import torch
import torch.nn as nn
import math

### ------------------------------------###
'''
Creating embeddings--> basically building a dictionary of words of size (vocab_size, feature_length)
vocab_size is the total vocabulary size (total number of words) and feature_length is number of features for each word
'''
total_words=1000
ft_size_word=512
embedding=nn.Embedding(total_words,ft_size_word)
## embedding(x) in the forward method, where x is the token

### ------------------------------------###


'''
using torch.aranage and unsqueeze
torch.arange(x,y) --> creating a tensor from x to y
torch_vector.unsqueeze(dim=1) --> adding an extra dimension at position dimension
'''
number_vector=torch.arange(0,3)
print(number_vector.shape)
number_vector_dim=number_vector.unsqueeze(dim=1)
print(number_vector_dim.shape)


### ------------------------------------###

'''
nn.Linear --> to build a fully connected layers (input_dim, output_dim, bias:bool=True)
'''
input_dim=128
output_dim=256
linear_layer=nn.Linear(input_dim, output_dim)
## forward method --> linear_layer(x)


### ------------------------------------###




src_tensor=torch.arange(0,3)
sos_token=torch.tensor([1])

src_sos=torch.cat([src_tensor,sos_token])

print("this is the src_token:",src_tensor)
print("this is the sos_token:",sos_token)
print("this is the src_sos:",src_sos)

### ------------------------------------###

enc=torch.tensor([1,2,3,4,5,7,7,7,7])
enc_mask=(enc!=7)

print("this is the enc tokens:",enc)
print("this is the enc_mask:",enc_mask)
print(enc_mask.shape)
enc_mask=enc_mask.unsqueeze(0)
print(enc_mask.shape)
print(enc_mask)
enc_mask=enc_mask.unsqueeze(0)
print(enc_mask.shape)
print(enc_mask)



alpha=torch.arange(0,4)
matrix=torch.ones(2,3,4)

print(alpha*matrix)

x=torch.empty(5,3)
print(x)



d_model=512
denominator=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
print(denominator.shape)
denominator=denominator.unsqueeze(1)

encoding_vector=torch.exp(number_vector*denominator)

print(encoding_vector.shape)


decoder_input=torch.empty(1,1)
print(f"empty decoder input:",decoder_input)

decoder_input.fill_(32)
print(f"Filled decoder input:",decoder_input)

next_word=torch.tensor([[42]])

next_decoder_input=torch.cat([decoder_input,next_word],dim=1)
print(f"this is the next decoder input:{next_decoder_input}")


# '''
# Creating a 1D,2D, 3D  zeros matrix in torch 
# '''
# twod_matrix=torch.zeros(2,3) ## 2 rows 3 columns
# twod_matrix=torch.zeros(2,3,dtype=torch.float64)

# oned_matrix=torch.zeros(3) 

# position =torch.arange(0,5)
# #print(position)
# position_dim=torch.arange(0,5,dtype=torch.float).unsqueeze(1)
# #print(position_dim)

# try_tensor=torch.rand(3,2,2)
# print(try_tensor)
# mean_val=try_tensor.mean(dim=-1,keepdim=True)
# print(mean_val)
# print(mean_val.shape)

# zero_mean=try_tensor-mean_val
# print(zero_mean)
# print(zero_mean.shape)