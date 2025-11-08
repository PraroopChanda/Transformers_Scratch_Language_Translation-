import numpy as np
import torch
import torch.nn as nn
import math 
import torch.nn.functional as F

'''
converting each word/token to a feature vector of given size
goal to make similar words closer in the embedding space
'''
class InputEmbeddings(nn.Module):
    def __init__(self,d_model:int, vocab_size:int)->None:
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        ## (batch,seq_len) --> (batch,seq_len, d_model)
        ## multiplting by d_model as per the paper
        return (self.embedding(x)) * math.sqrt(self.d_model)

'''
This is a non learnable layer, calculate the positional encodings on basis of seq_len and d_model
seq_len is basically the total number of words/token in a sentence
After computing, just add the same to no of words in the sentence(seq_length)
'''

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:int) ->None:
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)

        # defining a matrix (seq_len, d_model)
        PositionalEncoding=torch.zeros(self.seq_len,self.d_model)

        # vector of size (seq_len,1) --> will be used a position in calculating positional encodings
        #adding extra dimension for matrix multiplication
        positions=torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)

        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model)) #(d_model/2)

        # applying embeddings --> even using sin and odd using sin
        # matrix mutiplication (seq_len,1)*(d_model/2) ---> broadcasted to (seq_len,1) *(1,d_model/2)
        PositionalEncoding[:,0::2]=torch.sin(positions*div_term)  # sin( position * (10000 ** (2i/d_model))
        PositionalEncoding[:,1::2]=torch.cos(positions*div_term) # cos( position * (10000 ** (2i/d_model))

        ## adding extra dimension as batch dimension for adding easily --> will automatically broadcast in batch channel during adding in forward
        # (seq_len, d_model) --> (1,seq_len, d_model)
        PositionalEncoding=PositionalEncoding.unsqueeze(0) 

        ## register in the buffer as we want to save it with the model checkpoint and this is not a learnabale part
        self.register_buffer('pe',PositionalEncoding)

    def forward(self,x):
        # here x is (batch, seq_len,d_model)
        x=x+(self.pe[:,:x.shape[1],:]).requires_grad_(False)  # still (batch, seq_len,d_model)
        return self.dropout(x)

'''
FeedForward Layer --> basically a FCN going from d_model --> 2048 --> d_model
using RELU activation
'''
class FeedForwardBlock(nn.Module):
    def __init__(self,d_model:int, d_ff:int, dropout:float)-> None:
        super().__init__()
        self.d_model=d_model
        self.d_ff=d_ff
        self.linear_1=nn.Linear(d_model,d_ff) ## w1 and b1
        self.dropout=nn.Dropout(dropout)
        self.linear_2=nn.Linear(d_ff,d_model) ## w2 and b2

    def forward(self,x):
        ## (batch,seq_len,d_model)-->(batch,seq_len,d_ff) --> (batch,seq_len,d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
'''
Normalizing across the feature dimension --> manually calculating the mean and variance
also adding one addition and multiplication parameter
features --> is basically the number of features --> d_model in this case --> just putting for reference
'''
class LayerNormalization(nn.Module):
    def __init__(self,features:int, eps:float=10**-6)->None:
        super().__init__()
        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(features)) # both are learnable --> alpha for multiplication
        self.bias=nn.Parameter(torch.zeros(features)) # bias for addition

    def forward(self,x):
        # input (batch, seq_len, d_model)
        mean=torch.mean(x,dim=-1, keepdim=True) #(batch, seq_len, 1)
        std=torch.std(x,dim=-1, keepdim=True) #(batch, seq_len, 1)
        normalized_vector=(x-mean)/(std+self.eps)
        return self.alpha * normalized_vector + self.bias  ## here this * is element wise multiplication and not matrix @
    

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,d_model:int, h:int, dropout:float):
        super().__init__()
        self.d_model=d_model
        self.h=h ## no of heads
        assert d_model % h==0,"no of feature should be divisible by no of heads"
        self.d_k=d_model//h ## feature size of each head
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.w_o=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)

    @staticmethod
    def attention(query,key,value, mask,dropout:nn.Dropout):
        #query, key, value -->(batch,h,seq_len,d_k)
        d_k=query.shape[-1]
        attention_scores=query@key.transpose(-2,-1)/ math.sqrt(d_k) # (batch,h, seq_len, seq_len)
        if mask is not None:
            attention_scores.masked_fill(mask==0,-1e9)
        ##--> take softmax across last dimension    
        attention_scores=attention_scores.softmax(dim=-1) # (batch,h, seq_len, seq_len)
        if dropout is not None:
            attention_scores=dropout(attention_scores)
        #(batch,h, seq_len, seq_len) --> (batch,h,seq_len,d_k)
        return (attention_scores @ value),attention_scores       

    def forward(self,q,k,v,mask):
        ## --> multiplying the matrices to get key, query, value
        query=self.w_q(q) # (batch, seq_len,d_model)-->(batch, seq_len,d_model)
        key=self.w_k(k) # (batch, seq_len,d_model)-->(batch, seq_len,d_model)
        value=self.w_v(v) # (batch, seq_len,d_model)-->(batch, seq_len,d_model)

        ## --> splitting across number of features dimension for h number of heads and taking transpose for getting correct indexing
        ## --> Logic of transpose if that each head contains the full sequence so we change the indexing that way (see the figure of splits)
        query=query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2) #(batch, seq_len,d_model)-->(batch,h,seq_len,d_k)
        key=key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2) #(batch, seq_len,d_model)-->(batch,h,seq_len,d_k)
        value=value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2) #(batch, seq_len,d_model)-->(batch,h,seq_len,d_k)

        # calculating attention
        x,self.attention_scores=MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        # combine all heads together (not concatinate add them) --> use view again
        # (batch,h,seq_len,d_k) --> (batch,seq_len,h,d_k) --> (batch,seq_len,d_model)
        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)

        # do final matrix multiplication by output matrix ---> analogy combine all the information from all heads now
        # (batch,seq_len,d_model) --> (batch,seq_len,d_model)
        return self.w_o(x)
        
'''
Now building residual connections
if we see the diagram--> its always input + layer(x) --> and inside the layer there is block(norm(x))
'''        
class ResidualConnections(nn.Module):
    def __init__(self,features:int,dropout:float):
        super().__init__()
        self.norm=LayerNormalization(features)
        self.dropout=nn.Dropout(dropout)

    ## basically sublayer into norm of x
    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))

'''
now building one complete encoder block
'''
class EncoderBlock(nn.Module):
    def __init__(self,features:int, self_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardBlock,dropout:float)-> None:
        super().__init__()
        self.self_attention_block=self_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connections=nn.ModuleList([ResidualConnections(features,dropout) for _ in range(2)])

    def forward(self,x,src_mask):
        x=self.residual_connections[0](x,lambda x:self.self_attention_block(x,x,x,src_mask))
        x=self.residual_connections[1](x,self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self,features:int, layers:nn.ModuleList)->None:
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization(features)
    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)    
    
'''
Now building one decoder block using all the components defined above
'''    
class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float)-> None:
        super().__init__()
        self.self_attention_block=self_attention_block
        self.cross_attention_block=cross_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connections=nn.ModuleList([ResidualConnections(features,dropout) for _ in range(3)])

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x=self.residual_connections[0](x,lambda x:self.self_attention_block(x,x,x,tgt_mask))
        x=self.residual_connections[1](x,lambda x:self.cross_attention_block(x,encoder_output,encoder_output,src_mask)) ## in this case src mask will mask padding tokens after cross attenion 
        x=self.residual_connections[2](x,self.feed_forward_block)
        return x

'''
similarly now building the decoder which would have multiple decoder blocks
'''
class Decoder(nn.Module):
    def __init__(self, features:int, layers:nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization(features)
    def forward(self,x,encoder_output, src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)   
     
'''
Building a projection layer, which basically projects from d_model to vocab_size, 
and then we take a softmax --> for getting the translated word
'''
class Projection_layer(nn.Module):
    def __init__(self,d_model:int, vocab_size:int)->None:
        super().__init__()
        self.projection_layer=nn.Linear(d_model,vocab_size)

    def forward(self,x):
        # (B, seq_len,d_model) --> (B,seq_len, vocab_size)
        return self.projection_layer(x)

'''
Putting it all together and building a transformer
'''

class Transformer(nn.Module):
    def __init__(self,encoder:Encoder,decoder:Decoder,src_embed:InputEmbeddings,tgt_embed:InputEmbeddings,src_pos:PositionalEncoding,tgt_pos:PositionalEncoding,projection_layer:Projection_layer)->None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self,src,src_mask):
        # (batch, seq_len, d_model)
        src=self.src_embed(src)
        src=self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self,encoder_output:torch.Tensor,src_mask:torch.Tensor,tgt:torch.Tensor,tgt_mask:torch.Tensor):
        # (batch, seq_len, d_model)
        tgt=self.tgt_embed(tgt)
        tgt=self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
    
    def project(self,x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048)->Transformer:
    ## creating the embeding layers
    src_embed=InputEmbeddings(d_model,src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # create the positional encodings
    src_pos=PositionalEncoding(d_model,src_seq_len,dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # create the encoder blocks
    encoder_blocks=[]
    for _ in range(N):
        encoder_self_attention_block=MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block=FeedForwardBlock(d_model,d_ff,dropout)
        encoder_block=EncoderBlock(d_model,encoder_self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)     

    ## creating encoder and decoder
    encoder=Encoder(d_model,nn.ModuleList(encoder_blocks))
    decoder=Decoder(d_model,nn.ModuleList(decoder_blocks))

    # final projection layers
    projection_layer=Projection_layer(d_model,tgt_vocab_size)

    # create the transformer
    transformer=Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)

    #Initialize the parameters
    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)

    return transformer               