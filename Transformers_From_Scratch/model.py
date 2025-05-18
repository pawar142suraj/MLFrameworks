import torch
import torch.nn as nn 
import math 

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #Create a matrix of shae(seq_len,d_model)
        pe = torch.zeros(seq_len,d_model)

        #ccreate a vector of shape (seq_len,1)
        position = torch.arange(0,seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))

        #Apply the sin to even positions
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) #(1,seq_len,d_model) since we will have batch of sentences

        self.register_buffer('pe',pe) #buffer of the module->it will be saved with file along with state of module

    def forward(self,x):
        x = x+(self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6): #eps for numerical stability and avoid divide by zero
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #Multiplied ability to amplify if needed
        self.bias = nn.Parameter(torch.zeros(1)) #Additive ability to add bias

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim= True)
        std = x.std(dim=-1, keepdim = True)

        return self.alpha * (x-mean)/ (std+self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model :int, d_ff:int , dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_ff) -> (Batch, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x)))) 
    
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float)-> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) #Wq
        self.w_k = nn.Linear(d_model, d_model) #Wk
        self.w_v = nn.Linear(d_model, d_model) #Wv

        self.w_o = nn.Linear(d_model,d_model) #Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query,key,value,mask,dropout: nn.Dropout):
        d_k = query.shape[-1]

        #(Batch,h,seq_len,d_k) --> #(Batch,h,seq_len,seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) #(Batch,h,seq_len,seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores

    def forward(self, x, q, k , v, mask):
        query = self.w_q(q) #(Batch, seq_len, d_model) -> (Batch, seq_len, d_model) refer diagram for more info q is seq, d_model and w is d_model, d_model so w will be seq,d_model
        key = self.w_k(k) #(Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        value = self.w_v(v) #(Batch, seq_len, d_model) -> (Batch, seq_len, d_model)

        #divide into smaller matrices so that we can give it to each head
        ##(Batch, seq_len, d_model) -> #(Batch, seq_len, h, d_k) -> #(Batch, h, seq_len, d_k) (each head sees each word in the sentence but only smaller part of embedding)
        query = query.view( query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view( key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view( value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, mask, self.dropout)

        #(Batch,h,seq_len,d_k) --> #(Batch,seq_len,h,d_k) --> #(Batch,seq_len,d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)

        #(Batch,seq_len,d_model) -> (Batch, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):

    def __init__(self, dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block : MultiHeadAttention, feed_forward_block : FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, self.self_attention_block(x,x,x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

