from typing import Optional, Union, Tuple

import torch.nn.functional as F 
from torch import Tensor, nn
from einops import rearrange, einsum

def grouped_query_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_casual: bool = None, 
    masked_attention: bool = None
    dropout_probability: float = 0.1
    group_query: bool = True
    need_weights: bool = False
    average_attn_weights: bool = False
)

    ''' This function provides support for both multi-head 
    scaled dot product attention and grouped query scaled dot product attention 
    
    
    query: Tensor of size (b, n, h, d) 
    key: Tensor of size (b, s, h, d) 
    value: Tensor of size (b, s, h, d) 

    b: batch size 
    n: sequence_length (query)
    s: sequence length (key/value) why do we have to have different notations?
    h: number of heads
    d: dimension of query, key and value vectors 
    
    
    '''
    ''' Check to ensure attention knows how to handle computation correctly'''
    if (masked_attention is not None) and (is_casual is not None):
        raise ValueError('One of \'mask\' or \'is_casual\' must be None, both is provided')

    elif not query.ndim == value.ndim == key.ndim == 4:
        raise ValueError (
            'Expected q, k, v to have 4 dimensions got dimensions f"query: {query.shape}, f"key: {key.shape}, f"value:{value.shape}')

    query =rearrange(query, "b n h d -> b h n d")
    key = rearrange(key, "b s h d -> b h s d")
    value = rearrange(value, "b s h d -> b h s d")


    bq, nq, hq, dq = query.shape
    bk, sk, hk, dk = key.shape
    bv, sv, hv, dv = value.shape

    if (bq != bk != bv) and (dq != dk != dv):
        raise ValueError('query, key and value must have the same batch size and embedding dimension')

    elif (hk != hv) and (sk != sv):
        raise ValueError('num_of_heads, h in key and value is expected to be the same got different values')

    elif hq % hk != 0:
        raise ValueError("Number of query heads must be a multiple of key/value head")


    #if scale is None:
    scale = key.size(-1)**0.5
    
    g = hq//hk

    query = rearrange(query, 'b (h g) n d) -> b h g n d' )
    attention_scores = einsum(query, key, 'b h g n d, b h s d -> b h g n s')
    scaled_ attention_matrix= attention_scores / scale

    if is_casual:
            mask = torch.torch.ones((bq, nq, sk), device=query.device, dtype=torch.bool).tril_()

    if masked_attention is not None:
        '''rearrange the mask matrix so it matches up with the dimension of the input'''
        if mask.ndim == 2:
            mask.rearrange 
        if mask.ndim == 3:

        scaled_attention_matrix.masked_fill_(~mask, float('-inf')) 
   
    attention_weights = F.softmax(scaled_attention_matrix, dim=-1)

    if dropout > 0:
        attention = nn.Dropout(attention_weights, p=dropout)

    output = einsum(attention, value, 'b g h n s, b  h s d -> b g h n d')

    output = rearrange(output, 'b g h n d -> b n (h g) d')

    attention_weights: Optional = None

    if need_weights:
        attention_weights = rearrange(attention_weights, 'b g h n s -> b n s (h g)')

        if average_attention_weights:
            attention_weights = attention_weights.mean(dim=1)


    return output, attention_weights


class MultiHeadGroupedQueryAttention(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
        query_heads: int,
        kv_matrix_heads: int,
        dropout: float = 0.1,
        bias: bool = True,

     
        ):
        super().__init__(MultiHeadGroupedQueryAttention, self)
        self.embed_dim = embed_dim
        self.query_heads = query_heads
        self.kv_matrix_heads = kv_matrix_heads
        self.dropout = dropout
        self.head_dim = self.embed_dim // self.query_heads

     
        if self.query_heads % kv_matrix_heads != 0:
            raise ValueError('query_heads must be divisible by number of kv heads')
        elif (embed_dim % query_heads != 0) or (embed_dim % kv_matrix_heads != 0):
            raise ValueError(
                'embedding_dim f"{self.embed_dim} must be divisible by query_heads f"{self.query_heads} and key/value heads f"{self.kv_matrix_heads}')
    
        head_dim = self.embed_dim // self.query

        self.q_proj = nn.Linear(embed_dim, head_dim)




    def forward(self, )





