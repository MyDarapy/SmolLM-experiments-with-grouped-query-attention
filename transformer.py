from SwiGLU_FFN import SwiMLP
from grouped_query_attention import MultiHeadGroupedQueryAttention
from Embeddings import  EmbeddingLayer
import torch 
import torch.nn as nn

# Model Hyperparameters
context_length = 512
embed_dim = 896
query_heads = 16
kv_matrix_heads = 3
dropout_probability = 0.1
hidden_size = hidden_size = embed_size * 3
batch_size = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_blocks = 8
eval_iters = 200
evaluation_intervals = 200
vocab_size = 15_000

class Block(nn.Module):
    def __init__(self, embed_dim, query_heads, kv_matrix_heads, dropout_probability=0.1):
        super(Block, self).__init__()
        self.feedforward = SwiMLP(embed_dim, hidden_size, dropout_probability)
        self.multihead_gqa = MultiHeadGroupedQueryAttention(
            embed_dim, query_heads, kv_matrix_heads, dropout_probability)
        self.layernorm= nn.LayerNorm(embed_dim)
    
    def forward(self, input):
        attention_vectors = self.multihead_gqa(input, input, input)
        attention_vectors = attention_vectors + input
        attention_vectors = self.layernorm(attention_vectors)
        feedforward_output = self.feedforward(attention_vectors)
        output = self.layernorm(feedforward_output + attention_vectors)
        return output



class SmolLM(nn.Module):
    def __init__(self, vocab_size, n_blocks, block_size):
        super(SmolLM, self).__init__()
        self.embeddings = EmbeddingLayer(vocab_size, embed_dim, block_size)
        self.position_token = EmbeddingLayer(vocab_size, embed_dim, block_size)
        self.block = nn.Sequential([*Block(embed_dim, query_heads, kv_matrix_heads, dropout_probability=0.1) for _ in range(n_blocks)])
        self.layernorm = nn.LayerNorm(embed_dim) 
        self.output_projection.weight = self.embedding.weight  # Weight sharing

    
    def forward(self, input, target=None):
        batch, block_size = input.shape
        embedding = self.embeddings(input) + self.position_token(input)
        embedded = self.dropout(embedding)
        x = self.block(embedded)
        # Final output projection (from embeddings back to tokens). Embedding sharing
        logits = self.output_projection(x)

        if target is None:
            loss = None
        else: 
            b, n, d = logits.shape
            logits = rearrange(logits, 'b n d -> (b n) d')
            targets = rearrange(targets,'b n -> (b n)')
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss





