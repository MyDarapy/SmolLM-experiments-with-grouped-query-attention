import torch 
import torch.nn as nn

#Implement Sinusoidial position encoding
def get_position_encoding():
    pass 


block_size = 2048
embed_dim = 576
query_heads = 9
kv_matrix_heads = 3
dropout_probability = 0.1
hidden_size = 1536
batch_size = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_blocks = 8
eval_iters = 200
evaluation_intervals = 200
vocab_size = 5000

class EmbeddingLayer(nn.Module):
    def __init__ (self, vocab_size, embed_dim, block_size):
        super(EmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.block_size = block_size
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Embedding(block_size, embed_dim) #Learned Embeddings




    def forward(self, input_ids):
        batch, block_size = input_ids.shape 
        token_embeddings = self.embeddings(input_ids)
        position_embeddings = self.positional_encoding(torch.arange(block_size, device=device))

        token = token_embeddings + position_embeddings
        return token


class SkipConnections():
    pass
    

