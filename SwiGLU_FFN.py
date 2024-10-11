import torch 
import torch.nn as nn



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

class SwiMLP(nn.Module):
    def __init__(self, embed_dim, hidden_size, dropout_probability=0.1):
        super(SwiMLP, self).__init__()
        self.embed_dim = embed_dim

        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(embed_dim, hidden_size)
        self.linear2 = nn.Linear(embed_dim, hidden_size)
        self.linear3 = nn.Linear(hidden_size, embed_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_probability)


    def swiglu(self, x):
        swish = self.sigmoid(x)
        return x*swish

    def forward(self, input):
        x1 = self.linear1(input)
        x2 = self.linear2(input)
        output = x1*self.swiglu(x2)
        output = self.dropout(output)
        output = self.linear3(output)
        return output

