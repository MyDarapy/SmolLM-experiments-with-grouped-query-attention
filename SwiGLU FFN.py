import torch 
import torch.nn as nn

class SwiMLP(nn.Module):
    def __init__(self, embed_dim, hidden_size, dropout_probability=0.1):
        super(SwiMLP, self).__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, embed_dim)
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
        return output
