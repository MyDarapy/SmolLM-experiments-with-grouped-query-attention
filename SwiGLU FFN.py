class SwiMLP(nn.Module):
    def __init__(self, embed_dim, hidden_size, embed_dim):
        super(SwiMLP, self).__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, embed_dim)
        self.layernorm= nn.LayerNorm(embed_dim)

    def forward(self, )