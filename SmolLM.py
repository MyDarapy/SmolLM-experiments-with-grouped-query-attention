class SmolLM(nn.Module):
    def __init__(self):
        super(SmolLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Embedding(block_size, embed_dim) #Learned positional Embeddings
        self.block = nn.Sequential(*[Block() for _ in range(n_blocks)])
        self.layernorm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout_probability)
        self.output_projection = nn.Linear(embed_dim, vocab_size, bias=False)

        # Share weights with the embedding layer
        self.output_projection.weight = self.embeddings.weight




    def forward(self, input, targets=None):
        batch, block_size = input.shape
        token_embeddings = self.embeddings(input)
        position_embeddings = self.positional_encoding(torch.arange(block_size, device=device))
        token = token_embeddings + position_embeddings
        embedded = self.dropout(token)
        x = self.block(embedded)
        x = self.layernorm(x)
        # Final output projection (from embeddings back to tokens). Embedding sharing
        logits = self.output_projection(x)

        if targets is None:
            loss = None
        else:
            b, n, d = logits.shape
            logits = rearrange(logits, 'b n d -> (b n) d')
            targets = rearrange(targets,'b n -> (b n)')
            loss = F.cross_entropy(logits, targets)

        return logits, loss
