import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_head_attention import MultiHeadAttention
import math

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, emb_dim: int, feed_forward_dim: int, vocab_size: int, dropout: float = 0.1, pool_type: str = 'mean'):
        super(TransformerEncoder, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim = emb_dim
        self.feed_forward_dim = feed_forward_dim
        self.vocab_size = vocab_size
        self.pool_type = pool_type
        self.dropout_p = dropout
        self.setup_model()

    def setup_model(self):
        """
        Instantiate the model.
        """
        
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim) if self.vocab_size else None
        self.pos_encoding = PositionalEncoding(self.emb_dim, self.dropout_p, 5000)

        self.model = nn.Sequential(*[
            EncoderLayer(self.emb_dim, self.feed_forward_dim, self.dropout_p, n_heads=self.n_heads)
            for _ in range(self.n_layers)
        ])

    def pool(self, x):
        """
        Given a batch_size x sequence_length x emb_dim tensor, pool the input along the sequence_length dimension 
        Returns: A pooled vector of shape batch_size x emb_dim
        """
        if self.pool_type == 'cls':
            return x[:, 0, :]
        
        elif self.pool_type == 'max':
            return torch.max(x, 1).values
        
        elif self.pool_type == 'mean':
            return x.mean(dim=1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Run a forward pass on the model
        """
        x = self.embedding(x) * math.sqrt(self.emb_dim)
        x = self.pos_encoding(x)

        x = self.model(x)
        return self.pool(x)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, feed_forward_dim, dropout = 0.0, n_heads = 1):
        super(EncoderLayer, self).__init__()

        self.attn = MultiHeadAttention(emb_dim, n_heads)
        #Two layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(emb_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dim, emb_dim)
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # Attention Part
        attn_out, attn_weights = self.attn(x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x

#Test 
# sh = SingleHead(2, 4)

# Q = torch.tensor([[[1., 0.], [1., 1.], [0., 1.]]])
# K = torch.tensor([[[1., 0.], [0., 2.], [1., 1.]]])
# V = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]])

# print(torch.round(sh.self_attention(Q, K, V), decimals=1))

model = TransformerEncoder(n_layers=2, n_heads=1, emb_dim=64, feed_forward_dim=256, vocab_size=1000)
tokens = torch.randint(0, 1000, (8, 20))
print(model(tokens).shape)


"""
class SingleHead(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super(SingleHead, self).__init__()
        self.embed_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.W_q = nn.Linear(emb_dim, hidden_dim) # Randomly initialized weight matrices
        self.W_k = nn.Linear(emb_dim, hidden_dim)
        self.W_v = nn.Linear(emb_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, emb_dim)

    # Calculate the self attention
    def self_attention(self, Q, K, V):
        scores = Q @ K.transpose(-2, -1) # Only transpose the last two dimensions without the batch (B)
        norm_scores = scores / math.sqrt(K.size(-1))
        attention = torch.nn.functional.softmax(norm_scores, dim=-1) @ V
        return attention

    # Run the input to the layer B x L x D through each layer (computes x @ W.T + b) projections
    def forward(self, x):
       Q = self.W_q(x)
       K = self.W_k(x)
       V = self.W_v(x)
       out = self.self_attention(Q, K, V)
       return self.W_o(out)
"""