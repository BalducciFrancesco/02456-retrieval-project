import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, mask=None):
    """Scaled dot-product attention.

    The maths of a single head. It only uses the last two dims [T, d_k], so
    with q, k, v of shape [batch, heads, T, d_k] all heads run in one call.
    mask: 1 = real token, 0 = padding; must broadcast to the scores [..., T, T].
    Returns (output [..., T, d_k], weights [..., T, T]).
    """
    # scores[i, j] = how well query token i matches key token j, scaled by
    # sqrt(d_k) so large d_k does not push softmax into near one-hot outputs
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5  # [..., T, T]
    if mask is not None:
        # -inf becomes 0 after softmax, so no token reads from padding
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)  # each row i sums to 1: how token i splits its attention
    return weights @ v, weights              # output row i = weighted average of the value vectors


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int):
        super().__init__()
        assert emb_dim % n_heads == 0, "emb_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.d = emb_dim // n_heads  # numbers per token, per head
        self.W_q = nn.Linear(emb_dim, emb_dim)
        self.W_k = nn.Linear(emb_dim, emb_dim)
        self.W_v = nn.Linear(emb_dim, emb_dim)
        self.W_o = nn.Linear(emb_dim, emb_dim)  # mixes the heads at the end

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, T, _ = x.shape                       # [batch, T, emb_dim]
        x = x.view(batch, T, self.n_heads, self.d)  # [batch, T, heads, d]
        #  transpose done because before the matrix was grouped by token
        #  instead of by head  but q,k,v should be grouped by head
        return x.transpose(1, 2)                    # [batch, heads, T, d]

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        batch, T, _ = x.shape                 # [batch, T, emb_dim]
        q = self.separate_heads(self.W_q(x))  # [batch, heads, T, d]
        k = self.separate_heads(self.W_k(x))  # [batch, heads, T, d]
        v = self.separate_heads(self.W_v(x))  # [batch, heads, T, d]

        mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch, T):
                raise ValueError("attention_mask must have shape (batch, sequence_length)")
            mask = attention_mask.to(device=x.device)
            if not mask.any(dim=1).all():
                raise ValueError("Each sequence must contain at least one non-padding token")
            # mask [batch, T] -> [batch, 1, 1, T], so it covers every head and row
            mask = mask[:, None, None, :]
        # all heads in one call: the heads dim is just another batch dim
        projected_v, weights = scaled_dot_product_attention(q, k, v, mask)  # [batch, heads, T, d], [batch, heads, T, T]
        heads = [projected_v[:, h] for h in range(self.n_heads)]  # n_heads x [batch, T, d]
        z = torch.cat(heads, dim=-1)                               # Concat(head_1..head_h) -> [batch, T, emb_dim]
        return self.W_o(z), weights              # [batch, T, emb_dim], [batch, heads, T, T]
