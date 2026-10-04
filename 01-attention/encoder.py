import math

import torch
import torch.nn as nn


class OwnAttention(nn.Module):
    def __init__(self, d_model=512, d_k=64, num_heads=2):
        super().__init__()
        self.d_k = d_k
        self.num_heads = num_heads
        self.W_Q = nn.Linear(d_model, d_k * num_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * num_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_k * num_heads, bias=False)
        self.W_O = nn.Linear(d_k * num_heads, d_model, bias=False)

    def forward(self, x):
        seq_len = x.size(0)
        Q = self.W_Q(x).reshape(seq_len, self.num_heads, self.d_k).permute(1, 0, 2)  # (num_heads, seq_len, d_k)
        K = self.W_K(x).reshape(seq_len, self.num_heads, self.d_k).permute(1, 0, 2)  # (num_heads, seq_len, d_k)
        QK = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (num_heads, seq_len, seq_len)
        attn_weights = torch.softmax(QK, dim=-1)
        V = self.W_V(x).reshape(seq_len, self.num_heads, self.d_k).permute(1, 0, 2)
        Z = torch.matmul(attn_weights, V)
        Z_2 = Z.permute(1, 0, 2).reshape(seq_len, self.d_k * self.num_heads)
        # concat all attention heads
        output = self.W_O(Z_2)
        return output


class BuiltInAttention(nn.Module):
    def __init__(self, d_model=512, d_k=64, num_heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            bias=False,
            batch_first=False,
        )

    def forward(self, x):
        # nn.MultiheadAttention expects (seq_len, batch, d_model) when batch_first=False
        # our x is (seq_len, d_model), so we add a batch dim
        x = x.unsqueeze(1)  # (seq_len, 1, d_model)
        output, attn_weights = self.attn(x, x, x)  # self-attention: Q=K=V=x
        return output.squeeze(1), attn_weights
