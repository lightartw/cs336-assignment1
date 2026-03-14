import torch
import torch.nn as nn
import math
from einops import einsum, rearrange

from .util import scaled_dot_product_attention

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0*std, b=3.0*std) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... input, output input -> ... output")


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        input:  (batch_size, sequence_length)
        output: (batch_size, sequence_length, embedding_dim)
        """
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.empty((d_model,), device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)

        return ((x / rms) * self.weight).to(in_dtype)


class RoPE(nn.Module):
    freqs_cis: torch.Tensor 

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # freqs_cis 的形状为 (max_seq_len, d_k / 2)
        freqs_cis = self.precompute_freqs_cis(d_k, max_seq_len, theta, device)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)


    @staticmethod
    def precompute_freqs_cis(dim: int, seq_len: int, theta: float, device=None) -> torch.Tensor:
        """
        code from LLaMA 
        Returns:
            torch.Tensor:  (seq_len, dim / 2)
        """
        powers = torch.arange(0, dim, 2, device=device).float() / dim
        freqs = 1.0 / (theta ** powers)
        t = torch.arange(seq_len, device=device)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        if token_positions is not None:
            freqs_cis_pos = self.freqs_cis[token_positions]
        else:
            freqs_cis_pos = self.freqs_cis[:seq_len] 
        x_pairs = rearrange(x.float(), "... (d k) -> ... d k", k=2)
        
        x_complex = torch.view_as_complex(x_pairs)
        
        x_rotated = x_complex * freqs_cis_pos
        
        x_out = torch.view_as_real(x_rotated)
        x_out = rearrange(x_out, "... d k -> ... (d k)")
        
        return x_out.type_as(x) 

class MultiheadSelfAttention(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            num_heads: int,
            theta: float | None = None,
            max_seq_len: int | None = None,
            device=None,
            dtype=None
        ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads

        self.wq = Linear(d_model, d_model, device, dtype)
        self.wk = Linear(d_model, d_model, device, dtype)
        self.wv = Linear(d_model, d_model, device, dtype)

        self.wo = Linear(d_model, d_model, device, dtype)

        # RoPE
        self.rope = None
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len, device=device)

        self.reset_parameters()

    def reset_parameters(self):
        self.wq.reset_parameters()
        self.wk.reset_parameters()
        self.wv.reset_parameters()
        self.wo.reset_parameters()
    
    def forward(self, x: torch.Tensor):
        b, s, d = x.shape

        q = self.wq(x) 
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(b, s, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(b, s, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(b, s, self.num_heads, self.head_dim).transpose(1,2)

        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        mask = torch.tril(torch.ones(s, s, device=x.device)).bool()
        out = scaled_dot_product_attention(q, k, v, mask=mask)
        
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.wo(out)

        

class SwiGLU(nn.Module):
    def __init__(self, in_features: int, ff_features: int, device=None, dtype=None):
        super().__init__()

        self.d_model = in_features
        self.d_ff = ff_features

        self.linear1 = Linear(
            self.d_model,
            self.d_ff,
            device=device,
            dtype=dtype
        )

        self.linear2 = Linear(
            self.d_ff,
            self.d_model,
            device=device,
            dtype=dtype
        )

        self.linear3 = Linear(
            self.d_model,
            self.d_ff,
            device=device,
            dtype=dtype
        )

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        self.linear3.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.linear1(x)
        gate = gate * torch.sigmoid(gate)
        up_proj = self.linear3(x)

        return self.linear2(gate * up_proj)

