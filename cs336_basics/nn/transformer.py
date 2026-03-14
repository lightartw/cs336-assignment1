import torch
import torch.nn as nn
from .basic import RMSNorm, MultiheadSelfAttention, SwiGLU, Embedding, Linear

class TransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.rmsn1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.MHA = MultiheadSelfAttention(d_model, num_heads, theta, max_seq_len, device, dtype)
        self.rmsn2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.FFN = SwiGLU(d_model, d_ff, device, dtype)
    
    def reset_parameters(self):
        self.rmsn1.reset_parameters()
        self.MHA.reset_parameters()
        self.rmsn2.reset_parameters()
        self.FFN.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.MHA(self.rmsn1(x)) + x
        x = self.FFN(self.rmsn2(x)) + x
        return x

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = None,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device, dtype)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=theta,
                max_seq_len=context_length,
                device=device,
                dtype=dtype
            ) for _ in range(num_layers)
        ])

        self.post_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)
  
    def reset_parameters(self):
        self.embedding.reset_parameters()
        for block in self.blocks:
            assert isinstance(block, TransformerBlock)
            block.reset_parameters()
        self.post_norm.reset_parameters()
        self.lm_head.reset_parameters()
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.post_norm(x)
        logits = self.lm_head(x)

        return logits


