import torch
from torch import Tensor
from typing import Optional, Iterable
import math

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    max_val = torch.max(x, dim=dim, keepdim=True)[0]
    
    x_exp = torch.exp(x - max_val)
    partition_function = torch.sum(x_exp, dim=dim, keepdim=True)
    
    return x_exp / partition_function
    
def scaled_dot_product_attention(
    query: Tensor, 
    key: Tensor, 
    value: Tensor, 
    mask: Optional[Tensor] = None
) -> Tensor:
    """
    Args:
        query: (batch_size, ..., n, d_k)
        key:   (batch_size, ..., m, d_k)
        value: (batch_size, ..., m, d_v)
        mask:  shape (n, m) or broadcastable to (..., n, m).
               True means "attend", False means "ignore".
    Returns:
        (batch_size, ..., n, d_v)
    """
    d_k = query.size(-1)
    score = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        score = score.masked_fill(mask == False, float('-inf'))

    return torch.matmul(softmax(score, dim=-1), value)


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Args:
        logits: (batch_size, ..., vocab_size)
        targets: (batch_size, ...)
    """
    logits = logits.float()

    max_val = torch.max(logits, dim=-1, keepdim=True)[0]
    sum_exp = torch.sum(torch.exp(logits - max_val), dim=-1, keepdim=True)
    lse = max_val + torch.log(sum_exp)

    correct_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1))
    loss_per_token = -(correct_logits - lse)
    return loss_per_token.mean()

def learning_rate_schedule(t: int, amax: float, amin: float, tw: int, tc: int) -> float:
    if t < tw:
        at = (t / tw) * amax
    elif t >= tw and t <= tc:
        cosine = math.cos(((t - tw) / (tc - tw)) * math.pi)
        at = amin + 0.5 * (amax - amin) * (1 + cosine)
    else:
        at = amin
    return at

def gradient_clipping(params: Iterable[torch.nn.Parameter], maxl2norm: float, eps=10e-6):
    total_norm = torch.tensor(0.0)
    for p in params:
        if p.grad is not None:
            total_norm += torch.sum(p.grad.data ** 2)
    total_norm = torch.sqrt(total_norm)

    if total_norm > maxl2norm:
        scale = maxl2norm / (total_norm + eps)
        for p in params:
            if p.grad is not None:
                p.grad.data.mul_(scale)