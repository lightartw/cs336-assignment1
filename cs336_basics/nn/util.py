import torch
from torch import Tensor
from typing import Optional


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