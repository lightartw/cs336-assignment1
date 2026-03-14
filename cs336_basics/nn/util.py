import os
import typing
import torch
from torch import Tensor
from typing import Optional, Iterable, Tuple
import math
import numpy as np
import numpy.typing as npt
from pathlib import Path

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


# ============================== optimizer ================================

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

@torch.no_grad()
def gradient_clipping(params: Iterable[torch.nn.Parameter], maxl2norm: float, eps=10e-6):
    grads = tuple(p.grad for p in params if p.grad is not None)
    if not grads:
        return

    square_sum = torch.tensor(0.0, device=grads[0].device)
    for g in grads:
        square_sum += torch.sum(g ** 2)
    
    norm_all = torch.sqrt(square_sum)
    if norm_all > maxl2norm:
        clip_coef = maxl2norm / (norm_all + eps)
        for g in grads:
            g.mul_(clip_coef)

# ================================ training loop ==========================

def load_batch(
    dataset: npt.NDArray, 
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_index = len(dataset) - context_length - 1
    ix = torch.randint(max_index + 1, (batch_size, )).tolist()
    x_tensors = [torch.tensor(dataset[i : i + context_length], dtype=torch.long) for i in ix]
    y_tensors = [torch.tensor(dataset[i + 1 : i + 1 + context_length], dtype=torch.long) for i in ix]
    
    x = torch.stack(x_tensors).to(device)
    y = torch.stack(y_tensors).to(device)
    return x, y

def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    iteration: int, 
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    checkpoint = dict(model_state=model_state, optimizer_state=optimizer_state, iteration=iteration)
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer | None = None
):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["iteration"]



def load_and_verify_mmap(file_path: Path, vocab_size: int):
    mmap_array = np.memmap(file_path, dtype=np.uint16, mode='r')
    
    max_token_id = mmap_array.max()
    min_token_id = mmap_array.min()
    
    print(f"Min Token ID in data: {min_token_id}")
    print(f"Max Token ID in data: {max_token_id}")
    print(f"Vocabulary size: {vocab_size}")
    
    assert max_token_id < vocab_size, f"Error: 发现了非法的 Token ID ({max_token_id})，超出了词表范围！可能是 dtype 错误或文件损坏。"
    assert min_token_id >= 0, "Error: Token ID "
    
    print(f"First 10 tokens: {mmap_array[:10]}")
    return mmap_array

    
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    DATA_PATH = BASE_DIR / "results" / "TinyStories" / "encoded_data.bin"

    mmap_data = load_and_verify_mmap(DATA_PATH, vocab_size=10000)