import numpy as np
import torch
import numpy as np

from pathlib import Path
from typing import Optional
import re

import time
import wandb
from loguru import logger

from .config import Config
from .util import load_checkpoint, save_checkpoint, load_batch
from .util import cross_entropy, learning_rate_schedule, gradient_clipping
from .transformer import TransformerLM
from .optimizer import AdamW

@torch.no_grad()
def estimate_loss(model, data, batch_size, context_length, device, eval_iters=10):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = load_batch(data, batch_size, context_length, device)
        logits = model(X)
        loss = cross_entropy(logits, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()

def find_latest_checkpoint(out_dir: str) -> Optional[str]:
    out_path = Path(out_dir)
    if not out_path.exists():
        return None
    checkpoints = list(out_path.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None
 
    max_iter = -1
    max_ckpt = None
    for p in checkpoints:
        match = re.search(r'\d+', p.name)
        if match:
            iter_num = int(match.group())
            if iter_num > max_iter:
                max_iter = iter_num
                max_ckpt = p
    if max_ckpt:
        return str(max_ckpt)
    return None

def manage_checkpoints(out_dir: str, max_to_keep: int = 10):
    out_path = Path(out_dir)
    checkpoints = list(out_path.glob("checkpoint_*.pt"))
    
    if len(checkpoints) <= max_to_keep:
        return

    def get_iter(path):
        match = re.search(r'checkpoint_(\d+)\.pt', path.name)
        return int(match.group(1)) if match else -1
    
    checkpoints.sort(key=get_iter)

    num_to_delete = len(checkpoints) - max_to_keep
    for i in range(num_to_delete):
        try:
            checkpoints[i].unlink()
            logger.info(f"Deleted old checkpoint: {checkpoints[i].name}")
        except Exception as e:
            logger.error(f"Error deleting {checkpoints[i].name}: {e}") 

def train(config: Config | str):
    if isinstance(config, str):
        config = Config.from_json(config)
    tc = config.training 
    mc = config.model
    oc = config.optimizer

    # 1. prepare config
    run_name = f"lr-{oc.learning_rate}-batch-{tc.batch_size}"

    wandb.init(
        project=tc.wandb_project,
        config=config.model_dump(),
        name=run_name
    )
    Path(tc.out_dir).mkdir(parents=True, exist_ok=True)            
    ckpt_dir = Path(tc.out_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"loading model and optimizer, device: {mc.device}")
    model = TransformerLM(
        vocab_size=mc.vocab_size,
        context_length=mc.context_length,
        num_layers=mc.num_layers,
        d_model=mc.d_model,
        num_heads=mc.num_head,
        d_ff=mc.d_ff,
        theta=mc.theta,
        device=torch.device(mc.device),
        dtype=getattr(torch, mc.dtype)
    )
    model.to(mc.device)
    model.reset_parameters()
    
    oc = config.optimizer
    optimizer = AdamW(
        model.parameters(), 
        lr=oc.learning_rate, 
        weight_decay=oc.weight_decay,
        betas=(oc.beta1, oc.beta2),
        eps=oc.eps
    )
    
    # resume from checkpoint / start a new training loop
    start_iter = 0
    latest_ckpt = find_latest_checkpoint(str(ckpt_dir))
    if latest_ckpt:
        logger.info(f"resume training from {latest_ckpt}...")
        start_iter = load_checkpoint(latest_ckpt, model, optimizer) + 1

    logger.info("loading data...")
    train_data = np.memmap(tc.train_data_path, dtype=np.uint16, mode='r')
    val_data = None
    if tc.val_data_path:
        logger.info(f"Validation data path found, loading from {tc.val_data_path}")
        val_data = np.memmap(tc.val_data_path, dtype=np.uint16, mode='r')
    
    # main loop
    model.train()
    start_time = time.time()    
    training_start_time = start_time
    for iter_num in range(start_iter, tc.max_iters + 1):
        lr = learning_rate_schedule(
            t=iter_num,
            amax=oc.learning_rate,
            amin=oc.min_lr,
            tw=tc.warmup_iters,
            tc=tc.cosine_cycle_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # forward pass
        X, Y = load_batch(train_data, tc.batch_size, mc.context_length, mc.device)
        logits = model(X)
        loss = cross_entropy(logits, Y)

        # backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), maxl2norm=tc.max_l2_norm)
        optimizer.step()

        end_time = time.time()
        # logging
        total_tokens = iter_num * tc.batch_size * mc.context_length
        if iter_num % tc.log_interval == 0:
            dt = end_time - start_time
            start_time = end_time
            tokens_per_sec = (tc.batch_size * mc.context_length * tc.log_interval) / dt
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": lr,
                "perf/tokens_per_sec": tokens_per_sec,
                "time/elapsed_sec": end_time - training_start_time
                                                                    # "train/total_tokens": total_tokens
            }, step=total_tokens)
            logger.info(f"Iter {iter_num:5d} | Tokens: {total_tokens:.2e} | Loss: {loss.item():.4f} | LR: {lr:.2e} | Tok/s: {tokens_per_sec:.0f}") 

        # eval and save
        if iter_num % tc.eval_interval == 0:
            if val_data is not None:
                val_loss = estimate_loss(model, val_data, tc.batch_size, mc.context_length, mc.device)
                wandb.log({
                    "val/loss": val_loss,
                }, step=total_tokens)
                logger.success(f"Iter {iter_num} | Validation Loss: {val_loss:.4f}")

            save_path = ckpt_dir / f"checkpoint_{iter_num}.pt" 
            save_checkpoint(model, optimizer, iter_num, save_path)
            manage_checkpoints(str(ckpt_dir))
        
    wandb.finish()