from cs336_basics.bpe_train import BPETrainer
import os
import multiprocessing
import cProfile
import pstats
import pickle
import io

from pathlib import Path
import json
import time          
import tracemalloc
import logging

logger = logging.getLogger(__name__)

def train_tokenizer(
    data_path: str | Path,
    vocab_size: int,
    save_path: str | Path,
    special_tokens: list[str] | None = None
):
    """
    save profile_summury.txt, vocab.json, merges.txt 
    """
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]

    data_path = Path(data_path)
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return
    
    bpe = BPETrainer(str(data_path), vocab_size, special_tokens, verbose=True)

    # =========== start monitor ===============
    tracemalloc.start()
    profiler = cProfile.Profile()
    profiler.enable()
    
    vocab, merges = bpe.train()
    
    profiler.disable()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # =========== end monitor =================

    # save files 
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    ps.print_stats(30)
    (save_dir / "profile_summary.txt").write_text(s.getvalue(), encoding="utf-8")
    
    try:
        bpe.save(save_dir)
    except Exception:
        logger.exception("Failed to save tokenizer components")
        return

    # show metrics
    peak_mem_mb = peak_mem / (1024 * 1024)
    longest_token = max(vocab.values(), key=len)

    logger.info(f"Successfully trained tokenizer for {data_path.name}")
    logger.info(f"Final Vocab Size: {len(vocab)}")
    logger.info(f"Peak Memory: {peak_mem_mb:.2f}MB")
    logger.info(f"Longest Token: {longest_token!r} ({len(longest_token)} bytes)")
    logger.info(f"Artifacts: {save_dir}")

    return vocab, merges

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    multiprocessing.freeze_support()

    current_dir = Path(__file__).resolve().parent
    results_dir = current_dir / "results"
    data_dir = current_dir / "data"

    # train tokenizer
    special_tokens = ["<|endoftext|>"]
    
    
    train_tokenizer(
         data_path= data_dir / "TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
       save_path=results_dir / "TinyStories",
        special_tokens=special_tokens
    )
    
    
    #train_tokenizer(
    #    data_path= data_dir / "owt_train.txt",
    #    vocab_size=32000,
    #    save_path=results_dir / "owt_train",
    #    special_tokens=special_tokens
    #)