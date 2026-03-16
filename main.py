import os
import io
import time
import pstats
import cProfile
import contextlib
import logging
import tracemalloc
from pathlib import Path
from typing import List 

import typer
import regex as re
from cs336_basics.bpe_train import BPETrainer
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.nn.train import train as run_train
from cs336_basics.nn.config import Config
import json

app = typer.Typer(help="CS336 Train", add_completion=False)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

@contextlib.contextmanager
def profile_and_monitor(save_dir: Path, prefix: str = ""):
    save_dir.mkdir(parents=True, exist_ok=True)
    tracemalloc.start()
    profiler = cProfile.Profile()
    start_time = time.perf_counter()
    profiler.enable()
    
    try:
        yield 
    finally:
        profiler.disable()
        end_time = time.perf_counter()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        s = io.StringIO()
        # 按照累计时间排序，查看最耗时的函数
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime') 
        ps.print_stats(30)
        
        file_name = f"{prefix}_profile.txt"
        profile_path = save_dir / file_name
        profile_path.write_text(s.getvalue(), encoding="utf-8")
        
        duration = end_time - start_time
        peak_mem_mb = peak_mem / (1024 * 1024)
        
        logger.info("-" * 40)
        logger.info(f"任务阶段: {prefix}")
        logger.info(f"执行耗时: {duration:.2f}s")
        logger.info(f"峰值内存: {peak_mem_mb:.2f}MB")
        logger.info(f"性能分析详情已保存至: {profile_path}")
        logger.info("-" * 40)


@app.command()
def train_bpe(
    data_file: Path = typer.Option(..., "--data-path", "-d", help="训练语料路径", exists=True, readable=True), 
    vocab_size: int = typer.Option(10000, "--vocab-size", "-v", help="词表大小"),
    save_path: Path = typer.Option("./models", "--save-path", "-s", help="模型保存目录"),
    special_tokens: List[str] = typer.Option(["<|endoftext|>"], "--special-tokens", help="特殊 token 列表")
):
    """训练 BPE 分词器"""
    if not data_file.exists():
        logger.error(f"文件不存在: {data_file}")
        raise typer.Exit(code=1)

    bpe = BPETrainer(str(data_file), vocab_size, special_tokens, verbose=True)
    with profile_and_monitor(save_path):
        bpe.train()
    bpe.save(save_path)
    logger.info(f"训练完成并保存至 {save_path}")

@app.command()
def compare_ratio(
    vocab_file: Path,
    merges_file: Path,
    data_file: Path,
    num_samples: int = 10,
    special_tokens: List[str] = ["<|endoftext|>"]
):
    """
    实验要求 (a) & (b): 采样并计算压缩比及吞吐量。
    """
    # 1. 初始化
    tokenizer = Tokenizer.from_files(str(vocab_file), str(merges_file), special_tokens)
    split_token = special_tokens[0]
    pattern = re.compile(re.escape(split_token))
    
    total_bytes, total_tokens, sampled_count = 0, 0, 0
    remainder = ""
    chunk_size = 1024 * 1024  # 1MB 缓冲区

    # 2. 流式采样 & 3. 编码分析
    start_time = time.time() 
    
    with open(data_file, "r", encoding="utf-8", errors="replace") as f:
        while sampled_count < num_samples:
            chunk = f.read(chunk_size)
            if not chunk: break
            
            content = remainder + chunk
            parts = pattern.split(content)
            remainder = parts.pop() 
            
            for doc_text in parts:
                doc_text = doc_text.strip()
                if not doc_text: continue
                
                # 统计原始字节与编码后的 token 数
                total_bytes += len(doc_text.encode("utf-8"))
                total_tokens += len(tokenizer.encode(doc_text))
                
                sampled_count += 1
                if sampled_count >= num_samples: break

    end_time = time.time()
    duration = end_time - start_time

    # 结果输出与 Pile 数据集处理时间估算
    if total_tokens > 0 and duration > 0:
        ratio = total_bytes / total_tokens
        throughput = total_bytes / duration  # bytes/second
        
        # 估算处理 825GB (825 * 1024^3 bytes) 数据集的时间
        pile_size_bytes = 825 * (1024**3)
        estimated_hours = (pile_size_bytes / throughput) / 3600

        logger.info("-" * 30)
        logger.info(f"数据源: {data_file.name} | 采样数: {sampled_count}")
        logger.info(f"压缩比 (Bytes/Token): {ratio:.4f}")
        logger.info(f"吞吐量 (Throughput): {throughput / 1024 / 1024:.2f} MB/s")
        logger.info(f"处理 825GB Pile 数据集预计耗时: {estimated_hours:.2f} 小时")
        logger.info("-" * 30)


@app.command()
def encode_file(
    vocab_file: Path = typer.Option(..., "--vocab-file", help="词表 JSON 文件路径"),
    merges_file: Path = typer.Option(..., "--merges-file", help="BPE merges.txt 文件路径"),
    data_file: Path = typer.Option(..., "--data-file", help="待编码的原始文本文件"),
    output_file: Path = typer.Option(..., "--output-file", help="输出的二进制 .bin 文件路径"),
    special_tokens: List[str] = typer.Option(["<|endoftext|>"], "--special-tokens", help="特殊 Token 列表")
):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_files(str(vocab_file), str(merges_file), special_tokens)
    
    
    file_size_mb = os.path.getsize(data_file) / (1024 * 1024)
    logger.info(f"Starting encoding: {data_file} ({file_size_mb:.2f} MB)")

    try:
        save_path = output_file.parent
        with profile_and_monitor(save_path, prefix="encode"):
            tokenizer.encode_file(data_file=data_file, output_file=output_file)
        
        # 验证输出
        if output_file.exists():
            out_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            logger.info(f"Encoding complete! Saved to {output_file} ({out_size_mb:.2f} MB)")
        else:
            logger.warning("encode_file finished but output file was not created.")
            
    except Exception as e:
        logger.error(f"Error during file encoding: {e}", exc_info=True) 

# ================== train model =====================

def train_lr(config_json: str):
    with open(config_json, 'r') as f:
        config_dict = json.load(f)
    lr = config_dict['optimizer']['learning_rate']
    base_dir = config_dict['training']['out_dir']
    config_dict['training']['out_dir'] = os.path.join(base_dir, f"lr_{lr}")
    config_dict['training']['wandb_project'] = "tinystories-lr-experiment"
    
    os.makedirs(config_dict['training']['out_dir'], exist_ok=True)
    
    config_obj = Config.model_validate(config_dict)
    print(f"🚀 启动训练 | LR: {lr} | 目录: {config_dict['training']['out_dir']}")
    run_train(config_obj)

def train_batch(config_json: str):
    import math
    with open(config_json, 'r') as f:
        config_dict = json.load(f)

    base_batch = 64
    base_lr = 0.001
    base_iter = 5000
    base_dir = config_dict['training']['out_dir']
    config_dict['training']['wandb_project'] = "tinystories-batch-experiment"        
    os.makedirs(config_dict['training']['out_dir'], exist_ok=True)

    batches = [1, 32, 128]
    for batch in batches:
        lr = base_lr *  math.sqrt(batch / base_batch)
        max_iter = base_iter * (base_batch / batch)
        
        config_dict['training']['out_dir'] = os.path.join(base_dir, f"batch_{batch}_")
        config_dict['training']['batch_size'] = batch
        config_dict['optimizer']['learning_rate'] = lr
        config_dict['training']['max_iters'] = max_iter
        config_dict['training']['cosine_cycle_iters'] = max_iter
        config_dict['training']['warmup_iters'] = max_iter // 20
        config_dict['training']['eval_interval'] = max_iter // 100
        config_dict['training']['log_interval'] = max_iter // 500

        config_obj = Config.model_validate(config_dict)
        print(f"🚀 启动训练 | Batch: {batch} | 目录: {config_dict['training']['out_dir']}")
        run_train(config_obj)


@app.command()
def train(config_json: str):
    run_train(config_json)

if __name__ == "__main__":
    app()