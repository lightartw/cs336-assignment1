from cs336_basics.bpe import BPETokenizer
import os
import multiprocessing
import cProfile
import pstats
import pickle
import io

def load_bpe_results(results_path="results/bpe_results.pkl"):
    """
    从磁盘读取 BPE 训练结果。
    返回: (vocab, merges)
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"找不到结果文件: {results_path}")
        
    with open(results_path, "rb") as f:
        data = pickle.load(f)
        
    vocab = data["vocab"]
    merges = data["merges"]
    
    print(f"--- 成功加载结果 ---")
    print(f"词表大小 (Vocab Size): {len(vocab)}")
    print(f"合并次数 (Total Merges): {len(merges)}")
    print(vocab)

    return vocab, merges

load_bpe_results()


"""
if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    # 创建 results 文件夹
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    file_path = os.path.join(data_dir, "TinyStoriesV2-GPT4-train.txt")
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    
    # --- training ---
    bpe = BPETokenizer(file_path, vocab_size, special_tokens)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    vocab, merges = bpe.train_bpe()
    
    profiler.disable()

    # 保存原始数据到 results/bpe_profile.prof
    profile_path = os.path.join(results_dir, "bpe_profile.prof")
    profiler.dump_stats(profile_path)
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    ps.print_stats(30)
    profile_output = s.getvalue()
    print(profile_output)
    
    # 将文本格式的统计结果保存下来
    with open(os.path.join(results_dir, "profile_summary.txt"), "w") as f:
        f.write(profile_output)

    # --- 2. 保存结果到磁盘 (回答问题 a) ---
    output_data = {
        "vocab": vocab,
        "merges": merges
    }
    results_path = os.path.join(results_dir, "bpe_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(output_data, f)
    print(f"Vocab and merges saved to {results_path}")

    # --- 3. 找出最长的 Token (回答问题 a) ---
    longest_token = max(vocab.values(), key=len)
    print(f"Longest token: {longest_token}")
    print(f"Longest token length: {len(longest_token)} bytes")
"""