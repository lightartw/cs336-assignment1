import os
from typing import BinaryIO
import regex as re
import multiprocessing
from collections import Counter

class BPETokenizer:
    def __init__(self, input_path, vocab_size=300, special_tokens=["<|endoftext|>"]):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        
        self.num_processes = multiprocessing.cpu_count() // 2
        # internal data
        self.pretoken_counts = Counter()   # for merge
        self.pair_stats = Counter()

        raw_vocab_list = [bytes([i]) for i in range(256)]
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in raw_vocab_list: # 避免重复添加
                raw_vocab_list.append(token_bytes)
        self.vocabulary: dict[int, bytes] = {i: b for i, b in enumerate(raw_vocab_list)} 
        self.ENDOFTEXT = self.special_tokens[0].encode("utf-8") if self.special_tokens else b"<|endoftext|>"
        
    def train_bpe(self):
        # 0. initialization
        self.merges: list[tuple[bytes, bytes]] = []
        # 1.pretoken
        with open(self.input_path, "rb") as f:
            boundaries = self.find_chunk_boundaries(f, self.num_processes, self.ENDOFTEXT)
            chunks = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunks.append(f.read(end - start).decode("utf-8", errors="ignore").replace("\r\n", "\n"))   # windows specifice

        with multiprocessing.Pool(processes=self.num_processes) as pool:
            for partial_count in pool.imap_unordered(self.partial_token, chunks):
                self.pretoken_counts.update(partial_count)
        # 2.merge            
        for token, freq in self.pretoken_counts.items():
            for i in range(len(token) - 1):
                pair = (token[i], token[i+1])
                self.pair_stats[pair] += freq
        
        while len(self.vocabulary) < self.vocab_size:
            if not self.pair_stats:
                break    
            
            best_pair = max(self.pair_stats.keys(), key=lambda pair: (self.pair_stats[pair], pair))
            cur_freq = self.pair_stats[best_pair]
            # print(f"best pair: {best_pair}, freq: {cur_freq}")
            if cur_freq <= 0:
                break
            # update vocab & merge
            self.merges.append(best_pair)
            new_token_id = len(self.vocabulary)
            new_token_bytes = best_pair[0] + best_pair[1] 
            self.vocabulary[new_token_id] = new_token_bytes
            # real merge
            self.apply_merge(best_pair)

            # 进度条
            if len(self.vocabulary) % 100 == 0:
                print(f"Current Vocab Size: {len(self.vocabulary)} / {self.vocab_size}")

        return self.vocabulary, self.merges

    # =========================== help functions ================================== 

    def partial_token(self, chunk: str):
        # remove special tokens
        escaped_tokens = [re.escape(token) for token in self.special_tokens]
        pattern = "|".join(escaped_tokens)
        segments = re.split(pattern, chunk)

        # pretoken
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        token_cnt = Counter()
        for seg in segments:
            if not seg:
                continue
            for m in re.finditer(PAT, seg):
                token_tuple = tuple(bytes([b]) for b in m.group().encode('utf-8'))
                token_cnt[token_tuple] += 1
        return token_cnt
    
    def apply_merge(self, best_pair):
        new_counts = Counter() 
        p0, p1 = best_pair
        merged_unit = p0 + p1

        # find tokens to modify
        tokens_to_modify = []
        for token, freq in self.pretoken_counts.items():
            for k in range(len(token) - 1):
                if token[k] == p0 and token[k+1] == p1:
                    tokens_to_modify.append((token, freq))
                    break


        # modify pair_stats & pretoken_counts
        for old_token, freq in tokens_to_modify:
            new_token = []
            i = 0
            while i < len(old_token):
                if i < len(old_token) - 1 and old_token[i] == p0 and old_token[i+1] == p1:
                    # incremental update
                    #   minus
                    if i > 0:
                        left_pair = (old_token[i-1], old_token[i])
                        self.pair_stats[left_pair] -= freq
                        if self.pair_stats[left_pair] <= 0:
                            del self.pair_stats[left_pair]
                    if i + 1 < len(old_token) - 1:
                        right_pair = (old_token[i+1], old_token[i+2])
                        self.pair_stats[right_pair] -= freq
                        if self.pair_stats[right_pair] <= 0:
                            del self.pair_stats[right_pair] 
                    self.pair_stats[best_pair] -= freq
                    if self.pair_stats[best_pair] <= 0:
                        del self.pair_stats[best_pair]
                    
                    # merge
                    new_token.append(merged_unit)
                    
                    #   add
                    if len(new_token) > 1:
                        new_left_pair = (new_token[-2], new_token[-1])
                        self.pair_stats[new_left_pair] += freq
                    if i + 1 < len(old_token) - 1:
                        new_right_pair = (merged_unit, old_token[i+2])
                        self.pair_stats[new_right_pair] += freq

                    i += 2
                else:
                    new_token.append(old_token[i])
                    i += 1
            # update pretoken_counts
            del self.pretoken_counts[old_token]
            target_token = tuple(new_token)
            self.pretoken_counts[target_token] = self.pretoken_counts.get(target_token, 0) + freq

    def find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

if __name__ == "__main__":
    import cProfile
    import pstats
    
    multiprocessing.freeze_support()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    file_path = os.path.join(data_dir, "TinyStoriesV2-GPT4-valid.txt")
    special_tokens=["<|endoftext|>"]
    
    bpe = BPETokenizer(file_path, 300, special_tokens)
    
    print("begin training")
    
    # 在代码内部启动 profiler
    profiler = cProfile.Profile()
    profiler.enable()
    
    bpe.train_bpe()
    print("================vocabulary==============")
    print(bpe.vocabulary)
    print("==================tokens==============")
    print(bpe.pretoken_counts)
    
    profiler.disable()
    
    # 保存结果
    profiler.dump_stats('bpe_profile.prof')
    
    # 打印简要结果
    ps = pstats.Stats(profiler)
    
    print("\n" + "="*60)
    print("partial_token 性能统计:")
    print("="*60)
    ps.sort_stats('cumulative').print_stats('partial_token', 5)
    
    print("\n" + "="*60)
    print("merge_pretoken 性能统计:")
    print("="*60)
    ps.sort_stats('cumulative').print_stats('merge_pretoken', 5)

    
    print("finished!")