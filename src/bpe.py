import os
from typing import BinaryIO
import regex as re
import multiprocessing
from collections import Counter, defaultdict

from tqdm import tqdm

from .pre_tokenization import PreTokenizer

class BpeTrainer:
    def __init__(self, 
        input_path: str, 
        vocab_size: int, 
        special_tokens: list[str]=["<|endoftext|>"],
        pre_tokenizer=PreTokenizer,
        verbose: bool = False 
    ):
        # control monitor
        self.verbose = verbose
        # vocabulary 
        self.target_vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.voab: dict[int, bytes] = {}
        
        # pretoken
        self.pretokenizer = pre_tokenizer(input_path, special_tokens)
        self.pretoken_states: dict[bytes, list[bytes]] = {}
        # pair
        self.pair_to_pretokens: dict[tuple[bytes, bytes], set[bytes]] = defaultdict(set)  # for optimization
        self.pair_count: dict[tuple[bytes, bytes], int] = Counter()
    
    def _initialization(self):
        # vocabulary
        raw_vocab_list = [bytes([i]) for i in range(256)]
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in raw_vocab_list: # 避免重复添加
                raw_vocab_list.append(token_bytes)
        self.vocab = {i: b for i, b in enumerate(raw_vocab_list)} 

        self.merges: list[tuple[bytes, bytes]] = []
        # pretokenization
        self.pretoken_count: dict[bytes, int] = self.pretokenizer.pretoken(verbose=self.verbose)

        # init cur_token, pair_count, pair_to_pretokens            
        for pretoken, freq in self.pretoken_count.items():
            self.pretoken_states[pretoken] = [bytes([b]) for b in pretoken]
            token = self.pretoken_states[pretoken]
            for i in range(len(token) - 1):
                pair = (token[i], token[i+1])
                self.pair_count[pair] += freq
                self.pair_to_pretokens[pair].add(pretoken)
    
    def _apply_merge(self, best_pair: tuple[bytes, bytes]):
        # find tokens to modify & modify best_pair
        pretokens_to_modify = list(self.pair_to_pretokens[best_pair])
        merged_unit = best_pair[0] + best_pair[1]

        # modify pair_count & pretoken_counts
        for pretoken in pretokens_to_modify:
            state = self.pretoken_states[pretoken]
            freq = self.pretoken_count[pretoken]
            i = 0
            while i < len(state) - 1:
                if (state[i], state[i+1]) == best_pair: # lazy update, don't need discard
                    state[i] = merged_unit
                    state.pop(i+1)                    
                    # minus
                    if i > 0:
                        left_pair = (state[i-1], best_pair[0])
                        self.pair_count[left_pair] -= freq
                        if self.pair_count[left_pair] <= 0:
                            del self.pair_count[left_pair]

                        new_left_pair = (state[i-1], merged_unit)
                        self.pair_count[new_left_pair] += freq
                        self.pair_to_pretokens[new_left_pair].add(pretoken)

                    if i < len(state) - 1:
                        right_pair = (best_pair[1], state[i+1])
                        self.pair_count[right_pair] -= freq
                        if self.pair_count[right_pair] <= 0:
                            del self.pair_count[right_pair] 
                        
                        new_right_pair = (merged_unit, state[i+1])
                        self.pair_count[new_right_pair] += freq
                        self.pair_to_pretokens[new_right_pair].add(pretoken)
                else:
                    i += 1

        del self.pair_to_pretokens[best_pair] # best_pair disappear
        del self.pair_count[best_pair]

    def train(self):
        self._initialization()

        num_merges = self.target_vocab_size - len(self.vocab)

        with tqdm(total=num_merges, desc="BPE Training", disable=not self.verbose) as pbar:
            while len(self.vocab) < self.target_vocab_size:
                if not self.pair_count:
                    break    
                
                best_pair = max(self.pair_count.keys(), key=lambda pair: (self.pair_count[pair], pair))
                cur_freq = self.pair_count[best_pair]
                if cur_freq <= 0:
                    break
                # update vocab & merge
                self.merges.append(best_pair)
                new_token_id = len(self.vocab)
                new_token_bytes = best_pair[0] + best_pair[1] 
                self.vocab[new_token_id] = new_token_bytes
                # real merge
                self._apply_merge(best_pair)

                # update pbar
                pbar.set_description(f"Vocab: {len(self.vocab)}")
                pbar.update(1)
                pbar.set_postfix({"last_freq": cur_freq})

        if self.verbose:
            tqdm.write(f"Final vocabulary size: {len(self.vocab)}")

        return self.vocab, self.merges


if __name__ == "__main__":
    import cProfile
    import pstats
    
    multiprocessing.freeze_support()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    file_path = os.path.join(data_dir, "TinyStoriesV2-GPT4-valid.txt")
    special_tokens=["<|endoftext|>"]
    
    bpe = BpeTrainer(file_path, 300, special_tokens, verbose=True)
    
    print("begin training")    
    # 在代码内部启动 profiler
    profiler = cProfile.Profile()
    profiler.enable()
    
    bpe.train()
    print("================vocabulary==============")
    print(bpe.vocab)
    # print("==================tokens==============")
    # print(bpe.pretoken_count)
    
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