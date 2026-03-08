"""
python -m cs336_basics.bpe
"""
import os
import multiprocessing
from collections import Counter, defaultdict
import heapq

from tqdm import tqdm
import json
from pathlib import Path

from .pre_tokenization import PreTokenizer

# for min-heap compair
class ComparablePair:
    __slots__ = ("pair", )

    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair
    
    def __lt__(self, other):
        return self.pair > other.pair

    def __eq__(self, other):
        return self.pair == other.pair

    def __repr__(self):
        return str(self.pair)

class BPETrainer:
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
        self.vocab: dict[int, bytes] = {}
        self.ENDOFTEXT = b"<|endoftext|>"
        # pretoken
        self.input_path = input_path
        self.pretokenizer = pre_tokenizer(special_tokens)
        self.pretoken_states: dict[bytes, list[bytes]] = {}
        # pair
        self.pair_count: dict[tuple[bytes, bytes], int] = Counter()
        self.pair_to_pretokens: dict[tuple[bytes, bytes], set[bytes]] = defaultdict(set)  # for optimization
        self.pair_count_heap = []           # for optimization
    
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
        self.pretoken_count: dict[bytes, int] = self.pretokenizer.pretoken(
            self.input_path, self.ENDOFTEXT, verbose=self.verbose)

        # init cur_token, pair_count, pair_to_pretokens            
        for pretoken, freq in tqdm(
            self.pretoken_count.items(), 
            desc="Init BPE States", 
            disable=not self.verbose,
            total=len(self.pretoken_count)
        ):
            self.pretoken_states[pretoken] = [bytes([b]) for b in pretoken]
            token = self.pretoken_states[pretoken]
            for i in range(len(token) - 1):
                pair = (token[i], token[i+1])
                self.pair_count[pair] += freq
                self.pair_to_pretokens[pair].add(pretoken)

        # build pair_count_heap
        self.pair_count_heap = [
            (-freq, ComparablePair(p)) 
            for p, freq in tqdm(
                self.pair_count.items(), 
                desc="Building Heap", 
                disable=not self.verbose,
                total=len(self.pair_count)
            )
        ]
        heapq.heapify(self.pair_count_heap)
    
    def _apply_merge(self, best_pair: tuple[bytes, bytes]):
        # find tokens to modify & modify best_pair
        pretokens_to_modify = list(self.pair_to_pretokens[best_pair])
        merged_unit = best_pair[0] + best_pair[1]

        # save memory
        updated_pairs = set()

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
                        updated_pairs.add(left_pair)

                        new_left_pair = (state[i-1], merged_unit)
                        self.pair_count[new_left_pair] += freq
                        self.pair_to_pretokens[new_left_pair].add(pretoken)
                        updated_pairs.add(new_left_pair)

                    if i < len(state) - 1:
                        right_pair = (best_pair[1], state[i+1])
                        self.pair_count[right_pair] -= freq
                        updated_pairs.add(right_pair)

                        
                        new_right_pair = (merged_unit, state[i+1])
                        self.pair_count[new_right_pair] += freq
                        self.pair_to_pretokens[new_right_pair].add(pretoken)
                        updated_pairs.add(new_right_pair)
                else:
                    i += 1
        
        for pair in updated_pairs:
            count = self.pair_count.get(pair, 0)
            if count > 0: 
                heapq.heappush(self.pair_count_heap, (-count, ComparablePair(pair)))

        del self.pair_to_pretokens[best_pair] # best_pair disappear
        del self.pair_count[best_pair]

    def _get_best_pair(self) :
        """
        get best_pair based on pair_count
        """ 
        # max(self.pair_count.keys(), key=lambda pair: (self.pair_count[pair], pair))
        while self.pair_count_heap:
            neg_freq, comp_pair = heapq.heappop(self.pair_count_heap)
            pair = comp_pair.pair
            current_freq = self.pair_count[pair]
            
            # lazy update
            if current_freq > 0 and -neg_freq == current_freq:
                return pair
        return None

    def train(self):
        self._initialization()

        num_merges = self.target_vocab_size - len(self.vocab)
        with tqdm(total=num_merges, desc="BPE Training", disable=not self.verbose) as pbar:
            while len(self.vocab) < self.target_vocab_size:
                best_pair = self._get_best_pair()
                if best_pair == None:
                    break
                
                # update vocab & merge
                self.merges.append(best_pair) 
                new_token_bytes = best_pair[0] + best_pair[1] 
                self.vocab[len(self.vocab)] = new_token_bytes

                # real merge
                self._apply_merge(best_pair)
                # update pbar
                pbar.set_description(f"Vocab: {len(self.vocab)}")
                pbar.update(1)
                pbar.set_postfix({"last_freq": self.pair_count[best_pair]})

        if self.verbose:
            tqdm.write(f"Final vocabulary size: {len(self.vocab)}")
        return self.vocab, self.merges

    def save(self, save_dir: str | Path):
        """
        must be called after train!
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # 1. vocab
        vocab_path = save_path / "vocab.json"
        serializable_vocab = {
           str(token_id): token_bytes.decode('latin-1')
           for token_id, token_bytes in self.vocab.items() 
        }
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(serializable_vocab, f, indent=4, ensure_ascii=False)
        
        # 2. merges
        merges_path = save_path / "merges.txt"
        with open(merges_path, "w", encoding="utf-8") as f:
            for pair in self.merges:
                p0 = pair[0].decode('latin-1')
                p1 = pair[1].decode('latin-1')
                f.write(f"{p0} {p1}\n")


if __name__ == "__main__":
    import cProfile
    import pstats
    
    multiprocessing.freeze_support()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    file_path = os.path.join(data_dir, "TinyStoriesV2-GPT4-valid.txt")
    special_tokens=["<|endoftext|>"]
    
    bpe = BpeTrainer(file_path, 300, special_tokens, verbose=True)
    
    # 在代码内部启动 profiler
    profiler = cProfile.Profile()
    profiler.enable()
    
    bpe.train()
    
    profiler.disable()
    
    
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