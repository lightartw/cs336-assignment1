import os
from typing import BinaryIO
import regex as re
import multiprocessing
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator

from tqdm import tqdm
import time
import logging

class PreTokenizer:
    def __init__(self, special_tokens: list[str]=["<|endoftext|>"]):
        sorted_tokens = sorted(special_tokens, key=len, reverse=True)   # important
        self.special_tokens_bytes = [t.encode("utf-8") for t in sorted_tokens]

        if self.special_tokens_bytes:
            pattern_str = b"|".join([re.escape(t) for t in self.special_tokens_bytes])
            # 用于训练（split 不带括号，去掉特殊标记）
            self.train_split_pattern = re.compile(pattern_str)
            # 用于推理（split 带括号，保留特殊标记）
            self.inference_split_pattern = re.compile(b"(" + pattern_str + b")")
        else:
            self.train_split_pattern = None
            self.inference_split_pattern = None
        self.PAT = re.compile(rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        self.num_processes = multiprocessing.cpu_count()

    def pretoken(self, input_path: str, ENDOFTEXT: bytes, verbose: bool = False) -> dict[bytes, int]:
        pretoken_counts = Counter()
        start_time = time.time()

        # record time
        file_size = os.path.getsize(input_path)
        if verbose:
            logging.info(f"start, filesize: {file_size / 1024 / 1024:.2f} MB")

        with open(input_path, "rb") as f:
            boundaries = self.find_chunk_boundaries(f, self.num_processes * 20, ENDOFTEXT)
        
        chunk_args = []
        for i in range(len(boundaries) - 1):
            chunk_args.append((
                input_path, 
                boundaries[i], 
                boundaries[i+1], 
                self.train_split_pattern, 
                self.PAT
            ))

        with multiprocessing.Pool(processes=self.num_processes) as pool:
            result_iter = pool.imap_unordered(self.processes_chunk, chunk_args)
            # tqdm
            if verbose:
                result_iter = tqdm(result_iter, total=len(chunk_args), desc="pretokenize", unit="chunk")
            for partial_count in result_iter:
                pretoken_counts.update(partial_count)
        # end time
        end_time = time.time()
        if verbose:
            duration = end_time - start_time
            speed_md = (file_size / 1024 / 1024) / duration
            logging.info(f"finish! time: {duration:.2f} sec")
            logging.info(f"speed: {speed_md:.2f} MB/s")

        return pretoken_counts

    @staticmethod
    def processes_chunk(args) -> dict[bytes, int]:
        """
        Args: input_path: str,
              start: int
              end:   int
              split_pattern
              token_pattern
        """
        input_path, start, end, split_pattern, token_pattern = args
        pretoken_cnt = Counter()
        
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start)
        # remove special tokens
        segments = split_pattern.split(chunk) if split_pattern else [chunk]
        
        # pretoken
        for seg in segments:
            if not seg:  continue
            for m in token_pattern.finditer(seg):
                word = m.group()
                pretoken_cnt[word] += 1
        return pretoken_cnt

    def pre_tokenize(self, bytes_text: bytes) -> Iterator[bytes]:
        if self.inference_split_pattern:
            segments = self.inference_split_pattern.splititer(bytes_text)
        else:
            segments = [bytes_text]

        for seg in segments:
            if not seg:  continue
            if seg in self.special_tokens_bytes:
                yield seg
            else:
                for m in self.PAT.finditer(seg):
                    yield m.group()
    
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


#=================== test ======================

import psutil
import os

def print_process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Current Process Memory: {mem_info.rss / (1024 * 1024):.2f} MB")

# testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    file = "data/owt_valid.txt"
    special_tokens = ["<|endoftext|>"]
    ENDOFTEXT = b"<|endoftext|>"
    pretokenizer = PreTokenizer(special_tokens)
    # 在 pretoken 运行前后分别打印
    print_process_memory()
    pretoken_count = pretokenizer.pretoken(file, ENDOFTEXT, verbose=True)
    print_process_memory()
