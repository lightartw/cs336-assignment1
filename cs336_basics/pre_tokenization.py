import os
from typing import BinaryIO
import regex as re
import multiprocessing
from collections import Counter, defaultdict

from tqdm import tqdm
import time
import logging

class PreTokenizer:
    def __init__(self, input_path: str, special_tokens: list[str]=["<|endoftext|>"]):
        self.input_path = input_path
        self.special_tokens = special_tokens
        self.ENDOFTEXT = self.special_tokens[0].encode("utf-8") if self.special_tokens else b"<|endoftext|>"
        self.num_processes = multiprocessing.cpu_count()

    def pretoken(self, verbose: bool = False) -> dict[bytes, int]:
        pretoken_counts = Counter()

        # record time
        start_time = time.time()
        file_size = os.path.getsize(self.input_path)
        if verbose:
            logging.info(f"start, filesize: {file_size / 1024 / 1024:.2f} MB")

        with open(self.input_path, "rb") as f:
            boundaries = self.find_chunk_boundaries(f, self.num_processes * 20, self.ENDOFTEXT)
        chunk_args = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i+1]
            chunk_args.append((self.input_path, start, end, self.special_tokens))

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
        input_path, start, end, special_tokens = args

        pretoken_cnt = Counter()
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore").replace("\r\n", "\n")   # windows specifice
        # remove special tokens
        escaped_tokens = [re.escape(token) for token in special_tokens]
        pattern = "|".join(escaped_tokens)
        segments = re.split(pattern, chunk)

        # pretoken
        for seg in segments:
            if not seg:
                continue
            for m in re.finditer(PAT, seg):
                word = m.group().encode("utf-8")
                pretoken_cnt[word] += 1
        return pretoken_cnt

    
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

    file = "data/owt_train.txt"
    special_tokens = ["<|endoftext|>"]
    pretokenizer = PreTokenizer(file, special_tokens)
    # 在 pretoken 运行前后分别打印
    print_process_memory()
    pretoken_count = pretokenizer.pretoken(verbose=True)
    print_process_memory()
