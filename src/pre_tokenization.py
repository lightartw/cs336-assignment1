import os
from typing import BinaryIO
import regex as re
import multiprocessing
from collections import Counter, defaultdict

class PreTokenizer:
    def __init__(self, input_path: str, special_tokens=["<|endoftext|>"]):
        self.input_path = input_path
        self.num_processes = multiprocessing.cpu_count() // 2
        self.special_tokens = special_tokens
        self.ENDOFTEXT = self.special_tokens[0].encode("utf-8") if self.special_tokens else b"<|endoftext|>"

    def pretoken(self) -> dict[bytes, int]:
        with open(self.input_path, "rb") as f:
            boundaries = self.find_chunk_boundaries(f, self.num_processes, self.ENDOFTEXT)
            chunks = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunks.append(f.read(end - start).decode("utf-8", errors="ignore").replace("\r\n", "\n"))   # windows specifice

        pretoken_counts = Counter()
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            for partial_count in pool.imap_unordered(self.partial_token, chunks):
                pretoken_counts.update(partial_count)
        return pretoken_counts

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
                word = m.group().encode("utf-8")
                token_cnt[word] += 1
        return token_cnt
    
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
