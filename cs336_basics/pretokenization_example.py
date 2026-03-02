import os
from typing import BinaryIO
import regex as re
import multiprocessing
from collections import Counter

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
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

# ================================= BPE ====================================
# 1. Vocabulary：256 + endoftext
# =========================  parilization ==================================
# 1.5. split chunk with <|endoftext|>(remove special tokens)
# 2. pretoken : use regex, datastructure dict[tuple(bytes), int];
# ================================================================== 
# 3.merge: use a dict: for str, freq in pretoken:
#                          for every to succecieve word in vocabulary in str:
#                                 merge[word] += freq
#   get merged token , vocabulary.add(merged_token)
# for token in pretoken: change tuple(bytes) based on new vocabulary 

filename = ""
special_tokens = ["<|endoftext|>"]
ENDOFTEXT = b"<|endoftext|>"


def pretoken(chunk: str):
    # remove special tokens
    escaped_tokens = [re.escape(token) for token in special_tokens]
    pattern = "|".join(escaped_tokens)
    segments = re.split(pattern, chunk)

    # pretoken
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    token_cnt = Counter()
    for seg in segments:
        for m in re.finditer(PAT, seg):
            token_tuple = tuple(char.encode('utf-8') for char in m.group())
            token_cnt[token_tuple] += 1
    return token_cnt

def merge_vocab(best_pair, pretoken_counts):
    new_tokens = {}

    merged_unit = best_pair[0] + best_pair[1]
    for token, freq in pretoken_counts.items():
        new_token = []
        i = 0
        while i < len(token):
            if i < len(token) - 1 and token[i] == best_pair[0] and token[i+1] == best_pair[1]:
                new_token.append(merged_unit)
                i += 2
            else:
                new_token.append(token[i])
                i += 1
        new_counts[tuple(new_token)] = freq
    
    return new_counts


if __name__ == "__main__":
    num_processes = multiprocessing.cpu_count()
    num_merge = 3
    # 1.vocabulary
    vocabulary: list[bytes] = [bytes([i]) for i in range(256)] + [ENDOFTEXT]

    # 2.pretoken 
    with open(filename, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, ENDOFTEXT)
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))

    pretoken_counts = Counter()
    with multiprocessing.Pool(processes=num_processes) as pool:
        for partial_count in pool.imap_unordered(pretoken, chunks):
            pretoken_counts.update(partial_count)
     
    # 3. merge
    for _ in range(num_merge):    
        freq_counts = Counter()
        for token, freq in pretoken_counts.items():
            for i, j in zip(token, token[1:]):
                pair = (token[i], token[j])
                freq_counts[pair] += freq
        best_pair = max(
            freq_counts.keys(), 
            key=lambda x: (freq_counts[x], x)
        )
        # update vocab
        vocabulary.append(best_pair[0] + best_pair[1])
        # real merge
        pretoken_counts = merge_vocab(best_pair, pretoken_counts)