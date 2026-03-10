from collections.abc import Iterable, Iterator
from tqdm import tqdm
import json
from pathlib import Path
import numpy as np
from functools import lru_cache
from multiprocessing import Pool
import os

from .pre_tokenization import PreTokenizer
from .util import find_chunk_boundaries, get_file_iter

class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None=None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []

        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.vocab.values():
                self.vocab[len(self.vocab)] = token_bytes

        self.token_to_id: dict[bytes, int] = {token: id for id, token in self.vocab.items()}
        self.pair_to_rank: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }

        self.pretokenizer = PreTokenizer(self.special_tokens)
   
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None=None):
        # 1. vocab
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            serialized_vocab = json.load(f)
        vocab = {
            int(token_id): token_bytes.encode('latin-1')
            for token_id, token_bytes in serialized_vocab.items()
        }

        # 2. merges
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip('\n')
                if not line:
                    continue
                
                parts = line.split(' ')
                if len(parts) != 2:
                    continue
                    
                p0, p1 = parts
                merges.append((p0.encode('latin-1'), p1.encode('latin-1'))) 

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    @lru_cache(maxsize=100000)
    def _encode_one_token(self, pretoken: bytes) -> list[int]:
        if pretoken in self.token_to_id:
            return [self.token_to_id[pretoken]]

        # apply merge
        parts = [bytes([b]) for b in pretoken]

        while len(parts) > 1:
            best_pair = None
            min_rank = float('inf')
            
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i+1])
                rank = self.pair_to_rank.get(pair, float('inf'))
                if rank < min_rank:
                    min_rank = rank
                    best_pair = pair
            
            if best_pair is None:
                break
                
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i+1]) == best_pair:
                    new_parts.append(parts[i] + parts[i+1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts

        ids = [self.token_to_id[t] for t in parts] 
        return ids

    def encode(self, text: str) -> list[int]:
        byte_text = text.encode("utf-8")
        
        id_list = []
        for pretoken in self.pretokenizer.pre_tokenize(byte_text):
            ids = self._encode_one_token(pretoken)
            id_list.extend(ids)
        return id_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            byte_text = text.encode("utf-8")
            for pretoken in self.pretokenizer.pre_tokenize(byte_text):
                yield from self._encode_one_token(pretoken)
   
    def decode(self, ids: list[int]) -> str:
        byte_list = []
        for token_id in tqdm(ids, desc="Decoding IDs", leave=False):
            byte_list.append(self.vocab[token_id])
            
        return b"".join(byte_list).decode("utf-8", errors="replace")
    
    # =======================================================================

    def encode_to_bytes(self, text: str) -> bytes:
        byte_text = text.encode("utf-8")
        id_list = []
        for pretoken in self.pretokenizer.pre_tokenize(byte_text):
            ids = self._encode_one_token(pretoken)
            id_list.extend(ids)
        
        if not id_list:
            return b""
        return np.array(id_list, dtype=np.uint16).tobytes()
    
    def encode_file(self, data_file: Path, output_file: Path, batch: bool = True) -> None:
        """
        encode data in data_file and save it int output_file
        """
        assert len(self.vocab) < 65536, "Vocab size exceeds uint16 limit!"

        split_token_bytes = self.special_tokens[0].encode("utf-8")
        with open(data_file, "rb") as f:
            boundaries = find_chunk_boundaries(f, 400, split_token_bytes)

        file_iter = get_file_iter(data_file, boundaries)

        # batch encode
        with open(output_file, "wb") as f_out:
            if batch:
                num_workers = os.cpu_count() or 8
                with Pool(processes=num_workers) as pool:
                    results = pool.imap(self.encode_to_bytes, file_iter, chunksize=1)
                    for chunk_bytes in tqdm(results, desc="Multi-proc Encoding", total=len(boundaries)-1):
                        if chunk_bytes:
                            f_out.write(chunk_bytes)
            else:
                for text_chunk in tqdm(file_iter, desc="Single-proc Encoding", total=len(boundaries)-1):
                    chunk_bytes = self.encode_to_bytes(text_chunk)
                    if chunk_bytes:
                        f_out.write(chunk_bytes) 