import os
import multiprocessing
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator

from tqdm import tqdm
import json
from pathlib import Path

from .pre_tokenization import PreTokenizer


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

        self.pretokenizer = PreTokenizer(self.special_tokens)
        # self.cache: dict[bytes, list[int]]

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
                line = line.strip()
                if not line:
                    continue
                p0_str, p1_str = line.split()
                merges.append((p0_str.encode('latin-1'), p1_str.encode('latin-1')))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _encode_one_token(self, pretoken: bytes) -> list[int]:
        if pretoken in self.token_to_id:
            return [self.token_to_id[pretoken]]

        # apply merge
        pretoken_byte = [bytes([b]) for b in pretoken]

        for merge_pair in self.merges:
            if len(pretoken_byte) <= 1:
                break

            new_pretoken = []
            i = 0
            while i < len(pretoken_byte):
                if i < len(pretoken_byte) - 1 and (pretoken_byte[i], pretoken_byte[i+1]) == merge_pair:
                    new_pretoken.append(pretoken_byte[i] + pretoken_byte[i+1])
                    i += 2
                else:
                    new_pretoken.append(pretoken_byte[i])
                    i += 1
            pretoken_byte = new_pretoken

        id_list = [self.token_to_id[t] for t in pretoken_byte]
        return id_list

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
        byte_list = [self.vocab[token_id] for token_id in ids]
        return b"".join(byte_list).decode("utf-8", errors="replace")