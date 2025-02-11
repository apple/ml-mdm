# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import re

from mlx.data.core import CharTrie


def read_dictionary_bert(vocab_file):
    trie_key_scores = []
    trie = CharTrie()

    f = open(vocab_file, "rb")
    sep = "\u2581".encode()

    max_score = 0
    for line in f:
        line = line.rstrip()
        token, score = line.split(b"\t")
        score = -float(score)

        token = token.replace(sep, b" ")
        if trie.search(token):
            raise RuntimeError(b"token " + token + b" already exists")
        trie.insert(token)
        # special token?
        if token not in [b"[PAD]", b"[SEP]", b"[CLS]"]:
            trie_key_scores.append(0.0)
        else:
            trie_key_scores.append(score)
        max_score = max(max_score, score)

    eos, bos, pad = -1, -1, -1
    for i in range(trie.num_keys()):
        key = "".join(trie.key(i))
        if key == "[SEP]":
            eos = i
        if key == "[CLS]":
            bos = i
        if key == "[PAD]":
            pad = i

    return trie, trie_key_scores, eos, bos, pad


def read_dictionary_t5(vocab_file):
    trie_key_scores = []
    trie = CharTrie()

    f = open(vocab_file, "rb")
    sep = "\u2581".encode()

    max_score = 0
    for line in f:
        line = line.rstrip()
        token, score = line.split(b"\t")
        score = -float(score)

        token = token.replace(sep, b" ")
        if trie.search(token):
            raise RuntimeError(b"token " + token + b" already exists")
        trie.insert(token)
        trie_key_scores.append(score)
        max_score = max(max_score, score)

    eos, bos, pad = -1, -1, -1
    for i in range(trie.num_keys()):
        key = "".join(trie.key(i))
        if key == "</s>":
            eos = i
        if key == "<unk>":
            bos = i
        if key == "<pad>":
            pad = i

    return trie, trie_key_scores, eos, bos, pad


def read_dictionary(vocab_file):
    trie_key_scores = []
    trie = CharTrie()

    # make sure those are first
    special_tokens = [b"<pad>", b"<s>", b"</s>"]
    for token in special_tokens:
        trie.insert(token)
        trie_key_scores.append(0.0)

    f = open(vocab_file, "rb")
    sep = "\u2581".encode()

    max_score = 0
    for line in f:
        line = line.rstrip()
        token, score = line.split(b"\t")
        score = -float(score)

        # special token?
        if re.match(b"^<.*>$", token):
            if not token in special_tokens:
                special_tokens.append(token)
        else:
            token = token.replace(sep, b" ")
            if trie.search(token):
                raise RuntimeError(b"token " + token + b" already exists")
            trie.insert(token)
            trie_key_scores.append(score)
        max_score = max(max_score, score)

    for token in special_tokens:
        # hex token?
        hex_byte = re.match(b"^<0x(..)>$", token)
        if hex_byte:
            (token,) = hex_byte.groups()
            token = bytes.fromhex(token.decode())
        if not trie.search(token):
            trie.insert(token)
            trie_key_scores.append(max_score + 1.0)

    eos, bos, pad = -1, -1, -1
    for i in range(trie.num_keys()):
        key = "".join(trie.key(i))
        if key == "</s>":
            eos = i
        if key == "<s>":
            bos = i
        if key == "<pad>":
            pad = i

    return trie, trie_key_scores, eos, bos, pad


class Tokenizer:
    def __init__(self, vocab_file, mode=None):
        if mode == "t5":
            (
                self._trie,
                self._trie_key_scores,
                self.eos,
                self.bos,
                self.pad,
            ) = read_dictionary_t5(vocab_file)
        elif mode == "bert":
            (
                self._trie,
                self._trie_key_scores,
                self.eos,
                self.bos,
                self.pad,
            ) = read_dictionary_bert(vocab_file)
        else:
            (
                self._trie,
                self._trie_key_scores,
                self.eos,
                self.bos,
                self.pad,
            ) = read_dictionary(vocab_file)
        self.vocab_size = self._trie.num_keys()

    @property
    def trie(self):
        return self._trie

    @property
    def trie_key_scores(self):
        return self._trie_key_scores

    def tokens2text(self, tokens):
        return "".join([self._trie.key_string(tok) for tok in tokens])

    def token_id(self, token):
        node = self._trie.search(token)
        if node is None:
            raise ValueError(f"token: {token} not found in vocab.")
        return node.id
