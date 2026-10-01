"""GARGANTUA v1 tokens 包：词表与文本编解码。"""
from .vocab import Vocab, get_vocab
from .text_codec import encode, decode

__all__ = ["Vocab", "get_vocab", "encode", "decode"]
