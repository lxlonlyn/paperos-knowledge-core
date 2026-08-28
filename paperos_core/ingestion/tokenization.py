"""PaperOS-owned deterministic token estimate for authoritative chunk sizing.

Chunk boundaries must not depend on Cognee's provider/model resolver. Local
GGUF tokenizers can always fall back to individual UTF-8 bytes, so the encoded
byte length is a conservative upper bound on the number of model tokens for
ordinary text tokenization without injected special tokens. Using that bound
keeps ``chunk_hard_max_tokens`` deterministic across platforms and optional
tokenizer dependencies.
"""

from __future__ import annotations

from typing import Final


class AuthoritativeChunkTokenizer:
    """Count conservative, environment-independent chunk token units."""

    __slots__ = ()

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.encode("utf-8"))


AUTHORITATIVE_CHUNK_TOKENIZER: Final = AuthoritativeChunkTokenizer()


__all__ = ["AUTHORITATIVE_CHUNK_TOKENIZER", "AuthoritativeChunkTokenizer"]
