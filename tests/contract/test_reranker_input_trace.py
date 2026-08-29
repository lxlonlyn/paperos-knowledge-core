"""Direct contract for exact reranker input accounting and overflow rejection."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.runtime.local_inference.schemas import RerankResult


def _require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _node_contract() -> dict[str, object]:
    module = REPOSITORY_ROOT / "services/local_models/dist/reranker_input.js"
    _require(module.is_file(), "Run npm build before the reranker input contract")
    script = """
import {
  RerankerInputTooLargeError,
  validateRerankerInputTokenTrace,
} from "./services/local_models/dist/reranker_input.js";

const safe = validateRerankerInputTokenTrace(7, 13, 29, 32);
if (
  safe.queryTokenCount !== 7 ||
  safe.documentTokenCount !== 13 ||
  safe.specialPromptTokenCount !== 9 ||
  safe.effectiveInputTokenCount !== 29 ||
  safe.modelMaxInputTokens !== 32 ||
  safe.truncated !== false
) {
  throw new Error("safe reranker trace is inconsistent");
}
let overflow = null;
try {
  validateRerankerInputTokenTrace(7, 13, 33, 32);
} catch (error) {
  if (!(error instanceof RerankerInputTooLargeError)) throw error;
  overflow = {
    code: error.code,
    message: error.message,
    effectiveInputTokenCount: error.effectiveInputTokenCount,
    modelMaxInputTokens: error.modelMaxInputTokens,
  };
}
if (
  overflow === null ||
  overflow.code !== "reranker_input_too_large" ||
  overflow.effectiveInputTokenCount !== 33 ||
  overflow.modelMaxInputTokens !== 32
) {
  throw new Error("reranker overflow was not rejected with the stable error");
}
process.stdout.write(JSON.stringify({safe, overflow}));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _schema_contract(node: dict[str, object]) -> dict[str, object]:
    safe = node["safe"]
    assert isinstance(safe, dict)
    result = RerankResult.model_validate(
        {
            "candidate_id": "chunk_contract",
            "original_index": 0,
            "relevance_score": 0.5,
            "final_rank": 1,
            "document_token_count": 13,
            "input_token_count": 29,
            "effective_input_token_count": 29,
            "model_max_input_tokens": 32,
            "query_token_count": 7,
            "special_prompt_token_count": 9,
            "truncated": False,
            "window_count": 1,
            "winning_window_document_token_count": 13,
            "winning_window_index": 0,
            "winning_window_text": "contract span",
        }
    )
    _require(
        result.input_token_count == result.effective_input_token_count,
        "Legacy and explicit effective input counts diverged",
    )
    _require(
        result.effective_input_token_count <= result.model_max_input_tokens,
        "Successful reranker trace exceeds model context",
    )
    _require(
        result.query_token_count
        + result.winning_window_document_token_count
        + result.special_prompt_token_count
        == result.effective_input_token_count,
        "Reranker token components do not reconstruct effective input",
    )
    return result.model_dump(mode="json")


def _source_contract() -> dict[str, object]:
    source = (
        REPOSITORY_ROOT / "services/local_models/src/reranker.ts"
    ).read_text(encoding="utf-8")
    _require(
        source.index("const inputTraces") < source.index("this.context.rankAll"),
        "Context validation does not happen before model evaluation",
    )
    _require(
        "truncated: false as const" not in source,
        "Reranker trace still hard-codes truncated=false at result construction",
    )
    _require(
        "one prebuilt PaperOS RerankSpan" in source
        and "authoritative canonical parent Chunk" in source,
        "Structured reranker input boundary is not documented",
    )
    _require(
        "splitSentences" not in source
        and "PROVISIONAL_WINDOW_DOCUMENT_TOKENS" not in source,
        "Local reranker still performs query-time string windowing",
    )
    return {
        "validated_before_rank": True,
        "structured_projection": True,
        "one_score_per_span": True,
    }


def main() -> None:
    node = _node_contract()
    report = {
        "status": "passed",
        "node": node,
        "schema": _schema_contract(node),
        "source": _source_contract(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
