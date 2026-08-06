"""Provider-neutral LLM contract tests through Cognee's LLMGateway.

The complete PaperOS LLM flows (section extraction, document summary, answer
synthesis) run against two real providers using configuration-only switching:

* provider A: DeepSeek custom endpoint (live; opt-in via
  ``PAPEROS_RUN_LIVE_PROVIDER_CONTRACTS=1`` plus ``LLM_API_KEY``);
* provider B: a local OpenAI-compatible endpoint started by the test itself.

PaperOS code is identical for both; only the ``[llm]`` configuration changes.
These tests exercise actual HTTP structured-output calls and do not inspect
environment variables in place of real requests.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter
from paperos_core.adapters.cognee.config import configure_cognee
from paperos_core.adapters.llm import LLMClient
from paperos_core.config import load_settings
from paperos_core.domain.canonical import (
    CanonicalBundle,
    CanonicalSnapshot,
    Chunk,
    Document,
    Section,
)
from paperos_core.prompt_repository import PromptRepository


def _mock_app() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        body = await request.json()
        messages = body.get("messages", [])
        user_content = next(
            (item.get("content", "") for item in messages if item.get("role") == "user"),
            "",
        )
        try:
            payload = json.loads(user_content)
        except (TypeError, ValueError):
            payload = {}
        evidence = payload.get("evidence") or []
        chunk_ids = [
            item.get("chunk_id")
            for item in evidence
            if isinstance(item, dict) and item.get("chunk_id")
        ]
        first = chunk_ids[0] if chunk_ids else "chunk_missing"
        if "question" in payload:
            content = json.dumps({"answer": "Mock evidence-bound answer."})
        elif "summary" in str(payload.get("task", "")):
            content = json.dumps(
                {"text": "Mock document summary.", "source_chunk_ids": [first]}
            )
        else:
            content = json.dumps(
                {
                    "entities": [
                        {
                            "key": "e1",
                            "name": "Mock Entity",
                            "entity_type": "concept",
                            "description": "Mock entity.",
                            "source_chunk_ids": [first],
                            "confidence": 0.9,
                        }
                    ],
                    "claims": [
                        {
                            "key": "c1",
                            "text": "Mock claim.",
                            "source_chunk_ids": [first],
                        }
                    ],
                    "relations": [
                        {
                            "source_key": "e1",
                            "target_key": "e1",
                            "relation_type": "RELATED_TO",
                            "source_chunk_ids": [first],
                        }
                    ],
                }
            )
        return JSONResponse(
            {
                "id": "mock-completion",
                "object": "chat.completion",
                "model": body.get("model", "mock"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    return app


class _MockProvider:
    def __enter__(self) -> str:
        self.server = uvicorn.Server(
            uvicorn.Config(_mock_app(), host="127.0.0.1", port=0, log_level="warning")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        for _ in range(200):
            if self.server.started:
                break
            time.sleep(0.05)
        if not self.server.started:
            raise RuntimeError("mock provider did not start")
        socket = self.server.servers[0].sockets[0]
        return f"http://127.0.0.1:{socket.getsockname()[1]}/v1"

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


def _bundle() -> tuple[CanonicalBundle, list[Chunk]]:
    snapshot = CanonicalSnapshot(
        id="snapshot_contract",
        source_file_id="src_contract",
        parse_run_id="parse_contract",
        document_id="doc_contract",
        manifest_path=Path("/tmp/manifest.json"),
        dataset_id="papers",
    )
    document = Document(
        id="doc_contract",
        source_file_id=snapshot.source_file_id,
        parse_run_id=snapshot.parse_run_id,
        canonical_snapshot_id=snapshot.id,
        language="en",
        title="Contract paper",
    )
    first = Section(
        id="section_first",
        document_id=document.id,
        canonical_snapshot_id=snapshot.id,
        title="First",
        level=1,
        order=0,
        path="/First",
    )
    second = Section(
        id="section_second",
        document_id=document.id,
        canonical_snapshot_id=snapshot.id,
        title="Second",
        level=1,
        order=1,
        path="/Second",
    )

    def chunk(chunk_id: str, section: Section | None, text: str) -> Chunk:
        return Chunk(
            id=chunk_id,
            document_id=document.id,
            canonical_snapshot_id=snapshot.id,
            text=text,
            order=0,
            element_ids=[f"element_{chunk_id}"],
            element_span_ids=[f"element_{chunk_id}:0"],
            section_id=section.id if section else None,
            section_path=section.path if section else None,
            token_count=max(1, len(text.split())),
        )

    chunks = [
        chunk(
            "chunk_first_1",
            first,
            "Graph Neural Networks learn node representations by aggregating "
            "neighbor information across layers.",
        ),
        chunk(
            "chunk_first_2",
            first,
            "The proposed loss function combines supervised classification "
            "with a structural regularization term.",
        ),
        chunk(
            "chunk_second_1",
            second,
            "Experiments show the model outperforms strong baselines on "
            "standard citation datasets.",
        ),
    ]
    return CanonicalBundle(
        snapshot=snapshot,
        document=document,
        sections=[first, second],
        elements=[],
        references=[],
        warnings=[],
    ), chunks


async def _run_provider_flows(settings) -> None:
    configure_cognee(settings)
    CogneeCompatibilityAdapter.reset_configuration_caches()
    client = LLMClient(settings.llm, PromptRepository())
    bundle, chunks = _bundle()
    enrichment = await client.enrich(bundle, chunks)
    assert enrichment.entities
    assert enrichment.claims
    assert enrichment.relations
    assert len(enrichment.summaries) == 1
    assert enrichment.summaries[0].text
    assert enrichment.covered_chunk_ids == [chunk.id for chunk in chunks]
    assert enrichment.uncovered_chunk_ids == []
    assert enrichment.coverage_ratio == 1.0
    answer = await client.synthesize_answer(
        query="What is tested?",
        profile="truth",
        evidence=[
            {
                "evidence_id": "evidence_1",
                "chunk_id": "chunk_first_1",
                "text": "First section evidence one.",
            }
        ],
    )
    assert isinstance(answer, str) and answer.strip()


def test_provider_b_local_openai_compatible_contract(gate1_run_dir: Path) -> None:
    """Provider B: local OpenAI-compatible endpoint, no external network."""
    with _MockProvider() as endpoint:
        config_path = gate1_run_dir / "provider-contract-b" / "paperos.toml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            f'''[data]
directory = "{gate1_run_dir / 'provider-contract-b-data'}"
dataset = "papers"
[llm]
provider = "custom"
model = "openai/mock-contract-b"
endpoint = "{endpoint}"
''',
            encoding="utf-8",
        )
        settings = load_settings(
            config_path,
            environ={"LLM_API_KEY": "mock-key"},
        )
        asyncio.run(_run_provider_flows(settings))


@pytest.mark.skipif(
    os.getenv("PAPEROS_RUN_LIVE_PROVIDER_CONTRACTS") != "1",
    reason="set PAPEROS_RUN_LIVE_PROVIDER_CONTRACTS=1 and LLM_API_KEY for the live DeepSeek contract",
)
def test_provider_a_deepseek_live_contract(gate1_run_dir: Path) -> None:
    """Provider A: live DeepSeek custom endpoint through the same PaperOS flows."""
    config_path = gate1_run_dir / "provider-contract-a" / "paperos.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f'''[data]
directory = "{gate1_run_dir / 'provider-contract-a-data'}"
dataset = "papers"
[llm]
provider = "custom"
model = "deepseek/deepseek-v4-flash"
endpoint = "https://api.deepseek.com/v1"
''',
        encoding="utf-8",
    )
    settings = load_settings(
        config_path,
        environ={"LLM_API_KEY": os.environ.get("LLM_API_KEY", "")},
    )
    asyncio.run(_run_provider_flows(settings))
