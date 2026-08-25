"""Contracts for the portable, exact final-synthesis Query Replay."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

from paperos_core.adapters.cognee.llm import AnswerOutput, LLMClient
from paperos_core.retrieval.candidates import Evidence, QueryReplay, QueryResponse
from paperos_core.retrieval.synthesis import (
    FinalSynthesisContext,
    render_synthesis_prompt,
    synthesize_answer,
)


def _evidence(
    chunk_id: str,
    text: str,
    *,
    title: str,
    authors: list[str] | None = None,
    year: int | None = None,
    section_path: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
) -> Evidence:
    return Evidence(
        evidence_id=f"evidence:{chunk_id}",
        chunk_id=chunk_id,
        document_id=f"document_{chunk_id}",
        source_file_id=f"source_{chunk_id}",
        source_filename=f"{chunk_id}.pdf",
        title=title,
        authors=authors or [],
        year=year,
        section_path=section_path,
        page_start=page_start,
        page_end=page_end,
        text=text,
        channels=["vector"],
        knowledge_kind="source_fact",
        derived_from_ids=[],
    )


def test_renderer_preserves_query_canonical_text_order_and_available_metadata() -> None:
    original_query = "  NISE、ADADIV 与 EFIS 有何差异？\n请逐项比较。  "
    first = _evidence(
        "chunk_z",
        "完整 canonical text Z，不应被截断。",
        title="Paper Z",
        authors=["Ada A.", "Bo B."],
        year=2024,
        section_path="4 / Limitations",
        page_start=7,
        page_end=9,
    )
    second = _evidence(
        "chunk_a",
        "canonical text A",
        title="Paper A",
    )

    prompt = render_synthesis_prompt(
        FinalSynthesisContext(original_query=original_query, evidence=[first, second])
    )

    assert f"# Research Question\n\n{original_query}\n\n# Evidence" in prompt
    assert prompt.index("## Evidence 1") < prompt.index("## Evidence 2")
    assert prompt.index(first.text) < prompt.index(second.text)
    assert "Evidence ID: evidence:chunk_z" in prompt
    assert "Paper: Paper Z" in prompt
    assert "Authors: Ada A.; Bo B." in prompt
    assert "Year: 2024" in prompt
    assert "Section: 4 / Limitations" in prompt
    assert "Pages: 7-9" in prompt
    second_block = prompt.split("## Evidence 2", maxsplit=1)[1]
    assert "Authors:" not in second_block
    assert "Year:" not in second_block
    assert "Section:" not in second_block
    assert "Pages:" not in second_block
    assert prompt.endswith("State clearly when the supplied evidence is insufficient.")


def test_synthesis_wrapper_passes_the_rendered_prompt_without_rerendering() -> None:
    item = _evidence("chunk_exact", "canonical source", title="Exact Paper")
    prompt = render_synthesis_prompt(
        FinalSynthesisContext(original_query="原始问题", evidence=[item])
    )
    observed: dict[str, Any] = {}

    class Client:
        async def synthesize_answer(
            self, *, prompt: str, evidence: list[dict[str, Any]]
        ) -> str:
            observed["prompt"] = prompt
            observed["evidence"] = evidence
            return "answer [chunk_exact]"

    answer = asyncio.run(
        synthesize_answer(Client(), prompt=prompt, evidence=[item])  # type: ignore[arg-type]
    )

    assert observed["prompt"] is prompt
    assert observed["evidence"][0]["text"] == item.text
    assert answer == "answer [evidence:chunk_exact]"


def test_llm_adapter_sends_replay_as_the_exact_user_prompt(
    monkeypatch: Any,
) -> None:
    prompt = "# Task\n\nExact replay prompt\n"
    observed: dict[str, Any] = {}

    class Gateway:
        @staticmethod
        async def acreate_structured_output(**kwargs: Any) -> AnswerOutput:
            observed.update(kwargs)
            return AnswerOutput(answer="grounded", cited_chunk_ids=["chunk_exact"])

    cognee = ModuleType("cognee")
    infrastructure = ModuleType("cognee.infrastructure")
    llm_module = ModuleType("cognee.infrastructure.llm")
    llm_module.LLMGateway = Gateway  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cognee", cognee)
    monkeypatch.setitem(sys.modules, "cognee.infrastructure", infrastructure)
    monkeypatch.setitem(sys.modules, "cognee.infrastructure.llm", llm_module)

    client = LLMClient(
        SimpleNamespace(load=lambda _name: "stable provider instruction"),
        SimpleNamespace(),
    )
    answer = asyncio.run(
        client.synthesize_answer(
            prompt=prompt,
            evidence=[
                {
                    "evidence_id": "evidence:chunk_exact",
                    "chunk_id": "chunk_exact",
                    "text": "canonical source",
                }
            ],
        )
    )

    assert observed["text_input"] is prompt
    assert observed["system_prompt"] == "stable provider instruction"
    assert answer == "grounded [chunk_exact]"


def test_replay_is_the_only_new_serialized_query_result_object() -> None:
    replay = QueryReplay(original_query="original", replay_text="# Task\n...")

    assert set(QueryReplay.model_fields) == {"original_query", "replay_text"}
    assert QueryResponse.model_fields["replay"].annotation is QueryReplay
    assert replay.model_dump(mode="json") == {
        "original_query": "original",
        "replay_text": "# Task\n...",
    }
