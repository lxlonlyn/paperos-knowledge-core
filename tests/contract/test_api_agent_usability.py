"""Fast contracts for the public API and agent-oriented CLI surface."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import httpx
import pytest

from paperos_core.api.app import create_app
from paperos_core.config import RuntimeSettings
from paperos_core.errors import DocumentNotFoundError
from paperos_core.jobs.queue import JobQueue
from paperos_core.paths import build_data_paths
from paperos_core.storage.initializer import StorageInitializer
from scripts import agent_client


class _Documents:
    def inspect(self, document_id: str) -> SimpleNamespace:
        if document_id != "doc_known":
            raise DocumentNotFoundError(
                f"Document '{document_id}' does not exist.", affected=document_id
            )
        return SimpleNamespace(model_dump=lambda **_kwargs: {"id": document_id})


def test_api_job_contract_and_not_found_mapping(tmp_path: Path) -> None:
    async def scenario() -> None:
        settings = RuntimeSettings.model_validate(
            {"data": {"directory": tmp_path / "data"}}
        )
        paths = build_data_paths(settings.data_dir)
        StorageInitializer(paths).initialize()
        queue = JobQueue(paths)
        api = create_app(settings)
        api.state.paperos = SimpleNamespace(
            paths=paths,
            settings=settings,
            queue=queue,
            services=SimpleNamespace(documents=_Documents()),
        )
        transport = httpx.ASGITransport(app=api)

        async with httpx.AsyncClient(
            transport=transport, base_url="http://paperos.test"
        ) as client:
            responses = [
                await client.post(
                    "/api/v1/ingest",
                    files={"file": ("paper.pdf", b"%PDF-1.7\n", "application/pdf")},
                ),
                await client.post("/api/v1/documents/doc_known/reprocess"),
                await client.post("/api/v1/rebuild"),
                await client.post("/api/v1/improve"),
            ]
            for response in responses:
                assert response.status_code == 202
                assert set(response.json()) == {"id", "status"}
                assert response.json()["id"].startswith("opjob_")
                assert response.json()["status"] == "pending"

            ids = [response.json()["id"] for response in responses]
            claimed = queue.claim_next()
            assert claimed is not None and claimed.id == ids[0]
            queue.complete(ids[1], {"document_id": "doc_known"})
            queue.fail(ids[2], "private /staging/path must remain private")

            for job_id in ids:
                status_response = await client.get(f"/api/v1/jobs/{job_id}")
                assert status_response.status_code == 200
                assert status_response.json()["id"] == job_id
                assert "job_id" not in status_response.json()

            listed_response = await client.get("/api/v1/jobs", params={"limit": 50})
            assert listed_response.status_code == 200
            listed = listed_response.json()
            assert {job["status"] for job in listed} == {
                "pending",
                "running",
                "completed",
                "failed",
            }
            assert {job["id"] for job in listed} == set(ids)
            assert all("path" not in job["payload"] for job in listed)
            assert "/staging/path" not in json.dumps(listed)

            assert (await client.get("/api/v1/jobs", params={"limit": 0})).status_code == 422
            assert (
                await client.get("/api/v1/jobs", params={"limit": 1001})
            ).status_code == 422

            before = len(queue.list_jobs(limit=1000))
            missing_reprocess = await client.post(
                "/api/v1/documents/doc_missing/reprocess"
            )
            assert missing_reprocess.status_code == 404
            assert missing_reprocess.json()["error"]["code"] == "document_not_found"
            assert len(queue.list_jobs(limit=1000)) == before

            missing_document = await client.get("/api/v1/documents/doc_missing")
            assert missing_document.status_code == 404
            assert missing_document.json()["error"]["code"] == "document_not_found"
            missing_job = await client.get("/api/v1/jobs/opjob_missing")
            assert missing_job.status_code == 404
            assert (
                missing_job.json()["error"]["code"]
                == "operational_job_not_found"
            )

    asyncio.run(scenario())


def _response(
    method: str, path: str, payload: object, *, status_code: int = 200
) -> httpx.Response:
    request = httpx.Request(method, f"http://paperos.test{path}")
    return httpx.Response(status_code, json=payload, request=request)


class _FakeClient:
    def __init__(
        self,
        responses: list[tuple[str, str, httpx.Response]],
        *,
        base_url_log: list[str] | None = None,
    ) -> None:
        self.responses = responses
        self.base_url_log = base_url_log
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def factory(self, *, base_url: str, timeout: int) -> _FakeClient:
        assert timeout == 300
        if self.base_url_log is not None:
            self.base_url_log.append(base_url)
        return self

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        assert not self.responses, f"Unused HTTP responses: {self.responses}"

    def _take(self, method: str, path: str, kwargs: dict[str, object]) -> httpx.Response:
        expected_method, expected_path, response = self.responses.pop(0)
        assert (method, path) == (expected_method, expected_path)
        self.calls.append((method, path, kwargs))
        return response

    def post(self, path: str, **kwargs: object) -> httpx.Response:
        return self._take("POST", path, kwargs)

    def get(self, path: str, **kwargs: object) -> httpx.Response:
        return self._take("GET", path, kwargs)


def _install_client(
    monkeypatch: pytest.MonkeyPatch,
    fake: _FakeClient,
) -> None:
    monkeypatch.setattr(agent_client.httpx, "Client", fake.factory)
    monkeypatch.setattr(agent_client.time, "sleep", lambda _seconds: None)


def test_ingest_polling_uses_creation_id_and_failed_is_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    completed = _FakeClient(
        [
            (
                "POST",
                "/api/v1/ingest",
                _response(
                    "POST",
                    "/api/v1/ingest",
                    {"id": "opjob_stable", "status": "pending"},
                    status_code=202,
                ),
            ),
            (
                "GET",
                "/api/v1/jobs/opjob_stable",
                _response(
                    "GET",
                    "/api/v1/jobs/opjob_stable",
                    {"id": "opjob_stable", "status": "running"},
                ),
            ),
            (
                "GET",
                "/api/v1/jobs/opjob_stable",
                _response(
                    "GET",
                    "/api/v1/jobs/opjob_stable",
                    {"id": "opjob_stable", "status": "completed", "result": {}},
                ),
            ),
        ]
    )
    _install_client(monkeypatch, completed)
    assert (
        agent_client.run(["--base-url", "http://paperos.test", "ingest", str(pdf)])
        == 0
    )
    output = capsys.readouterr()
    assert output.err.count("Job: opjob_stable") == 1
    assert json.loads(output.out)["status"] == "completed"

    failed = _FakeClient(
        [
            (
                "POST",
                "/api/v1/ingest",
                _response(
                    "POST",
                    "/api/v1/ingest",
                    {"id": "opjob_failed", "status": "pending"},
                    status_code=202,
                ),
            ),
            (
                "GET",
                "/api/v1/jobs/opjob_failed",
                _response(
                    "GET",
                    "/api/v1/jobs/opjob_failed",
                    {
                        "id": "opjob_failed",
                        "status": "failed",
                        "error": {
                            "code": "operational_job_failed",
                            "message": "The operation could not be completed.",
                        },
                    },
                ),
            ),
        ]
    )
    _install_client(monkeypatch, failed)
    assert (
        agent_client.run(["--base-url", "http://paperos.test", "ingest", str(pdf)])
        != 0
    )
    output = capsys.readouterr()
    assert "paperos error: operational_job_failed" in output.err
    assert "Traceback" not in output.err


def test_no_wait_jobs_and_configured_base_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    no_wait = _FakeClient(
        [
            (
                "POST",
                "/api/v1/ingest",
                _response(
                    "POST",
                    "/api/v1/ingest",
                    {"id": "opjob_nowait", "status": "pending"},
                    status_code=202,
                ),
            )
        ]
    )
    _install_client(monkeypatch, no_wait)
    assert (
        agent_client.run(
            ["--base-url", "http://paperos.test", "ingest", str(pdf), "--no-wait"]
        )
        == 0
    )
    output = capsys.readouterr()
    assert "Job: opjob_nowait" in output.err

    settings = RuntimeSettings.model_validate(
        {"data": {"directory": tmp_path / "data"}, "api": {"port": 18473}}
    )
    monkeypatch.setattr(agent_client, "load_settings", lambda: settings)
    base_urls: list[str] = []
    jobs = _FakeClient(
        [
            (
                "GET",
                "/api/v1/jobs",
                _response("GET", "/api/v1/jobs", []),
            )
        ],
        base_url_log=base_urls,
    )
    _install_client(monkeypatch, jobs)
    assert agent_client.run(["jobs", "--limit", "50"]) == 0
    assert base_urls == ["http://127.0.0.1:18473"]
    assert jobs.calls[0][2]["params"] == {"limit": 50}


def _query_payload(
    *,
    query_id: str = "query:history/1",
    replay_text: str = "# Replay\n\nUse this evidence.",
    research_replay_text: str = "# Research Replay\n\nContinue the research.",
) -> dict[str, object]:
    return {
        "id": query_id,
        "query": "normalized query",
        "dataset": "papers-a",
        "answer": "Evidence-bound answer.",
        "answer_model": "test/model",
        "replay": {
            "original_query": "server copy",
            "replay_text": replay_text,
            "research_replay_text": research_replay_text,
        },
        "candidates": [],
        "evidence": [],
    }


@pytest.mark.parametrize(
    ("output_flag", "expected"),
    [
        ([], "Evidence-bound answer.\n"),
        (["--replay"], "# Research Replay\n\nContinue the research.\n"),
        (["--synthesis-replay"], "# Replay\n\nUse this evidence.\n"),
    ],
)
def test_query_text_modes_and_history(
    output_flag: list[str],
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = RuntimeSettings.model_validate(
        {"data": {"directory": tmp_path / "data"}, "api": {"port": 18473}}
    )
    monkeypatch.setattr(agent_client, "load_settings", lambda: settings)
    fake = _FakeClient(
        [
            (
                "POST",
                "/api/v1/query",
                _response("POST", "/api/v1/query", _query_payload()),
            )
        ]
    )
    _install_client(monkeypatch, fake)
    original = "What did the paper actually show?"
    assert (
        agent_client.run(
            [
                "query",
                original,
                "--document-id",
                "doc_1",
                "--work-id",
                "work_1",
                "--expand-context",
                *output_flag,
            ]
        )
        == 0
    )
    output = capsys.readouterr()
    assert output.out == expected
    assert [(method, path) for method, path, _kwargs in fake.calls] == [
        ("POST", "/api/v1/query")
    ]
    if not output_flag:
        assert "Research Replay" not in output.out

    history_path = settings.data_dir / "query_history" / "queries.jsonl"
    entry = json.loads(history_path.read_text(encoding="utf-8"))
    assert entry["original_query"] == original
    assert entry["document_ids"] == ["doc_1"]
    assert entry["work_ids"] == ["work_1"]
    assert entry["expand_context"] is True
    assert entry["history_id"].startswith("history_")
    assert entry["query_response_id"] == "query:history/1"
    assert entry["replay_file"] == f"replay/{entry['history_id']}.md"
    replay_path = settings.data_dir / "query_history" / entry["replay_file"]
    assert replay_path.read_text(encoding="utf-8") == (
        "# Research Replay\n\nContinue the research."
    )


def test_repeated_query_response_identity_keeps_distinct_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = RuntimeSettings.model_validate(
        {"data": {"directory": tmp_path / "data"}}
    )
    monkeypatch.setattr(agent_client, "load_settings", lambda: settings)
    query_clients: list[_FakeClient] = []

    for research_replay_text in ("Replay A", "Replay B"):
        fake = _FakeClient(
            [
                (
                    "POST",
                    "/api/v1/query",
                    _response(
                        "POST",
                        "/api/v1/query",
                        _query_payload(
                            query_id="query_same",
                            research_replay_text=research_replay_text,
                        ),
                    ),
                )
            ]
        )
        query_clients.append(fake)
        _install_client(monkeypatch, fake)
        assert agent_client.run(["query", "Same request"]) == 0
        capsys.readouterr()

    assert sum(len(client.calls) for client in query_clients) == 2
    assert all(
        [(method, path) for method, path, _kwargs in client.calls]
        == [("POST", "/api/v1/query")]
        for client in query_clients
    )

    history_path = settings.data_dir / "query_history" / "queries.jsonl"
    records = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    first, second = records
    assert first["query_response_id"] == "query_same"
    assert second["query_response_id"] == "query_same"
    assert first["history_id"] != second["history_id"]
    assert first["replay_file"] != second["replay_file"]

    first_replay = settings.data_dir / "query_history" / first["replay_file"]
    second_replay = settings.data_dir / "query_history" / second["replay_file"]
    assert first_replay.read_text(encoding="utf-8") == "Replay A"
    assert second_replay.read_text(encoding="utf-8") == "Replay B"


def test_no_evidence_query_still_saves_research_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = RuntimeSettings.model_validate(
        {"data": {"directory": tmp_path / "data"}}
    )
    monkeypatch.setattr(agent_client, "load_settings", lambda: settings)
    no_evidence_answer = "No PaperOS evidence was retrieved."
    research_replay = (
        "Original question: unresolved?\n"
        "PaperOS did not retrieve supporting evidence.\n"
        "This does not establish a negative answer."
    )
    payload = _query_payload(
        replay_text="",
        research_replay_text=research_replay,
    )
    payload["answer"] = no_evidence_answer
    payload["answer_model"] = "paperos/no-evidence"
    fake = _FakeClient(
        [
            (
                "POST",
                "/api/v1/query",
                _response("POST", "/api/v1/query", payload),
            )
        ]
    )
    _install_client(monkeypatch, fake)

    assert agent_client.run(["query", "unresolved?"]) == 0

    output = capsys.readouterr()
    assert output.out == f"{no_evidence_answer}\n"
    assert research_replay not in output.out
    assert [(method, path) for method, path, _kwargs in fake.calls] == [
        ("POST", "/api/v1/query")
    ]
    history_path = settings.data_dir / "query_history" / "queries.jsonl"
    record = json.loads(history_path.read_text(encoding="utf-8"))
    replay_path = settings.data_dir / "query_history" / record["replay_file"]
    assert replay_path.read_text(encoding="utf-8") == research_replay


def test_history_failure_warns_without_retrying_successful_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = RuntimeSettings.model_validate(
        {"data": {"directory": tmp_path / "data"}}
    )
    monkeypatch.setattr(agent_client, "load_settings", lambda: settings)

    def fail_history(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected history failure")

    monkeypatch.setattr(agent_client, "_save_query_history", fail_history)
    fake = _FakeClient(
        [
            (
                "POST",
                "/api/v1/query",
                _response("POST", "/api/v1/query", _query_payload()),
            )
        ]
    )
    _install_client(monkeypatch, fake)

    assert agent_client.run(["query", "Question?"]) == 0

    output = capsys.readouterr()
    assert output.out == "Evidence-bound answer.\n"
    assert output.err == (
        "paperos warning: query history was not saved (OSError).\n"
    )
    assert [(method, path) for method, path, _kwargs in fake.calls] == [
        ("POST", "/api/v1/query")
    ]


def test_query_json_and_typed_http_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = RuntimeSettings.model_validate(
        {"data": {"directory": tmp_path / "data"}}
    )
    monkeypatch.setattr(agent_client, "load_settings", lambda: settings)
    query = _FakeClient(
        [
            (
                "POST",
                "/api/v1/query",
                _response("POST", "/api/v1/query", _query_payload()),
            )
        ]
    )
    _install_client(monkeypatch, query)
    assert agent_client.run(["query", "Question?", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == _query_payload()
    assert [(method, path) for method, path, _kwargs in query.calls] == [
        ("POST", "/api/v1/query")
    ]
    history_path = settings.data_dir / "query_history" / "queries.jsonl"
    record = json.loads(history_path.read_text(encoding="utf-8"))
    replay_path = settings.data_dir / "query_history" / record["replay_file"]
    assert replay_path.read_text(encoding="utf-8") == (
        "# Research Replay\n\nContinue the research."
    )

    error_client = _FakeClient(
        [
            (
                "GET",
                "/api/v1/jobs/opjob_missing",
                _response(
                    "GET",
                    "/api/v1/jobs/opjob_missing",
                    {
                        "error": {
                            "code": "operational_job_not_found",
                            "message": "The requested operational job was not found.",
                            "retryable": False,
                        }
                    },
                    status_code=404,
                ),
            )
        ]
    )
    _install_client(monkeypatch, error_client)
    assert (
        agent_client.run(
            ["--base-url", "http://paperos.test", "job", "opjob_missing"]
        )
        != 0
    )
    output = capsys.readouterr()
    assert output.out == ""
    assert "paperos error: operational_job_not_found" in output.err
    assert "Traceback" not in output.err
