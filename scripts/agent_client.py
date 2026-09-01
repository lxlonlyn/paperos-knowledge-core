"""Small HTTP-only PaperOS client suitable for agents and debugger use."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from paperos_core.config import RuntimeSettings, load_settings
from paperos_core.errors import PaperOSError, public_diagnostic

_POLL_INTERVAL_SECONDS = 1
_FAILED_JOB_EXIT_CODE = 2
_CLIENT_ERROR_EXIT_CODE = 1


class _ClientResponseError(Exception):
    """The server response did not satisfy the documented public contract."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url")
    subcommands = parser.add_subparsers(dest="command", required=True)

    ingest = subcommands.add_parser("ingest")
    ingest.add_argument("pdf", type=Path)
    ingest.add_argument("--dataset")
    ingest.add_argument("--no-wait", action="store_true")

    query = subcommands.add_parser("query")
    query.add_argument("question")
    query.add_argument("--dataset")
    query.add_argument("--document-id", action="append", dest="document_ids")
    query.add_argument("--work-id", action="append", dest="work_ids")
    query.add_argument("--expand-context", action="store_true")
    query.add_argument("--expand-graph", action="store_true")
    output = query.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true")
    output.add_argument("--replay", action="store_true")

    job = subcommands.add_parser("job")
    job.add_argument("job_id")
    jobs = subcommands.add_parser("jobs")
    jobs.add_argument("--limit", type=int, default=100, choices=range(1, 1001))
    subcommands.add_parser("health")
    subcommands.add_parser("documents")
    return parser


def _configured_base_url(settings: RuntimeSettings) -> str:
    return f"http://{settings.api.host}:{settings.api.port}"


def _response_payload(response: httpx.Response) -> object:
    response.raise_for_status()
    return response.json()


def _object_response(response: httpx.Response) -> dict[str, Any]:
    payload = _response_payload(response)
    if not isinstance(payload, dict):
        raise _ClientResponseError("PaperOS returned a non-object response.")
    return {str(key): value for key, value in payload.items()}


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _print_public_error(error: object) -> None:
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message")
        if isinstance(code, str) and isinstance(message, str):
            print(f"paperos error: {code}", file=sys.stderr)
            print(message, file=sys.stderr)
            return
    print("paperos error: request_failed", file=sys.stderr)
    print("The request could not be completed.", file=sys.stderr)


def _handle_http_status(error: httpx.HTTPStatusError) -> None:
    try:
        payload = error.response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        _print_public_error(payload.get("error"))
    else:
        _print_public_error(None)


def _save_query_history(
    settings: RuntimeSettings,
    payload: dict[str, Any],
    *,
    original_query: str,
    document_ids: list[str] | None,
    work_ids: list[str] | None,
    expand_context: bool,
    expand_graph: bool,
) -> None:
    history_root = settings.data_dir / "query_history"
    history_id = f"history_{uuid.uuid4().hex}"
    replay = payload.get("replay")
    replay_text = replay.get("replay_text") if isinstance(replay, dict) else None
    query_id_value = payload.get("id")
    replay_file: str | None = None
    if isinstance(replay_text, str) and replay_text:
        replay_name = f"{history_id}.md"
        replay_path = history_root / "replay" / replay_name
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text(replay_text, encoding="utf-8")
        replay_file = (Path("replay") / replay_name).as_posix()

    history_root.mkdir(parents=True, exist_ok=True)
    entry = {
        "history_id": history_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "query_response_id": query_id_value,
        "original_query": original_query,
        "answer": payload.get("answer"),
        "answer_model": payload.get("answer_model"),
        "dataset": payload.get("dataset"),
        "document_ids": document_ids or [],
        "work_ids": work_ids or [],
        "expand_context": expand_context,
        "expand_graph": expand_graph,
        "replay_file": replay_file,
    }
    with (history_root / "queries.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _query_output(payload: dict[str, Any], *, full_json: bool, replay_only: bool) -> None:
    if full_json:
        _print_json(payload)
        return
    if replay_only:
        replay = payload.get("replay")
        replay_text = replay.get("replay_text") if isinstance(replay, dict) else ""
        print(replay_text if isinstance(replay_text, str) else "")
        return
    answer = payload.get("answer")
    print(answer if isinstance(answer, str) else "")


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings: RuntimeSettings | None = None
    try:
        if args.base_url is None:
            settings = load_settings()
            base_url = _configured_base_url(settings)
        else:
            base_url = str(args.base_url).rstrip("/")

        with httpx.Client(base_url=base_url, timeout=300) as client:
            if args.command == "ingest":
                with args.pdf.open("rb") as stream:
                    response = client.post(
                        "/api/v1/ingest",
                        params={"dataset": args.dataset} if args.dataset else None,
                        files={"file": (args.pdf.name, stream, "application/pdf")},
                    )
                payload = _object_response(response)
                job_id = payload["id"]
                if not isinstance(job_id, str):
                    raise _ClientResponseError(
                        "PaperOS returned an invalid operational job ID."
                    )
                print(f"Job: {job_id}", file=sys.stderr, flush=True)
                if args.no_wait:
                    _print_json(payload)
                    return 0
                while payload.get("status") in {"pending", "running"}:
                    time.sleep(_POLL_INTERVAL_SECONDS)
                    payload = _object_response(client.get(f"/api/v1/jobs/{job_id}"))
                _print_json(payload)
                if payload.get("status") == "failed":
                    _print_public_error(payload.get("error"))
                    return _FAILED_JOB_EXIT_CODE
                return 0

            if args.command == "query":
                payload_body: dict[str, object] = {"query": args.question}
                if args.dataset:
                    payload_body["dataset"] = args.dataset
                if args.document_ids:
                    payload_body["document_ids"] = args.document_ids
                if args.work_ids:
                    payload_body["work_ids"] = args.work_ids
                payload_body["expand_context"] = args.expand_context
                payload_body["expand_graph"] = args.expand_graph
                payload = _object_response(client.post("/api/v1/query", json=payload_body))
                try:
                    history_settings = settings or load_settings()
                    _save_query_history(
                        history_settings,
                        payload,
                        original_query=args.question,
                        document_ids=args.document_ids,
                        work_ids=args.work_ids,
                        expand_context=args.expand_context,
                        expand_graph=args.expand_graph,
                    )
                except (OSError, PaperOSError, ValueError) as error:
                    print(
                        f"paperos warning: query history was not saved ({type(error).__name__}).",
                        file=sys.stderr,
                    )
                _query_output(payload, full_json=args.json, replay_only=args.replay)
                return 0

            output_payload: object
            if args.command == "job":
                output_payload = _object_response(
                    client.get(f"/api/v1/jobs/{args.job_id}")
                )
            elif args.command == "jobs":
                output_payload = _response_payload(
                    client.get("/api/v1/jobs", params={"limit": args.limit})
                )
            elif args.command == "documents":
                output_payload = _response_payload(client.get("/api/v1/documents"))
            else:
                output_payload = _object_response(client.get("/api/v1/health"))
        _print_json(output_payload)
        return 0
    except httpx.HTTPStatusError as error:
        _handle_http_status(error)
        return _CLIENT_ERROR_EXIT_CODE
    except httpx.RequestError:
        print("paperos error: connection_failed", file=sys.stderr)
        print(f"Unable to reach PaperOS at {base_url}.", file=sys.stderr)
        return _CLIENT_ERROR_EXIT_CODE
    except PaperOSError as error:
        diagnostic = public_diagnostic(error.code)
        _print_public_error(diagnostic)
        return _CLIENT_ERROR_EXIT_CODE
    except (KeyError, _ClientResponseError):
        print("paperos error: invalid_response", file=sys.stderr)
        print("PaperOS returned an invalid response.", file=sys.stderr)
        return _CLIENT_ERROR_EXIT_CODE


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
