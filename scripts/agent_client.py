"""Small HTTP-only PaperOS client suitable for agents and debugger use."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    subcommands = parser.add_subparsers(dest="command", required=True)
    ingest = subcommands.add_parser("ingest")
    ingest.add_argument("pdf", type=Path)
    ingest.add_argument("--dataset")
    query = subcommands.add_parser("query")
    query.add_argument("question")
    query.add_argument("--profile", default="comprehensive")
    job = subcommands.add_parser("job")
    job.add_argument("job_id")
    subcommands.add_parser("health")
    subcommands.add_parser("documents")
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=300) as client:
        if args.command == "ingest":
            with args.pdf.open("rb") as stream:
                response = client.post(
                    "/api/v1/ingest",
                    params={"dataset": args.dataset} if args.dataset else None,
                    files={"file": (args.pdf.name, stream, "application/pdf")},
                )
            response.raise_for_status()
            payload = response.json()
            while payload.get("status") in {"pending", "running"}:
                time.sleep(1)
                status_response = client.get(f"/api/v1/jobs/{payload['job_id']}")
                status_response.raise_for_status()
                payload = status_response.json()
        elif args.command == "query":
            response = client.post(
                "/api/v1/query",
                json={"query": args.question, "profile": args.profile},
            )
            response.raise_for_status()
            payload = response.json()
        elif args.command == "job":
            response = client.get(f"/api/v1/jobs/{args.job_id}")
            response.raise_for_status()
            payload = response.json()
        elif args.command == "documents":
            response = client.get("/api/v1/documents")
            response.raise_for_status()
            payload = response.json()
        else:
            response = client.get("/api/v1/health")
            response.raise_for_status()
            payload = response.json()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
