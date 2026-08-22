"""Remove chunk-corpus-review artifacts that were wrongly written to production DATA_DIR.

Run once after the 2026-08-22 incident where chunk_corpus_review.py used the
default config data directory instead of validation/runs/chunk.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
REGISTRY = DATA / "jobs" / "registry.sqlite3"

# Source files created 2026-08-22 by chunk_corpus_review (dataset chunk-corpus-review).
POLLUTED_SOURCE_IDS = (
    "src_e8b6fc363eabba81ab1b33cd8a38d73a",
    "src_112cae73ae2bed3bdd3719fb2e0a2124",
    "src_afb359df0abb2bc71e95474832446f20",
    "src_dbc577278449898ca30b54ed86b44d63",
    "src_30c5539c0796260a5080af02d5a3c7b6",
    "src_0a6d556646aec7d48b873ddd1d800a5b",
)


def main() -> None:
    placeholders = ",".join("?" for _ in POLLUTED_SOURCE_IDS)
    with sqlite3.connect(REGISTRY) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        parse_run_ids = [
            row[0]
            for row in connection.execute(
                f"SELECT id FROM parse_runs WHERE source_file_id IN ({placeholders})",
                POLLUTED_SOURCE_IDS,
            )
        ]
        if parse_run_ids:
            run_placeholders = ",".join("?" for _ in parse_run_ids)
            connection.execute(
                f"DELETE FROM parser_artifacts WHERE parse_run_id IN ({run_placeholders})",
                parse_run_ids,
            )
        connection.execute(
            f"DELETE FROM canonical_snapshots WHERE source_file_id IN ({placeholders})",
            POLLUTED_SOURCE_IDS,
        )
        connection.execute(
            f"DELETE FROM parse_runs WHERE source_file_id IN ({placeholders})",
            POLLUTED_SOURCE_IDS,
        )
        connection.execute(
            f"DELETE FROM ingestion_jobs WHERE source_file_id IN ({placeholders})",
            POLLUTED_SOURCE_IDS,
        )
        connection.execute(
            f"DELETE FROM source_files WHERE id IN ({placeholders})",
            POLLUTED_SOURCE_IDS,
        )
        connection.commit()

    for source_id in POLLUTED_SOURCE_IDS:
        for subdir in ("raw", "parsed", "canonical"):
            path = DATA / subdir / source_id
            if path.exists():
                shutil.rmtree(path)
                print(f"removed {path}")

    print("production registry cleanup complete for", len(POLLUTED_SOURCE_IDS), "sources")


if __name__ == "__main__":
    main()
