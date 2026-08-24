"""Audit Claim and ABOUT production artifacts for the configured data root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.config import load_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/paperos.toml"))
    parser.add_argument("--expect-enabled", action="store_true")
    args = parser.parse_args()
    settings = load_settings(args.config)
    graph_root = settings.data_dir / "cognee" / "graphs"
    claim_count = 0
    about_count = 0
    for path in sorted(graph_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        claim_count += sum(
            node.get("__type__") == "ClaimDataPoint"
            for node in payload.get("nodes", [])
        )
        about_count += sum(
            relation.get("relation_type") == "ABOUT"
            for relation in payload.get("relations", [])
        )
    enabled = settings.ingestion.claim_enrichment_enabled
    passed = enabled == args.expect_enabled and (
        enabled or (claim_count == 0 and about_count == 0)
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "claim_enrichment_enabled": enabled,
        "claim_count": claim_count,
        "about_edge_count": about_count,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
