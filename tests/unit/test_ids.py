from paperos_core.domain.ids import (
    canonical_snapshot_id,
    chunk_id,
    document_id,
    source_file_id,
)


def test_source_file_id_is_stable_and_versioned() -> None:
    digest = "39a662c2bb98e7400f4273f0066a52e62ab931325197a6f9c7ce7f4e09c0dd3f"
    first = source_file_id(digest, id_version="1")
    second = source_file_id(digest.upper(), id_version="1")
    assert first == second
    assert first.startswith("src_")
    assert source_file_id(digest, id_version="2") != first


def test_canonical_ids_are_stable_and_versioned() -> None:
    source = "src_example"
    parse = "parse_example"
    assert document_id(source) == document_id(source)
    assert canonical_snapshot_id(parse) == canonical_snapshot_id(parse)
    assert chunk_id(document_id(source), 3, ["element_a", "element_b"]) == chunk_id(
        document_id(source), 3, ["element_a", "element_b"]
    )
    assert canonical_snapshot_id(parse, id_version="2") != canonical_snapshot_id(parse)
