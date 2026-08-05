"""Document summary retrieval through Cognee public search."""

from __future__ import annotations

from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter
from paperos_core.adapters.cognee.search import CogneeSearchAdapter
from paperos_core.retrieval.candidates import Candidate
from paperos_core.retrieval.corpus import CorpusView
from paperos_core.retrieval.semantic import summary_retrieve


async def global_context_retrieve(
    search: CogneeSearchAdapter,
    compat: CogneeCompatibilityAdapter,
    corpus: CorpusView,
    query: str,
    *,
    dataset_name: str,
    limit: int,
    document_ids: set[str],
) -> list[Candidate]:
    return await summary_retrieve(
        search,
        compat,
        corpus,
        query,
        dataset_name=dataset_name,
        limit=limit,
        document_ids=document_ids,
    )
