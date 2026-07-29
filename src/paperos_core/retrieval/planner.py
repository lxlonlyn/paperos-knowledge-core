"""Query planning boundary."""

from paperos_core.config import PaperOSConfig
from paperos_core.retrieval.candidates import QueryPlan, QueryRequest
from paperos_core.retrieval.profiles import build_query_plan


class QueryPlanner:
    def __init__(self, config: PaperOSConfig) -> None:
        self.config = config

    def plan(self, request: QueryRequest) -> QueryPlan:
        return build_query_plan(request, self.config)
