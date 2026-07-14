from __future__ import annotations

import re
import uuid

from research_pipeline.agents.base import BaseAgent
from research_pipeline.llm_client import LLMClient
from research_pipeline.schemas import AgentMessage, Depth, ResearchPlan, SubQuery

STOPWORDS = {"the", "a", "an", "of", "in", "on", "for", "and", "to", "is", "are", "how", "what"}

# Angle templates used to decompose a topic into sub-queries deterministically
# (stands in for an LLM decomposition call; swap `_llm_decompose` in for a real call).
ANGLES = [
    "overview and definition",
    "recent developments",
    "key players and organizations",
    "challenges and risks",
    "economic or societal impact",
    "future outlook",
    "regulatory and policy landscape",
    "case studies and examples",
]

DEPTH_TO_NUM_SUBQUERIES = {Depth.SHALLOW: 3, Depth.MODERATE: 5, Depth.DEEP: 8}
DEPTH_TO_STRATEGY = {
    Depth.SHALLOW: "breadth_first",
    Depth.MODERATE: "breadth_first",
    Depth.DEEP: "iterative_deepening",
}


class PlannerAgent(BaseAgent):
    inbox = "planner"
    group = "planner_group"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._llm_client = kwargs.get("llm_client") or LLMClient()

    def handle(self, message: AgentMessage) -> None:
        if message.msg_type != "research.requested":
            return
        payload = message.payload
        topic = payload["topic"]
        depth = Depth(payload.get("depth", "moderate"))
        max_sources = payload["max_sources"]

        n = DEPTH_TO_NUM_SUBQUERIES[depth]
        strategy = DEPTH_TO_STRATEGY[depth]
        sub_queries = self._decompose(topic, n)

        plan = ResearchPlan(
            request_id=message.request_id,
            topic=topic,
            strategy=strategy,
            sub_queries=sub_queries,
            max_sources=max_sources,
        )
        self.logger.info(
            "Plan created for request=%s topic=%r strategy=%s n_subqueries=%d",
            message.request_id, topic, strategy, len(sub_queries),
        )
        self.emit(
            channel="searcher",
            request_id=message.request_id,
            recipient="SearcherAgent",
            msg_type="plan.created",
            payload=plan.model_dump(),
        )

    def _decompose(self, topic: str, n: int) -> list[SubQuery]:
        """Decompose a topic into sub-queries, preferring a real LLM call when
        available and falling back to the deterministic template path for
        offline reproducibility."""
        if getattr(self._llm_client, "is_available", lambda: False)():
            try:
                decomposed = self._llm_client.generate(
                    system=(
                        "You are an expert research planner. Return exactly n lines, each line "
                        "being one concise research sub-query for the given topic. Do not add bullets or commentary."
                    ),
                    user=f"Topic: {topic}\nNumber of sub-queries: {n}",
                    max_tokens=220,
                    temperature=0.2,
                )
                lines = [line.strip() for line in decomposed.splitlines() if line.strip()]
                if len(lines) >= n:
                    return [
                        SubQuery(query_id=str(uuid.uuid4())[:8], query_text=line, priority=i + 1)
                        for i, line in enumerate(lines[:n])
                    ]
            except Exception as exc:
                self.logger.warning("LLM-based topic decomposition failed, falling back to template: %s", exc)

        core_terms = [w for w in re.findall(r"[a-zA-Z]+", topic.lower()) if w not in STOPWORDS]
        core = " ".join(core_terms) if core_terms else topic
        angles = ANGLES[:n]
        return [
            SubQuery(query_id=str(uuid.uuid4())[:8], query_text=f"{core} {angle}", priority=i + 1)
            for i, angle in enumerate(angles)
        ]
