from __future__ import annotations

import time
from collections import Counter

from research_pipeline.agents.base import BaseAgent
from research_pipeline.schemas import (
    AgentMessage, Critique, Metadata, ReportSection, ResearchPlan, ResearchReport, SourceRecord,
)

MAX_RESEARCH_ITERATIONS = 2
MIN_ACCEPTABLE_CONFIDENCE = 0.55


class CriticAgent(BaseAgent):
    inbox = "critic"
    group = "critic_group"

    def handle(self, message: AgentMessage) -> None:
        if message.msg_type != "synthesis.done":
            return
        payload = message.payload
        plan = ResearchPlan.model_validate(payload["plan"])
        sources = [SourceRecord.model_validate(s) for s in payload["sources"]]
        sections = [ReportSection.model_validate(s) for s in payload["sections"]]
        summary = payload["summary"]
        iteration = plan.iteration
        start_time = plan.start_time

        gaps = [s.heading for s in sections if not s.citations]
        bias_flags = self._detect_bias(sources)
        confidence = self._score_confidence(sections, sources, gaps)

        self.logger.info(
            "Critique for request=%s: confidence=%.2f gaps=%d bias_flags=%d iteration=%d",
            message.request_id, confidence, len(gaps), len(bias_flags), iteration,
        )

        needs_research = confidence < MIN_ACCEPTABLE_CONFIDENCE or gaps
        if needs_research and iteration < MAX_RESEARCH_ITERATIONS:
            self.logger.info(
                "Triggering re-search loop (iteration %d -> %d) for request=%s",
                iteration, iteration + 1, message.request_id,
            )
            next_plan = plan.model_copy(update={
                "max_sources": min(50, plan.max_sources + 5),
                "iteration": iteration + 1,
            })
            self.emit(
                channel="searcher",
                request_id=message.request_id,
                recipient="SearcherAgent",
                msg_type="plan.created",
                payload=next_plan.model_dump(),
            )
            return

        critique = Critique(
            confidence_score=round(confidence, 3),
            gaps=gaps,
            bias_flags=bias_flags,
        )
        report = ResearchReport(
            topic=plan.topic,
            summary=summary,
            sections=sections,
            sources=sources,
            critique=critique,
            metadata=Metadata(
                total_urls_visited=len(sources),
                agent_interactions=payload.get("agent_interactions", 4),
                wall_clock_seconds=round(time.time() - start_time, 3),
            ),
        )
        self.emit(
            channel=f"output.{message.request_id}",
            request_id=message.request_id,
            recipient="Supervisor",
            msg_type="report.done",
            payload=report.model_dump(),
        )

    def _detect_bias(self, sources: list[SourceRecord]) -> list[str]:
        flags = []
        if not sources:
            return ["no sources available to assess bias"]
        domain_counts = Counter(s.url.split("/")[2] for s in sources)
        total = len(sources)
        for domain, count in domain_counts.items():
            share = count / total
            if share > 0.4:
                flags.append(f"over-reliance on single domain '{domain}' ({share:.0%} of sources)")
        low_relevance = sum(1 for s in sources if s.relevance_score < 0.15)
        if low_relevance / total > 0.5:
            flags.append("more than half of sources have low relevance scores")
        return flags

    def _score_confidence(
        self, sections: list[ReportSection], sources: list[SourceRecord], gaps: list[str]
    ) -> float:
        if not sections:
            return 0.0
        coverage_ratio = (len(sections) - len(gaps)) / len(sections)
        avg_relevance = sum(s.relevance_score for s in sources) / len(sources) if sources else 0.0
        source_diversity = min(1.0, len({s.url.split("/")[2] for s in sources}) / max(1, len(sources) * 0.3))
        return 0.5 * coverage_ratio + 0.35 * avg_relevance + 0.15 * source_diversity
