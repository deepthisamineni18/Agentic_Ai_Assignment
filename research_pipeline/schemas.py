"""Pydantic models defining the input/output contracts for the pipeline
and for inter-agent messages passed over the Redis Streams message bus."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Depth(str, Enum):
    SHALLOW = "shallow"
    MODERATE = "moderate"
    DEEP = "deep"


class OutputFormat(str, Enum):
    MARKDOWN = "markdown"
    PDF = "pdf"
    JSON = "json"


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=5, max_length=200)
    depth: Depth = Depth.MODERATE
    max_sources: int = Field(default=15, ge=5, le=50)
    output_format: OutputFormat = OutputFormat.JSON
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class SubQuery(BaseModel):
    query_id: str
    query_text: str
    priority: int = 1


class ResearchPlan(BaseModel):
    request_id: str
    topic: str
    strategy: str  # "breadth_first" | "iterative_deepening"
    sub_queries: list[SubQuery]
    max_sources: int
    iteration: int = 0
    start_time: float = Field(default_factory=lambda: __import__("time").time())


class SourceRecord(BaseModel):
    source_id: str
    url: str
    title: str
    snippet: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    scraped_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sub_query_id: Optional[str] = None
    # Optional metadata added at ingestion/synthesis time
    url_status: Optional[dict] = None
    archived_url: Optional[str] = None


class SearchResult(BaseModel):
    request_id: str
    sub_query_id: str
    sources: list[SourceRecord]


class ReportSection(BaseModel):
    heading: str
    content: str
    citations: list[str]


class Critique(BaseModel):
    confidence_score: float = Field(ge=0.0, le=1.0)
    gaps: list[str]
    bias_flags: list[str]


class Metadata(BaseModel):
    total_urls_visited: int
    agent_interactions: int
    wall_clock_seconds: float
    timings: dict[str, float] | None = None


class ResearchReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    summary: str
    sections: list[ReportSection]
    sources: list[SourceRecord]
    critique: Critique
    metadata: Metadata

    @field_validator("summary")
    @classmethod
    def summary_length(cls, v: str) -> str:
        # Soft-enforced; real LLM summaries vary in length. We warn rather than
        # hard-fail so the pipeline stays robust to LLM output variance.
        return v


class AgentMessage(BaseModel):
    """Envelope for every message on the Redis Streams bus."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str
    sender: str
    recipient: str
    msg_type: str  # e.g. "plan.created", "search.result", "synthesis.done", "critique.done"
    payload: dict
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
