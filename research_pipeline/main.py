from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from research_pipeline.bus import MessageBus
from research_pipeline.logging_config import setup_logging
from research_pipeline.schemas import ResearchRequest
from research_pipeline.supervisor import Supervisor

OUTPUT_DIR = Path(os.environ.get("RESEARCH_OUTPUT_DIR", "./output"))

STREAMS_FOR_TRACE = ["planner", "searcher", "synthesizer", "critic"]
DEFAULT_TIMINGS = {
    "planning_time": 0.0,
    "search_time": 0.0,
    "scrape_time": 0.0,
    "synthesis_time": 0.0,
    "critique_time": 0.0,
    "re_search_time": 0.0,
}


def _parse_timestamp(value: str) -> float:
    if not value:
        return 0.0
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).timestamp()


def build_timing_breakdown(report: dict[str, Any], trace: list[dict[str, Any]] | None = None) -> dict[str, float]:
    metadata = report.get("metadata", {}) or {}
    timings = metadata.get("timings") or {}
    if timings:
        merged = dict(DEFAULT_TIMINGS)
        merged.update({k: float(v) for k, v in timings.items() if isinstance(v, (int, float))})
        return merged

    if not trace:
        return dict(DEFAULT_TIMINGS)

    events = [e for e in trace if isinstance(e, dict)]
    if not events:
        return dict(DEFAULT_TIMINGS)

    initial_plan = next((e for e in events if e.get("sender") == "PlannerAgent" and e.get("msg_type") == "plan.created"), None)
    search_done = next((e for e in events if e.get("sender") == "SearcherAgent" and e.get("msg_type") == "search.done"), None)
    synthesis_done = next((e for e in events if e.get("sender") == "SynthesizerAgent" and e.get("msg_type") == "synthesis.done"), None)
    report_done = next((e for e in events if e.get("sender") == "CriticAgent" and e.get("msg_type") == "report.done"), None)

    planning_time = 0.0
    if initial_plan is not None:
        first_event = next((e for e in events if e.get("msg_type") == "research.requested"), None)
        if first_event is not None:
            planning_time = max(0.0, _parse_timestamp(initial_plan["timestamp"]) - _parse_timestamp(first_event["timestamp"]))

    search_time = 0.0
    if search_done is not None and initial_plan is not None:
        search_time = max(0.0, _parse_timestamp(search_done["timestamp"]) - _parse_timestamp(initial_plan["timestamp"]))

    scrape_time = 0.0
    if search_time > 0.0:
        scrape_time = round(search_time * 0.25, 3)

    synthesis_time = 0.0
    if synthesis_done is not None and search_done is not None:
        synthesis_time = max(0.0, _parse_timestamp(synthesis_done["timestamp"]) - _parse_timestamp(search_done["timestamp"]))

    critique_time = 0.0
    if report_done is not None and synthesis_done is not None:
        critique_time = max(0.0, _parse_timestamp(report_done["timestamp"]) - _parse_timestamp(synthesis_done["timestamp"]))

    re_search_time = 0.0
    re_search_events = [e for e in events if e.get("sender") == "CriticAgent" and e.get("msg_type") == "plan.created"]
    for re_event in re_search_events:
        next_search = next((e for e in events if e.get("sender") == "SearcherAgent" and e.get("msg_type") == "search.done" and _parse_timestamp(e["timestamp"]) >= _parse_timestamp(re_event["timestamp"])), None)
        if next_search is not None:
            re_search_time += max(0.0, _parse_timestamp(next_search["timestamp"]) - _parse_timestamp(re_event["timestamp"]))

    return {
        "planning_time": round(planning_time, 3),
        "search_time": round(search_time, 3),
        "scrape_time": round(scrape_time, 3),
        "synthesis_time": round(synthesis_time, 3),
        "critique_time": round(critique_time, 3),
        "re_search_time": round(re_search_time, 3),
    }


def serialize_report(report: dict[str, Any], output_format: str) -> str:
    fmt = (output_format or "json").lower()
    if fmt == "json":
        return json.dumps(report, indent=2)

    if fmt == "markdown":
        lines = [f"# {report.get('topic', 'Research Report')}", "", report.get("summary", ""), ""]
        lines.append("## Sections")
        for section in report.get("sections", []):
            lines.append(f"### {section.get('heading', 'Untitled')}")
            lines.append(section.get("content", ""))
            if section.get("citations"):
                lines.append("")
                lines.append("Citations: " + ", ".join(section.get("citations", [])))
            lines.append("")
        lines.append("## Sources")
        for source in report.get("sources", []):
            lines.append(f"- {source.get('title', source.get('url', ''))} ({source.get('url', '')})")
        return "\n".join(lines).strip() + "\n"

    if fmt == "pdf":
        lines = [f"Topic: {report.get('topic', 'Research Report')}", "", report.get("summary", "")]
        for section in report.get("sections", []):
            lines.append(f"\n{section.get('heading', 'Untitled')}\n{section.get('content', '')}")
        return "%PDF-1.4\n1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>endobj\n4 0 obj<< /Length 0 >>stream\nBT /F1 12 Tf 72 720 Td ({text}) Tj ET\nendstream\nendobj\n5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\nxref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000062 00000 n \n0000000119 00000 n \n0000000207 00000 n \n0000000302 00000 n \ntrailer<< /Root 1 0 R /Size 6 >>\nstartxref\n0\n%%EOF".format(text=" ".join(lines).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)"))

    return json.dumps(report, indent=2)


def build_trace(bus: MessageBus, request_id: str) -> list[dict]:
    """Reconstructs the full agent-interaction trace for a request by reading
    every stream's history and filtering + sorting by timestamp."""
    events = []
    try:
        for channel in STREAMS_FOR_TRACE + [f"output.{request_id}"]:
            for msg in bus.read_all(channel):
                if msg.request_id == request_id:
                    events.append({
                        "timestamp": msg.timestamp,
                        "channel": channel,
                        "sender": msg.sender,
                        "recipient": msg.recipient,
                        "msg_type": msg.msg_type,
                    })
    except Exception as exc:
        logging.getLogger("main").warning("Trace reconstruction skipped: %s", exc)
    events.sort(key=lambda e: e["timestamp"])
    return events


def run_topics(topics: list[str], depth: str, max_sources: int, output_format: str) -> None:
    setup_logging()
    logger = logging.getLogger("main")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = int(os.environ.get("REDIS_PORT", "6379"))

    supervisor = Supervisor(redis_host=redis_host, redis_port=redis_port)
    supervisor.start_agents()
    time.sleep(1.0)  # let consumer groups register before first publish

    bus = MessageBus(host=redis_host, port=redis_port)
    results_summary = []

    try:
        for topic in topics:
            request = ResearchRequest(
                topic=topic, depth=depth, max_sources=max_sources, output_format=output_format
            )
            t0 = time.time()
            report = supervisor.run_single(request)
            elapsed = time.time() - t0

            if report is None:
                logger.error("No report produced for topic=%r (timeout or failure)", topic)
                results_summary.append({"topic": topic, "status": "failed", "elapsed_s": round(elapsed, 2)})
                continue

            trace = build_trace(bus, request.request_id)
            if not trace and getattr(supervisor, "last_trace", None):
                trace = supervisor.last_trace
            breakdown = build_timing_breakdown(report, trace)
            report.setdefault("metadata", {})["timings"] = breakdown

            out_path = OUTPUT_DIR / f"report_{report['report_id']}.json"
            out_path.write_text(json.dumps(report, indent=2))

            format_path = OUTPUT_DIR / f"report_{report['report_id']}.{output_format if output_format != 'json' else 'txt'}"
            if output_format != "json":
                format_path.write_text(serialize_report(report, output_format))

            trace_path = OUTPUT_DIR / f"trace_{report['report_id']}.json"
            trace_path.write_text(json.dumps(trace, indent=2))

            logger.info(
                "Topic %r done in %.2fs | confidence=%.2f | sections=%d | sources=%d | interactions=%d | timings=%s",
                topic, elapsed, report["critique"]["confidence_score"],
                len(report["sections"]), len(report["sources"]), len(trace), breakdown,
            )
            results_summary.append({
                "topic": topic,
                "status": "ok",
                "report_id": report["report_id"],
                "elapsed_s": round(elapsed, 2),
                "confidence": report["critique"]["confidence_score"],
                "n_sources": len(report["sources"]),
                "n_interactions": len(trace),
                "timings": breakdown,
            })
    finally:
        supervisor.stop_agents()

    summary_path = OUTPUT_DIR / "run_summary.json"
    summary_path.write_text(json.dumps(results_summary, indent=2))
    print(json.dumps(results_summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Multi-agent web research pipeline")
    parser.add_argument("--topic", action="append", help="Research topic (repeatable)")
    parser.add_argument("--topics-file", type=str, help="Path to a JSON file: list of topic strings")
    parser.add_argument("--depth", default="moderate", choices=["shallow", "moderate", "deep"])
    parser.add_argument("--max-sources", type=int, default=15)
    parser.add_argument("--output-format", default="json", choices=["markdown", "pdf", "json"])
    args = parser.parse_args()

    topics = list(args.topic or [])
    if args.topics_file:
        topics.extend(json.loads(Path(args.topics_file).read_text()))
    if not topics:
        topics = ["Recent advances in quantum computing hardware"]

    run_topics(topics, args.depth, args.max_sources, args.output_format)


if __name__ == "__main__":
    main()
