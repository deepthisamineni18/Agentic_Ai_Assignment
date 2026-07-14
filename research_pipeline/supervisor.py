"""Supervisor: spawns each agent as an independent OS process, tracks their
state, retries failed message handling (via Redis Streams' pending-entries
list / XCLAIM), and enforces a global 5-minute timeout per research request.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from research_pipeline.agents.critic import CriticAgent
from research_pipeline.agents.planner import PlannerAgent
from research_pipeline.agents.searcher import SearcherAgent
from research_pipeline.agents.synthesizer import SynthesizerAgent
from research_pipeline.bus import MessageBus
from research_pipeline.schemas import AgentMessage, ResearchRequest

logger = logging.getLogger("supervisor")

GLOBAL_TIMEOUT_SECONDS = 300  # 5 minutes per research request
AGENT_CLASSES = {
    "planner": PlannerAgent,
    "searcher": SearcherAgent,
    "synthesizer": SynthesizerAgent,
    "critic": CriticAgent,
}
MAX_RETRIES_PER_AGENT = 3


def _agent_process_entrypoint(agent_key: str, redis_host: str, redis_port: int, stop_event):
    """Target function for each agent's independent process."""
    from research_pipeline.logging_config import setup_logging
    setup_logging()
    bus = MessageBus(host=redis_host, port=redis_port)
    agent_cls = AGENT_CLASSES[agent_key]
    consumer_name = f"{agent_key}-{uuid.uuid4().hex[:6]}"
    agent = agent_cls(bus=bus, consumer_name=consumer_name)
    agent.run_forever(stop_flag=lambda: stop_event.is_set())


@dataclass
class AgentHandle:
    key: str
    process: mp.Process
    restarts: int = 0


class _InProcessBus:
    def __init__(self, agent_map: dict[str, Any], trace: list[dict[str, Any]]):
        self.agent_map = agent_map
        self.trace = trace
        self.report_payload: dict | None = None

    def ensure_group(self, channel: str, group: str) -> None:
        return None

    def publish(self, channel: str, message: AgentMessage) -> str:
        self.trace.append({
            "timestamp": message.timestamp,
            "channel": channel,
            "sender": message.sender,
            "recipient": message.recipient,
            "msg_type": message.msg_type,
        })
        agent = self.agent_map.get(message.recipient)
        if agent is not None:
            agent.handle(message)
        elif channel.startswith("output."):
            self.report_payload = message.payload
        return "in-process"

    def consume(self, channel: str, group: str, consumer: str, count: int = 10, block_ms: int = 2000):
        return []

    def ack(self, channel: str, group: str, entry_id: str) -> None:
        return None


class Supervisor:
    """Owns the lifecycle of all agent processes and the top-level request loop."""

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.bus = MessageBus(host=redis_host, port=redis_port)
        self._stop_event = mp.Event()
        self._handles: dict[str, AgentHandle] = {}
        self._use_inprocess = not self._redis_available()
        self.last_trace: list[dict[str, Any]] = []
        self.last_report: dict | None = None

    def _redis_available(self) -> bool:
        try:
            self.bus.r.ping()
            return True
        except Exception:
            return False

    # ---- process lifecycle ----------------------------------------------
    def start_agents(self) -> None:
        if self._use_inprocess:
            logger.info("Redis unavailable; running requests in-process")
            return
        for key in AGENT_CLASSES:
            self._spawn(key)
        logger.info("All %d agents started", len(self._handles))

    def _spawn(self, key: str) -> None:
        p = mp.Process(
            target=_agent_process_entrypoint,
            args=(key, self.redis_host, self.redis_port, self._stop_event),
            daemon=True,
            name=f"agent-{key}",
        )
        p.start()
        self._handles[key] = AgentHandle(key=key, process=p)
        logger.info("Spawned agent process '%s' pid=%s", key, p.pid)

    def health_check_and_restart(self) -> None:
        """Restart any agent process that has died unexpectedly (bounded retries)."""
        for key, handle in list(self._handles.items()):
            if not handle.process.is_alive() and not self._stop_event.is_set():
                if handle.restarts >= MAX_RETRIES_PER_AGENT:
                    logger.error("Agent '%s' exceeded max restarts (%d); leaving down",
                                 key, MAX_RETRIES_PER_AGENT)
                    continue
                logger.warning("Agent '%s' died (exitcode=%s); restarting (attempt %d/%d)",
                                key, handle.process.exitcode, handle.restarts + 1, MAX_RETRIES_PER_AGENT)
                self._spawn(key)
                self._handles[key].restarts = handle.restarts + 1

    def stop_agents(self) -> None:
        if self._use_inprocess:
            return
        self._stop_event.set()
        for handle in self._handles.values():
            handle.process.join(timeout=5)
            if handle.process.is_alive():
                handle.process.terminate()
        logger.info("All agents stopped")

    # ---- request handling -------------------------------------------------
    def submit(self, request: ResearchRequest) -> None:
        if self._use_inprocess:
            logger.info("Submitting request %s in-process", request.request_id)
            return
        output_channel = f"output.{request.request_id}"
        self.bus.ensure_group(output_channel, "output_group")
        msg = AgentMessage(
            request_id=request.request_id,
            sender="Supervisor",
            recipient="PlannerAgent",
            msg_type="research.requested",
            payload=request.model_dump(mode="json"),
        )
        self.bus.publish("planner", msg)
        logger.info("Submitted research request %s: topic=%r", request.request_id, request.topic)

    def wait_for_report(self, request_id: str, timeout: float = GLOBAL_TIMEOUT_SECONDS) -> dict | None:
        """Poll this request's dedicated output stream until a report.done
        arrives or the global timeout elapses. Each request gets its own
        stream (`output.<request_id>`) so concurrent requests never steal
        each other's completion messages via consumer-group load balancing."""
        deadline = time.time() + timeout
        output_channel = f"output.{request_id}"
        consumer = f"waiter-{uuid.uuid4().hex[:6]}"
        while time.time() < deadline:
            self.health_check_and_restart()
            messages = self.bus.consume(output_channel, "output_group", consumer, block_ms=1000)
            for entry_id, msg in messages:
                self.bus.ack(output_channel, "output_group", entry_id)
                if msg.request_id == request_id and msg.msg_type == "report.done":
                    return msg.payload
        logger.error("Timed out after %.0fs waiting for report on request %s", timeout, request_id)
        return None

    def _run_inprocess(self, request: ResearchRequest) -> dict | None:
        trace: list[dict[str, Any]] = []
        agents = {
            "PlannerAgent": PlannerAgent(bus=_InProcessBus({}, trace), consumer_name="local-planner"),
            "SearcherAgent": SearcherAgent(bus=_InProcessBus({}, trace), consumer_name="local-searcher"),
            "SynthesizerAgent": SynthesizerAgent(bus=_InProcessBus({}, trace), consumer_name="local-synth"),
            "CriticAgent": CriticAgent(bus=_InProcessBus({}, trace), consumer_name="local-critic"),
        }
        bus = _InProcessBus(agents, trace)
        for agent in agents.values():
            agent.bus = bus
        message = AgentMessage(
            request_id=request.request_id,
            sender="Supervisor",
            recipient="PlannerAgent",
            msg_type="research.requested",
            payload=request.model_dump(mode="json"),
        )
        agents["PlannerAgent"].handle(message)
        self.last_trace = trace
        self.last_report = bus.report_payload
        return bus.report_payload

    def run_single(self, request: ResearchRequest) -> dict | None:
        if self._use_inprocess:
            return self._run_inprocess(request)
        self.submit(request)
        return self.wait_for_report(request.request_id)
