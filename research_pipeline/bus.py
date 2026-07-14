"""Message bus abstraction. Backed by Redis Streams in production; agents
communicate exclusively through this interface (never via direct function
calls or shared memory), satisfying the "independent process" requirement.

Each logical channel is a Redis Stream, e.g. "stream:planner", "stream:searcher".
Consumers use consumer groups so messages are acknowledged and not reprocessed
on restart, and so the Supervisor can detect stalled consumers.
"""
from __future__ import annotations

import json
import logging
from typing import Iterable, Optional

import redis

from research_pipeline.schemas import AgentMessage

logger = logging.getLogger("bus")


class MessageBus:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.r = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    # ---- stream helpers -------------------------------------------------
    @staticmethod
    def stream_name(channel: str) -> str:
        return f"stream:{channel}"

    def ensure_group(self, channel: str, group: str) -> None:
        stream = self.stream_name(channel)
        try:
            self.r.xgroup_create(stream, group, id="0", mkstream=True)
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    def publish(self, channel: str, message: AgentMessage) -> str:
        stream = self.stream_name(channel)
        msg_id = self.r.xadd(stream, {"data": message.model_dump_json()})
        logger.debug("PUBLISH %s -> %s [%s]", message.sender, channel, message.msg_type)
        return msg_id

    def consume(
        self,
        channel: str,
        group: str,
        consumer: str,
        count: int = 10,
        block_ms: int = 2000,
    ) -> list[tuple[str, AgentMessage]]:
        """Read new messages for this consumer group. Returns list of (stream_msg_id, AgentMessage)."""
        self.ensure_group(channel, group)
        stream = self.stream_name(channel)
        resp = self.r.xreadgroup(group, consumer, {stream: ">"}, count=count, block=block_ms)
        out = []
        for _stream_name, entries in resp or []:
            for entry_id, fields in entries:
                try:
                    msg = AgentMessage.model_validate_json(fields["data"])
                    out.append((entry_id, msg))
                except Exception:
                    logger.exception("Failed to parse message %s on %s", entry_id, channel)
        return out

    def ack(self, channel: str, group: str, entry_id: str) -> None:
        self.r.xack(self.stream_name(channel), group, entry_id)

    def pending_count(self, channel: str, group: str) -> int:
        try:
            info = self.r.xpending(self.stream_name(channel), group)
            return info["pending"] if info else 0
        except redis.exceptions.ResponseError:
            return 0

    def read_all(self, channel: str) -> list[AgentMessage]:
        """Read the full history of a stream (used for trace reconstruction
        and debugging; Streams retain entries after ack unless trimmed)."""
        stream = self.stream_name(channel)
        entries = self.r.xrange(stream, min="-", max="+")
        out = []
        for _entry_id, fields in entries:
            try:
                out.append(AgentMessage.model_validate_json(fields["data"]))
            except Exception:
                continue
        return out

    def flush_all(self) -> None:
        """Danger: clears all streams. Used between test runs only."""
        for key in self.r.keys("stream:*"):
            self.r.delete(key)
