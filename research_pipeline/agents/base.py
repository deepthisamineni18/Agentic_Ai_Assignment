"""Abstract base class all agents implement. Each agent runs as an
independent OS process (see supervisor.py) and communicates exclusively
through the MessageBus - never via direct imports of other agents' state."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from research_pipeline.bus import MessageBus
from research_pipeline.schemas import AgentMessage


class BaseAgent(ABC):
    #: channel this agent listens on
    inbox: str
    #: consumer group name (one per agent type, so multiple replicas can share load)
    group: str

    def __init__(self, bus: MessageBus, consumer_name: str, poll_block_ms: int = 2000):
        self.bus = bus
        self.consumer_name = consumer_name
        self.poll_block_ms = poll_block_ms
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def handle(self, message: AgentMessage) -> None:
        """Process a single inbound message. Implementations should publish
        their output(s) back onto the bus via self.bus.publish(...)."""
        raise NotImplementedError

    def emit(self, channel: str, request_id: str, recipient: str, msg_type: str, payload: dict) -> None:
        msg = AgentMessage(
            request_id=request_id,
            sender=self.__class__.__name__,
            recipient=recipient,
            msg_type=msg_type,
            payload=payload,
        )
        self.bus.publish(channel, msg)

    def run_forever(self, stop_flag=None) -> None:
        """Main process loop: poll inbox, handle, ack. `stop_flag` is an
        optional callable; when it returns True the loop exits (used by the
        supervisor for graceful shutdown / tests)."""
        self.logger.info("Agent %s starting, listening on '%s'", self.consumer_name, self.inbox)
        while True:
            if stop_flag is not None and stop_flag():
                self.logger.info("Agent %s received stop signal", self.consumer_name)
                break
            try:
                messages = self.bus.consume(
                    self.inbox, self.group, self.consumer_name, block_ms=self.poll_block_ms
                )
            except Exception:
                self.logger.exception("Error polling bus, backing off")
                time.sleep(1)
                continue

            for entry_id, message in messages:
                try:
                    self.logger.debug("Handling %s from %s", message.msg_type, message.sender)
                    self.handle(message)
                except Exception:
                    self.logger.exception(
                        "Agent %s failed to handle message %s (req=%s)",
                        self.consumer_name, entry_id, message.request_id,
                    )
                    # Message is intentionally left un-acked so the Supervisor's
                    # retry/claim logic (XCLAIM/XPENDING) can reassign it.
                    continue
                self.bus.ack(self.inbox, self.group, entry_id)
