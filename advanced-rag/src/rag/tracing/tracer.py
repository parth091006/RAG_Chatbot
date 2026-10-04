from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TraceEvent:
    name: str
    duration_ms: float
    metadata: dict[str, str] = field(default_factory=dict)


class Tracer:
    def __init__(self):
        self.events: list[TraceEvent] = []

    def add_event(self, name: str, duration_ms: float, **metadata: str) -> None:
        self.events.append(TraceEvent(name=name, duration_ms=duration_ms, metadata=metadata))

    def summary(self) -> dict[str, float]:
        return {event.name: event.duration_ms for event in self.events}
