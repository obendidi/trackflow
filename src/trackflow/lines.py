"""Line counting: count tracks whose foot point crosses a line segment drawn in image pixels."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from trackflow.types import Track

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class CountingLine:
    """Named line segment in image pixels.

    A crossing to the right of the direction from ``start`` to ``end`` counts as ``in``, the
    other way as ``out``. For a line drawn left to right, ``in`` means moving down the frame.
    """

    name: str
    start: Point
    end: Point

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("A counting line needs a name.")
        if not all(math.isfinite(v) for v in (*self.start, *self.end)):
            raise ValueError(f"Line {self.name!r} needs finite coordinates.")
        if self.start == self.end:
            raise ValueError(f"Line {self.name!r} needs two different points.")

    @classmethod
    def parse(cls, spec: str, default_name: str) -> CountingLine:
        """Parse ``[name=]x1,y1,x2,y2``, as passed to ``trackflow run --line``."""
        name, _, coords = spec.rpartition("=")
        try:
            x1, y1, x2, y2 = (float(v) for v in coords.split(","))
        except ValueError:
            raise ValueError(
                f"--line {spec!r} must be x1,y1,x2,y2 or name=x1,y1,x2,y2 in image pixels."
            ) from None
        return cls(name.strip() or default_name, (x1, y1), (x2, y2))

    def side(self, point: Point) -> float:
        """Positive on the ``in`` side, negative on the ``out`` side, zero on the line."""
        (x1, y1), (x2, y2) = self.start, self.end
        return (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)

    def spans(self, p: Point, q: Point) -> bool:
        """Whether the infinite line through ``p`` and ``q`` meets this segment."""
        dx, dy = q[0] - p[0], q[1] - p[1]
        a = dx * (self.start[1] - p[1]) - dy * (self.start[0] - p[0])
        b = dx * (self.end[1] - p[1]) - dy * (self.end[0] - p[0])
        return a * b <= 0


@dataclass(slots=True)
class _TrackSide:
    side: float  # side the track was last confirmed on
    point: Point  # its latest foot point on that side
    pending_since_s: float | None = None  # when it was first seen on the other side
    pending_spans: bool = False  # whether that move went through the segment


class LineCounter:
    """Counts crossings of each line, per direction, for one run or stream.

    A crossing counts once the track has stayed on the new side for ``hold_s`` seconds. When
    people overlap, a box's bottom edge can jump back and forth across a line for a few
    frames, and each jump would otherwise count.
    """

    def __init__(self, lines: Sequence[CountingLine], hold_s: float = 0.3) -> None:
        names = [line.name for line in lines]
        if len(set(names)) != len(names):
            raise ValueError(f"Counting line names must be unique, got {names}.")
        self.lines = list(lines)
        self.hold_s = hold_s
        self.counts = {name: {"in": 0, "out": 0} for name in names}
        self._sides: dict[tuple[str, int], _TrackSide] = {}

    def update(self, tracks: Sequence[Track], timestamp_s: float) -> list[dict[str, Any]]:
        """Record this frame's tracks and return a ``line_crossed`` event per new crossing."""
        events: list[dict[str, Any]] = []
        for tr in tracks:
            point = tr.foot_point
            for line in self.lines:
                side = line.side(point)
                if side == 0:
                    continue
                key = (line.name, tr.track_id)
                state = self._sides.get(key)
                if state is None:
                    self._sides[key] = _TrackSide(side, point)
                    continue
                if (state.side > 0) == (side > 0):
                    state.point, state.pending_since_s = point, None
                    continue
                if state.pending_since_s is None:
                    state.pending_since_s = timestamp_s
                    state.pending_spans = line.spans(state.point, point)
                if timestamp_s - state.pending_since_s < self.hold_s:
                    continue
                if state.pending_spans:
                    direction = "in" if side > 0 else "out"
                    self.counts[line.name][direction] += 1
                    events.append(
                        {
                            "type": "line_crossed",
                            "line": line.name,
                            "direction": direction,
                            "track_id": tr.track_id,
                            "class_name": tr.class_name,
                        }
                    )
                state.side, state.point, state.pending_since_s = side, point, None
        return events
