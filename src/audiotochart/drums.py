"""Drum hit data structure.

A simple frozen dataclass representing a single drum hit with
instrument name, time, and optional confidence score.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DrumHit:
    """A single detected drum hit.

    Attributes:
        time_sec: Time of the hit in seconds.
        instrument: Name of the drum instrument (e.g. "kick", "snare").
        confidence: Confidence score between 0 and 1. Defaults to 1.0.
    """

    time_sec: float
    instrument: str
    confidence: float = 1.0
