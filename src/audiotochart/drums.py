from dataclasses import dataclass


@dataclass(frozen=True)
class DrumHit:
    time_sec: float
    instrument: str
    confidence: float = 1.0
