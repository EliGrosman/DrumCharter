from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

TOM_CLASSES = frozenset({3, 5, 7})
CYMBAL_CLASSES = frozenset({2, 4, 6})
TRIPLE_WINDOW_FRAMES = 5


@dataclass(frozen=True)
class CQSComponents:
    coverage: float
    identity: float
    restraint: float
    playability: float
    grid_coherence: float | None
    cqs: float
    n_gt: int
    n_pred: int
    n_matched: int
    n_spurious: int

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "coverage": self.coverage,
            "identity": self.identity,
            "restraint": self.restraint,
            "playability": self.playability,
            "grid_coherence": self.grid_coherence,
            "cqs": self.cqs,
            "n_gt": self.n_gt,
            "n_pred": self.n_pred,
            "n_matched": self.n_matched,
            "n_spurious": self.n_spurious,
        }


@dataclass(frozen=True)
class MatchResult:
    matches: list[tuple[int, int, int, int]]
    missed: list[tuple[int, int]]
    spurious: list[tuple[int, int]]


def picks_to_events(picks_per_class: dict[int, np.ndarray]) -> list[tuple[int, int]]:
    events: list[tuple[int, int]] = []
    for class_idx, frames in picks_per_class.items():
        for frame in frames:
            events.append((int(frame), int(class_idx)))
    events.sort()
    return events


def labels_to_events(
    labels: np.ndarray,
    *,
    threshold: float = 0.5,
    max_frames: int | None = None,
) -> list[tuple[int, int]]:
    t_frames = labels.shape[0] if max_frames is None else min(labels.shape[0], max_frames)
    events: list[tuple[int, int]] = []
    for class_idx in range(labels.shape[1]):
        for frame in np.flatnonzero(labels[:t_frames, class_idx] > threshold):
            events.append((int(frame), int(class_idx)))
    events.sort()
    return events


def match_events(
    gt_events: list[tuple[int, int]],
    pred_events: list[tuple[int, int]],
    *,
    tol_frames: int = 2,
) -> MatchResult:
    if not gt_events:
        return MatchResult(matches=[], missed=[], spurious=list(pred_events))
    if not pred_events:
        return MatchResult(matches=[], missed=list(gt_events), spurious=[])

    gt_frames = np.fromiter((f for f, _c in gt_events), dtype=np.int64, count=len(gt_events))
    gt_classes = np.fromiter((c for _f, c in gt_events), dtype=np.int32, count=len(gt_events))
    gt_used = np.zeros(len(gt_events), dtype=bool)
    matches: list[tuple[int, int, int, int]] = []
    leftover: list[tuple[int, int]] = []

    def best_in_window(pred_frame: int, pred_class: int, *, same_class_only: bool) -> int:
        lo = int(np.searchsorted(gt_frames, pred_frame - tol_frames))
        hi = int(np.searchsorted(gt_frames, pred_frame + tol_frames + 1))
        best = -1
        best_dist = tol_frames + 1
        for idx in range(lo, hi):
            if gt_used[idx]:
                continue
            if same_class_only and int(gt_classes[idx]) != pred_class:
                continue
            dist = abs(int(gt_frames[idx]) - pred_frame)
            if dist < best_dist:
                best = idx
                best_dist = dist
        return best

    for pred_frame, pred_class in pred_events:
        match_idx = best_in_window(pred_frame, pred_class, same_class_only=True)
        if match_idx >= 0:
            gt_used[match_idx] = True
            matches.append(
                (
                    int(gt_frames[match_idx]),
                    int(gt_classes[match_idx]),
                    pred_frame,
                    pred_class,
                )
            )
        else:
            leftover.append((pred_frame, pred_class))

    spurious: list[tuple[int, int]] = []
    for pred_frame, pred_class in leftover:
        match_idx = best_in_window(pred_frame, pred_class, same_class_only=False)
        if match_idx >= 0:
            gt_used[match_idx] = True
            matches.append(
                (
                    int(gt_frames[match_idx]),
                    int(gt_classes[match_idx]),
                    pred_frame,
                    pred_class,
                )
            )
        else:
            spurious.append((pred_frame, pred_class))

    missed = [gt_events[idx] for idx in range(len(gt_events)) if not gt_used[idx]]
    return MatchResult(matches=matches, missed=missed, spurious=spurious)


def coverage(match: MatchResult) -> float:
    n_gt = len(match.matches) + len(match.missed)
    return 1.0 if n_gt == 0 else len(match.matches) / n_gt


def identity(match: MatchResult) -> float:
    if not match.matches:
        return 1.0
    total = 0.0
    for _gt_frame, gt_class, _pred_frame, pred_class in match.matches:
        if gt_class == pred_class:
            total += 1.0
        elif (gt_class in TOM_CLASSES and pred_class in TOM_CLASSES) or (
            gt_class in CYMBAL_CLASSES and pred_class in CYMBAL_CLASSES
        ):
            total += 0.5
    return total / len(match.matches)


def restraint(match: MatchResult) -> float:
    n_gt = len(match.matches) + len(match.missed)
    if n_gt == 0:
        return 1.0 if not match.spurious else 0.0
    return max(0.0, 1.0 - len(match.spurious) / n_gt)


def playability(
    pred_events: list[tuple[int, int]],
    *,
    triple_window_frames: int = TRIPLE_WINDOW_FRAMES,
) -> float:
    hand_events = [(frame, class_idx) for frame, class_idx in pred_events if class_idx != 0]
    if len(hand_events) < 3:
        return 1.0

    flagged: set[int] = set()
    for idx_0 in range(len(hand_events) - 2):
        frame_0, class_0 = hand_events[idx_0]
        idx_1 = idx_0 + 1
        while idx_1 < len(hand_events) and hand_events[idx_1][0] - frame_0 <= triple_window_frames:
            _frame_1, class_1 = hand_events[idx_1]
            idx_2 = idx_1 + 1
            while idx_2 < len(hand_events) and hand_events[idx_2][0] - frame_0 <= triple_window_frames:
                _frame_2, class_2 = hand_events[idx_2]
                if len({class_0, class_1, class_2}) == 3:
                    flagged.update({idx_0, idx_1, idx_2})
                idx_2 += 1
            idx_1 += 1

    return 1.0 - len(flagged) / len(hand_events)


def compute_cqs(
    picks_per_class: dict[int, np.ndarray],
    labels: np.ndarray,
    *,
    tol_frames: int = 2,
) -> CQSComponents:
    t_frames = labels.shape[0]
    clipped = {
        class_idx: frames[(frames >= 0) & (frames < t_frames)]
        for class_idx, frames in picks_per_class.items()
    }
    pred_events = picks_to_events(clipped)
    gt_events = labels_to_events(labels, max_frames=t_frames)
    match = match_events(gt_events, pred_events, tol_frames=tol_frames)

    cov = coverage(match)
    ident = identity(match)
    restr = restraint(match)
    play = playability(pred_events)
    present = [cov, ident, restr, play]
    eps = 1e-6
    cqs = float(np.exp(float(np.mean([np.log(max(value, eps)) for value in present]))))
    return CQSComponents(
        coverage=cov,
        identity=ident,
        restraint=restr,
        playability=play,
        grid_coherence=None,
        cqs=cqs,
        n_gt=len(gt_events),
        n_pred=len(pred_events),
        n_matched=len(match.matches),
        n_spurious=len(match.spurious),
    )


def aggregate_cqs(per_song: Iterable[CQSComponents]) -> dict[str, float | int | None]:
    rows = list(per_song)
    if not rows:
        return {
            "n_songs": 0,
            "cqs": 0.0,
            "coverage": 0.0,
            "identity": 0.0,
            "restraint": 0.0,
            "playability": 0.0,
            "grid_coherence": None,
            "n_songs_with_grid": 0,
        }

    return {
        "n_songs": len(rows),
        "cqs": float(np.mean([row.cqs for row in rows])),
        "coverage": float(np.mean([row.coverage for row in rows])),
        "identity": float(np.mean([row.identity for row in rows])),
        "restraint": float(np.mean([row.restraint for row in rows])),
        "playability": float(np.mean([row.playability for row in rows])),
        "grid_coherence": None,
        "n_songs_with_grid": 0,
    }
