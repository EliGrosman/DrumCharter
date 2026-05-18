"""Label matrix construction for drum transcription.

Converts parsed drum onset lists into binary frame-level label arrays
used as training targets.
"""

from __future__ import annotations

import numpy as np

from audiotochart.training.rb_midi import NUM_CLASSES, DrumOnset


def onsets_to_label_matrix(
    onsets: list[DrumOnset],
    num_frames: int,
    *,
    num_classes: int = NUM_CLASSES,
    fps: int = 100,
) -> np.ndarray:
    """Convert a list of drum onsets into a binary frame-level label matrix.

    Each onset is placed at the nearest frame index; multiple onsets at
    the same frame are set to 1 independently per class.

    Args:
        onsets: Parsed drum onset events.
        num_frames: Total number of time frames in the output.
        num_classes: Number of drum classes (default 8).
        fps: Frames per second for time-to-frame conversion.

    Returns:
        A float32 array of shape (num_frames, num_classes) with 1.0 at
        onset positions and 0.0 elsewhere.
    """
    labels = np.zeros((num_frames, num_classes), dtype=np.float32)

    for onset in onsets:
        frame = round(onset.time * fps)
        if 0 <= frame < num_frames:
            labels[frame, onset.class_id] = 1.0

    return labels
