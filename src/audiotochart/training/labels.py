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
    labels = np.zeros((num_frames, num_classes), dtype=np.float32)

    for onset in onsets:
        frame = round(onset.time * fps)
        if 0 <= frame < num_frames:
            labels[frame, onset.class_id] = 1.0

    return labels
