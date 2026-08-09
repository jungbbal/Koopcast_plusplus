
from __future__ import annotations

from typing import List

import numpy as np


def windows_from_raw(
    raw: np.ndarray,
    obs_len: int,
    pred_len: int,
    skip: int = 1,
    min_ped: int = 1,
) -> List[np.ndarray]:
    
    if len(raw) == 0:
        return []

    seq_len = obs_len + pred_len
    frames = np.unique(raw[:, 0])
    frame_data = {f: raw[raw[:, 0] == f] for f in frames}

    scenes: List[np.ndarray] = []
    for start in range(0, len(frames) - seq_len + 1, skip):
        window_frames = frames[start : start + seq_len]
        ped_sets = [set(frame_data[f][:, 1].tolist()) for f in window_frames]
        common_peds = set.intersection(*ped_sets) if ped_sets else set()
        if len(common_peds) < min_ped:
            continue

        scene = np.zeros((len(common_peds), seq_len, 2), dtype=np.float32)
        for i, pid in enumerate(sorted(common_peds)):
            for t, f in enumerate(window_frames):
                row = frame_data[f]
                match = row[row[:, 1] == pid][0]
                scene[i, t] = match[2:4]
        scenes.append(scene)
    return scenes
