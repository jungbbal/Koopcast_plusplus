
from __future__ import annotations

import math

import torch

from tsca_net.functional import compute_entropy


def test_entropy_distinguishes_peaky_from_uniform():
    H = W = 64
    uniform = torch.full((1, 1, 1, H, W), 0.02)
    peaky = torch.full((1, 1, 1, H, W), 0.001)
    peaky[0, 0, 0, 10:12, 10:12] = 0.98  # small, confident peak

    ent_uniform = compute_entropy(uniform).item()
    ent_peaky = compute_entropy(peaky).item()
    max_ent = math.log(H * W)

    assert ent_uniform > ent_peaky, "a uniform heatmap must have higher entropy than a peaked one"
    
    assert ent_peaky < 0.9 * max_ent


def test_entropy_is_finite_and_non_negative():
    hm = torch.rand(2, 3, 1, 32, 32)
    ent = compute_entropy(hm)
    assert torch.isfinite(ent).all()
    assert (ent >= 0).all()
    assert ent.shape == (2, 3, 1)
