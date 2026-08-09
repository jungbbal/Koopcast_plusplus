

from __future__ import annotations

import torch

from tsca_net.modules.akgr import AdaptiveKANLayer, KANLayer
from tsca_net.modules.cpcp import CPCPModule, SocialGraphBuilder
from tsca_net.modules.tsca import TSCAModule


def test_tsca_module_shapes(tiny_cfg):
    tsca = TSCAModule(tiny_cfg)
    B, N, K, T = 2, 3, 5, tiny_cfg.obs_len
    goals = torch.rand(B, N, K, 2)
    history = torch.rand(B, N, T, 2)

    mat, gates = tsca(goals, history)
    assert mat.shape == (B, N, K, T, 2)
    assert gates.shape == (B, N, K, T)
    assert (gates >= 0).all() and (gates <= 1).all()


def test_social_graph_excludes_self_and_respects_radius(tiny_cfg):
    graph = SocialGraphBuilder(tiny_cfg)
    # 3 agents on a line, spacing 1m; radius 1.5m connects neighbours only.
    pos = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]]])
    adj, mask = graph(pos, radius=1.5)

    assert not mask[0, 0, 0]  # self excluded
    assert mask[0, 0, 1]  # within radius
    assert not mask[0, 0, 2]  # outside radius
    assert torch.allclose(adj.sum(dim=-1), torch.tensor([[1.0, 1.0, 0.0]]), atol=1e-5)


def test_cpcp_module_shapes(tiny_cfg):
    cpcp = CPCPModule(tiny_cfg)
    graph = SocialGraphBuilder(tiny_cfg)
    B, N, T = 2, 4, tiny_cfg.obs_len

    pred_pos = torch.rand(B, N, 2)
    histories = torch.rand(B, N, T, 2)
    valid_mask = torch.ones(B, N)
    adj, _ = graph(pred_pos, valid_mask)

    feat = cpcp(pred_pos, histories, valid_mask, adj)
    assert feat.shape == (B, N, tiny_cfg.lstm_hidden_dim)
    assert torch.isfinite(feat).all()


def test_b_spline_basis_count_matches_coeff_shape():
    grid, order = 5, 3
    layer = KANLayer(in_dim=8, out_dim=4, grid=grid, order=order)
    x = torch.rand(6, 8) * 2 - 1
    y = layer(x)
    assert y.shape == (6, 4)
    assert torch.isfinite(y).all()


def test_adaptive_kan_layer_end_to_end(tiny_cfg):
    layer = AdaptiveKANLayer(tiny_cfg)
    B, N = 2, 3
    x = torch.randn(B, N, tiny_cfg.kan_hidden_dim)
    entropy = torch.rand(B, N, 1)
    hidden = torch.randn(B, N, tiny_cfg.kan_hidden_dim)

    out = layer(x, entropy, hidden)
    assert out.shape == (B, N, tiny_cfg.kan_output_dim)
    assert torch.isfinite(out).all()
