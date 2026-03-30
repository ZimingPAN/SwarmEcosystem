import numpy as np
import torch

from RL4KMC.world_models import KMCObservationShape, flatten_kmc_observation, unflatten_kmc_observation


def test_flatten_and_unflatten_roundtrip_shapes():
    shape = KMCObservationShape(max_vacancies=2, top_k=3, node_feat_dim=14, stats_dim=10)
    obs = {
        "V_features_local": np.arange(28, dtype=np.float32).reshape(2, 14),
        "topk_update_info": {
            "diff_k": np.arange(18, dtype=np.float32).reshape(2, 3, 3),
            "dist_k": np.arange(6, dtype=np.float32).reshape(2, 3),
        },
    }
    share_obs = np.arange(10, dtype=np.float32)

    flat = flatten_kmc_observation(obs, shape=shape, share_obs=share_obs)
    v_feat, diff_k, dist_k, stats = unflatten_kmc_observation(torch.from_numpy(flat), shape=shape)

    assert flat.shape == (shape.flat_dim,)
    assert tuple(v_feat.shape) == (1, 2, 14)
    assert tuple(diff_k.shape) == (1, 2, 3, 3)
    assert tuple(dist_k.shape) == (1, 2, 3)
    assert tuple(stats.shape) == (1, 10)


def test_flatten_observation_zero_pads_missing_entries():
    shape = KMCObservationShape(max_vacancies=3, top_k=2, node_feat_dim=14, stats_dim=4)
    obs = {
        "V_features_local": np.ones((1, 14), dtype=np.float32),
        "topk_update_info": {
            "diff_k": np.ones((1, 2, 3), dtype=np.float32),
            "dist_k": np.ones((1, 2), dtype=np.float32),
        },
    }
    flat = flatten_kmc_observation(obs, shape=shape, share_obs=np.array([1, 2], dtype=np.float32))
    v_feat, diff_k, dist_k, stats = unflatten_kmc_observation(torch.from_numpy(flat), shape=shape)

    assert torch.allclose(v_feat[0, 0], torch.ones(14))
    assert torch.allclose(v_feat[0, 1:], torch.zeros(2, 14))
    assert torch.allclose(diff_k[0, 1:], torch.zeros(2, 2, 3))
    assert torch.allclose(dist_k[0, 1:], torch.zeros(2, 2))
    assert torch.allclose(stats[0], torch.tensor([1.0, 2.0, 0.0, 0.0]))
