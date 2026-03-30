import numpy as np
import torch

from RL4KMC.world_models import (
    DefectGraphObservationShape,
    bcc_shell_cutoff_sq,
    build_defect_graph_observation,
    unflatten_defect_graph_observation,
)


class _MockEnv:
    dims = np.array([40, 40, 40], dtype=np.float32)
    V_TYPE = 2
    CU_TYPE = 1

    def get_vacancy_array(self):
        return np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float32)

    def get_cu_array(self):
        return np.array([[1, 1, 1], [39, 39, 39], [3, 1, 1]], dtype=np.float32)


def test_build_defect_graph_observation_shapes_and_masks():
    env = _MockEnv()
    shape = DefectGraphObservationShape(max_vacancies=3, max_defects=10, max_shells=1, node_feat_dim=4, stats_dim=2)
    flat = build_defect_graph_observation(env, shape=shape, share_obs=np.array([1.0, 2.0], dtype=np.float32))
    node_attr, node_mask, stats = unflatten_defect_graph_observation(torch.from_numpy(flat), shape=shape)

    assert flat.shape == (shape.flat_dim,)
    assert tuple(node_attr.shape) == (1, 3, 10, 4)
    assert tuple(node_mask.shape) == (1, 3, 10)
    assert tuple(stats.shape) == (1, 2)
    assert torch.allclose(stats[0], torch.tensor([1.0, 2.0]))
    assert int(node_mask[0, 0].sum().item()) == 3
    assert torch.allclose(node_mask[0, 2], torch.zeros(10))
    assert torch.allclose(node_attr[0, 0, 0, :3], torch.zeros(3))
    assert node_attr[0, 0, 0, 3].item() == 2.0
    selected_offsets = {
        tuple(int(v) for v in node_attr[0, 0, idx, :3].tolist())
        for idx in range(int(node_mask[0, 0].sum().item()))
    }
    assert selected_offsets == {(0, 0, 0), (1, 1, 1), (-1, -1, -1)}


def test_bcc_shell_cutoff_matches_first_16_shells():
    assert bcc_shell_cutoff_sq(1, (80, 80, 80)) == 3
    assert bcc_shell_cutoff_sq(2, (80, 80, 80)) == 4
    assert bcc_shell_cutoff_sq(16, (80, 80, 80)) == 44
