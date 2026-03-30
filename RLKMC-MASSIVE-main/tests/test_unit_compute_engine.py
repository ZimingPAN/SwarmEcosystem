import types

import torch

from RL4KMC.envs.distributed_kmc import KMCObs


class _FakeEnv:
    def __init__(self, *_a, **_k):
        self.time = 0.0


class _Embed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, V_feat, diff_k, dist_k):
        self.batch_sizes.append(int(V_feat.shape[0]))
        return V_feat.sum(dim=1, keepdim=True) + diff_k + dist_k


def _make_obs(*, changed_vids=None, vid_list=None, diff_val=1.0, dist_val=2.0):
    vids = list(vid_list or [0])
    return KMCObs(
        topk_update_info={
            "vid_list": vids,
            "diff_k": torch.full((len(vids), 1), float(diff_val), dtype=torch.float32),
            "dist_k": torch.full((len(vids), 1), float(dist_val), dtype=torch.float32),
        },
        updated_cu=None,
        updated_vacancy=None,
        cu_move_from=None,
        cu_move_to=None,
        cu_id=None,
        cu_topk_id=None,
        vac_id=0,
        changed_vids=list(changed_vids or []),
        dir_idx=0,
        energy_change=0.0,
    )


def _build_engine(monkeypatch, *, incremental=False):
    from RL4KMC.runner.engine import compute_engine as ce

    monkeypatch.setattr(ce, "DistributedKMCEnv", _FakeEnv, raising=True)
    embed = _Embed()
    args = types.SimpleNamespace()
    engine = ce.KMCComputeEngine(
        args=args,
        embed=embed,
        embed_device=torch.device("cpu"),
        worker_id=0,
        enable_incremental_policy=incremental,
    )
    return engine, embed


def test_update_topk_cache_from_obs_updates_selected_vids(monkeypatch):
    engine, _embed = _build_engine(monkeypatch)
    engine.diff_k_cache = torch.zeros((4, 1), dtype=torch.float32)
    engine.dist_k_cache = torch.zeros((4, 1), dtype=torch.float32)

    obs = _make_obs(vid_list=[1, 3], diff_val=5.0, dist_val=7.0)
    engine.update_topk_cache_from_obs(obs)

    assert float(engine.diff_k_cache[1].item()) == 5.0
    assert float(engine.diff_k_cache[3].item()) == 5.0
    assert float(engine.dist_k_cache[1].item()) == 7.0
    assert float(engine.dist_k_cache[3].item()) == 7.0


def test_get_logits_incremental_only_recomputes_changed_vids(monkeypatch):
    engine, embed = _build_engine(monkeypatch, incremental=True)
    engine.diff_k_cache = torch.zeros((4, 1), dtype=torch.float32)
    engine.dist_k_cache = torch.zeros((4, 1), dtype=torch.float32)

    first_feat = torch.tensor(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=torch.float32
    )
    logits0 = engine.get_logits(first_feat).clone()
    assert embed.batch_sizes == [4]

    second_feat = torch.tensor(
        [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]], dtype=torch.float32
    )
    engine.changed_vids_global = {1, 2}
    logits1 = engine.get_logits(second_feat)

    assert embed.batch_sizes == [4, 2]
    assert float(logits1[0].item()) == float(logits0[0].item())
    assert float(logits1[1].item()) != float(logits0[1].item())
    assert float(logits1[2].item()) != float(logits0[2].item())


def test_run_one_task_bench_mode_updates_stats(monkeypatch):
    engine, _embed = _build_engine(monkeypatch)

    class TimingStats:
        def __init__(self):
            self.calls = []

        def add_task_window(self, s, e):
            self.calls.append(("window", float(e) - float(s) >= 0.0))

        def add_task(self, n):
            self.calls.append(("task", int(n)))

        def add_jumps(self, n):
            self.calls.append(("jumps", int(n)))

    timing = TimingStats()
    engine.timing_stats = timing

    monkeypatch.setattr(engine, "reinit_env_for_task", lambda *_a, **_k: setattr(engine, "env", types.SimpleNamespace(time=0.0)))

    step_calls = {"n": 0}

    def fake_step(_timer=None):
        step_calls["n"] += 1
        if step_calls["n"] <= 2:
            return _make_obs(changed_vids=[0], vid_list=[0])
        return None

    monkeypatch.setattr(engine, "step_traditional_kmc", fake_step)
    monkeypatch.setattr(engine, "update_topk_cache_from_obs", lambda *_a, **_k: None)

    jumps = engine.run_one_task(
        {"temp": 400, "time": 2.0, "cu_density": 0.0, "v_density": 1.0},
        bench_step=5,
        use_traditional_kmc=True,
    )

    assert jumps == 2
    assert ("task", 1) in timing.calls
    assert ("jumps", 2) in timing.calls
