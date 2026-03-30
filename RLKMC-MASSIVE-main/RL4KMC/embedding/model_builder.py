"""Model/embed construction + evaluation weight restore utilities.

Goal: keep `DistributedKMCRunner` thin by moving model class selection, device
resolution, distributed weight sync, and cross-rank digest verification into a
single cohesive module.

This module intentionally does NOT depend on runner types.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import numpy as np
import torch

from RL4KMC.embedding.SGDNTC_Model import SGDNTC_Model
from RL4KMC.runner.services.worker import CONFIG
from RL4KMC.utils.util import safe_barrier
from environ import ENVIRON

_LOGGER = logging.getLogger(__name__)
if not _LOGGER.handlers:
    _LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class EmbedBuildResult:
    embed: torch.nn.Module
    device: torch.device
    model_path: str
    digest: str


def resolve_infer_device() -> torch.device:
    """Select the default device for inference/model.

    Note: this is intentionally separate from comm_device.
    """
    return torch.device("cpu")
    # TODO
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resolve_embed_class(model_type: str) -> type:
    if model_type == "SGDNTC_Model":
        return SGDNTC_Model
    else:
        raise ValueError(f"Unsupported model_type: {model_type!r}")

def build_embed(
    args,
    device: torch.device,
) -> EmbedBuildResult:
    model_type = CONFIG.runner.model_type
    EmbedClass = resolve_embed_class(model_type)
    embed = EmbedClass(args=args, device=device)

    state_dict = torch.load(ENVIRON.model_path, map_location=device, weights_only=True)

    # 创建一个新的字典，去掉 'embed.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("embed."):
            new_key = k[6:]  # 去掉 'embed.'（共6个字符）
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    embed.load_state_dict(new_state_dict, strict=False)

    h = hashlib.sha256()
    sd_now = embed.state_dict()
    for k in sorted(sd_now.keys()):
        t = sd_now[k].detach().cpu().contiguous().numpy().view(np.uint8)
        h.update(str(k).encode("utf-8"))
        h.update(t.tobytes())
    digest = h.hexdigest()

    return EmbedBuildResult(
        embed=embed,
        device=device,
        model_path=str(ENVIRON.model_path),
        digest=digest,
    )
