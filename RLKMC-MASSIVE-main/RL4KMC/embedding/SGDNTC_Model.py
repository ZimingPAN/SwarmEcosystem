import math
import logging
import torch
import torch.nn as nn

from RL4KMC.utils.env import EnvKeys, env_int, env_str


_LOGGER = logging.getLogger(__name__)
if not _LOGGER.handlers:
    _LOGGER.addHandler(logging.NullHandler())

# PyTorch < 2.0 doesn't provide torch.nn.functional.scaled_dot_product_attention.
# Provide a compatible fallback for CPU-only/legacy environments.
# NOTE: In older versions, the private _scaled_dot_product_attention returns a tuple
# (attn_output, attn_weights). We normalize to always return just attn_output.
if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    from torch.nn.functional import scaled_dot_product_attention as _sdpa
else:
    from torch.nn.functional import _scaled_dot_product_attention as _sdpa #type: ignore


def scaled_dot_product_attention(
    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs
):

    # Vendor/legacy SDPA kernels (especially on non-standard CPU accelerators)
    # can be numerically unstable or even hard-crash. Provide a manual fallback.
    impl = str(env_str(EnvKeys.EMBED_SDPA_IMPL, "torch") or "torch").lower()  # torch|manual

    orig_dtype = q.dtype

    def _safe_softmax(x, dim=-1):
        # Avoid torch.softmax kernel; use a stable exp/sum implementation.
        x_max = x.max(dim=dim, keepdim=True).values
        x = x - x_max
        # After subtracting max, values should be <= 0. Clamp to avoid denorm/NaN paths.
        x = x.clamp(min=-50.0, max=0.0)
        exp_x = torch.exp(x)
        denom = exp_x.sum(dim=dim, keepdim=True).clamp(min=1e-20)
        return exp_x / denom

    def _stable_softmax(x, dim=-1):
        # preprocess x to avoid extreme values
        x = x - x.max(dim=dim, keepdim=True).values
        x = x.clamp(min=-50.0, max=0.0)
        return torch.softmax(x, dim=dim)

    def _manual_sdpa(qf, kf, vf, maskf):
        d = qf.shape[-1]
        scores = torch.matmul(qf, kf.transpose(-2, -1)) / math.sqrt(d)
        if is_causal:
            Lq = scores.shape[-2]
            Lk = scores.shape[-1]
            causal = torch.triu(
                torch.ones(Lq, Lk, device=scores.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal, -1.0e4)
        if maskf is not None:
            if maskf.dtype == torch.bool:
                scores = scores.masked_fill(maskf, -1.0e4)
            else:
                scores = scores + maskf

        attn = _stable_softmax(scores, dim=-1)
        # attn = _stable_softmax(scores, dim=-1)
        # attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, vf)

    # Always do the math in fp32 for stability, then cast back.
    qf = q.float()
    kf = k.float()
    vf = v.float()
    maskf = (
        attn_mask.float()
        if attn_mask is not None and attn_mask.dtype != torch.bool
        else attn_mask
    )

    if impl == "manual":
        out = _manual_sdpa(qf, kf, vf, maskf)
    else:
        try:
            out = _sdpa(
                qf,
                kf,
                vf,
                attn_mask=maskf,
                dropout_p=dropout_p,
                is_causal=is_causal,
                **kwargs,
            )
        except TypeError:
            # Older/private SDPA variants don't accept is_causal/extra kwargs.
            out = _sdpa(qf, kf, vf, attn_mask=maskf, dropout_p=dropout_p)
        # torch<2.0 private API returns (attn_output, attn_weights)
        if isinstance(out, tuple):
            out = out[0]

    return out.to(orig_dtype)


# -------------------- Small MLP --------------------
def MLP(in_dim, hid, out_dim=None):
    out_dim = out_dim or hid
    return nn.Sequential(nn.Linear(in_dim, hid), nn.GELU(), nn.Linear(hid, out_dim))


class FiLM(nn.Module):
    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.scale = nn.Linear(cond_dim, feat_dim)
        self.shift = nn.Linear(cond_dim, feat_dim)

    def forward(self, x, cond):
        s = self.scale(cond)
        t = self.shift(cond)
        return (1.0 + s) * x + t


class CrossModalFusion(nn.Module):
    def __init__(self, D_loc=14, hidden=64, K=8, attn_heads=4):
        super().__init__()
        self.K = K
        self.H = hidden
        self.loc_enc = MLP(D_loc, hidden)
        self.cu_vec_enc = MLP(3, hidden)
        self.q_loc = nn.Linear(hidden, hidden)
        self.k_cu = nn.Linear(hidden, hidden)
        self.v_cu = nn.Linear(hidden, hidden)
        self.q_cu = nn.Linear(hidden, hidden)
        self.k_loc = nn.Linear(hidden, hidden)
        self.v_loc = nn.Linear(hidden, hidden)
        self.film_loc = FiLM(hidden, hidden)
        self.film_cu = FiLM(hidden, hidden)
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(self, loc_feat, cu_diff, dist_bias=None):
        B, N, K, _ = cu_diff.shape
        H = self.H
        loc_h = self.loc_enc(loc_feat)
        cu_hk = self.cu_vec_enc(cu_diff.reshape(B * N * K, 3)).reshape(B, N, K, H)
        q_loc = self.q_loc(loc_h).unsqueeze(2)
        k_cu = self.k_cu(cu_hk)
        v_cu = self.v_cu(cu_hk)
        q_flat = q_loc.reshape(B * N, 1, H)
        k_flat = k_cu.reshape(B * N, K, H)
        v_flat = v_cu.reshape(B * N, K, H)
        attn_mask = None
        # if _LOGGER.isEnabledFor(logging.DEBUG):
        #     _LOGGER.debug("CrossModalFusion.forward: before SDPA")
        #     _LOGGER.debug(
        #         "CrossModalFusion.forward: q/k/v dtype=%s/%s/%s",
        #         str(q_flat.dtype),
        #         str(k_flat.dtype),
        #         str(v_flat.dtype),
        #     )
        #     _LOGGER.debug(
        #         "CrossModalFusion.forward: q/k/v shape=%s/%s/%s",
        #         tuple(q_flat.shape),
        #         tuple(k_flat.shape),
        #         tuple(v_flat.shape),
        #     )

        if dist_bias is not None:
            attn_mask = (-dist_bias.reshape(B * N, 1, K)).to(loc_feat.dtype)
        # if _LOGGER.isEnabledFor(logging.DEBUG):
        #     _LOGGER.debug(
        #         "CrossModalFusion.forward: attn_mask dtype=%s shape=%s",
        #         str(attn_mask.dtype) if attn_mask is not None else "None",
        #         tuple(attn_mask.shape) if attn_mask is not None else "None",
        #     )

        attn_out = scaled_dot_product_attention(
            q_flat, k_flat, v_flat, attn_mask=attn_mask, dropout_p=0.0
        )
        attn_out = attn_out.reshape(B, N, H)
        loc2 = self.film_loc(loc_h, attn_out)
        # _LOGGER.debug("CrossModalFusion.forward: after SDPA")
        cu_summary = attn_out
        q_cu = self.q_cu(cu_summary)
        k_loc = self.k_loc(loc_h)
        v_loc = self.v_loc(loc_h)
        # _LOGGER.debug("CrossModalFusion.forward: before SDPA cu->loc")
        qc_flat = q_cu.reshape(B * N, 1, H)
        kl_flat = k_loc.reshape(B * N, 1, H)
        vl_flat = v_loc.reshape(B * N, 1, H)
        cu_att_out = scaled_dot_product_attention(
            qc_flat, kl_flat, vl_flat, attn_mask=None, dropout_p=0.0
        )
        cu_att_out = cu_att_out.reshape(B, N, H)
        # _LOGGER.debug("CrossModalFusion.forward: after SDPA cu->loc")
        cu2 = self.film_cu(cu_summary, cu_att_out)
        gate_in = torch.cat([loc2, cu2], dim=-1)
        g = self.gate(gate_in)
        fused = g * loc2 + (1.0 - g) * cu2
        fused = self.out_proj(fused)
        # _LOGGER.debug("CrossModalFusion.forward: after out_proj")
        return fused, {"gate": g.detach(), "loc2": loc2.detach(), "cu2": cu2.detach()}


class SimpleTransformer(nn.Module):
    def __init__(self, hidden_dim, n_head=4, n_layer=0):
        super().__init__()
        self.layers = nn.ModuleList()

    def forward(self, x):
        return x


# -------------------- Main SGDNTC Model --------------------
class SGDNTC_Model(nn.Module):
    def __init__(self, args, device: torch.device):
        super().__init__()
        self.device = device
        self.V_FEAT_DIM = getattr(args, "V_FEAT_DIM", 14)
        self.hidden = getattr(args, "hidden_size", 64)
        self.K = getattr(args, "K", 16)
        self.lattice = torch.tensor(
            getattr(args, "LATTICE_SIZE", (100.0, 100.0, 100.0)),
            device=self.device,
            dtype=torch.float32,
        )
        self.n_head = getattr(args, "n_head", 4)
        self.num_layers = 0
        self.output_dim = getattr(args, "output_dim", 8)
        self.fusion = CrossModalFusion(
            D_loc=self.V_FEAT_DIM, hidden=self.hidden, K=self.K, attn_heads=self.n_head
        )
        self.transformer = None
        self.predictor = nn.Linear(self.hidden, self.output_dim)
        self.to(self.device)

    def forward(self, V_feat: torch.Tensor, diff_k: torch.Tensor, dist_k: torch.Tensor, vv_edge_index=None):
        """
        V_feat: [B,N,H_loc]
        diff_k: [B,N,K,3]
        dist_k: [B,N,K]
        """
        # _LOGGER.debug("SGDNTC_Model.forward called")
        squeeze = False
        if V_feat.dim() == 2:
            V_feat = V_feat.unsqueeze(0)
            diff_k = diff_k.unsqueeze(0)
            dist_k = dist_k.unsqueeze(0)
            squeeze = True

        param_dtype = self.predictor.weight.dtype
        B, N = int(V_feat.shape[0]), int(V_feat.shape[1])
        chunk_size = int(env_int(EnvKeys.EMBED_CHUNK_SIZE, 0))
        if chunk_size <= 0:
            auto_chunk = int(env_int(EnvKeys.EMBED_CHUNK_AUTO, 200000))
            if auto_chunk > 0 and N > auto_chunk:
                chunk_size = auto_chunk
        # _LOGGER.debug("embedding: chunk_size=%s", int(chunk_size))
        if chunk_size > 0 and N > chunk_size:
            out_chunks = []
            for start in range(0, N, chunk_size):
                end = min(N, start + chunk_size)
                V_c = V_feat[:, start:end].to(device=self.device, dtype=param_dtype)
                diff_c = diff_k[:, start:end].to(device=self.device, dtype=param_dtype)
                dist_c = dist_k[:, start:end].to(device=self.device, dtype=param_dtype)
                fused_c, _ = self.fusion(V_c, diff_c, dist_bias=dist_c)
                if self.transformer is not None:
                    fused_c = self.transformer(fused_c)
                out_chunks.append(self.predictor(fused_c))
            scores = torch.cat(out_chunks, dim=1)
            return scores.squeeze(0) if squeeze else scores

        V_feat = V_feat.to(device=self.device, dtype=param_dtype)
        diff_k = diff_k.to(device=self.device, dtype=param_dtype)
        dist_k = dist_k.to(device=self.device, dtype=param_dtype)
        try:
            v_nan = torch.isnan(V_feat).any().item()
            v_inf = torch.isinf(V_feat).any().item()
            d_nan = torch.isnan(diff_k).any().item()
            d_inf = torch.isinf(diff_k).any().item()
            r_nan = torch.isnan(dist_k).any().item()
            r_inf = torch.isinf(dist_k).any().item()
            if v_nan or v_inf or d_nan or d_inf or r_nan or r_inf:
                _LOGGER.warning(
                    "[embed_in] V_feat(NaN=%s,Inf=%s,shape=%s,dtype=%s) diff_k(NaN=%s,Inf=%s,shape=%s,dtype=%s) dist_k(NaN=%s,Inf=%s,shape=%s,dtype=%s)",
                    v_nan,
                    v_inf,
                    tuple(V_feat.shape),
                    str(V_feat.dtype),
                    d_nan,
                    d_inf,
                    tuple(diff_k.shape),
                    str(diff_k.dtype),
                    r_nan,
                    r_inf,
                    tuple(dist_k.shape),
                    str(dist_k.dtype),
                )
        except Exception:
            pass
        fused, diag = self.fusion(V_feat, diff_k, dist_bias=dist_k)
        if self.transformer is not None:
            fused = self.transformer(fused)
        scores = self.predictor(fused)
        return scores.squeeze(0) if squeeze else scores
