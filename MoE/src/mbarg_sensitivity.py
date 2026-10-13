from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

def quantize_symmetric(x: torch.Tensor, bits: int, eps: float = 1e-8) -> torch.Tensor:
    if bits < 2:
        raise ValueError(f"bits must be >= 2, got {bits}")

    qmax = (1 << (bits - 1)) - 1
    alpha = x.detach().abs().max()
    scale = torch.clamp(alpha / qmax, min=eps)
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return q * scale

@dataclass(frozen=True)
class SensitivityStats:

    jr: torch.Tensor
    snr_hist: torch.Tensor
    uncertainty: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    q: torch.Tensor

    def as_dict(self):
        return {
            "jr": self.jr.detach().cpu().tolist(),
            "snr_hist": self.snr_hist.detach().cpu().tolist(),
            "uncertainty": self.uncertainty.detach().cpu().tolist(),
            "u": self.u.detach().cpu().tolist(),
            "v": self.v.detach().cpu().tolist(),
            "q": self.q.detach().cpu().tolist(),
        }

def _flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError(f"expected at least 2 dimensions [batch, ...], got {tuple(x.shape)}")
    return x.reshape(x.shape[0], -1)

def jump_ratio(history: torch.Tensor, update: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if history.shape != update.shape:
        raise ValueError(f"history and update must have the same shape, got {history.shape} and {update.shape}")
    h = _flatten_tokens(history.float())
    d = _flatten_tokens(update.float())
    return torch.linalg.vector_norm(d, dim=-1) / (torch.linalg.vector_norm(h, dim=-1) + eps)

def historical_snr(
    history: torch.Tensor,
    update: torch.Tensor,
    *,
    bits: int = 4,
    eps: float = 1e-8,
) -> torch.Tensor:
    if history.shape != update.shape:
        raise ValueError(f"history and update must have the same shape, got {history.shape} and {update.shape}")
    residual = history.float() + update.float()
    quantized = quantize_symmetric(residual, bits=bits)
    reconstructed_history = quantized - update.float()
    h = _flatten_tokens(history.float())
    error = _flatten_tokens(reconstructed_history - history.float())
    signal_power = h.square().mean(dim=-1)
    error_power = error.square().mean(dim=-1)
    return 10.0 * torch.log10((signal_power + eps) / (error_power + eps))

def robust_zscore(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    median = x.median()
    mad = (x - median).abs().median()
    scale = 1.4826 * mad + eps
    return (x - median) / scale

def summarize_sensitivity(
    jr_batches: Sequence[torch.Tensor] | Iterable[torch.Tensor],
    snr_batches: Sequence[torch.Tensor] | Iterable[torch.Tensor],
    *,
    eps: float = 1e-8,
) -> SensitivityStats:
    jr_list = list(jr_batches)
    snr_list = list(snr_batches)
    if not jr_list or len(jr_list) != len(snr_list):
        raise ValueError("jr_batches and snr_batches must be non-empty and have equal length")

    jr_rows, snr_rows = [], []
    for jr, snr in zip(jr_list, snr_list):
        jr = torch.as_tensor(jr).float()
        snr = torch.as_tensor(snr).float()
        if jr.ndim == 1:
            jr = jr.unsqueeze(0)
        if snr.ndim == 1:
            snr = snr.unsqueeze(0)
        if jr.ndim != 2 or snr.ndim != 2 or jr.shape != snr.shape:
            raise ValueError("each JR/SNR item must be [layers] or [batch, layers]")
        jr_rows.append(jr.mean(dim=0))
        snr_rows.append(snr.mean(dim=0))

    jr_matrix = torch.stack(jr_rows)
    snr_matrix = torch.stack(snr_rows)
    log_jr = torch.log(jr_matrix.clamp_min(0.0) + eps)
    neg_snr = -snr_matrix
    jr = torch.median(jr_matrix, dim=0).values
    snr = torch.median(snr_matrix, dim=0).values
    u = robust_zscore(torch.median(log_jr, dim=0).values, eps=eps)
    v = robust_zscore(torch.median(neg_snr, dim=0).values, eps=eps)

    jr_mad = (log_jr - log_jr.median(dim=0).values).abs().median(dim=0).values
    snr_mad = (neg_snr - neg_snr.median(dim=0).values).abs().median(dim=0).values
    uncertainty = 1.4826 * (jr_mad + snr_mad)
    q = uncertainty / (uncertainty.median() + eps)
    return SensitivityStats(jr=jr, snr_hist=snr, uncertainty=uncertainty, u=u, v=v, q=q.clamp_min(0.0))

def stats_from_residual_batches(
    residual_batches: Sequence[tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]],
    *,
    eps: float = 1e-8,
) -> SensitivityStats:
    jr_batches, snr_batches = [], []
    for histories, updates in residual_batches:
        if len(histories) != len(updates):
            raise ValueError("each residual batch must contain matching histories and updates")
        jr_batches.append(torch.stack([jump_ratio(h, d, eps=eps) for h, d in zip(histories, updates)]))
        snr_batches.append(torch.stack([historical_snr(h, d, eps=eps) for h, d in zip(histories, updates)]))
    return summarize_sensitivity(jr_batches, snr_batches, eps=eps)

compute_jump_ratio = jump_ratio
compute_historical_snr = historical_snr

