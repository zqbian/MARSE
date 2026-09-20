"""Inference-only MARSE model definition.

This file intentionally contains only the modules used by the released
inference path. Training losses, data generation, and experimental branches
remain outside this package.
"""

from itertools import combinations
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn


class PairwiseSpatialFeatures(nn.Module):
    def __init__(self, n_channels: int = 6):
        super().__init__()
        self.mic_pairs = list(combinations(range(n_channels), 2))

    def forward(self, stft: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                torch.angle(stft[:, mic_i]) - torch.angle(stft[:, mic_j])
                for mic_i, mic_j in self.mic_pairs
            ],
            dim=1,
        )


class RegionAngleSampler(nn.Module):
    def __init__(self, n_samples: int = 11):
        super().__init__()
        self.n_samples = n_samples

    def forward(self, theta_l: torch.Tensor, theta_h: torch.Tensor) -> torch.Tensor:
        raw_span = theta_h - theta_l
        span = torch.remainder(raw_span, 360.0)
        full_span = torch.isclose(
            raw_span.abs(), torch.full_like(raw_span, 360.0), atol=1e-4, rtol=0.0
        )
        span = torch.where(full_span, torch.full_like(span, 360.0), span)
        interpolation = torch.linspace(
            0.0, 1.0, self.n_samples, device=theta_l.device, dtype=theta_l.dtype
        )
        full_interpolation = torch.arange(
            self.n_samples, device=theta_l.device, dtype=theta_l.dtype
        ) / float(self.n_samples)
        interpolation = torch.where(
            full_span.unsqueeze(-1),
            full_interpolation.unsqueeze(0),
            interpolation.unsqueeze(0),
        )
        angles = theta_l.unsqueeze(-1) + span.unsqueeze(-1) * interpolation
        return torch.remainder(angles + 180.0, 360.0) - 180.0


class DirectionFeatures(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        sound_speed: float = 343.0,
        mic_radius: float = 0.05,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.sound_speed = sound_speed
        self.mic_pairs = list(combinations(range(6), 2))
        positions = mic_radius * torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=torch.float32,
        )
        pair_diffs = [positions[j] - positions[i] for i, j in self.mic_pairs]
        self.register_buffer("pair_diffs", torch.stack(pair_diffs), persistent=False)
        self.reference_pair_index = 0

    def forward(
        self,
        observed_ipd: torch.Tensor,
        sampled_angles: torch.Tensor,
        n_freq: int,
    ) -> torch.Tensor:
        dtype = observed_ipd.dtype
        device = observed_ipd.device
        theta = torch.deg2rad(sampled_angles)
        directions = torch.stack(
            (torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)), dim=-1
        )
        delays = torch.einsum(
            "pd,bnd->bnp", self.pair_diffs.to(device=device, dtype=dtype), directions
        ) / self.sound_speed
        frequencies = torch.linspace(
            0.0, self.sample_rate / 2.0, n_freq, device=device, dtype=dtype
        )
        theoretical_phase = (
            2.0 * torch.pi * delays.unsqueeze(-1) * frequencies.view(1, 1, 1, -1)
        )
        residual = observed_ipd.unsqueeze(1) - theoretical_phase.unsqueeze(-1)
        features = torch.cos(residual).permute(0, 3, 4, 1, 2)

        order = torch.argsort(delays[..., self.reference_pair_index], dim=-1)
        gather_index = order[:, None, None, :, None].expand(
            -1, features.shape[1], features.shape[2], -1, features.shape[4]
        )
        return torch.gather(features, dim=3, index=gather_index)


class RegionAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.region_rnn = nn.LSTM(15, 32, batch_first=True)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        batch, n_freq, n_time, n_angles, n_features = sequence.shape
        sequence = sequence.reshape(-1, n_angles, n_features)
        looped = torch.cat((sequence, sequence[:, :1]), dim=1)
        output, _ = self.region_rnn(looped)
        descriptor = torch.cat((output[:, -2], output[:, -1]), dim=-1)
        return descriptor.reshape(batch, n_freq, n_time, 64)


class NarrowBandBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_mhsa = nn.LayerNorm(128)
        self.mhsa = nn.MultiheadAttention(128, 4, batch_first=True)
        self.dropout_mhsa = nn.Dropout(0.0)
        self.norm_tconv = nn.LayerNorm(128)
        self.tconvffn = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.SiLU(),
            nn.Conv1d(256, 256, 3, padding=1, groups=8),
            nn.SiLU(),
            nn.Conv1d(256, 256, 3, padding=1, groups=8),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv1d(256, 256, 3, padding=1, groups=8),
            nn.SiLU(),
            nn.Conv1d(256, 128, 1),
        )
        self.dropout_tconv = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_freq, n_time, hidden = x.shape
        attn_input = self.norm_mhsa(x).reshape(batch * n_freq, n_time, hidden)
        attn_output, _ = self.mhsa(
            attn_input, attn_input, attn_input, need_weights=False
        )
        x = x + self.dropout_mhsa(
            attn_output.reshape(batch, n_freq, n_time, hidden)
        )
        conv_input = self.norm_tconv(x).permute(0, 1, 3, 2).reshape(
            batch * n_freq, hidden, n_time
        )
        conv_output = self.tconvffn(conv_input)
        conv_output = conv_output.reshape(batch, n_freq, hidden, n_time).permute(
            0, 1, 3, 2
        )
        return x + self.dropout_tconv(conv_output)


class CrossArrayAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 128
        self.head_dim = 64
        self.n_heads = 2
        self.window_size = 5
        self.attn_dim = self.head_dim * self.n_heads
        self.scale = self.head_dim ** -0.5
        self.norm1 = nn.LayerNorm(self.input_dim)
        self.norm2 = nn.LayerNorm(self.input_dim)
        self.proj_q = nn.Linear(self.input_dim, self.attn_dim)
        self.proj_k = nn.Linear(self.input_dim, self.attn_dim)
        self.proj_v = nn.Linear(self.input_dim, self.attn_dim)
        self.proj_out = nn.Linear(self.attn_dim, self.input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.input_dim, 2 * self.input_dim),
            nn.GELU(),
            nn.Linear(2 * self.input_dim, self.input_dim),
        )
        self.gate_attn = nn.Parameter(torch.tensor([0.5]))
        self.gate_ffn = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_arrays, n_freq, n_time, hidden = x.shape
        pad = self.window_size // 2
        normed = self.norm1(x)
        flat = normed.reshape(batch * n_arrays, n_freq, n_time, hidden)
        queries = self.proj_q(flat).reshape(
            batch, n_arrays, n_freq, n_time, self.n_heads, self.head_dim
        )
        keys = self.proj_k(flat).reshape(
            batch, n_arrays, n_freq, n_time, self.n_heads, self.head_dim
        )
        values = self.proj_v(flat).reshape(
            batch, n_arrays, n_freq, n_time, self.n_heads, self.head_dim
        )
        keys = F.pad(keys, (0, 0, 0, 0, pad, pad)).unfold(
            3, self.window_size, 1
        ).permute(0, 1, 2, 3, 6, 4, 5)
        values = F.pad(values, (0, 0, 0, 0, pad, pad)).unfold(
            3, self.window_size, 1
        ).permute(0, 1, 2, 3, 6, 4, 5)

        enhanced = x.clone()
        for array_idx in range(n_arrays):
            other_arrays = [idx for idx in range(n_arrays) if idx != array_idx]
            query = queries[:, array_idx].reshape(
                batch * n_freq, n_time, self.n_heads, self.head_dim
            ).unsqueeze(3)
            n_keys = (n_arrays - 1) * self.window_size
            key = torch.cat([keys[:, idx] for idx in other_arrays], dim=3).reshape(
                batch * n_freq,
                n_time,
                n_keys,
                self.n_heads,
                self.head_dim,
            ).permute(0, 1, 3, 2, 4)
            value = torch.cat(
                [values[:, idx] for idx in other_arrays], dim=3
            ).reshape(
                batch * n_freq,
                n_time,
                n_keys,
                self.n_heads,
                self.head_dim,
            ).permute(0, 1, 3, 2, 4)
            weights = torch.softmax(
                torch.matmul(query, key.transpose(-1, -2)) * self.scale, dim=-1
            )
            attended = torch.matmul(weights, value).squeeze(3).reshape(
                batch, n_freq, n_time, self.attn_dim
            )
            enhanced[:, array_idx] = (
                x[:, array_idx] + self.gate_attn * self.proj_out(attended)
            )
        return enhanced + self.gate_ffn * self.ffn(self.norm2(enhanced))


class MARSE(nn.Module):
    """Fixed inference architecture used by the released MARSE checkpoint."""

    n_arrays = 3
    n_channels_per_array = 6
    reference_channel = 0
    sample_rate = 16000

    def __init__(self):
        super().__init__()
        self.spatial_extractor = PairwiseSpatialFeatures(self.n_channels_per_array)
        self.angle_sampler = RegionAngleSampler(11)
        self.direction_extractor = DirectionFeatures(sample_rate=self.sample_rate)
        self.base_encoder = nn.Linear(12, 32)
        self.query_state_encoder = nn.Linear(3, 256)
        self.region_aggregator = RegionAggregator()
        self.dropout = nn.Dropout(0.0)
        self.lstm1 = nn.LSTM(96, 256, bidirectional=True, batch_first=False)
        self.narrow_input_proj = nn.Linear(512, 128)
        self.narrow_blocks = nn.ModuleList([NarrowBandBlock(), NarrowBandBlock()])
        self.ff = nn.Linear(128, 2)
        self.cross_array_attention = CrossArrayAttention()
        self.local_angular_encoder = nn.Sequential(
            nn.Linear(5, 64), nn.Tanh(), nn.Linear(64, 64)
        )
        self.dynamic_reference_selector = nn.Sequential(
            nn.Linear(196, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        self.dynamic_reference_silence_gate = nn.Sequential(
            nn.Linear(591, 64), nn.Tanh(), nn.Linear(64, 1)
        )

        self.last_per_array_stacked_masks = None
        self.last_selector_weights = None
        self.last_silence_prob = None

    @staticmethod
    def _query_features(theta_l: torch.Tensor, theta_h: torch.Tensor) -> torch.Tensor:
        raw_width = theta_h - theta_l
        width = torch.remainder(raw_width, 360.0)
        full_circle = torch.isclose(
            raw_width.abs(), torch.full_like(raw_width, 360.0), atol=1e-4, rtol=0.0
        )
        width = torch.where(full_circle, torch.full_like(width, 360.0), width)
        center = torch.remainder(theta_l + 0.5 * width + 180.0, 360.0) - 180.0
        center = torch.deg2rad(center)
        return torch.stack((torch.sin(center), torch.cos(center), width / 180.0), -1)

    def _local_queries(
        self, theta_l: torch.Tensor, theta_h: torch.Tensor
    ) -> torch.Tensor:
        raw_span = theta_h - theta_l
        full_span = torch.isclose(
            raw_span.abs(), torch.full_like(raw_span, 360.0), atol=1e-4, rtol=0.0
        )
        true_zero = torch.isclose(
            raw_span, torch.zeros_like(raw_span), atol=1e-4, rtol=0.0
        )
        span = torch.remainder(raw_span, 360.0)
        width = torch.where(
            full_span,
            torch.full_like(span, 360.0),
            torch.where(true_zero, torch.zeros_like(span), span),
        )
        theta_l_rad = torch.deg2rad(theta_l)
        theta_h_rad = torch.deg2rad(theta_h)
        raw = torch.stack(
            (
                torch.sin(theta_l_rad),
                torch.cos(theta_l_rad),
                torch.sin(theta_h_rad),
                torch.cos(theta_h_rad),
                width / 360.0,
            ),
            dim=-1,
        )
        return self.local_angular_encoder(raw)

    def _encode_array(
        self, features: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        batch, n_freq, n_time, n_features = features.shape
        sequence = features.permute(1, 0, 2, 3).reshape(
            n_freq, batch * n_time, n_features
        )
        cell = self.query_state_encoder(query).unsqueeze(0).unsqueeze(2)
        cell = cell.repeat(2, 1, n_time, 1).reshape(2, batch * n_time, 256)
        sequence, _ = self.lstm1(sequence, (torch.zeros_like(cell), cell))
        sequence = sequence.reshape(n_freq, batch, n_time, 512).permute(1, 0, 2, 3)
        sequence = self.narrow_input_proj(sequence)
        for block in self.narrow_blocks:
            sequence = block(sequence)
        return sequence

    def forward(
        self,
        x: torch.Tensor,
        theta_l: torch.Tensor,
        theta_h: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate a selected-array compressed CRM.

        Args:
            x: Stacked real/imaginary STFT, shape ``[B, 3, 12, F, T]``.
            theta_l: Lower local azimuth bounds in degrees, shape ``[B, 3]``.
            theta_h: Upper local azimuth bounds in degrees, shape ``[B, 3]``.
        """
        if x.ndim != 5 or x.shape[1:3] != (3, 12):
            raise ValueError(f"Expected x with shape [B, 3, 12, F, T], got {x.shape}.")
        if theta_l.shape != theta_h.shape or theta_l.shape != x.shape[:2]:
            raise ValueError("theta_l and theta_h must both have shape [B, 3].")

        batch, n_arrays, _, n_freq, n_time = x.shape
        theta_l = theta_l.to(device=x.device, dtype=x.dtype)
        theta_h = theta_h.to(device=x.device, dtype=x.dtype)
        local_queries = self._local_queries(theta_l, theta_h)

        flat = x.reshape(batch * n_arrays, 12, n_freq, n_time)
        real, imag = flat[:, :6], flat[:, 6:]
        stft = torch.complex(real, imag)
        base = torch.cat((real, imag), dim=1).permute(0, 2, 3, 1)
        observed_ipd = self.spatial_extractor(stft)
        flat_l = theta_l.reshape(-1)
        flat_h = theta_h.reshape(-1)
        sampled_angles = self.angle_sampler(flat_l, flat_h)
        directions = self.direction_extractor(observed_ipd, sampled_angles, n_freq)
        region = self.region_aggregator(directions)
        features = torch.cat((self.base_encoder(base), region), dim=-1)
        query = self._query_features(flat_l, flat_h)
        latents = self._encode_array(features, query)
        latents = latents.reshape(batch, n_arrays, n_freq, n_time, 128)
        latents = self.cross_array_attention(latents)

        logits = self.ff(latents)
        masks = torch.tanh(logits.permute(0, 1, 4, 2, 3))
        evidence = torch.cat(
            (logits.mean(dim=(2, 3)), logits.abs().mean(dim=(2, 3))), dim=-1
        )
        summaries = torch.cat(
            (latents.mean(dim=(2, 3)), evidence, local_queries), dim=-1
        )
        selector_logits = self.dynamic_reference_selector(summaries).squeeze(-1)
        selector_weights = torch.softmax(selector_logits, dim=1)
        gate_input = torch.cat((summaries.reshape(batch, -1), selector_weights), dim=-1)
        silence_prob = torch.sigmoid(
            self.dynamic_reference_silence_gate(gate_input).squeeze(-1)
        )

        selected = selector_weights.argmax(dim=1)
        batch_index = torch.arange(batch, device=x.device)
        self.last_per_array_stacked_masks = masks
        self.last_selector_weights = selector_weights
        self.last_silence_prob = silence_prob
        return masks[batch_index, selected]


def load_model(
    checkpoint: Union[str, Path], device: Union[str, torch.device] = "cpu"
) -> MARSE:
    """Load MARSE from a project Lightning checkpoint or a plain state dict."""
    device = torch.device(device)
    payload = torch.load(str(checkpoint), map_location=device)
    state = payload.get("state_dict", payload)
    state = {
        (key[6:] if key.startswith("model.") else key): value
        for key, value in state.items()
    }

    model = MARSE()
    expected = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in expected and expected[key].shape == value.shape
    }
    missing = sorted(set(expected) - set(compatible))
    if missing:
        raise RuntimeError(
            "Checkpoint is missing inference parameters: " + ", ".join(missing)
        )
    model.load_state_dict(compatible, strict=True)
    model.to(device).eval()
    return model


def decompress_cirm(mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Invert the bounded complex-ratio-mask representation."""
    compressed = torch.complex(mask[:, 0], mask[:, 1])
    numerator = 1.0 - compressed
    denominator = 1.0 + compressed
    denominator = denominator + eps * (denominator.abs() < eps)
    return -torch.log(numerator / denominator)
