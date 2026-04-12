"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from datetime import timedelta
from typing import Optional

import numpy as np
import torch
from einops import rearrange
from torch import nn, einsum
import math

from aurora.batch import Batch
from aurora.model.fourier import (
    absolute_time_expansion,
    lead_time_expansion,
    levels_expansion,
    pos_expansion,
    scale_expansion,
    memory_sequential_expansion
)
from aurora.model.levelcond import LevelConditioned
from aurora.model.patchembed import LevelPatchEmbed
from aurora.model.perceiver import MLP, PerceiverResampler
from aurora.model.posencoding import pos_scale_enc
from aurora.model.util import (
    check_lat_lon_dtype,
    init_weights,
)

class DriftAwareLightMemory(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        self.drift_extractor = DriftCorrectionExtractor(embed_dim)

        # self.memory_spatial_embed = nn.Linear(embed_dim, embed_dim)
        self.memory_sequential_embed = nn.Linear(embed_dim, embed_dim)

        # lightweight temporal retrieval
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.memory_proj = nn.Linear(embed_dim, embed_dim)

        # drift-aware temporal score
        self.cur_drift_proj = nn.Linear(embed_dim, embed_dim)
        self.mem_drift_proj = nn.Linear(embed_dim, embed_dim)

        # fusion gate
        self.fuse_gate = nn.Linear(embed_dim * 2, embed_dim)

        # weights
        self.lambda_drift = 0.3

    def build_physical_trajectory(self, x, memory_snapshot):
        if memory_snapshot is None:
            return x
        
        prev_state = memory_snapshot[:, -1]  # [B, L, D]
        return prev_state

    def forward(
        self,
        x,                      # [B, L, D]
        memory_snapshot,        # [B, T, L, D] or None
        memory_sequential_expansion,
        dtype=None,
    ):
        """
        x: current hidden state [B, L, D]
        memory_snapshot: historical drift-correction memory [B, T, L, D]
        """
        if dtype is None:
            dtype = x.dtype


        x_phys = self.build_physical_trajectory(
            x, memory_snapshot
        )  # [B, L, D]

        # current raw memory / drift-correction feature
        raw_memory, drift_residual = self.drift_extractor(x, x_phys)  # [B, L, D], [B, L, D]

        if memory_snapshot is not None:
            memory = memory_snapshot      # [B, T, L, D]
        else:
            memory = raw_memory.unsqueeze(1)  # [B, 1, L, D]

        Bm, Tm, Lm, Dm = memory.shape

        # --------------------------------------------------
        # 2) add sequential / rollout embedding
        # --------------------------------------------------
        memory_sequential_indexes = torch.arange(
            1, Tm + 1, device=x.device, dtype=dtype
        ).view(1, Tm).expand(Bm, Tm)

        memory_sequential_encode = memory_sequential_expansion(
            memory_sequential_indexes, self.embed_dim
        ).to(dtype=dtype)  # [B, T, D]

        memory_sequential_pos_emb = self.memory_sequential_embed(
            memory_sequential_encode
        ).unsqueeze(2)  # [B, T, 1, D]

        memory = memory + memory_sequential_pos_emb

        # ==================================================
        # 3) LIGHTWEIGHT Enhancement
        # ==================================================

        # current query summary: [B, D]
        q_global = (x + raw_memory).mean(dim=1)
        q_global = self.query_proj(q_global)

        # rollout summaries: [B, T, D]
        memory_global = memory.mean(dim=2)
        memory_global = self.memory_proj(memory_global)

        # ----- content score over rollout dimension only -----
        # [B, T]
        content_score = torch.einsum(
            'bd,btd->bt', q_global, memory_global
        ) / math.sqrt(Dm)

        # ----- drift consistency score -----
        cur_drift = self.cur_drift_proj(drift_residual.mean(dim=1))     # [B, D]

        if Tm > 1:
            mem_delta = torch.zeros_like(memory)
            mem_delta[:, 1:] = memory[:, 1:] - memory[:, :-1]
            mem_delta[:, 0] = memory[:, 0]
        else:
            mem_delta = memory
        mem_drift = self.mem_drift_proj(mem_delta.mean(dim=2))         # [B, T, D]

        drift_score = -((cur_drift.unsqueeze(1) - mem_drift) ** 2).mean(dim=-1)  # [B, T]

        # total sequential score
        time_score = content_score + self.lambda_drift * drift_score
        time_attn = torch.softmax(time_score, dim=1)  # [B, T]

        # ==================================================
        # 4) aggregate memory along rollout dimension
        # ==================================================
        # [B, L, D]
        enhanced_memory = torch.einsum('bt,btld->bld', time_attn, memory)

        # ==================================================
        # 5) gated fusion
        # ==================================================
        fuse_input = torch.cat([x, enhanced_memory], dim=-1)     # [B, L, 2D]
        gate = torch.sigmoid(self.fuse_gate(fuse_input))         # [B, L, D]

        x = x + raw_memory + gate * enhanced_memory
        
        aux = {
            "x_phys": x_phys,
            "drift_residual": drift_residual,
            "content_score": content_score,
            "drift_score": drift_score,
            "time_attn": time_attn,
        }

        return x, x.detach(), aux

class DriftCorrectionExtractor(nn.Module):
    """
    Explicit drift-correction feature extractor.

    Drift is defined as the departure from a physically plausible trajectory:
        delta = x - x_phys

    The extractor encodes this departure into a correction feature.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # encode drift residual itself
        self.delta_proj = nn.Linear(embed_dim, embed_dim)

        # optional context branches
        self.x_proj = nn.Linear(embed_dim, embed_dim)
        self.phys_proj = nn.Linear(embed_dim, embed_dim)

        # confidence / injection gate
        self.gate_proj = nn.Linear(embed_dim * 3, embed_dim)

        # final fusion
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, x_phys):
        """
        x      : [B, L, D] current latent field
        x_phys : [B, L, D] physically plausible continuation

        returns
        -------
        drift_feat  : [B, L, D]
        delta       : [B, L, D] explicit drift residual
        """
        delta = x - x_phys                           # explicit physical departure

        delta_feat = self.delta_proj(delta)
        x_feat = self.x_proj(x)
        phys_feat = self.phys_proj(x_phys)

        gate_input = torch.cat([delta, x, x_phys], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))

        drift_feat = gate * delta_feat + (1.0 - gate) * (x_feat - phys_feat)
        drift_feat = self.out_proj(drift_feat)

        return drift_feat, delta
    

class Perceiver3DEncoder(nn.Module):
    """Multi-scale multi-source multi-variable encoder based on the Perceiver architecture."""

    def __init__(
        self,
        surf_vars: tuple[str, ...],
        static_vars: tuple[str, ...] | None,
        atmos_vars: tuple[str, ...],
        patch_size: int = 4,
        latent_levels: int = 8,
        embed_dim: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        drop_rate: float = 0.0,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        max_history_size: int = 2,
        perceiver_ln_eps: float = 1e-5,
        stabilise_level_agg: bool = False,
        level_condition: Optional[tuple[int | float, ...]] = None,
        dynamic_vars: bool = False,
        atmos_static_vars: bool = False,
        simulate_indexing_bug: bool = False,
    ) -> None:
        """Initialise.

        Args:
            surf_vars (tuple[str, ...]): All supported surface-level variables.
            static_vars (tuple[str, ...], optional): All supported static variables.
            atmos_vars (tuple[str, ...]): All supported atmospheric variables.
            patch_size (int, optional): Patch size. Defaults to `4`.
            latent_levels (int): Number of latent pressure levels. Defaults to `8`.
            embed_dim (int, optional): Embedding dim. used in the aggregation blocks. Defaults
                to `1024`.
            num_heads (int, optional): Number of attention heads used in aggregation blocks.
                Defaults to `16`.
            head_dim (int, optional): Dimension of attention heads used in aggregation blocks.
                Defaults to `64`.
            drop_rate (float, optional): Drop out rate for input patches. Defaults to `0.0`.
            depth (int, optional): Number of Perceiver cross-attention and feed-forward blocks.
                Defaults to `2`.
            mlp_ratio (float, optional): Ratio of hidden dimensionality to embedding dimensionality
                for MLPs. Defaults to `4.0`.
            max_history_size (int, optional): Maximum number of history steps to consider. Defaults
                to `2`.
            perceiver_ln_eps (float, optional): Epsilon value for layer normalisation in the
                Perceiver. Defaults to `1e-5`.
            stabilise_level_agg (bool, optional): Stabilise the level aggregation by inserting an
                additional layer normalisation. Defaults to `False`.
            level_condition (tuple[int | float, ...], optional): Make the patch embeddings dependent
                on pressure level. If you want to enable this feature, provide a tuple of all
                possible pressure levels.
            dynamic_vars (bool, optional): Use dynamically generated static variables, like time
                of day. Defaults to `False`.
            atmos_static_vars (bool, optional): Also concatenate the static variables to the
                atmospheric variables. Defaults to `False`.
            simulate_indexing_bug (bool, optional): Simulate an indexing bug that's present for the
                air pollution version of Aurora. This is necessary to obtain numerical equivalence
                to the original implementation. Defaults to `False`.
        """
        super().__init__()

        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.level_condition = level_condition
        self.dynamic_vars = dynamic_vars
        self.atmos_static_vars = atmos_static_vars
        self.simulate_indexing_bug = simulate_indexing_bug

        # Add in the dynamic variables first.
        if self.dynamic_vars:
            if static_vars is None:
                static_vars = ()
            static_vars += ("tod_cos", "tod_sin", "dow_cos", "dow_sin", "doy_cos", "doy_sin")

        # We treat the static variables as surface variables in the model (and possibly even as
        # atmospheric variables!).
        if static_vars:
            surf_vars += static_vars
            if self.atmos_static_vars:
                # In this case, we prefix the static variables to avoid name clashes. E.g., `z` is
                # both a static variable and an atmospheric variable.
                atmos_vars += tuple(f"static_{v}" for v in static_vars)

        # Latent tokens
        # assert latent_levels > 1, "At least two latent levels are required."
        self.latent_levels = latent_levels
        # One latent level will be used by the surface level.
        # self.atmos_latents = nn.Parameter(torch.randn(latent_levels - 1, embed_dim))

        # Learnable embedding to encode the surface level.
        self.surf_level_encoding = nn.Parameter(torch.randn(embed_dim))
        self.surf_mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=drop_rate)
        self.surf_norm = nn.LayerNorm(embed_dim)

        # Position, scale, and time embeddings
        self.pos_embed = nn.Linear(embed_dim, embed_dim)
        self.scale_embed = nn.Linear(embed_dim, embed_dim)
        self.lead_time_embed = nn.Linear(embed_dim, embed_dim)
        self.absolute_time_embed = nn.Linear(embed_dim, embed_dim)
        self.atmos_levels_embed = nn.Linear(embed_dim, embed_dim)
        self.drift_aware_memory_process = DriftAwareLightMemory(embed_dim)

        # Patch embeddings:
        assert max_history_size > 0, "At least one history step is required."
        self.surf_token_embeds = LevelPatchEmbed(surf_vars, patch_size, embed_dim, max_history_size)
        # if not self.level_condition:
        #     self.atmos_token_embeds = LevelPatchEmbed(
        #         atmos_vars, patch_size, embed_dim, max_history_size
        #     )
        # else:
        #     self.atmos_token_embeds = LevelConditioned(
        #         lambda: LevelPatchEmbed(atmos_vars, patch_size, embed_dim, max_history_size),
        #         levels=self.level_condition,
        #         levels_dim=-5,
        #     )

        # Learnable pressure level aggregation:
        self.level_agg = PerceiverResampler(
            latent_dim=embed_dim,
            context_dim=embed_dim,
            depth=depth,
            head_dim=head_dim,
            num_heads=num_heads,
            drop=drop_rate,
            mlp_ratio=mlp_ratio,
            ln_eps=perceiver_ln_eps,
            ln_k_q=stabilise_level_agg,
        )

        # Drop patches after encoding.
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.apply(init_weights)

        # Initialize the latents like in the Huggingface implementation of the Perceiver:
        #
        #   https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/models/perceiver/modeling_perceiver.py#L628
        #
        # torch.nn.init.trunc_normal_(self.atmos_latents, std=0.02)
        torch.nn.init.trunc_normal_(self.surf_level_encoding, std=0.02)

    def aggregate_levels(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate pressure level information.

        Args:
            x (torch.Tensor): Tensor of shape `(B, C_A, L, D)` where `C_A` refers to the number
                of pressure levels.

        Returns:
            torch.Tensor: Tensor of shape `(B, C, L, D)` where `C` is the number of
                aggregated pressure levels.
        """
        B, _, L, _ = x.shape
        latents = self.atmos_latents.to(dtype=x.dtype)
        latents = latents.unsqueeze(1).expand(B, -1, L, -1)  # (C_A, D) to (B, C_A, L, D)

        x = torch.einsum("bcld->blcd", x)
        x = x.flatten(0, 1)  # (B * L, C_A, D)
        latents = torch.einsum("bcld->blcd", latents)
        latents = latents.flatten(0, 1)  # (B * L, C_A, D)

        x = self.level_agg(latents, x)  # (B * L, C, D)
        x = x.unflatten(dim=0, sizes=(B, L))  # (B, L, C, D)
        x = torch.einsum("blcd->bcld", x)  # (B, C, L, D)
        return x

    def forward(self, batch: Batch, lead_time: timedelta, memory_snapshot: list) -> torch.Tensor:
        """Peform encoding.

        Args:
            batch (:class:`aurora.Batch`): Batch to encode.
            lead_time (timedelta): Lead time.

        Returns:
            torch.Tensor: Encoding of shape `(B, L, D)`.
        """
        surf_vars = tuple(batch.surf_vars.keys())
        static_vars = tuple(batch.static_vars.keys())
        atmos_vars = tuple(batch.atmos_vars.keys())
        atmos_levels = batch.metadata.atmos_levels

        x_surf = torch.stack(tuple(batch.surf_vars.values()), dim=2)
        # x_static = torch.stack(tuple(batch.static_vars.values()), dim=2)
        # x_atmos = torch.stack(tuple(batch.atmos_vars.values()), dim=2)

        B, T, C, H, W = x_surf.size()
        assert x_surf.shape[:2] == (B, T), f"Expected shape {(B, T)}, got {x_surf.shape[:2]}."

        # if static_vars is None:
        #     assert x_static is None, "Static variables given, but not configured."
        # else:
        #     assert x_static is not None, "Static variables not given."
        #     x_static = x_static.expand((B, T, -1, -1, -1))

        #     if self.dynamic_vars:
        #         ones = torch.ones((1, T, 1, H, W), device=x_static.device, dtype=x_static.dtype)
        #         time = batch.metadata.time
        #         x_dynamic = torch.cat(
        #             [
        #                 torch.cat(
        #                     (
        #                         ones * np.cos(2 * np.pi * time[b].hour / 24),
        #                         ones * np.sin(2 * np.pi * time[b].hour / 24),
        #                         ones * np.cos(2 * np.pi * time[b].weekday() / 7),
        #                         ones * np.sin(2 * np.pi * time[b].weekday() / 7),
        #                         ones * np.cos(2 * np.pi * time[b].day / 365.25),
        #                         ones * np.sin(2 * np.pi * time[b].day / 365.25),
        #                     ),
        #                     dim=-3,
        #                 )
        #                 for b in range(B)
        #             ],
        #             dim=0,
        #         )
        #         dynamic_vars = ("tod_cos", "tod_sin", "dow_cos", "dow_sin", "doy_cos", "doy_sin")
        #         x_surf = torch.cat((x_surf, x_static, x_dynamic), dim=2)
        #         surf_vars = surf_vars + static_vars + dynamic_vars

        #         # Add to atmospheric variables too.
        #         if self.atmos_static_vars:
        #             # in this case, we prefix the static variables to avoid name clashes. e.g., `z`
        #             # is both a static variable and an atmospheric variable.
        #             atmos_vars += tuple(f"static_{v}" for v in static_vars + dynamic_vars)
        #             inds = (-1, -1, -1, len(atmos_levels), -1, -1)
        #             x_atmos = torch.cat(
        #                 (
        #                     x_atmos,
        #                     # Repeat for every pressure level.
        #                     x_static[..., None, :, :].expand(*inds),
        #                     x_dynamic[..., None, :, :].expand(*inds),
        #                 ),
        #                 dim=2,
        #             )
        #     else:
        #         x_surf = torch.cat((x_surf, x_static), dim=2)  # (B, T, V_S + V_Static, H, W)
        #         surf_vars = surf_vars + static_vars

        #         # Add to atmospheric variables too.
        #         if self.atmos_static_vars:
        #             atmos_vars = atmos_vars + static_vars
        #             x_atmos = torch.cat(
        #                 (
        #                     x_atmos,
        #                     # Repeat for every pressure level.
        #                     x_static[..., None, :, :].expand(-1, -1, -1, len(atmos_levels), -1, -1),
        #                 ),
        #                 dim=2,
        #             )

        lat, lon = batch.metadata.lat, batch.metadata.lon
        check_lat_lon_dtype(lat, lon)
        lat, lon = lat.to(dtype=torch.float32), lon.to(dtype=torch.float32)
        assert lat.shape[-1] == H and lon.shape[-1] == W

        # Patch embed the surface level.
        x_surf = rearrange(x_surf, "b t v h w -> b v t h w")
        x_surf = self.surf_token_embeds(x_surf, surf_vars)  # (B, L, D)
        dtype = x_surf.dtype  # When using mixed precision, we need to keep track of the dtype.

        # In the original implementation, both `z` and `static_z` point towards the same index,
        # meaning that they select the same slice. Simulate this bug.
        if self.simulate_indexing_bug and "z" in atmos_vars:
            i_z = atmos_vars.index("z")
            i_static_z = atmos_vars.index("static_z")
            x_atmos = torch.cat(
                (
                    x_atmos[:, :, :i_static_z],
                    x_atmos[:, :, i_z : i_z + 1],
                    x_atmos[:, :, i_static_z + 1 :],
                ),
                dim=2,
            )

        # Patch embed the atmospheric levels.
        # if not self.level_condition:
        #     x_atmos = rearrange(x_atmos, "b t v c h w -> (b c) v t h w")
        #     x_atmos = self.atmos_token_embeds(x_atmos, atmos_vars)
        #     x_atmos = rearrange(x_atmos, "(b c) l d -> b c l d", b=B, c=C)
        # else:
        #     # In this case we need to keep the levels dimension separate.
        #     x_atmos = rearrange(x_atmos, "b t v c h w -> b c v t h w")
        #     x_atmos = self.atmos_token_embeds(x_atmos, atmos_vars, levels=atmos_levels)
        #     # The levels dimension is now already in the right place.

        # Add surface level encoding. This helps the model distinguish between surface and
        # atmospheric levels.
        x_surf = x_surf + self.surf_level_encoding[None, None, :].to(dtype=dtype)
        # Since the surface level is not aggregated, we add a Perceiver-like MLP only.
        x_surf = x_surf + self.surf_norm(self.surf_mlp(x_surf))

        # Add atmospheric pressure encoding of shape (C_A, D) and subsequent embedding.
        # atmos_levels_tensor = torch.tensor(atmos_levels, device=x_atmos.device)
        # atmos_levels_encode = levels_expansion(atmos_levels_tensor, self.embed_dim).to(dtype=dtype)
        # atmos_levels_embed = self.atmos_levels_embed(atmos_levels_encode)[None, :, None, :]
        # x_atmos = x_atmos + atmos_levels_embed  # (B, C_A, L, D)

        # Aggregate over pressure levels.
        # x_atmos = self.aggregate_levels(x_atmos)  # (B, C_A, L, D) to (B, C, L, D)

        # Concatenate the surface level with the amospheric levels.
        # x = torch.cat((x_surf.unsqueeze(1), x_atmos), dim=1)
        x = x_surf.unsqueeze(1)

        # Add position and scale embeddings to the 3D tensor.
        pos_encode_list, scale_encode_list = [], []
        for la, lo in zip(lat, lon):
            pos_encode, scale_encode = pos_scale_enc(
                self.embed_dim,
                la,
                lo,
                self.patch_size,
                pos_expansion=pos_expansion,
                scale_expansion=scale_expansion,
            )
            pos_encode_list.append(pos_encode)
            scale_encode_list.append(scale_encode)
        pos_encode = torch.stack(pos_encode_list, dim=0).unsqueeze(1)  # (B, 1, L, D)
        scale_encode = torch.stack(scale_encode_list, dim=0).unsqueeze(1)  # (B, 1, L, D)
        pos_encode = self.pos_embed(pos_encode.to(dtype=dtype))
        scale_encode = self.scale_embed(scale_encode.to(dtype=dtype))
        x = x + pos_encode + scale_encode

        # Flatten the tokens.
        x = x.reshape(B, -1, self.embed_dim)  # (B, C + 1, L, D) to (B, L', D)

        # Add lead time embedding.
        lead_hours = lead_time.total_seconds() / 3600
        lead_times = lead_hours * torch.ones(B, dtype=dtype, device=x.device)
        lead_time_encode = lead_time_expansion(lead_times, self.embed_dim).to(dtype=dtype)
        lead_time_emb = self.lead_time_embed(lead_time_encode)  # (B, D)
        x = x + lead_time_emb.unsqueeze(1)  # (B, L', D) + (B, 1, D)

        # Add absolute time embedding.
        absolute_times_list = [t.timestamp() / 3600 for t in batch.metadata.time]  # Times in hours
        absolute_times = torch.tensor(absolute_times_list, dtype=torch.float32, device=x.device)
        absolute_time_encode = absolute_time_expansion(absolute_times, self.embed_dim)
        absolute_time_embed = self.absolute_time_embed(absolute_time_encode.to(dtype=dtype))
        x = x + absolute_time_embed.unsqueeze(1)  # (B, L, D) + (B, 1, D)

        # Memory processing.
        x, enhanced_x_memory, aux = self.drift_aware_memory_process(x, memory_snapshot, memory_sequential_expansion)
        
        x = self.pos_drop(x)
        return x, enhanced_x_memory.detach()
