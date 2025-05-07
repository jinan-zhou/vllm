# SPDX-License-Identifier: Apache-2.0

"""
TimesFM

Added by Nutanix AI Team, 2025
"""


from typing import Optional
import math

import torch
from torch import nn
import torch.nn.functional as F
from transformers import TimesFmConfig

from vllm.config import VllmConfig
from vllm.attention import Attention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    MergedColumnParallelLinear,
)


class TimesFmMLP(nn.Module):
    def __init__(
        self,
        config: TimesFmConfig,
        prefix: str,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            prefix=f"{prefix}.gate_proj",
            quant_config=quant_config,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, 
            hidden_size,
            bias=False,
            prefix=f"{prefix}.down_proj",
            quant_config=quant_config,
        )
        self.layer_norm = nn.LayerNorm(
            normalized_shape=hidden_size,
            eps=1e-6,
        )
        
    def forward(self, x, paddings=None):
        gate_inp = self.layer_norm(x)
        gate = self.gate_proj(gate_inp)
        gate = F.relu(gate)
        outputs = self.down_proj(gate)
        if paddings is not None:
            outputs = outputs * (1.0 - paddings[:, :, None])
        return outputs + x


class TimesFmResidualBlock(nn.Module):
    def __init__(
        self,
        input_dims,
        hidden_dims,
        output_dims,
        prefix: str,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims

        self.input_layer = MergedColumnParallelLinear(
            input_dims,
            hidden_dims,
            bias=False,
            prefix=f"{prefix}.input_layer",
            quant_config=quant_config,
        )
        self.activation = nn.SiLU()
        self.output_layer = RowParallelLinear(
            hidden_dims,
            output_dims,
            bias=False,
            prefix=f"{prefix}.output_layer",
            quant_config=quant_config,
        )
        self.residual_layer = RowParallelLinear(
            input_dims,
            output_dims,
            bias=False,
            prefix=f"{prefix}.residual_layer",
            quant_config=quant_config,
        )

    def forward(self, x):
        hidden = self.input_layer(x)
        hidden = self.activation(hidden)
        output = self.output_layer(hidden)
        residual = self.residual_layer(x)
        return output + residual


class TimesFmRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class TimesFmPositionalEmbedding(nn.Module):
    def __init__(self, config: TimesFmConfig):
        super().__init__()
        min_timescale = config.min_timescale
        max_timescale = config.max_timescale
        self.embedding_dims = config.hidden_size
        num_timescales = self.embedding_dims // 2
        log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / max(num_timescales - 1, 1)
        self.register_buffer(
            "inv_timescales",
            min_timescale * torch.exp(torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment),
        )
        
    def forward(self, seq_length=None, position=None):
        if position is None and seq_length is None:
            raise ValueError("Either position or seq_length must be provided")

        if position is None:
            # [1, seqlen]
            position = torch.arange(seq_length, dtype=torch.float32, device=self.inv_timescales.device).unsqueeze(0)
        elif position.ndim != 2:
            raise ValueError(f"position must be 2-dimensional, got shape {position.shape}")

        scaled_time = position.view(*position.shape, 1) * self.inv_timescales.view(1, 1, -1)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)

        # Padding to ensure correct embedding dimension
        signal = F.pad(signal, (0, 0, 0, self.embedding_dims % 2))
        return signal
