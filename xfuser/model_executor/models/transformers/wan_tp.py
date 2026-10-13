"""Wan tensor parallelism applied after xDiT materializes the pipeline.

The runner owns loading, process groups, placement and inference. These wrappers
shard loaded FFN/self/text-cross-attention parameters, normalize Q/K across TP
ranks and reduce Row Linear outputs while preserving the native processors.
"""

import torch
from torch import nn
from torch.nn import functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.transformers.transformer_wan import WanAttention

from xfuser.core.distributed.parallel_state import get_tp_group, get_ulysses_parallel_world_size
from xfuser.model_executor.layers.base_layer import xFuserLayerBaseWrapper


def _shard_linear(linear: nn.Linear, dim: int) -> None:
    """Give a loaded Linear independent storage for its TP-local weight slice."""
    group = get_tp_group()
    width = linear.weight.shape[dim] // group.world_size
    weight = linear.weight
    linear.weight = nn.Parameter(
        weight.detach().narrow(dim, group.rank_in_group * width, width).clone(
            memory_format=torch.contiguous_format
        ),
        requires_grad=weight.requires_grad,
    )
    if dim == 0:
        linear.out_features = width
        if linear.bias is not None:
            bias = linear.bias
            linear.bias = nn.Parameter(
                bias.detach().narrow(0, group.rank_in_group * width, width).clone(),
                requires_grad=bias.requires_grad,
            )
    else:
        linear.in_features = width


class xFuserWanTPRowLinear(xFuserLayerBaseWrapper):
    """Reduce partial outputs with one fused bias contribution on TP rank 0."""

    def __init__(self, linear: nn.Linear):
        super().__init__(module=linear)
        _shard_linear(self.module, dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        group = get_tp_group()
        bias = self.module.bias if group.rank_in_group == 0 else None
        output = F.linear(hidden_states, self.module.weight, bias)
        return group.all_reduce(output)


class xFuserWanTPFeedForward(xFuserLayerBaseWrapper):
    """Keep native GELU/dropout and shard only the FFN intermediate channels.

    Input/output retain the full hidden dimension. Column shards the first
    projection and its bias; Row shards the second projection's input dimension
    and keeps its full bias. Native FeedForward still executes its net in order.
    """

    def __init__(self, feedforward: FeedForward):
        super().__init__(module=feedforward)
        column = self.module.net[0].proj
        if column.out_features % get_tp_group().world_size:
            raise ValueError("Wan FFN intermediate dimension must be divisible by TP degree")
        _shard_linear(column, dim=0)
        self.module.net[2] = xFuserWanTPRowLinear(self.module.net[2])

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.module(hidden_states, *args, **kwargs)


class xFuserWanTPRMSNorm(xFuserLayerBaseWrapper):
    """Normalize a channel shard using the original full normalized dimension.

    Keep the loaded RMSNorm's normalized_shape and eps as the global contract;
    only its affine weight is sharded. Match PyTorch 2.8 RMSNorm's opmath order:
    accumulate, normalize and multiply the weight before restoring input dtype.
    This wrapper is not nn.RMSNorm, so xDiT's local fused norm/RoPE kernel uses
    its existing reference path, which calls this distributed normalization.
    """

    def __init__(self, norm: nn.RMSNorm):
        super().__init__(module=norm)
        group = get_tp_group()
        weight = self.module.weight
        width = weight.numel() // group.world_size
        self.module.weight = nn.Parameter(
            weight.detach().narrow(0, group.rank_in_group * width, width).clone(),
            requires_grad=weight.requires_grad,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        opmath_dtype = (
            torch.float32 if input_dtype in (torch.float16, torch.bfloat16) else input_dtype
        )
        upcasted_input = hidden_states.to(opmath_dtype)
        square_sum = upcasted_input.square().sum(dim=-1, keepdim=True)
        square_sum = get_tp_group().all_reduce(square_sum)
        mean_square = square_sum / self.module.normalized_shape[0]
        output = upcasted_input * torch.rsqrt(mean_square + self.module.eps)
        return (output * self.module.weight).to(input_dtype)


def parallelize_wan_attention(attn: WanAttention) -> None:
    """Shard loaded Wan T2V self/text-cross attention, keeping its processor.

    Call once after loading/placement and before offload, compile or inference.
    Q/K/V own the same contiguous head range. Head metadata describes that local
    range; the Transformer's config remains the source of global model dimensions.
    Cross-attention keeps full text context on each TP rank and uses the native
    non-USP attention path, without RoPE or sequence/head layout communication.
    """
    group = get_tp_group()
    head_partition_degree = group.world_size
    if not attn.processor.is_cross_attention:
        head_partition_degree *= get_ulysses_parallel_world_size()
    if attn.heads % head_partition_degree:
        raise ValueError(
            f"Wan attention heads ({attn.heads}) must be divisible by the head partition "
            f"degree ({head_partition_degree}); self-attention uses TP times Ulysses, "
            "text cross-attention uses TP only."
        )
    if attn.fused_projections:
        raise ValueError("Wan attention TP must be applied before projection fusion")

    for linear in (attn.to_q, attn.to_k, attn.to_v):
        _shard_linear(linear, dim=0)
    attn.norm_q = xFuserWanTPRMSNorm(attn.norm_q)
    attn.norm_k = xFuserWanTPRMSNorm(attn.norm_k)
    attn.to_out[0] = xFuserWanTPRowLinear(attn.to_out[0])

    attn.heads //= group.world_size
    attn.inner_dim = attn.to_q.out_features
    attn.kv_inner_dim = attn.to_k.out_features
    if not attn.processor.is_cross_attention:
        # Only self-attention uses USP/head balancing. Slicing the global
        # identity would retain offsets on nonzero TP ranks and index past
        # the local heads. Cross-attention has no head_perm buffer.
        attn.head_perm = torch.arange(attn.heads, dtype=torch.long, device=attn.head_perm.device)
