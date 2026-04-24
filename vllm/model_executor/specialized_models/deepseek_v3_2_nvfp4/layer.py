# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MLA attention and decoder layer for DeepSeek V3.2 on SM100 (Blackwell).

MLAAttention:
  KV cache update -> W_UK_T absorption -> sparse attn kernel -> W_UV up-proj
  MLAAttention kept only as a registration stub for KV cache / backend.

DecoderLayer:
  Single decoder layer: norm -> attn -> norm -> MoE/MLP.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV32IndexerCache,
    yarn_get_mscale,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_moe_finalize_allreduce
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

from .kernels import fused_norm_rope, fused_q
from .sparse_indexer import sparse_attn_indexer

logger = init_logger(__name__)


def dsa(
    positions: torch.Tensor,
    q_c: torch.Tensor,
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    index_k: torch.Tensor,
    index_weights: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer = get_forward_context().no_compile_layers[layer_name]
    attn = layer.attn
    mla = attn.mla_attn

    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        output.zero_()
        return output

    mla_attn_metadata = attn_metadata.get(mla.layer_name)
    if mla_attn_metadata is None:
        output.zero_()
        return output

    num_actual_toks = mla_attn_metadata.num_actual_tokens  # type: ignore[attr-defined]
    if num_actual_toks == 0:
        output.zero_()
        return output

    # Step 2. fused norm + rope + cache writes
    slot_mapping = None
    indexer_k_cache = None
    mla_kv_cache = None
    mla_k_scale = None
    idx_meta = attn_metadata.get(attn.indexer_k_cache.prefix)
    if idx_meta is not None:
        slot_mapping = idx_meta.slot_mapping  # type: ignore[attr-defined]
        indexer_k_cache = attn.indexer_k_cache.kv_cache
        mla_kv_cache = attn.mla_attn.kv_cache
        mla_k_scale = attn.mla_attn._k_scale

    q_c = fused_norm_rope(
        positions,
        q_c,
        attn.q_a_layernorm_weight,
        layer.rms_norm_eps,
        kv_c,
        attn.kv_a_layernorm_weight,
        attn.rms_norm_eps,
        k_pe,
        attn.rotary_emb.cos_sin_cache,
        index_k,
        attn.indexer_k_norm.weight,
        attn.indexer_k_norm.bias,
        attn.rms_norm_eps,
        attn.indexer_rope_emb.cos_sin_cache,
        attn.topk_indices_buffer,
        slot_mapping=slot_mapping,
        indexer_k_cache=indexer_k_cache,
        mla_kv_cache=mla_kv_cache,
        mla_kv_cache_dtype=attn.mla_attn.kv_cache_dtype,
        mla_k_scale=mla_k_scale,
    )

    # Step 3. q_c -> index_q, q
    step3_out = torch.mm(q_c, layer._fused_step3_q_w.T)
    index_q, q = step3_out.split(layer._q_split_sizes, dim=-1)
    index_q = index_q.view(-1, attn.index_n_heads, attn.index_head_dim)
    q = q.view(-1, attn.num_local_heads, attn.qk_head_dim)

    # Step 4. Q RoPE + W_UK_T absorption + FP8 packing
    q_nope, q_pe = q.split(
        [mla.qk_nope_head_dim, mla.qk_rope_head_dim],
        dim=-1,
    )
    q_nope = q_nope.transpose(0, 1)
    ql_nope = torch.bmm(q_nope, mla.W_UK_T)
    ql_nope = ql_nope.transpose(0, 1)

    index_q_fp8, index_weights, mqa_q = fused_q(
        positions,
        q_pe,
        attn.rotary_emb.cos_sin_cache,
        index_q,
        attn.indexer_rope_emb.cos_sin_cache,
        ql_nope,
        mla._q_scale,
        index_weights,
        attn.indexer_softmax_scale,
        attn.index_n_heads**-0.5,
    )

    # Steps 5-6. Sparse indexer + MLA sparse decode attention
    sparse_attn_indexer(
        attn.indexer_k_cache.prefix,
        attn.indexer_k_cache.kv_cache,
        index_q_fp8,
        index_weights,
        attn.topk_tokens,
        attn.index_head_dim,
        layer.max_model_len,
        layer.indexer_workspace_size,
        attn.topk_indices_buffer,
    )

    mqa_q = mqa_q[:num_actual_toks]
    kv_cache = mla.kv_cache
    if mla.kv_cache_dtype.startswith("fp8") and mla.kv_cache_dtype != "fp8_ds_mla":
        kv_cache = kv_cache.view(torch.float8_e4m3fn)
    attn_out, _ = mla.impl.forward_mqa(mqa_q, kv_cache, mla_attn_metadata, mla)
    x = attn_out.view(-1, mla.num_heads, mla.kv_lora_rank).transpose(0, 1)

    out = output[:num_actual_toks].view(-1, mla.num_heads, mla.v_head_dim)
    out = out.transpose(0, 1)
    torch.bmm(x, mla.W_UV, out=out)
    return output


def dsa_fake(
    positions: torch.Tensor,
    q_c: torch.Tensor,
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    index_k: torch.Tensor,
    index_weights: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    del positions, q_c, kv_c, k_pe, index_k, index_weights, layer_name
    return output


direct_register_custom_op(
    op_name="monolithic_attn",
    op_func=dsa,
    fake_impl=dsa_fake,
    mutates_args=["output"],
    dispatch_key=current_platform.dispatch_key,
)


def _get_moe_finalize_allreduce_workspace(
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
):
    from vllm.distributed import get_tp_group
    from vllm.distributed.device_communicators.flashinfer_all_reduce import (
        get_fi_ar_quant_workspace,
    )
    from vllm.distributed.parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    return get_fi_ar_quant_workspace(
        world_size=get_tensor_model_parallel_world_size(),
        rank=get_tensor_model_parallel_rank(),
        max_token_num=max_token_num,
        hidden_dim=hidden_dim,
        dtype=dtype,
        group=get_tp_group().device_group,
    )


def deepseek_moe_finalize_allreduce_rmsnorm(
    allreduce_in: torch.Tensor,
    residual_in: torch.Tensor,
    rms_gamma: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_scale_factor: torch.Tensor,
    norm_out: torch.Tensor,
    residual_out: torch.Tensor,
    shared_expert_output: torch.Tensor | None,
    rms_eps: float,
    max_token_num: int,
    launch_with_pdl: bool,
) -> torch.Tensor:
    import flashinfer.comm as flashinfer_comm

    workspace = _get_moe_finalize_allreduce_workspace(
        max_token_num=max_token_num,
        hidden_dim=residual_in.shape[-1],
        dtype=allreduce_in.dtype,
    )
    if workspace is None:
        raise RuntimeError(
            "FlashInfer TRTLLM allreduce workspace is unavailable for "
            "DeepSeek V3.2 MoE finalize fusion."
        )

    flashinfer_comm.allreduce_fusion(
        input=allreduce_in,
        workspace=workspace,
        pattern=flashinfer_comm.AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm,
        launch_with_pdl=launch_with_pdl,
        residual_in=residual_in,
        residual_out=residual_out,
        norm_out=norm_out,
        rms_gamma=rms_gamma,
        rms_eps=rms_eps,
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        expert_scale_factor=expert_scale_factor,
        shared_expert_output=shared_expert_output,
    )
    return norm_out


def deepseek_moe_finalize_allreduce_rmsnorm_fake(
    allreduce_in: torch.Tensor,
    residual_in: torch.Tensor,
    rms_gamma: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_scale_factor: torch.Tensor,
    norm_out: torch.Tensor,
    residual_out: torch.Tensor,
    shared_expert_output: torch.Tensor | None,
    rms_eps: float,
    max_token_num: int,
    launch_with_pdl: bool,
) -> torch.Tensor:
    del (
        allreduce_in,
        residual_in,
        rms_gamma,
        expanded_idx_to_permuted_idx,
        expert_scale_factor,
        residual_out,
        shared_expert_output,
        rms_eps,
        max_token_num,
        launch_with_pdl,
    )
    return norm_out


direct_register_custom_op(
    op_name="deepseek_moe_finalize_allreduce_rmsnorm",
    op_func=deepseek_moe_finalize_allreduce_rmsnorm,
    fake_impl=deepseek_moe_finalize_allreduce_rmsnorm_fake,
    mutates_args=["norm_out", "residual_out"],
    dispatch_key=current_platform.dispatch_key,
)


class DeepseekV32DecoderLayer(nn.Module):
    """
    Single decoder layer: norm -> attn -> norm -> MoE/MLP.
    Norms are raw weight + direct kernel call.
    Gate inlined as raw weight, experts kept as FusedMoE for quantization.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        layer_idx: int,
        topk_indices_buffer: torch.Tensor,
        prefix: str = "",
    ) -> None:
        super().__init__()
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        self.layer_name = prefix
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.tp_size = get_tensor_model_parallel_world_size()

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        self.indexer_workspace_size = get_max_prefill_buffer_size(vllm_config)
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )

        # Use the regular vLLM RMSNorm modules so the compiler sees the
        # canonical residual-add + RMSNorm pattern.
        dtype = torch.get_default_dtype()
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
        )

        # Fused QKV A-projection lives inside self_attn namespace
        # for weight loading compatibility with original checkpoint paths
        from vllm.model_executor.models.deepseek_v2 import (
            DeepSeekV2FusedQkvAProjLinear,
        )

        self.self_attn = nn.Module()
        self.self_attn.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
            config.hidden_size,
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn.fused_qkv_a_proj",
        )

        # MLA Attention
        self.attn = DeepseekV32MLAAttention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            cache_config=cache_config,
            quant_config=quant_config,
            topk_indices_buffer=topk_indices_buffer,
            prefix=f"{prefix}.self_attn",
        )

        # MoE or Dense MLP
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        self.is_moe = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        from vllm.model_executor.models.deepseek_v2 import (
            DeepseekV2MLP,
            DeepseekV2MoE,
        )

        if self.is_moe:
            self.mlp = DeepseekV2MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual, _ = self.forward_maybe_fused_moe_norm(
            positions,
            hidden_states,
            residual,
            input_layernorm_applied=False,
            next_layernorm=None,
        )
        return hidden_states, residual

    def forward_maybe_fused_moe_norm(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        input_layernorm_applied: bool,
        next_layernorm: RMSNorm | None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Forward layer, optionally fusing MoE finalize into next RMSNorm."""
        if residual is None:
            assert not input_layernorm_applied
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        elif not input_layernorm_applied:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Step 1. hidden_states -> q_c, kv_c, k_pe
        #                       -> index_k, index_weights
        out = self.self_attn.fused_qkv_a_proj(hidden_states)
        if isinstance(out, tuple):
            out = out[0]

        q_c, kv_c, k_pe = out.split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        index_k, index_weights = torch.mm(
            hidden_states, self._fused_indexer_weights.T
        ).split(
            self._indexer_weights_split_sizes,
            dim=-1,
        )

        # Steps 2-6. Combined: fused norm/rope + Q projections + sparse MLA.
        mla = self.attn.mla_attn
        output_shape = (hidden_states.shape[0], mla.num_heads * mla.v_head_dim)
        output_dtype = mla.W_UV.dtype
        attn_out = torch.empty(
            output_shape,
            dtype=output_dtype,
            device=hidden_states.device,
        )
        attn_out = torch.ops.vllm.monolithic_attn(
            positions,
            q_c,
            kv_c,
            k_pe,
            index_k,
            index_weights,
            attn_out,
            self.layer_name,
        )

        hidden_states, _ = self.attn.o_proj(attn_out)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        if self.is_moe and next_layernorm is not None:
            fused = self._try_fused_moe_finalize_allreduce_rmsnorm(
                hidden_states,
                residual,
                next_layernorm,
            )
            if fused is not None:
                hidden_states, residual = fused
                return hidden_states, residual, True

        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual, False

    def _can_use_fused_moe_finalize_allreduce(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        next_layernorm: RMSNorm,
    ) -> bool:
        if self.tp_size <= 1:
            return False
        if not current_platform.is_cuda():
            return False
        if not has_flashinfer_moe_finalize_allreduce():
            return False
        if hidden_states.dtype != torch.bfloat16:
            return False
        if hidden_states.shape[0] > self.max_num_batched_tokens:
            return False
        if not hidden_states.is_contiguous() or not residual.is_contiguous():
            return False
        if hidden_states.shape != residual.shape:
            return False
        if hidden_states.shape[-1] != self.hidden_size:
            return False
        if not next_layernorm.has_weight:
            return False
        if next_layernorm.variance_size_override is not None:
            return False

        mlp = self.mlp
        if getattr(mlp, "is_sequence_parallel", False):
            return False
        if getattr(mlp, "ep_size", 1) != 1:
            return False
        if getattr(mlp, "is_rocm_aiter_moe_enabled", False):
            return False

        experts = mlp.experts
        moe_parallel_config = experts.moe_parallel_config
        if moe_parallel_config.dp_size != 1 or moe_parallel_config.pcp_size != 1:
            return False
        if getattr(experts, "_routed_input_transform", None) is not None:
            return False

        quant_method = experts.quant_method
        moe_kernel = getattr(quant_method, "moe_kernel", None)
        if moe_kernel is None or not getattr(moe_kernel, "is_monolithic", False):
            return False
        if not hasattr(moe_kernel, "apply_monolithic_without_finalize"):
            return False

        try:
            workspace = _get_moe_finalize_allreduce_workspace(
                max_token_num=self.max_num_batched_tokens,
                hidden_dim=self.hidden_size,
                dtype=hidden_states.dtype,
            )
            if workspace is None:
                return False
            return workspace.is_buffer_size_sufficient(
                self.tp_size,
                hidden_states.shape[0],
                self.hidden_size,
                hidden_states.dtype,
            )
        except Exception as e:
            logger.warning_once(
                "Disabling DeepSeek V3.2 fused MoE finalize allreduce RMSNorm: %s",
                e,
            )
            return False

    def _try_fused_moe_finalize_allreduce_rmsnorm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        next_layernorm: RMSNorm,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self._can_use_fused_moe_finalize_allreduce(
            hidden_states,
            residual,
            next_layernorm,
        ):
            return None

        mlp = self.mlp
        experts = mlp.experts
        experts.ensure_moe_quant_config_init()
        router_logits, _ = mlp.gate(hidden_states)

        shared_output = (
            mlp.shared_experts(hidden_states)
            if mlp.shared_experts is not None
            else None
        )

        moe_kernel = experts.quant_method.moe_kernel
        assert moe_kernel is not None
        allreduce_in, expert_scale_factor, expanded_idx_to_permuted_idx = (
            moe_kernel.apply_monolithic_without_finalize(
                hidden_states,
                experts.w13_weight,
                experts.w2_weight,
                router_logits,
                activation=experts.activation,
                global_num_experts=experts.global_num_experts,
                expert_map=experts.expert_map,
                apply_router_weight_on_input=experts.apply_router_weight_on_input,
                num_expert_group=experts.num_expert_group,
                topk_group=experts.topk_group,
                e_score_correction_bias=experts.e_score_correction_bias,
                routed_scaling_factor=experts.routed_scaling_factor,
            )
        )

        # Match DeepseekV2MoE.forward(): TRTLLM monolithic kernels use a
        # routed_scaling_factor of 1.0, so vLLM applies the model scaling after
        # expert reduction. Scaling the router weights is equivalent before
        # FlashInfer's deferred finalize reduction.
        if hidden_states.dtype != torch.float16:
            expert_scale_factor = expert_scale_factor * mlp.routed_scaling_factor
        elif shared_output is not None:
            shared_output = shared_output * (1.0 / mlp.routed_scaling_factor)

        norm_out = torch.empty_like(residual)
        residual_out = torch.empty_like(residual)
        norm_out = torch.ops.vllm.deepseek_moe_finalize_allreduce_rmsnorm(
            allreduce_in,
            residual,
            next_layernorm.weight.data,
            expanded_idx_to_permuted_idx,
            expert_scale_factor,
            norm_out,
            residual_out,
            shared_output,
            next_layernorm.variance_epsilon,
            self.max_num_batched_tokens,
            True,
        )
        return norm_out, residual_out

    def fuse_indexer_weights(self) -> None:
        """Fuse Step 1 and Step 3 BF16 linears used by the inlined path.

        Call after model weights are loaded.
        """
        attn = self.attn
        wk = attn.indexer_wk.weight.data  # [128, 7168]
        wp = attn.indexer_weights_proj.weight.data  # [64, 7168]
        if wk.dtype != wp.dtype:
            raise ValueError(
                "Cannot fuse indexer weights: expected matching dtypes for "
                "indexer_wk and indexer_weights_proj."
            )
        self._fused_indexer_weights = nn.Parameter(
            torch.cat([wk, wp], dim=0),  # [192, 7168]
            requires_grad=False,
        )
        self._indexer_weights_split_sizes = [wk.shape[0], wp.shape[0]]

        wq_b = attn.indexer_wq_b.weight.data
        q_b = attn.q_b_proj.weight.data
        if wq_b.dtype != q_b.dtype:
            raise ValueError(
                "Cannot fuse Step 3 weights: expected matching dtypes for "
                "indexer_wq_b and q_b_proj."
            )
        self._fused_step3_q_w = nn.Parameter(
            torch.cat([wq_b, q_b], dim=0),
            requires_grad=False,
        )
        self._q_split_sizes = [wq_b.shape[0], q_b.shape[0]]


class DeepseekV32MLAAttention(nn.Module):
    """
    MLA attention for DeepSeek V3.2 targeting SM100.
    MLA forward fully inlined. MLAAttention kept only for KV cache
    registration and backend/impl initialization.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        max_position_embeddings: int,
        cache_config: CacheConfig,
        quant_config: QuantizationConfig | None,
        topk_indices_buffer: torch.Tensor,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.num_local_heads = num_heads // get_tensor_model_parallel_world_size()
        self.scaling = self.qk_head_dim**-0.5
        self.rms_norm_eps = config.rms_norm_eps

        # Q path
        self.q_a_layernorm_weight = nn.Parameter(
            torch.ones(q_lora_rank, dtype=torch.get_default_dtype())
        )
        self.q_b_proj = ColumnParallelLinear(
            q_lora_rank,
            num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )

        # KV path
        self.kv_a_layernorm_weight = nn.Parameter(
            torch.ones(kv_lora_rank, dtype=torch.get_default_dtype())
        )
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )

        # Output projection (TP sync point)
        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # RoPE
        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )
        if config.rope_parameters["rope_type"] == "deepseek_yarn":
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        # V3.2 Sparse Indexer (inlined)
        self.indexer_rope_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=not getattr(config, "indexer_rope_interleave", False),
        )
        self.topk_tokens = config.index_topk
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.indexer_softmax_scale = config.index_head_dim**-0.5
        self.indexer_quant_block_size = 128
        self.topk_indices_buffer = topk_indices_buffer

        self.indexer_wq_b = ReplicatedLinear(
            q_lora_rank,
            config.index_head_dim * config.index_n_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer.wq_b",
        )
        self.indexer_wk = ReplicatedLinear(
            hidden_size,
            config.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer.wk",
        )
        self.indexer_k_norm = LayerNorm(config.index_head_dim, eps=1e-6)
        self.indexer_weights_proj = ReplicatedLinear(
            hidden_size,
            config.index_n_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.indexer.weights_proj",
        )

        idx_dim = config.index_head_dim
        indexer_cache_head_dim = idx_dim + idx_dim // 128 * 4
        self.indexer_k_cache = DeepseekV32IndexerCache(
            head_dim=indexer_cache_head_dim,
            dtype=torch.uint8,
            prefix=f"{prefix}.indexer.k_cache",
            cache_config=cache_config,
        )
        self.indexer_op = SparseAttnIndexer(
            self.indexer_k_cache,
            self.indexer_quant_block_size,
            "ue8m0",
            self.topk_tokens,
            config.index_head_dim,
            vllm_config.model_config.max_model_len,
            get_max_prefill_buffer_size(vllm_config),
            self.topk_indices_buffer,
        )

        # MLAAttention stub: only for KV cache registration + backend init.
        # We never call its forward(); we inline everything below.
        class _IndexerProxy:
            def __init__(proxy_self):
                proxy_self.topk_indices_buffer = topk_indices_buffer
                proxy_self.indexer_op = self.indexer_op

        self._indexer_proxy = _IndexerProxy()
        self.mla_attn = MLAAttention(
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            kv_b_proj=self.kv_b_proj,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mla_attn",
            use_sparse=True,
            indexer=self._indexer_proxy,
        )


def remap_weight_name(name: str) -> str:
    """Remap checkpoint names that differ from the module layout."""
    replacements = [
        (
            "self_attn.q_a_layernorm.weight",
            "attn.q_a_layernorm_weight",
        ),
        (
            "self_attn.kv_a_layernorm.weight",
            "attn.kv_a_layernorm_weight",
        ),
        ("self_attn.q_b_proj", "attn.q_b_proj"),
        ("self_attn.kv_b_proj", "attn.kv_b_proj"),
        ("self_attn.o_proj", "attn.o_proj"),
        ("self_attn.indexer.", "attn.indexer_"),
    ]
    for old, new in replacements:
        if old in name:
            return name.replace(old, new)
    return name
