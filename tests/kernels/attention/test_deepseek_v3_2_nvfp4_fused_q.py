# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.specialized_models.deepseek_v3_2_nvfp4.kernels import (
    fused_q,
)
from vllm.platforms import current_platform

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="DeepSeek v3.2 specialized kernels require compute capability 10+.",
        allow_module_level=True,
    )


def _make_cos_sin_cache(
    rotary_dim: int,
    max_position: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000
        ** (
            torch.arange(
                0,
                rotary_dim,
                2,
                device="cuda",
                dtype=torch.float32,
            )
            / rotary_dim
        )
    )
    positions = torch.arange(max_position, device="cuda", dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)


def _reference_index_q_path(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    half_rot_dim = index_q_cos_sin_cache.shape[-1] // 2
    cos_sin = index_q_cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos[:, None, :].to(torch.float32)
    sin = sin[:, None, :].to(torch.float32)

    x1 = index_q[..., :half_rot_dim].to(torch.float32)
    x2 = index_q[..., half_rot_dim : 2 * half_rot_dim].to(torch.float32)

    # Match the old kernel path: RoPE writes through the source dtype before
    # the full head is reloaded and quantized.
    r1 = (x1 * cos - x2 * sin).to(index_q.dtype).to(torch.float32)
    r2 = (x2 * cos + x1 * sin).to(index_q.dtype).to(torch.float32)
    q_nope = index_q[..., 2 * half_rot_dim :].to(torch.float32)

    full_index_q = torch.cat((r1, r2, q_nope), dim=-1)
    amax = full_index_q.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    index_q_scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))

    index_q_fp8 = (full_index_q / index_q_scale).to(torch.float8_e4m3fn)
    index_weights_out = index_weights.to(torch.float32) * index_q_scale.squeeze(-1)
    index_weights_out *= index_weights_softmax_scale
    index_weights_out *= index_weights_head_scale
    return index_q_fp8, index_weights_out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 17, 257])
@torch.inference_mode()
def test_fused_q_index_path_matches_reference(
    dtype: torch.dtype,
    num_tokens: int,
) -> None:
    torch.manual_seed(0)
    torch.set_default_device("cuda")

    max_position = 4096
    q_pe_dim = 64
    ql_nope_dim = 512
    index_head_dim = 128
    num_q_heads = 16
    num_index_q_heads = 64

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.int64)
    q_pe = torch.randn((num_tokens, num_q_heads, q_pe_dim), dtype=dtype)
    ql_nope = torch.randn((num_tokens, num_q_heads, ql_nope_dim), dtype=dtype)
    index_q = torch.randn(
        (num_tokens, num_index_q_heads, index_head_dim),
        dtype=dtype,
    )
    index_weights = torch.randn(
        (num_tokens, num_index_q_heads),
        dtype=torch.float32,
    )
    q_scale = torch.tensor([0.125], dtype=torch.float32)

    q_pe_cos_sin_cache = _make_cos_sin_cache(q_pe_dim, max_position, dtype)
    index_q_cos_sin_cache = _make_cos_sin_cache(q_pe_dim, max_position, dtype)

    index_q_ref = index_q.clone()
    index_weights_ref = index_weights.clone()

    index_q_fp8, index_weights_out, _ = fused_q(
        positions,
        q_pe,
        q_pe_cos_sin_cache,
        index_q,
        index_q_cos_sin_cache,
        ql_nope,
        q_scale,
        index_weights,
        index_head_dim**-0.5,
        num_index_q_heads**-0.5,
    )

    ref_index_q_fp8, ref_index_weights_out = _reference_index_q_path(
        positions,
        index_q_ref,
        index_q_cos_sin_cache,
        index_weights_ref,
        index_head_dim**-0.5,
        num_index_q_heads**-0.5,
    )

    torch.testing.assert_close(index_q, index_q_ref)
    assert torch.equal(
        index_q_fp8.view(torch.uint8),
        ref_index_q_fp8.view(torch.uint8),
    )
    torch.testing.assert_close(index_weights_out, ref_index_weights_out)
