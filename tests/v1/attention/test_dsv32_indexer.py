# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DeepseekV32IndexerMetadataBuilder — MTP3 native-vs-flatten
gating. Covers the DeepGEMM nv_dev next_n=4 feature:
https://github.com/vllm-project/vllm/issues/35878
"""

from unittest import mock

import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.config import SpeculativeConfig
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadataBuilder,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec


def _make_builder(num_speculative_tokens: int) -> DeepseekV32IndexerMetadataBuilder:
    # Use a small open model to avoid gated-repo auth during CI/local runs.
    vllm_config = create_vllm_config(model_name="Qwen/Qwen3-0.6B", block_size=64)
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=num_speculative_tokens,
    )
    kv_cache_spec = FullAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=None,
    )
    DeepseekV32IndexerBackend.get_builder_cls()  # smoke-check
    return DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=torch.device("cuda", 0),
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
def test_mtp3_batch_expansion_is_default(monkeypatch: pytest.MonkeyPatch):
    """Without the opt-in env var, num_speculative_tokens=3 must batch-expand."""
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM_MTP3", raising=False)
    builder = _make_builder(num_speculative_tokens=3)

    assert builder.use_native_mtp3 is False
    assert builder.natively_supported_next_n == [1, 2]
    assert builder.use_flattening is True
    assert builder.scheduler_metadata_buffer_mtp3 is None


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
def test_mtp3_native_requires_sm100(monkeypatch: pytest.MonkeyPatch):
    """Env var alone is not enough — kernel only works on Blackwell."""
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM_MTP3", "1")
    with mock.patch.object(
        current_platform.__class__,
        "is_device_capability_family",
        return_value=False,
    ):
        builder = _make_builder(num_speculative_tokens=3)

    assert builder.use_native_mtp3 is False
    assert builder.use_flattening is True
    assert builder.scheduler_metadata_buffer_mtp3 is None


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
def test_mtp3_native_when_enabled_on_sm100(monkeypatch: pytest.MonkeyPatch):
    """With env var + Blackwell, MTP=3 uses the native next_n=4 path and
    allocates a half-SM scheduler metadata buffer."""
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM_MTP3", "1")
    with mock.patch.object(
        current_platform.__class__,
        "is_device_capability_family",
        return_value=True,
    ):
        builder = _make_builder(num_speculative_tokens=3)

    assert builder.use_native_mtp3 is True
    assert builder.natively_supported_next_n == [1, 2, 4]
    assert builder.use_flattening is False
    assert builder.scheduler_metadata_buffer_mtp3 is not None
    assert builder.scheduler_metadata_buffer_mtp3.shape == (
        builder.num_sms // 2 + 1,
        2,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
def test_mtp3_env_var_ignored_for_other_next_n(monkeypatch: pytest.MonkeyPatch):
    """The env var only flips next_n=4; MTP=1 (next_n=2) stays on the
    pre-existing native path without the MTP3 scheduler buffer."""
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM_MTP3", "1")
    with mock.patch.object(
        current_platform.__class__,
        "is_device_capability_family",
        return_value=True,
    ):
        builder = _make_builder(num_speculative_tokens=1)

    assert builder.use_native_mtp3 is False
    assert builder.natively_supported_next_n == [1, 2]
    assert builder.use_flattening is False  # next_n=2 is natively supported
    assert builder.scheduler_metadata_buffer_mtp3 is None
