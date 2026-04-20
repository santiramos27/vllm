# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark: native next_n=4 vs 4x batch-expanded next_n=1 for
DeepGEMM's fp8_paged_mqa_logits.

This isolates the sparse-MLA indexer kernel from the rest of the MTP=3
serving path so we can see whether the nv_dev next_n=4 kernel is faster
than the historical batch-expansion fallback, and at what shapes.

Run:
    VLLM_USE_DEEP_GEMM_MTP3=1 .venv/bin/python \\
        benchmarks/kernels/bench_deepgemm_mqa_logits_mtp3.py

Requires DeepGEMM's `nv_dev` branch (the kernel that supports next_n=4)
installed in the active venv and an SM100+ GPU.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import torch

from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_paged_mqa_logits,
    get_num_sms,
    get_paged_mqa_logits_metadata,
)
from vllm.utils.import_utils import has_deep_gemm

# DeepSeek V3.2 sparse-attention indexer dimensions (see HF config):
#   index_n_heads = 64, index_head_dim = 128
# The paged KV cache block is fixed to 64 tokens for the indexer path.
HEADS = 64
HEAD_DIM = 128
BLOCK_SIZE = 64


def _pack_kv_cache_to_fp8(kv: torch.Tensor) -> torch.Tensor:
    """Mirror vLLM's indexer KV layout: [num_blocks, block_size, 1, D+4 bytes].

    The indexer kernel expects FP8 keys packed with a float32 per-token scale
    appended to each cache row. We don't care about values here — only that the
    byte layout matches what DeepGEMM reads.
    """
    num_blocks, block_size, num_heads, head_dim = kv.shape
    assert num_heads == 1
    x_amax = kv.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0  # fp8_e4m3 max
    x_scaled = (kv * (1.0 / sf)).to(torch.float8_e4m3fn)
    out = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=kv.device,
        dtype=torch.uint8,
    )
    out[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(dtype=torch.uint8)
    out[:, block_size * head_dim :] = sf.view(num_blocks, block_size).view(
        dtype=torch.uint8
    )
    return out.view(num_blocks, block_size, num_heads, head_dim + 4)


@dataclass
class KernelInputs:
    """Everything fp8_paged_mqa_logits needs, pre-built on GPU."""

    q_fp8: torch.Tensor  # [B_eff, next_n, H, D]
    kv_fp8: torch.Tensor  # [num_blocks, block_size, 1, D+4]
    weights: torch.Tensor  # [B_eff * next_n, H]
    context_lens: torch.Tensor  # [B_eff]
    block_tables: torch.Tensor  # [B_eff, max_blocks]
    schedule_metadata: torch.Tensor  # [num_sched_slots, 2]
    max_model_len: int


def _build_inputs(
    batch_size: int, context_len: int, seed: int = 0
) -> tuple[KernelInputs, KernelInputs]:
    """Build matched inputs for native (next_n=4) and expanded (4B, next_n=1).

    Key invariant: both paths must see the same *queries* attending to the
    same KV positions. Their batch dimension differs in how they present
    that work to the kernel.

    Native layout (what the nv_dev kernel wants):
      q:        [B, 4, H, D]      — query group of 4 per request
      ctx_lens: [B]                — kernel infers per-slot offsets from next_n
      bt:       [B, max_blocks]    — one row per request

    Expanded layout (what the historical fallback feeds the next_n=1 kernel):
      q:        [4B, 1, H, D]              — 4B independent single-token "requests"
      ctx_lens: [4B] = [L-3, L-2, L-1, L,  — each of the 4 query tokens within
                       L-3, L-2, L-1, L,     a request gets its own effective
                       ...]                  context length
      bt:       [4B, max_blocks]           — each original row replicated 4x
    """
    torch.manual_seed(seed)
    random.seed(seed)

    # Enough blocks for each request to hold `context_len` tokens, with a
    # pool large enough to pick distinct block indices per request.
    blocks_per_req = (context_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = max(blocks_per_req * batch_size * 2, 1024)

    # Single shared KV cache — the *data* doesn't matter for timing, only
    # the memory footprint and access pattern.
    kv_bf16 = torch.randn(
        (num_blocks, BLOCK_SIZE, 1, HEAD_DIM), device="cuda", dtype=torch.bfloat16
    )
    kv_fp8 = _pack_kv_cache_to_fp8(kv_bf16)

    # Native queries: [B, 4, H, D]. We cast to fp8_e4m3 to match what the
    # indexer passes in production.
    q_native = torch.randn(
        (batch_size, 4, HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    weights_native = torch.randn(
        (batch_size * 4, HEADS), device="cuda", dtype=torch.float32
    )
    context_lens_native = torch.full(
        (batch_size,), context_len, device="cuda", dtype=torch.int32
    )
    # Build the block table: each request gets a disjoint slice of
    # `blocks_per_req` block ids. Random order so reads scatter across the
    # cache rather than hitting contiguous memory.
    pool = list(range(num_blocks))
    random.shuffle(pool)
    block_tables_native = torch.tensor(
        [
            pool[i * blocks_per_req : (i + 1) * blocks_per_req]
            for i in range(batch_size)
        ],
        device="cuda",
        dtype=torch.int32,
    )

    num_sms = get_num_sms()
    # Native path: the next_n=4 kernel uses 2-way CTA multicast, so the
    # scheduler metadata is sized for half the SM slots. This matches
    # what the vLLM indexer does on the native MTP3 branch.
    sched_native = get_paged_mqa_logits_metadata(
        context_lens_native, BLOCK_SIZE, num_sms // 2
    )

    native = KernelInputs(
        q_fp8=q_native,
        kv_fp8=kv_fp8,
        weights=weights_native,
        context_lens=context_lens_native,
        block_tables=block_tables_native,
        schedule_metadata=sched_native,
        max_model_len=context_len,
    )

    # Expanded path: unroll each request into 4 pseudo-requests of 1 query.
    # `q_native` already has the right underlying data; a .reshape is all we
    # need, but we clone so the two benchmarks don't alias the same storage.
    q_expanded = q_native.reshape(batch_size * 4, 1, HEADS, HEAD_DIM).clone()
    weights_expanded = weights_native.clone()  # already [B*4, H]

    # Per-slot context lengths: for request i attending greedily to the last
    # token, the j-th query (j=0..3) sees L - (4-1) + j = L-3+j tokens.
    # Concretely, context for slot (i, j) is L_i - 3 + j.
    offsets = torch.arange(-3, 1, device="cuda", dtype=torch.int32)
    context_lens_expanded = (
        context_lens_native.unsqueeze(1) + offsets.unsqueeze(0)
    ).reshape(-1)
    # Block table: repeat each request's row 4 times (one per expanded slot).
    block_tables_expanded = torch.repeat_interleave(
        block_tables_native, repeats=4, dim=0
    )
    sched_expanded = get_paged_mqa_logits_metadata(
        context_lens_expanded, BLOCK_SIZE, num_sms
    )

    expanded = KernelInputs(
        q_fp8=q_expanded,
        kv_fp8=kv_fp8,
        weights=weights_expanded,
        context_lens=context_lens_expanded,
        block_tables=block_tables_expanded,
        schedule_metadata=sched_expanded,
        max_model_len=context_len,
    )

    return native, expanded


def _run(inputs: KernelInputs) -> torch.Tensor:
    """Single invocation of fp8_paged_mqa_logits with `inputs`. Returns the
    logits tensor so the caller can hold a reference (preventing dead-code
    elimination by the scheduler) and — if desired — compare outputs.
    """
    return fp8_paged_mqa_logits(
        inputs.q_fp8,
        inputs.kv_fp8,
        inputs.weights,
        inputs.context_lens,
        inputs.block_tables,
        inputs.schedule_metadata,
        inputs.max_model_len,
        clean_logits=False,
    )


def _time_us(inputs: KernelInputs, warmup: int, iters: int) -> float:
    """Time `iters` back-to-back kernel calls using CUDA events, after
    `warmup` untimed warmup calls. Returns the *per-call* median in
    microseconds.

    We take the median (not the mean) to suppress outliers from the first
    captured call, JIT recompiles inside DeepGEMM, and any host-side hiccup.
    """
    for _ in range(warmup):
        _run(inputs)
    torch.accelerator.synchronize()

    start_events = [torch.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events[i].record()
        _run(inputs)
        end_events[i].record()
    torch.accelerator.synchronize()

    times_us = sorted(
        start_events[i].elapsed_time(end_events[i]) * 1_000.0 for i in range(iters)
    )
    # median
    return times_us[iters // 2]


def _validate_equivalence(batch_size: int = 4, context_len: int = 2048) -> None:
    """Sanity check: both kernels should produce the same logits (where both
    define a value). If they diverge, the microbenchmark numbers below
    aren't comparing the same work.

    fp8_paged_mqa_logits returns shape [B_eff * next_n, max_model_len]. The
    native and expanded outputs have the same first dim but are laid out the
    same way (row i,j = attention output for query j of request i), so we can
    compare directly once we mask out positions beyond each query's context.
    """
    native, expanded = _build_inputs(batch_size, context_len, seed=42)
    logits_native = _run(native)
    logits_expanded = _run(expanded)
    assert logits_native.shape == logits_expanded.shape, (
        logits_native.shape,
        logits_expanded.shape,
    )

    # Positions that each query is allowed to attend to: j < ctx[i,j] where
    # ctx[i,j] = context_len - 3 + j for j in 0..3.
    positions = (
        torch.arange(context_len, device="cuda").unsqueeze(0).expand(batch_size * 4, -1)
    )
    mask = positions < expanded.context_lens.unsqueeze(1)
    a = logits_native.masked_fill(~mask, 0.0).double()
    b = logits_expanded.masked_fill(~mask, 0.0).double()
    # Cosine-similarity-style metric matching tests/kernels/.../calc_diff
    denom = (a * a + b * b).sum()
    diff = 1.0 - (2.0 * (a * b).sum() / denom).item()
    print(f"[validate] batch={batch_size} ctx={context_len}  cos_diff={diff:.3e}")
    if diff > 1e-3:
        raise RuntimeError(
            f"Native and expanded outputs disagree (diff={diff}). "
            "Refusing to report perf numbers that compare different work."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64, 128],
        help="Number of MTP-3 requests. Native kernel sees B, expanded kernel sees 4B.",
    )
    parser.add_argument(
        "--context-lens",
        type=int,
        nargs="+",
        default=[2048, 8192, 32768, 131072],
        help="Per-request KV length. Kernel work scales roughly linearly.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip the equivalence check (saves ~a second of startup).",
    )
    args = parser.parse_args()

    if not current_platform.is_cuda():
        raise SystemExit("CUDA required")
    if not has_deep_gemm():
        raise SystemExit("DeepGEMM not available — install the nv_dev wheel")
    if not current_platform.is_device_capability_family(100):
        print(
            "WARNING: this kernel path is only validated on SM100+ (Blackwell). "
            "Proceeding anyway."
        )

    # Equivalence check first. If this fails the timing numbers are
    # meaningless because we'd be comparing different work.
    if not args.skip_validate:
        _validate_equivalence()
        print()

    header = (
        f"{'B':>4}  {'ctx':>6}  {'native_us':>10}  {'expand_us':>10}  {'speedup':>8}"
    )
    print(header)
    print("-" * len(header))
    for bs in args.batch_sizes:
        for ctx in args.context_lens:
            try:
                native, expanded = _build_inputs(bs, ctx)
            except torch.cuda.OutOfMemoryError:
                print(f"{bs:>4}  {ctx:>6}   OOM on setup")
                torch.accelerator.empty_cache()  # noqa
                continue
            try:
                t_native = _time_us(native, args.warmup, args.iters)
                t_expand = _time_us(expanded, args.warmup, args.iters)
            except torch.accelerator.OutOfMemoryError:
                print(f"{bs:>4}  {ctx:>6}   OOM during timing")
                del native, expanded
                torch.accelerator.empty_cache()
                continue
            speedup = t_expand / t_native
            print(
                f"{bs:>4}  {ctx:>6}  {t_native:>10.1f}  {t_expand:>10.1f}  "
                f"{speedup:>8.2f}x"
            )
            del native, expanded  # noqa
            torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
