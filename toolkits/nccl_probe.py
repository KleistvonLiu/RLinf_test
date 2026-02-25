#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lightweight NCCL diagnostics for single-GPU and multi-GPU runs.

Usage examples:
    Single GPU:
        python toolkits/nccl_probe.py --iters 20 --numel 1048576

    Multi GPU (single node):
        torchrun --standalone --nproc_per_node=2 toolkits/nccl_probe.py \
            --iters 20 --numel 1048576
"""

from __future__ import annotations

import argparse
import os
import socket
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass
class DistContext:
    """Runtime distributed context."""

    rank: int
    world_size: int
    local_rank: int
    created_pg: bool


def _find_free_port() -> int:
    """Find an available localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _percentile(values: list[float], q: float) -> float:
    """Return q-quantile for a non-empty list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    w = pos - lo
    return sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="NCCL all-reduce diagnostic probe")
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="Distributed backend.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Measured all-reduce iterations.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=5,
        help="Warmup all-reduce iterations.",
    )
    parser.add_argument(
        "--numel",
        type=int,
        default=1_048_576,
        help="Number of float32 elements in the test tensor.",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=30,
        help="Timeout in seconds for process group operations.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index used when LOCAL_RANK is unavailable.",
    )
    parser.add_argument(
        "--barrier-per-iter",
        action="store_true",
        help="Run barrier() after each all-reduce.",
    )
    parser.add_argument(
        "--skip-value-check",
        action="store_true",
        help="Skip checking all-reduce result values.",
    )
    parser.add_argument(
        "--safe-nccl-env",
        action="store_true",
        default=False,
        help="Set conservative NCCL env defaults in current process.",
    )
    parser.add_argument(
        "--force-nccl-p2p-disable",
        action="store_true",
        default=False,
        help="Force NCCL_P2P_DISABLE=1 in current process.",
    )
    parser.add_argument(
        "--force-nccl-ib-disable",
        action="store_true",
        default=False,
        help="Force NCCL_IB_DISABLE=1 in current process.",
    )
    parser.add_argument(
        "--force-nccl-shm-disable",
        action="store_true",
        default=False,
        help="Force NCCL_SHM_DISABLE=1 in current process.",
    )
    parser.add_argument(
        "--cuda-only-smoke-test",
        action="store_true",
        default=False,
        help="Run a CUDA-only smoke test before distributed initialization.",
    )
    args = parser.parse_args()

    if args.iters <= 0:
        raise ValueError("--iters must be greater than 0")
    if args.warmup_iters < 0:
        raise ValueError("--warmup-iters must be >= 0")
    if args.numel <= 0:
        raise ValueError("--numel must be greater than 0")
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be greater than 0")
    return args


def init_dist(args: argparse.Namespace) -> DistContext:
    """Initialize process group from torchrun env or single-process fallback."""
    rank_env = os.environ.get("RANK")
    world_size_env = os.environ.get("WORLD_SIZE")
    local_rank_env = os.environ.get("LOCAL_RANK")

    created_pg = False
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(local_rank_env) if local_rank_env is not None else args.device
        return DistContext(rank, world_size, local_rank, created_pg=False)

    timeout = timedelta(seconds=args.timeout_s)
    if rank_env is not None and world_size_env is not None:
        rank = int(rank_env)
        world_size = int(world_size_env)
        local_rank = int(local_rank_env) if local_rank_env is not None else args.device
        dist.init_process_group(
            backend=args.backend,
            init_method="env://",
            timeout=timeout,
        )
        created_pg = True
        return DistContext(rank, world_size, local_rank, created_pg)

    # Single-process fallback for local NCCL sanity check.
    local_rank = args.device
    init_method = f"tcp://127.0.0.1:{_find_free_port()}"
    dist.init_process_group(
        backend=args.backend,
        init_method=init_method,
        rank=0,
        world_size=1,
        timeout=timeout,
    )
    created_pg = True
    return DistContext(rank=0, world_size=1, local_rank=local_rank, created_pg=created_pg)


def apply_nccl_env_overrides(args: argparse.Namespace) -> None:
    """Apply NCCL env overrides for this process."""
    if args.backend != "nccl":
        return

    if args.safe_nccl_env:
        os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
        os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
        os.environ.setdefault("TORCH_NCCL_AVOID_RECORD_STREAMS", "1")

    if args.force_nccl_p2p_disable:
        os.environ["NCCL_P2P_DISABLE"] = "1"
    if args.force_nccl_ib_disable:
        os.environ["NCCL_IB_DISABLE"] = "1"
    if args.force_nccl_shm_disable:
        os.environ["NCCL_SHM_DISABLE"] = "1"


def cuda_smoke_test(local_rank: int) -> None:
    """Run a lightweight CUDA-only compute/memory smoke test."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for smoke test.")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    x = torch.randn(4096, 4096, device=device, dtype=torch.float32)
    y = torch.randn(4096, 4096, device=device, dtype=torch.float32)
    z = x @ y
    _ = float(z.mean().item())
    torch.cuda.synchronize(device)


def _print_env(rank: int) -> None:
    """Print concise environment diagnostics."""
    info = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nccl_debug": os.environ.get("NCCL_DEBUG"),
        "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
        "nccl_ib_disable": os.environ.get("NCCL_IB_DISABLE"),
        "nccl_socket_ifname": os.environ.get("NCCL_SOCKET_IFNAME"),
    }
    try:
        info["nccl"] = str(torch.cuda.nccl.version())
    except Exception:
        info["nccl"] = "unavailable"
    print(f"[rank {rank}] env={info}", flush=True)


def run_probe(args: argparse.Namespace, ctx: DistContext) -> int:
    """Run all-reduce diagnostics and return process exit code."""
    if args.backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but backend=nccl was requested.")
        torch.cuda.set_device(ctx.local_rank)
        device = torch.device(f"cuda:{ctx.local_rank}")
    else:
        device = torch.device("cpu")

    _print_env(ctx.rank)
    print(
        f"[rank {ctx.rank}] start probe: backend={args.backend}, world_size={ctx.world_size}, "
        f"local_rank={ctx.local_rank}, numel={args.numel}, warmup={args.warmup_iters}, "
        f"iters={args.iters}, barrier_per_iter={args.barrier_per_iter}",
        flush=True,
    )

    tensor = torch.ones(args.numel, device=device, dtype=torch.float32)
    expected = float(ctx.world_size)

    # Warmup
    for _ in range(args.warmup_iters):
        tensor.fill_(1.0)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if args.barrier_per_iter:
            dist.barrier()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    latency_ms: list[float] = []
    max_abs_err = 0.0
    for i in range(args.iters):
        tensor.fill_(1.0)
        start_t = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if args.barrier_per_iter:
            dist.barrier()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        latency_ms.append(elapsed_ms)

        if not args.skip_value_check:
            cur_err = float((tensor - expected).abs().max().item())
            max_abs_err = max(max_abs_err, cur_err)
            if cur_err > 1e-5:
                raise RuntimeError(
                    f"[rank {ctx.rank}] value check failed at iter={i}, "
                    f"max_abs_err={cur_err}, expected={expected}"
                )

    summary = torch.tensor(
        [
            statistics.mean(latency_ms),
            _percentile(latency_ms, 0.50),
            _percentile(latency_ms, 0.95),
            max(latency_ms),
            max_abs_err,
        ],
        device=device,
        dtype=torch.float32,
    )
    gathered = [torch.zeros_like(summary) for _ in range(ctx.world_size)]
    dist.all_gather(gathered, summary)

    if ctx.rank == 0:
        print("[rank 0] probe summary (mean_ms, p50_ms, p95_ms, max_ms, max_abs_err):", flush=True)
        for r, stats in enumerate(gathered):
            m, p50, p95, mx, err = [float(x) for x in stats.tolist()]
            print(
                f"  rank={r}: mean={m:.3f} p50={p50:.3f} p95={p95:.3f} max={mx:.3f} err={err:.3e}",
                flush=True,
            )

    dist.barrier()
    print(f"[rank {ctx.rank}] NCCL probe PASSED", flush=True)
    return 0


def main() -> int:
    """Program entry point."""
    args = parse_args()
    apply_nccl_env_overrides(args)
    ctx = init_dist(args)
    try:
        if args.cuda_only_smoke_test and args.backend == "nccl":
            cuda_smoke_test(ctx.local_rank)
        return run_probe(args, ctx)
    except Exception as exc:
        print(f"[rank {ctx.rank}] NCCL probe FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if ctx.created_pg and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
