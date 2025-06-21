import json
import os
import random
import torch
import torch.distributed as dist
from functools import partial

import deep_ep
from utils import init_dist, bench, bench_kineto, calc_diff, hash_tensor, per_token_cast_back


def test_main(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
              rank: int, num_ranks: int, group: dist.ProcessGroup, buffer: deep_ep.Buffer, seed: int = 0):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # NOTES: the integers greater than 256 exceeds the BF16 precision limit
    rank_offset = 128
    assert num_ranks - rank_offset < 257, 'Too many ranks (exceeding test precision limit)'

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * (rank - rank_offset)
    x[:, -128:] = torch.arange(num_tokens, device='cuda').to(torch.bfloat16).view(-1, 1)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda').abs()

    # Randomly mask some positions
    for i in range(10):
        topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1

    # noinspection PyShadowingNames
    def large_gemm_with_hook(hook):
        mat_0 = torch.randn((8192, 8192), dtype=torch.float)
        mat_1 = torch.randn((8192, 8192), dtype=torch.float)
        mat_0 @ mat_1
        hook()

    # noinspection PyShadowingNames
    def test_func(zero_copy: bool, return_recv_hook: bool):
        recv_x, recv_count, handle, event, hook = \
            buffer.low_latency_dispatch(x, topk_idx, num_tokens, num_experts,
                                        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                        use_fp8=True, async_finish=False, return_recv_hook=return_recv_hook)
        large_gemm_with_hook(hook) if return_recv_hook else None
        if zero_copy:
            buffer.get_next_low_latency_combine_buffer(handle)[:, :, :] = simulated_gemm_x
        combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x, topk_idx, topk_weights, handle,
                                                             zero_copy=zero_copy, return_recv_hook=return_recv_hook)
        large_gemm_with_hook(hook) if return_recv_hook else None

    # Calculate bandwidth
    num_fp8_bytes, num_bf16_bytes = (hidden + hidden / 128 * 4 + 16), hidden * 2
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += num_fp8_bytes * num_selections
        num_combine_comm_bytes += num_bf16_bytes * num_selections

    # Dispatch + combine testing
    avg_t, min_t, max_t = bench(partial(test_func, zero_copy=False, return_recv_hook=False))
    print(f'[rank {rank}] Dispatch + combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, '
          f'avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us', flush=True)

    output_data = {}

    # Separate profiling
    for return_recv_hook in (False, True):
        group.barrier()
        bench_output = bench_kineto(partial(test_func, zero_copy=True, return_recv_hook=return_recv_hook),
                                    kernel_names=('dispatch', 'combine'), barrier_comm_profiling=True,
                                    suppress_kineto_output=True, duplicate_name_period=2 if return_recv_hook else None)
        if not return_recv_hook:
            dispatch_t, combine_t = bench_output
            data = dict(
                dispatch_bandwidth=num_dispatch_comm_bytes / 1e9 / dispatch_t,
                combine_bandwidth=num_combine_comm_bytes / 1e9 / combine_t,
                dispatch_t_us=dispatch_t * 1e6,
                combine_t_us=combine_t * 1e6,
            )
            print(f'[rank {rank}] Dispatch bandwidth: {data["dispatch_bandwidth"] :.2f} GB/s, avg_t={data["dispatch_t_us"] :.2f} us | '
                  f'Combine bandwidth: {data["combine_bandwidth"] :.2f} GB/s, avg_t={data["combine_t_us"] :.2f} us', flush=True)
        else:
            dispatch_t, combine_t, detail_times = bench_output
            data = dict(
                dispatch_t_us=dispatch_t * 2 * 1e6,
                combine_t_us=combine_t * 2 * 1e6,
                dispatch_send_t_us=detail_times["dispatch"][0] * 1e6,
                dispatch_recv_t_us=detail_times["dispatch"][1] * 1e6,
                combine_send_t_us=detail_times["combine"][0] * 1e6,
                combine_recv_t_us=detail_times["combine"][1] * 1e6,
            )
            print(f'[rank {rank}] Dispatch send/recv time: {data["dispatch_t_us"] :.2f} = {data["dispatch_send_t_us"] :.2f} + {data["dispatch_recv_t_us"] :.2f} us | '
                  f'Combine send/recv time: {data["combine_t_us"] :.2f} = {data["combine_send_t_us"] :.2f} + {data["combine_recv_t_us"] :.2f} us', flush=True)

        output_data |= {("hook_" if return_recv_hook else "std_") + k: v for k, v in data.items()}

    print('MAIN_OUTPUT=' + json.dumps(dict(
        rank=rank,
        num_tokens=num_tokens,
        hidden=hidden,
        num_experts=num_experts,
        num_topk=num_topk,
        num_ranks=num_ranks,
        **output_data,
    )))

    return hash_value


# noinspection PyUnboundLocalVariable
def test_loop(local_rank: int, num_local_ranks: int):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    # num_tokens, hidden, num_topk, num_experts = 4096, 7168, 8, (256 // num_ranks) * num_ranks
    num_tokens = int(os.environ.get("DEEPEP_TEST_NUM_TOKENS", "4096"))
    hidden = int(os.environ.get("DEEPEP_TEST_HIDDEN", "7168"))
    num_topk = int(os.environ.get("DEEPEP_TEST_NUM_TOPK", "8"))
    num_experts = int(os.environ.get("DEEPEP_TEST_NUM_EXPERTS", str((256 // num_ranks) * num_ranks)))

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
    if local_rank == 0:
        print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)
    buffer = deep_ep.Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                            num_qps_per_rank=num_experts // num_ranks,
                            allow_mnnvl=bool(int(os.environ.get("DEEPEP_TEST_ALLOW_MNNVL", "0"))))
    test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer, seed=1)

    # Destroy the communication group
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    # TODO: you may modify NUMA binding for less CPU overhead
    num_processes = int(os.getenv("DEEPEP_TEST_NUM_PROCESSES", "8"))
    torch.multiprocessing.spawn(test_loop, args=(num_processes,), nprocs=num_processes)
