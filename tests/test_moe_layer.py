import json
import os
import random
import torch
import torch.distributed as dist
from functools import partial

import deep_ep
from utils import init_dist, bench, bench_kineto, calc_diff, hash_tensor, per_token_cast_back

# --------------------------------------------- main -----------------------------------------------------


def test_main(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
              rank: int, num_ranks: int, group: dist.ProcessGroup, buffer: deep_ep.Buffer, seed: int = 0):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0

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
    def large_gemm():
        mat_0 = torch.randn((8192, 8192), dtype=torch.float)
        mat_1 = torch.randn((8192, 8192), dtype=torch.float)
        mat_0 @ mat_1

    # noinspection PyShadowingNames
    def test_func():
        hidden_states_fp8, recv_count, handle, dispatch_event, dispatch_hook = \
            buffer.low_latency_dispatch(x, topk_idx, num_tokens, num_experts,
                                        use_fp8=True, async_finish=False, return_recv_hook=True)
        assert dispatch_event is None
        large_gemm()
        dispatch_hook()

        combined_x, combine_event, combine_hook = buffer.low_latency_combine(
            simulated_gemm_x, topk_idx, topk_weights, handle,
            return_recv_hook=True,
            async_finish=True, # NOTE
        )
        combine_event.current_stream_wait()
        large_gemm()
        combine_hook()

    bench(test_func)


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

# --------------------------------------------- layer -----------------------------------------------------

def forward_layer():
    hidden_states_fp8, hidden_states_scale = hidden_states_fp8
    assert self.quant_method is not None
    assert self.activation == "silu"
    if num_recv_tokens_per_expert is None:
        return hidden_states_fp8.bfloat16()
    all_tokens = sum(num_recv_tokens_per_expert)
    if all_tokens <= 0:
        return hidden_states_fp8.bfloat16()
    M, K = hidden_states_fp8.size()
    N = self.w13_weight.size(1)
    scale_block_size = 128

    hidden_states_fp8_shape = hidden_states_fp8.shape
    hidden_states_fp8_device = hidden_states_fp8.device
    hidden_states_fp8_dtype = hidden_states_fp8.dtype

    input_tensor = [
        torch.empty(
            (all_tokens, K),
            device=hidden_states_fp8.device,
            dtype=hidden_states_fp8.dtype,
        ),
        torch.empty(
            (all_tokens, K // 128),
            device=hidden_states_fp8.device,
            dtype=torch.float32,
        ),
    ]
    m_indices = torch.empty(
        all_tokens, device=hidden_states_fp8.device, dtype=torch.int32
    )
    output_index = torch.empty_like(topk_idx)

    num_recv_tokens_per_expert_gpu = torch.tensor(
        num_recv_tokens_per_expert,
        dtype=torch.int32,
        pin_memory=True,
        device="cpu",
    ).cuda(non_blocking=True)
    expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)

    ep_scatter(
        hidden_states_fp8,
        hidden_states_scale,
        topk_idx,
        num_recv_tokens_per_expert_gpu,
        expert_start_loc,
        input_tensor[0],
        input_tensor[1],
        m_indices,
        output_index,
    )
    dispose_tensor(hidden_states_fp8)

    gateup_output = torch.empty(
        (all_tokens, N),
        device=hidden_states_fp8_device,
        dtype=torch.bfloat16,
    )
    input_tensor[1] = tma_align_input_scale(input_tensor[1])
    deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
        input_tensor, self.w13_weight_fp8, gateup_output, m_indices
    )
    del input_tensor
    down_input = torch.empty(
        (
            all_tokens,
            N // 2,
        ),
        device=gateup_output.device,
        dtype=torch.bfloat16,
    )
    silu_and_mul(gateup_output.view(-1, N), down_input)
    del gateup_output
    down_output = torch.empty(
        (all_tokens, K),
        device=hidden_states_fp8_device,
        dtype=torch.bfloat16,
    )
    down_input_fp8, down_input_scale = sglang_per_token_group_quant_fp8(
        down_input, scale_block_size
    )
    del down_input
    down_input_scale = tma_align_input_scale(down_input_scale)
    deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
        (down_input_fp8, down_input_scale),
        self.w2_weight_fp8,
        down_output,
        m_indices,
    )
    del down_input_fp8, down_input_scale

    gather_out = torch.empty(
        hidden_states_fp8_shape,
        device=hidden_states_fp8_device,
        dtype=torch.bfloat16,
    )
    ep_gather(down_output, topk_idx, topk_weights, output_index, gather_out)

    return gather_out

def forward_deepgemm_masked(
        self,
        hidden_states_fp8: Tuple[torch.Tensor, torch.Tensor],
        masked_m: torch.Tensor,
        expected_m: int,
):
    assert self.quant_method is not None
    assert self.activation == "silu"

    # GroupGemm-0
    num_groups, m, k = hidden_states_fp8[0].size()
    n = self.w13_weight.size(1)
    expected_m = min(expected_m, m)
    gateup_output = torch.empty(
        (num_groups, m, n), device=hidden_states_fp8[0].device, dtype=torch.bfloat16
    )
    deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
        hidden_states_fp8,
        self.w13_weight_fp8,
        gateup_output,
        masked_m,
        expected_m,
        recipe=(1, 128, 128) if deep_gemm_wrapper.DEEPGEMM_BLACKWELL else None,
    )
    dispose_tensor(hidden_states_fp8[0])

    # Act
    down_input = torch.empty(
        (
            gateup_output.shape[0],
            gateup_output.shape[1],
            gateup_output.shape[2] // 2,
        ),
        device=gateup_output.device,
        dtype=self.fp8_dtype,
    )
    scale_block_size = 128
    down_input_scale = torch.empty(
        (
            gateup_output.shape[0],
            gateup_output.shape[1],
            gateup_output.shape[2] // 2 // scale_block_size,
        ),
        device=gateup_output.device,
        dtype=torch.float32,
    )
    silu_and_mul_masked_post_quant_fwd(
        gateup_output,
        down_input,
        down_input_scale,
        scale_block_size,
        masked_m,
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
    )
    del gateup_output

    # GroupGemm-1
    n = self.w2_weight.size(1)
    down_input_fp8 = (
        down_input,
        (
            down_input_scale
            if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
            else deep_gemm_wrapper.get_col_major_tma_aligned_tensor(
                down_input_scale
            )
        ),
    )
    down_output = torch.empty(
        (num_groups, m, n), device=down_input.device, dtype=torch.bfloat16
    )
    deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
        down_input_fp8,
        self.w2_weight_fp8,
        down_output,
        masked_m,
        expected_m,
        recipe=(1, 128, 128) if deep_gemm_wrapper.DEEPGEMM_BLACKWELL else None,
    )

    return down_output


if __name__ == '__main__':
    # TODO: you may modify NUMA binding for less CPU overhead
    num_processes = int(os.getenv("DEEPEP_TEST_NUM_PROCESSES", "8"))
    torch.multiprocessing.spawn(test_loop, args=(num_processes,), nprocs=num_processes)
