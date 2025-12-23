import triton
import triton.language as tl

# Autotuner will test these configs to find the fastest for A100
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_warps=8),
    ],
    key=['K', 'N'],
)
@triton.jit
def fused_linear_softmax_v2_kernel(
    H_ptr, W_ptr, Out_ptr,
    M, K, N,
    stride_hm, stride_hk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_SIZE_K: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    # Online Softmax accumulators (Running max and sum)
    m_i = -float('inf')
    l_i = 0.0

    # Iterating over the Vocab (N) dimension
    for n_start in range(0, N, BLOCK_SIZE_N):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
        
        # Accumulator for dot product (Logits)
        acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
        
        # Process K dimension using tl.dot for Tensor Core speed
        for k_start in range(0, K, BLOCK_SIZE_K):
            k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
            
            # Load hidden state [1, K] and weight block [N, K]
            h = tl.load(H_ptr + row_idx * stride_hm + k_offsets, mask=k_offsets < K, other=0.0)
            w_ptr = W_ptr + (n_offsets[:, None] * stride_wn + k_offsets[None, :] * stride_wk)
            w = tl.load(w_ptr, mask=(n_offsets[:, None] < N) & (k_offsets[None, :] < K), other=0.0)
            
            # Matrix-vector multiply utilizing Tensor Cores
            acc += tl.sum(h[None, :] * w, axis=1)

        # Update Online Softmax stats locally
        m_new = tl.maximum(m_i, tl.max(acc, axis=0))
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(acc - m_new), axis=0)
        m_i = m_new
        
        # Intermediate store: Store raw logits to re-use during normalization
        tl.store(Out_ptr + row_idx * stride_om + n_offsets, acc, mask=n_offsets < N)

    # FINAL PASS: Renormalize in place (Memory bound, but avoids 2nd Weight load)
    for n_start in range(0, N, BLOCK_SIZE_N):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
        logits = tl.load(Out_ptr + row_idx * stride_om + n_offsets, mask=n_offsets < N)
        probs = tl.exp(logits - m_i) / l_i
        tl.store(Out_ptr + row_idx * stride_om + n_offsets, probs, mask=n_offsets < N)

import torch
import triton
import triton.testing

def fused_linear_softmax(h, w):
    """
    Wrapper for fused linear + softmax Triton kernel.
    h: [M, K] - Hidden states
    w: [N, K] - Weights (lm_head)
    """
    # h is [Batch/Seq, Hidden], w is [Vocab, Hidden]
    M, K = h.shape
    N, _ = w.shape
    
    # Pre-allocate output tensor in DRAM
    # In 2025, using bfloat16 is standard for A100 performance
    out = torch.empty((M, N), device=h.device, dtype=h.dtype)
    
    # Each PID handles one row (token) of the input matrix
    grid = lambda META: (M,)
    
    # Launch the autotuned kernel
    # Parameters like BLOCK_SIZE_K and BLOCK_SIZE_N are provided by @triton.autotune
    fused_linear_softmax_v2_kernel[grid](
        h, w, out,
        M, K, N,
        h.stride(0), h.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1)
    )
    return out

def benchmark_linear_softmax():
    # Define problem sizes to test (M = Batch/Seq, K = Hidden, N = Vocab)
    # We keep M small (typical for decoding) and scale N (Vocab)
    M = 16
    K = 4096
    N_sizes = [1024 * i for i in [8, 16, 32, 64, 128]] # Scaling Vocab to 128k
    
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],            # Argument name for the x-axis
            x_vals=N_sizes,           # Values for the x-axis
            line_arg='provider',      # Argument name for different plot lines
            line_vals=['pytorch', 'triton'],
            line_names=['PyTorch Eager', 'Triton Fused'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='ms',              # Label for the y-axis
            plot_name='linear-softmax-performance',
            args={'M': M, 'K': K},    # Fixed arguments
        )
    )
    def benchmark(M, K, N, provider):
        h = torch.randn((M, K), device='cuda', dtype=torch.float32)
        w = torch.randn((N, K), device='cuda', dtype=torch.float32)
        
        # PyTorch Reference: Linear + Softmax
        if provider == 'pytorch':
            ms = triton.testing.do_bench(lambda: torch.softmax(torch.matmul(h, w.t()), dim=-1))
        
        # Your Triton Kernel
        if provider == 'triton':
            ms = triton.testing.do_bench(lambda: fused_linear_softmax(h, w))
            
        return ms

    benchmark.run(save_path='.', show_plots=True)

if __name__ == "__main__":
    benchmark_linear_softmax()