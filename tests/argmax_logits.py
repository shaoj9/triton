import torch
import triton
import triton.language as tl
import triton.testing

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 512}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_N': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_N': 2048}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE_N': 4096}, num_warps=16, num_stages=2),
    ],
    key=['N', 'K'], # Re-tune if vocabulary size or hidden dim changes
)
@triton.jit
def argmax_logit_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    K: tl.constexpr, N: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr
):
    # Map program to a specific row of input X
    row_idx = tl.program_id(0)
    X_ptr += row_idx * stride_xm
    
    # Initialize tracking variables for argmax
    max_val = float("-inf")
    argmax_idx = -1

    # Iterate over the vocabulary (N) in blocks
    for start_n in range(0, N, BLOCK_SIZE_N):
        cols = start_n + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N

        # Compute dot product for the current tile: Logit = X @ W^T + B
        # Loading X (1, K) and W_tile (BLOCK_SIZE_N, K)
        acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
        for k in range(0, K, 32): # Inner loop over hidden dim K
            k_offsets = k + tl.arange(0, 32)
            x_tile = tl.load(X_ptr + k_offsets)
            w_tile = tl.load(W_ptr + cols[:, None] * stride_wn + k_offsets[None, :] * stride_wk, mask=mask[:, None])
            acc += tl.sum(x_tile[None, :] * w_tile, axis=1)
        
        # Add bias
        bias = tl.load(B_ptr + cols, mask=mask)
        logits = acc + bias

        # Local argmax within the block
        local_max = tl.max(logits, axis=0)
        local_argmax = tl.argmax(logits, axis=0)

        # Update global argmax if current tile has a higher value
        if local_max > max_val:
            max_val = local_max
            argmax_idx = start_n + local_argmax

    # Write the resulting index to memory
    tl.store(Out_ptr + row_idx, argmax_idx)

def compute_large_argmax(x, weight, bias):
    M, K = x.shape
    N, _ = weight.shape
    out = torch.empty((M,), device=x.device, dtype=torch.int64)
    
    # Grid: one program per row of the input
    grid = (M,)
    argmax_logit_kernel[grid](
        x, weight, bias, out,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        K=K, N=N, BLOCK_SIZE_N=1024
    )
    return out



def benchmark_argmax():
    # Test configurations for large vocab sizes
    M, K = 16, 4096  # Batch size and hidden dimension
    N_sizes = [32000, 64000, 128000, 256000] # Large vocabularies
    
    results = []

    for N in N_sizes:
        # Initialize inputs
        x = torch.randn((M, K), device='cuda', dtype=torch.float16)
        weight = torch.randn((N, K), device='cuda', dtype=torch.float16)
        bias = torch.randn((N,), device='cuda', dtype=torch.float16)

        # PyTorch Reference: Requires full logit matrix in HBM
        def torch_fn():
            logits = torch.matmul(x, weight.t()) + bias
            return torch.argmax(logits, dim=-1)

        # Triton Kernel: Computes argmax in tiles
        def triton_fn():
            return compute_large_argmax(x, weight, bias)

        # Measure performance
        ms_torch = triton.testing.do_bench(torch_fn, warmup=25, rep=100)
        ms_triton = triton.testing.do_bench(triton_fn, warmup=25, rep=100)

        results.append((N, ms_torch, ms_triton))
        print(f"Vocab: {N} | Torch: {ms_torch:.3f} ms | Triton: {ms_triton:.3f} ms")

    return results

if __name__ == "__main__":
    benchmark_argmax()