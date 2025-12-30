import time
import torch
import triton
import triton.language as tl

@triton.jit
def fused_softmax_entropy_chunked_kernel(
    input_ptr, softmax_ptr, entropy_ptr, 
    stride_row, VOCAB_SIZE, 
    BLOCK_V: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * stride_row
    
    # Online Softmax Stats
    m_i = -float('inf')
    l_i = 0.0
    
    # Pass 1: Find Max and Log-Sum-Exp across all chunks
    for start_v in range(0, VOCAB_SIZE, BLOCK_V):
        v_offsets = start_v + tl.arange(0, BLOCK_V)
        mask = v_offsets < VOCAB_SIZE
        logits = tl.load(row_start_ptr + v_offsets, mask=mask, other=-float('inf'))
        
        m_ij = tl.max(logits, axis=0)
        new_m_i = tl.maximum(m_i, m_ij)
        l_i = l_i * tl.exp(m_i - new_m_i) + tl.sum(tl.exp(logits - new_m_i))
        m_i = new_m_i

    lse = tl.log(l_i) + m_i
    
    # Pass 2: Calculate Softmax and Entropy per chunk
    row_entropy = 0.0
    for start_v in range(0, VOCAB_SIZE, BLOCK_V):
        v_offsets = start_v + tl.arange(0, BLOCK_V)
        mask = v_offsets < VOCAB_SIZE
        logits = tl.load(row_start_ptr + v_offsets, mask=mask, other=-float('inf'))
        
        log_p = logits - lse
        p = tl.exp(log_p)
        
        # Store Softmax results to global memory
        tl.store(softmax_ptr + row_idx * VOCAB_SIZE + v_offsets, p, mask=mask)
        
        # Accumulate entropy: sum(p * log_p)
        row_entropy += tl.sum(tl.where(mask, p * log_p, 0.0), axis=0)
        
    # Store Final scalar result: exp(entropy)
    tl.store(entropy_ptr + row_idx, tl.exp(row_entropy))

def fused_softmax_entropy_triton(x, chunk_size=4096):
    """
    Python wrapper for the fused chunked Softmax + Entropy + Exp kernel.
    Args:
        x: Input logits of shape [Batch*Seq, Vocab]
        chunk_size: Tile size for the vocabulary dimension (dim=-1)
    """
    n_rows, n_cols = x.shape
    # Output: Full Softmax tensor [N, V] and scalar Entropy [N]
    softmax_out = torch.empty_like(x)
    entropy_out = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    
    # Grid: One program per row (token)
    grid = (n_rows,)
    
    # Note: Ensure BLOCK_V is a power of 2 for Triton efficiency
    # num_warps is typically 8 or 16 for high-bandwidth tasks
    fused_softmax_entropy_chunked_kernel[grid](
        x, softmax_out, entropy_out,
        x.stride(0), n_cols,
        BLOCK_V=chunk_size,
        num_warps=16
    )
    return softmax_out, entropy_out

def benchmark_128k_vocab():
    # Llama 3 Scale: 8k Tokens, 128,256 Vocabulary
    N_TOKENS = 8192
    VOCAB_SIZE = 128256
    device = "cuda"
    dtype = torch.float32 # Use float32 for stable entropy reduction

    # Input data
    x = torch.randn(N_TOKENS, VOCAB_SIZE, device=device, dtype=dtype)
    
    print(f"Benchmarking: {N_TOKENS} tokens, {VOCAB_SIZE} vocabulary size")

    # --- PyTorch Baseline ---
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Materializes [8192, 128256] logits which is ~4.2GB in float32
    probs = torch.nn.functional.softmax(x, dim=-1)
    log_probs = torch.nn.functional.log_softmax(x, dim=-1)
    entropy = torch.exp(torch.sum(probs * log_probs, dim=-1))
    
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) * 1000
    pytorch_mem = torch.cuda.max_memory_allocated() / 1e9

    # Reset memory stats for Triton
    torch.cuda.reset_peak_memory_stats()

    # --- Triton Fused Kernel ---
    torch.cuda.synchronize()
    start_time = time.time()
    
    triton_sm, triton_ent = fused_softmax_entropy_triton(x)
    
    torch.cuda.synchronize()
    triton_time = (time.time() - start_time) * 1000
    triton_mem = torch.cuda.max_memory_allocated() / 1e9

    # Results
    print(f"\n[PyTorch] Time: {pytorch_time:.2f}ms | Peak VRAM: {pytorch_mem:.2f}GB")
    print(f"[Triton]  Time: {triton_time:.2f}ms | Peak VRAM: {triton_mem:.2f}GB")
    print(f"Speedup: {pytorch_time / triton_time:.2f}x")
    
    # Correctness Check
    torch.testing.assert_close(triton_ent, entropy, atol=1e-3, rtol=1e-3)
    print("✅ Numerical Verification Passed")

if __name__ == "__main__":
    benchmark_128k_vocab()