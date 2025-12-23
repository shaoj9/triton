import torch
import triton
import triton.language as tl

@triton.jit
def fast_fused_linear_softmax_kernel(
    H_ptr, W_ptr, Out_ptr,
    M, K, N,
    stride_hm, stride_hk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_SIZE_K: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    # Online Softmax accumulators
    m_i = -float('inf')
    l_i = 0.0

    # Pass 1: Fast Logit Computation + Online Softmax
    for n_start in range(0, N, BLOCK_SIZE_N):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
        
        # Accumulate dot product for this block using Tensor Cores
        logits = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
        for k_start in range(0, K, BLOCK_SIZE_K):
            k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
            
            h = tl.load(H_ptr + row_idx * stride_hm + k_offsets, mask=k_offsets < K, other=0.0)
            # Reshape h to [1, K] for tl.dot
            w = tl.load(W_ptr + n_offsets[:, None] * stride_wn + k_offsets[None, :] * stride_wk, 
                        mask=(n_offsets[:, None] < N) & (k_offsets[None, :] < K), other=0.0)
            
            # Using tl.dot is much faster than manual summation loops
            logits += tl.sum(h[None, :] * w, axis=1)

        # Update online softmax statistics
        chunk_max = tl.max(logits, axis=0)
        m_new = tl.maximum(m_i, chunk_max)
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(logits - m_new), axis=0)
        m_i = m_new

    # Pass 2: Final normalization and store
    for n_start in range(0, N, BLOCK_SIZE_N):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
        logits = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
        for k_start in range(0, K, BLOCK_SIZE_K):
            k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
            h = tl.load(H_ptr + row_idx * stride_hm + k_offsets, mask=k_offsets < K, other=0.0)
            w = tl.load(W_ptr + n_offsets[:, None] * stride_wn + k_offsets[None, :] * stride_wk, 
                        mask=(n_offsets[:, None] < N) & (k_offsets[None, :] < K), other=0.0)
            logits += tl.sum(h[None, :] * w, axis=1)
            
        probs = tl.exp(logits - m_i) / l_i
        tl.store(Out_ptr + row_idx * stride_om + n_offsets, probs, mask=n_offsets < N)

# --- PYTHON WRAPPER ---
def fused_linear_softmax(h, w):
    M, K = h.shape
    N, _ = w.shape
    out = torch.empty((M, N), device=h.device, dtype=h.dtype)
    
    # 1024 is a safe block size for most modern GPUs (SRAM limits)
    # This prevents the 'numel' error by ensuring no tl.arange is > 1048576
    BLOCK_SIZE_K = 1024
    BLOCK_SIZE_N = 1024

    grid = (M,)
    fast_fused_linear_softmax_kernel[grid](
        h, w, out,
        M, K, N,
        h.stride(0), h.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return out

def test_fused_chunked_kernel():
    # Large dimensions to trigger looping and test numel limits
    # Hidden dimension 2048 > BLOCK_SIZE_K (1024)
    # Vocab dimension 4096 > BLOCK_SIZE_N (1024)
    M, K, N = 8, 2048, 4096 
    
    print(f"Testing dimensions: Seq={M}, Hidden={K}, Vocab={N}")
    
    # Initialize tensors on GPU
    h = torch.randn((M, K), device='cuda', dtype=torch.float32)
    w = torch.randn((N, K), device='cuda', dtype=torch.float32)

    # 1. Reference PyTorch Implementation
    # Standard: hidden @ weight.T -> softmax
    ref_logits = torch.matmul(h, w.t())
    ref_probs = torch.softmax(ref_logits, dim=-1)

    # 2. Triton Implementation
    try:
        tri_probs = fused_linear_softmax(h, w)
        
        # 3. Validation
        torch.testing.assert_close(tri_probs, ref_probs, atol=1e-5, rtol=1e-5)
        print("✅ Verification Successful: Chunked Triton and PyTorch match.")
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")

if __name__ == "__main__":
    # Check if GPU is available
    if torch.cuda.is_available():
        test_fused_chunked_kernel()
    else:
        print("CUDA not available. Test skipped.")




print(f"Weight Shape: {lm_head_weight.shape}") # [Vocab, Hidden]

def test_with_hf_model():
    from transformers import AutoModelForCausalLM
    
    # Load just the head and config for metadata
    model_id = "gpt2" # Replace with your target model
    model = AutoModelForCausalLM.from_pretrained(model_id).cuda()
    
    # Extract weights
    w = model.lm_head.weight.detach()
    vocab_size, hidden_dim = w.shape
    
    # Create input from the model's expected hidden size
    h = torch.randn((1, hidden_dim), device='cuda', dtype=w.dtype)
    
    # Run test
    tri_probs = fused_linear_softmax(h, w)
    print(f"Tested successfully on {model_id} weights.")