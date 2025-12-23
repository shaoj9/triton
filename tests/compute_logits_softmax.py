import torch
import triton
import triton.language as tl

@triton.jit
def fused_linear_softmax_kernel(
    H_ptr, W_ptr, Out_ptr,
    M, K, N,
    stride_hm, stride_hk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input sequence
    row_idx = tl.program_id(0)
    
    # Load the hidden state vector for this row into SRAM
    k_offsets = tl.arange(0, BLOCK_SIZE)
    h_row_ptr = H_ptr + row_idx * stride_hm + k_offsets
    h_row = tl.load(h_row_ptr, mask=k_offsets < K, other=0.0)

    # Online Softmax accumulators
    m_i = -float('inf')  # Running max
    l_i = 0.0            # Running sum of exp
    
    # Pass 1: Compute logits and update max/sum in chunks
    for n_start in range(0, N, BLOCK_SIZE_N):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_offsets < N
        
        # Compute logit for this chunk (dot product h @ W_chunk^T)
        acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
        for k_start in range(0, K, BLOCK_SIZE_K):
            k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
            k_mask = k_offsets < K
            
            h = tl.load(H_ptr + row_idx * stride_hm + k_offsets, mask=k_mask, other=0.0)
            w = tl.load(W_ptr + n_offsets[:, None] * stride_wn + k_offsets[None, :] * stride_wk, 
                        mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            acc += tl.sum(h[None, :] * w, axis=1)

        # Update online softmax stats for this chunk
        chunk_max = tl.max(acc, axis=0)
        m_new = tl.maximum(m_i, chunk_max)
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(acc - m_new), axis=0)
        m_i = m_new

    # Pass 2: Final normalization and store
    for n_start in range(0, N, BLOCK_SIZE):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        n_mask = n_offsets < N
        
        # Re-compute logits for storage (or store in SRAM if memory allows)
        w_ptr = W_ptr + (n_offsets[:, None] * stride_wn + k_offsets[None, :] * stride_wk)
        w_block = tl.load(w_ptr, mask=(n_mask[:, None] & (k_offsets[None, :] < K)), other=0.0)
        logits = tl.sum(w_block * h_row[None, :], axis=1)
        
        # Normalize and store
        probs = tl.exp(logits - m_i) / l_i
        tl.store(Out_ptr + row_idx * stride_om + n_offsets, probs, mask=n_mask)

def fused_linear_softmax(h, w):
    M, K = h.shape
    N, _ = w.shape  # w is [Vocab, Hidden]
    out = torch.empty((M, N), device=h.device, dtype=h.dtype)
    
    # Block size should be a power of two
    BLOCK_SIZE = triton.next_power_of_2(K)
    
    grid = (M,)
    fused_linear_softmax_kernel[grid](
        h, w, out,
        M, K, N,
        h.stride(0), h.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def test_fused_kernel():
    torch.manual_seed(0)
    M, K, N = 16, 128, 1024  # Batch/Seq, Hidden, Vocab
    h = torch.randn((M, K), device='cuda', dtype=torch.float32)
    w = torch.randn((N, K), device='cuda', dtype=torch.float32)

    # Reference PyTorch implementation
    ref_logits = torch.matmul(h, w.t())
    ref_probs = torch.softmax(ref_logits, dim=-1)

    # Triton implementation
    tri_probs = fused_linear_softmax(h, w)

    # Verify results
    torch.testing.assert_close(tri_probs, ref_probs, atol=1e-5, rtol=1e-5)
    print("Verification Successful: Triton and PyTorch results match.")

if __name__ == "__main__":
    test_fused_kernel()

import torch
from transformers import AutoModelForCausalLM

# Load a model (example: a small GPT-2 or Llama)
model_name = "gpt2" 
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

# Access the final linear layer (lm_head)
# Most LLMs use 'lm_head'; check model.named_modules() if unsure
lm_head_weight = model.lm_head.weight.detach() 

print(f"Weight Shape: {lm_head_weight.shape}") # [Vocab, Hidden]

def test_with_real_weights():
    # 1. Setup Data from Model
    # Assume lm_head_weight is [N, K] where N=Vocab, K=Hidden
    vocab_size, hidden_dim = lm_head_weight.shape
    seq_len = 16
    
    # Create dummy hidden states matching the model's dimension
    h = torch.randn((seq_len, hidden_dim), device='cuda', dtype=torch.float32)
    w = lm_head_weight.to(torch.float32) # Ensure matching dtypes

    # 2. Reference PyTorch Implementation
    # PyTorch Linear uses x @ W^T + bias; here we use Matmul [M, K] @ [K, N]
    ref_logits = torch.matmul(h, w.t())
    ref_probs = torch.softmax(ref_logits, dim=-1)

    # 3. Triton Implementation
    tri_probs = fused_linear_softmax(h, w)

    # 4. Verify
    torch.testing.assert_close(tri_probs, ref_probs, atol=1e-5, rtol=1e-5)
    print("Verification with real model weights Successful.")