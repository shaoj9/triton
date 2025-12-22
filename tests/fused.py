import torch
import triton
import triton.language as tl

@triton.jit
def fused_logits_softmax_kernel(
    H_ptr, W_ptr, Out_ptr,
    M, N, K, # M=Batch*Seq, N=Vocab, K=HiddenDim
    stride_hm, stride_hk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_SIZE_K: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr
):
    # Each program handles one row (e.g., one token's hidden state)
    row_idx = tl.program_id(0)
    
    # Initialize Online Softmax statistics in registers
    m_i = -float('inf') 
    d_i = 0.0           
    
    # Pointers to the specific row of hidden states (H)
    # We assume BLOCK_SIZE_K equals K (HiddenDim) for simplicity here
    h_offs = tl.arange(0, BLOCK_SIZE_K)
    h_row_ptr = H_ptr + row_idx * stride_hm + h_offs
    h_row = tl.load(h_row_ptr, mask=h_offs < K, other=0.0)

    # 1. Tiling over the Vocabulary Dimension (N)
    for n_start in range(0, N, BLOCK_SIZE_N):
        n_offs = n_start + tl.arange(0, BLOCK_SIZE_N)
        
        # Load weight tile (W is [N, K])
        # W_tile pointers for [BLOCK_SIZE_N, BLOCK_SIZE_K]
        w_ptr = W_ptr + (n_offs[:, None] * stride_wn + h_offs[None, :] * stride_wk)
        w_tile = tl.load(w_ptr, mask=(n_offs[:, None] < N) & (h_offs[None, :] < K), other=0.0)
        
        # Compute Logit Tile: [BLOCK_SIZE_N]
        # Equivalent to h_row @ w_tile.T
        logit_tile = tl.sum(h_row[None, :] * w_tile, axis=1)

        # 2. Update Online Softmax Statistics
        tile_max = tl.max(logit_tile, axis=0)
        new_m = tl.maximum(m_i, tile_max)
        # Numerical scaling factor to adjust previous sum to new max
        alpha = tl.exp(m_i - new_m)
        d_i = d_i * alpha + tl.sum(tl.exp(logit_tile - new_m))
        m_i = new_m

    # 3. Final Pass: Normalize and Write (or compute Cross-Entropy)
    # Note: For efficiency, one would often store only the loss or labels here.
    # To output full softmax, a second tile loop is needed to divide by d_i.


def compute_fused_logits(hidden_states, weights):
    # hidden_states: [Batch*Seq, HiddenDim]
    # weights: [Vocab, HiddenDim]
    M, K = hidden_states.shape
    N, _ = weights.shape
    
    # Output buffer (if you need probabilities; if only loss, this is a scalar)
    output = torch.empty((M, N), device=hidden_states.device, dtype=hidden_states.dtype)
    
    # BLOCK_SIZE_K must match your hidden dimension or be tiled further
    grid = (M,)
    fused_logits_softmax_kernel[grid](
        hidden_states, weights, output,
        M, N, K,
        hidden_states.stride(0), hidden_states.stride(1),
        weights.stride(0), weights.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_K=triton.next_power_of_2(K),
        BLOCK_SIZE_N=512
    )
    return output


import torch
import triton

def test_fused_correctness(M=128, N=4096, K=1024):
    torch.manual_seed(0)
    # Inputs: Hidden states (H) and Weights (W)
    h = torch.randn((M, K), device='cuda', dtype=torch.float32)
    w = torch.randn((N, K), device='cuda', dtype=torch.float32)
    
    # Baseline: Materialize logits then apply softmax
    logits_ref = torch.matmul(h, w.t())
    expected_out = torch.softmax(logits_ref, dim=-1)
    
    # Triton tiled version (using the wrapper defined previously)
    actual_out = compute_fused_logits(h, w)
    
    # Accuracy check
    try:
        torch.testing.assert_close(actual_out, expected_out, atol=1e-5, rtol=1e-5)
        print("✅ Correctness: Tiled output matches PyTorch baseline.")
    except AssertionError as e:
        print(f"❌ Correctness Failed: Max difference {torch.max(torch.abs(actual_out - expected_out))}")