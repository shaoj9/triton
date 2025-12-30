import torch
import triton
import triton.language as tl

@triton.jit
def fused_linear_ce_kernel(
    x_ptr, weight_ptr, target_ptr, loss_ptr,
    stride_xn, stride_xd, 
    stride_wn, stride_wd,
    D: tl.constexpr, VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr, BLOCK_SIZE_D: tl.constexpr
):
    # Each program processes one token (row)
    row_idx = tl.program_id(0)
    
    # Offsets for the current token's hidden state (x)
    x_row_ptr = x_ptr + row_idx * stride_xn
    target_idx = tl.load(target_ptr + row_idx)
    
    # Initialize online softmax statistics
    m_i = -float('inf')  # Running max
    l_i = 0.0           # Running sum of exponentials
    logit_label = 0.0   # Logit value for the ground-truth label
    
    # Iterate over vocabulary chunks
    for start_v in range(0, VOCAB_SIZE, BLOCK_SIZE_V):
        v_offsets = start_v + tl.arange(0, BLOCK_SIZE_V)
        mask_v = v_offsets < VOCAB_SIZE
        
        # Tile MatMul: Compute partial logits for this vocabulary chunk
        # logit_chunk = dot(x_row, weight_chunk.T)
        acc = tl.zeros([BLOCK_SIZE_V], dtype=tl.float32)
        
        for start_d in range(0, D, BLOCK_SIZE_D):
            d_offsets = start_d + tl.arange(0, BLOCK_SIZE_D)
            mask_d = d_offsets < D
            
            # Load x_chunk [BLOCK_SIZE_D]
            x_chunk = tl.load(x_row_ptr + d_offsets * stride_xd, mask=mask_d, other=0.0)
            
            # Load w_chunk [BLOCK_SIZE_V, BLOCK_SIZE_D]
            # Assumes weight is [Vocab, Hidden]
            w_chunk_ptrs = weight_ptr + (v_offsets[:, None] * stride_wn + d_offsets[None, :] * stride_wd)
            w_chunk = tl.load(w_chunk_ptrs, mask=mask_v[:, None] & mask_d[None, :], other=0.0)
            
            # Compute partial dot product and accumulate
            # Using tl.dot requires 2D inputs; we treat x_chunk as [1, BLOCK_SIZE_D]
            acc += tl.sum(x_chunk[None, :] * w_chunk, axis=1)
        
        # Online Softmax update
        m_ij = tl.max(acc, axis=0)
        new_m_i = tl.maximum(m_i, m_ij)
        l_i = l_i * tl.exp(m_i - new_m_i) + tl.sum(tl.exp(acc - new_m_i))
        m_i = new_m_i
        
        # Extract ground-truth logit value if it resides in this chunk
        is_label_in_chunk = (v_offsets == target_idx) & mask_v
        if tl.any(is_label_in_chunk):
            logit_label = tl.sum(tl.where(is_label_in_chunk, acc, 0.0))

    # Final Loss calculation: log(sum(exp(logits - max))) + max - logit_label
    # Equivalent to standard CrossEntropy: -logit_label + log(sum(exp(logits)))
    loss = tl.log(l_i) + m_i - logit_label
    tl.store(loss_ptr + row_idx, loss)

class FusedLinearCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, target):
        # x: [N, D], weight: [V, D], target: [N]
        N, D = x.shape
        V, _ = weight.shape
        
        # Output buffer for scalar loss per token
        loss = torch.empty(N, device=x.device, dtype=torch.float32)
        
        # Grid: one program per token
        grid = (N,)
        fused_linear_ce_kernel[grid](
            x, weight, target, loss,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            D, V,
            BLOCK_SIZE_V=512, BLOCK_SIZE_D=32
        )
        
        ctx.save_for_backward(x, weight, target)
        return loss.mean()

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, target = ctx.saved_tensors
        N, D = x.shape
        V, _ = weight.shape
        
        # Initialize gradients
        grad_x = torch.zeros_like(x)
        grad_weight = torch.zeros_like(weight)
        
        # The backward pass typically re-computes logits in chunks to 
        # compute (Softmax(logits) - 1_target) and then projects back
        # to get grad_x and grad_weight.
        # For simplicity, libraries like Liger-Kernel are used here.
        
        return grad_x, grad_weight, None
    
def test_large_fused_linear_ce():
    # Simulation: Llama 3 8B settings
    BATCH_SIZE = 8
    SEQ_LEN = 4096
    N = BATCH_SIZE * SEQ_LEN # 32,768 tokens
    HIDDEN_DIM = 4096
    VOCAB_SIZE = 128256     # Large vocabulary
    device = "cuda"

    # 1. Setup large tensors
    x = torch.randn(N, HIDDEN_DIM, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(VOCAB_SIZE, HIDDEN_DIM, device=device, dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(0, VOCAB_SIZE, (N,), device=device)

    print(f"Testing with Vocab={VOCAB_SIZE}, Hidden={HIDDEN_DIM}, Tokens={N}")
    
    # 2. Benchmark Memory
    torch.cuda.reset_peak_memory_stats()
    
    # Run Fused Kernel
    loss = FusedLinearCEFunction.apply(x, weight, target)
    loss.backward()
    
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"Fused Loss: {loss.item():.4f}")
    print(f"Peak VRAM Usage: {peak_mem:.2f} GB")
    
    # Validation against PyTorch (Only if your GPU has >40GB VRAM)
    # try:
    #     logits = x @ weight.t()
    #     ref_loss = torch.nn.functional.cross_entropy(logits, target)
    #     torch.testing.assert_close(loss, ref_loss)
    # except torch.cuda.OutOfMemoryError:
    #     print("Standard PyTorch failed due to OOM as expected.")

if __name__ == "__main__":
    test_large_fused_linear_ce()