import torch
import triton
import triton.language as tl

@triton.jit
def fused_linear_ce_kernel(
    X_ptr, W_ptr, Y_ptr, Loss_ptr, 
    stride_xn, stride_xd, stride_wn, stride_wd,
    VOCAB_SIZE, BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_V: tl.constexpr
):
    # Each program handles one token (row)
    row_idx = tl.program_id(0)
    
    # 1. Pointers for current token hidden state
    x_row_ptr = X_ptr + row_idx * stride_xn
    target_idx = tl.load(Y_ptr + row_idx)
    
    # Running statistics for online softmax (log-sum-exp)
    m_i = -float('inf')
    l_i = 0.0
    logit_label = 0.0

    # 2. Iterate over vocabulary chunks
    for start_v in range(0, VOCAB_SIZE, BLOCK_SIZE_V):
        v_offsets = start_v + tl.arange(0, BLOCK_SIZE_V)
        
        # Compute partial logits: dot(x, W_chunk)
        # Note: In practice, this uses a nested loop for BLOCK_SIZE_D
        logits_chunk = tl.zeros([BLOCK_SIZE_V], dtype=tl.float32)
        for start_d in range(0, D, BLOCK_SIZE_D):
            # ... Load x_chunk and w_chunk, then tl.dot ...
        
        # 3. Update Online Softmax (Safe Log-Sum-Exp)
        m_ij = tl.max(logits_chunk, axis=0)
        new_m_i = tl.maximum(m_i, m_ij)
        l_i = l_i * tl.exp(m_i - new_m_i) + tl.sum(tl.exp(logits_chunk - new_m_i))
        m_i = new_m_i
        
        # 4. Extract ground-truth logit if it's in this chunk
        mask = (v_offsets == target_idx)
        if tl.any(mask):
            logit_label = tl.sum(tl.where(mask, logits_chunk, 0.0))

    # 5. Final Loss Calculation: Loss = log(sum(exp(logits))) - logit_label
    loss = tl.log(l_i) + m_i - logit_label
    tl.store(Loss_ptr + row_idx, loss)

class FusedLinearCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, target):
        # x: [Batch*Seq, Hidden], weight: [Vocab, Hidden], target: [Batch*Seq]
        N, D = x.shape
        V, _ = weight.shape
        
        # Output buffer for scalar loss per row
        loss = torch.empty(N, device=x.device, dtype=torch.float32)
        
        # Meta-parameters for Triton (tuning required for specific hardware)
        grid = (N,)
        # Note: In a real implementation, you'd call your @triton.jit kernel here
        # fused_linear_ce_kernel[grid](x, weight, target, loss, ...)
        
        # Save for backward (requires careful memory management)
        ctx.save_for_backward(x, weight, target)
        return loss.mean()

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, target = ctx.saved_tensors
        # Standard cross-entropy gradient: (softmax(logits) - 1_target) / Batch
        # The fused backward pass recomputes logits in chunks to calculate grad_x and grad_w
        # grad_x = (P - Y) @ W
        # grad_w = (P - Y).T @ x
        # See Liger-Kernel or Ian's Blog for full chunked backward logic
        return grad_x, grad_w, None
    
def test_fused_linear_ce():
    # Large weights: Vocab=128k, Hidden=4096, Batch*Seq=8192
    BATCH_SEQ = 8192
    HIDDEN_SIZE = 4096
    VOCAB_SIZE = 128256 # Llama 3 size
    device = "cuda"

    # Initialize data
    x = torch.randn(BATCH_SEQ, HIDDEN_SIZE, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(VOCAB_SIZE, HIDDEN_SIZE, device=device, dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(0, VOCAB_SIZE, (BATCH_SEQ,), device=device)

    # 1. Standard PyTorch (Will use ~20GB for intermediate logits)
    # logits = torch.matmul(x, weight.t()) # Memory Spike Here!
    # loss_ref = torch.nn.functional.cross_entropy(logits, target)
    
    # 2. Fused Triton Kernel (Memory efficient)
    loss_fused = FusedLinearCEFunction.apply(x, weight, target)
    loss_fused.backward()

    print(f"Fused Loss: {loss_fused.item()}")
    print(f"X Gradient Shape: {x.grad.shape}")
    print(f"Weight Gradient Shape: {weight.grad.shape}")

if __name__ == "__main__":
    test_fused_linear_ce()