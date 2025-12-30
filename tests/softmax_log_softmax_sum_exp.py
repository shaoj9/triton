import torch
import triton
import triton.language as tl
import time
@triton.jit
def fused_entropy_exp_kernel(
    input_ptr, output_ptr, 
    stride_row, n_cols, 
    BLOCK_SIZE: tl.constexpr
):
    # 1. Row Indexing
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * stride_row
    
    # 2. Load Logits
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    logits = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # 3. Online Softmax Logic (Safe Log-Sum-Exp)
    row_max = tl.max(logits, axis=0)
    shifted_logits = logits - row_max
    exp_logits = tl.exp(shifted_logits)
    sum_exp = tl.sum(exp_logits, axis=0)
    lse = tl.log(sum_exp) + row_max
    
    # 4. Fused Entropy Calculation
    # log_p = x - LSE
    # p = exp(x - LSE)
    log_p = logits - lse
    p = tl.exp(log_p)
    entropy_sum = tl.sum(p * log_p, axis=0)
    
    # 5. Apply Final Exp (New Operation Added)
    # This transforms the negative entropy into its exponential form
    final_result = tl.exp(entropy_sum)
    
    # 6. Store single scalar result per row
    tl.store(output_ptr + row_idx, final_result)

# --- 1. Triton Kernel Definition ---
@triton.jit
def fused_entropy_exp_kernel(
    input_ptr, output_ptr, 
    stride_row, n_cols, 
    BLOCK_SIZE: tl.constexpr
):
    # Row Indexing
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * stride_row
    
    # Load Logits with masking
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    logits = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # --- Online Softmax Logic ---
    # 1. Find Max for numerical stability
    row_max = tl.max(logits, axis=0)
    
    # 2. Calculate Log-Sum-Exp (LSE)
    # LSE = log(sum(exp(x - max))) + max
    shifted_logits = logits - row_max
    exp_logits = tl.exp(shifted_logits)
    sum_exp = tl.sum(exp_logits, axis=0)
    lse = tl.log(sum_exp) + row_max
    
    # --- Fused Entropy Calculation ---
    # log_p = x - LSE
    # p = exp(x - LSE)
    log_p = logits - lse
    p = tl.exp(log_p)
    
    # Calculate sum(p * log_p)
    # Note: We mask p*log_p to 0 where input was masked to avoid NaNs
    term = p * log_p
    entropy_sum = tl.sum(tl.where(mask, term, 0.0), axis=0)
    
    # --- Final Exp Fusion ---
    # result = exp(sum(p * log_p))
    final_result = tl.exp(entropy_sum)
    
    # Store scalar result
    tl.store(output_ptr + row_idx, final_result)

# --- 2. Python Wrapper ---
def fused_entropy_exp(x):
    """
    Computes exp(sum(softmax(x) * log_softmax(x))) in one fused pass.
    """
    n_rows, n_cols = x.shape
    output = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    
    # Block size must be power of 2
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Heuristic: If block size is too large for one block, standard split
    # For this demo, we assume n_cols fits in one block (<= 128k on H100, <=64k usually)
    # For extremely large vocab, you would need the chunked loop version.
    num_warps = 8
    if BLOCK_SIZE >= 2048: num_warps = 16
    if BLOCK_SIZE >= 4096: num_warps = 32

    grid = (n_rows,)
    fused_entropy_exp_kernel[grid](
        x, output,
        x.stride(0), n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    return output

# --- 3. Validation & Benchmark ---
def test_and_benchmark():
    torch.manual_seed(0)
    # Settings: Llama 3 sized vocabulary
    BATCH_SIZE = 2048 # Tokens
    VOCAB_SIZE = 32768 # Reasonable vocab size for single block
    dtype = torch.float32 # Use float32 for precision in sum
    device = "cuda"
    
    print(f"Config: Batch={BATCH_SIZE}, Vocab={VOCAB_SIZE}, Dtype={dtype}")
    
    x = torch.randn(BATCH_SIZE, VOCAB_SIZE, device=device, dtype=dtype)
    
    # --- Correctness Check ---
    print("Running correctness check...")
    
    # 1. PyTorch Native Implementation
    # This reads/writes memory 5+ times: 
    # softmax -> log_softmax -> mul -> sum -> exp
    t0 = time.time()
    probs = torch.nn.functional.softmax(x, dim=-1)
    log_probs = torch.nn.functional.log_softmax(x, dim=-1)
    entropy_sum = torch.sum(probs * log_probs, dim=-1)
    torch_out = torch.exp(entropy_sum)
    torch.cuda.synchronize()
    
    # 2. Fused Triton Implementation
    # Reads x once, writes output once.
    triton_out = fused_entropy_exp(x)
    triton.cuda.synchronize()
    
    # Compare
    max_diff = (torch_out - triton_out).abs().max()
    print(f"Max Difference: {max_diff:.2e}")
    if max_diff < 1e-4:
        print("✅ Correctness Passed")
    else:
        print("❌ Correctness Failed")
        
    # --- Benchmark ---
    print("\nRunning Benchmark (avg of 1000 runs)...")
    
    # Warmup
    for _ in range(10):
        fused_entropy_exp(x)
        
    # Measure Triton
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(1000):
        fused_entropy_exp(x)
    end.record()
    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end) / 1000
    
    # Measure PyTorch
    start.record()
    for _ in range(1000):
        p = x.softmax(dim=-1)
        lp = x.log_softmax(dim=-1)
        (p * lp).sum(dim=-1).exp()
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / 1000
    
    print(f"PyTorch Time: {torch_ms:.3f} ms")
    print(f"Triton Time:  {triton_ms:.3f} ms")
    print(f"Speedup:      {torch_ms / triton_ms:.2f}x")

if __name__ == "__main__":
    test_and_benchmark()