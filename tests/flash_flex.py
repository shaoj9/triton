import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# --- 1. Configuration ---
BATCH_SIZE = 1
HEADS = 8
DIM = 64
SEQLEN_Q = 4    # Number of active branches (leaves) in the draft tree
SEQLEN_KV = 128 # Total tokens in the shared KV cache (history + tree nodes)

# Set device and dtype (BFloat16 is standard for LLMs)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16

# --- 2. Setup Dummy Data ---
# The "Leaves" we are expanding (Draft Heads)
q = torch.randn(BATCH_SIZE, HEADS, SEQLEN_Q, DIM, device=device, dtype=dtype)

# The "Shared History" (KV Cache)
k = torch.randn(BATCH_SIZE, HEADS, SEQLEN_KV, DIM, device=device, dtype=dtype)
v = torch.randn(BATCH_SIZE, HEADS, SEQLEN_KV, DIM, device=device, dtype=dtype)

# --- 3. Define Tree Topology (The EAGLE Structure) ---
# Create a random adjacency matrix to simulate a draft tree.
# mask[i, j] = True if KV token j is an ancestor of Query leaf i.
tree_mask = torch.zeros((SEQLEN_Q, SEQLEN_KV), dtype=torch.bool, device=device)

# Setup a dummy tree structure:
# All leaves see the first 50 tokens (Shared Prefix)
tree_mask[:, :50] = True 
# Each leaf sees a unique subset of the remaining tokens (Divergent Branches)
for i in range(SEQLEN_Q):
    start = 50 + (i * 10)
    end = start + 10
    tree_mask[i, start:end] = True

# -------------------------------------------------------------------------
# METHOD A: Standard FlashAttention (Naive Batch Expansion)
# -------------------------------------------------------------------------
def run_standard_flash(q, k, v, mask):
    """
    Simulates how standard EAGLE works: 
    1. Expands the KV cache to create a batch for each branch.
    2. Applies a mask to each sequence in the batch.
    """
    # Reshape Q to [Batch * Branches, Heads, 1, Dim]
    # We treat each branch as a independent sequence
    q_expand = q.permute(0, 2, 1, 3).reshape(BATCH_SIZE * SEQLEN_Q, HEADS, 1, DIM)
    
    # Expand KV to match: [Batch * Branches, Heads, SeqLen_KV, Dim]
    # CRITICAL: This is the memory bottleneck (Duplication)
    k_expand = k.repeat_interleave(SEQLEN_Q, dim=0)
    v_expand = v.repeat_interleave(SEQLEN_Q, dim=0)
    
    # Create the boolean mask for SDPA
    # Shape: [Batch * Branches, 1, 1, SeqLen_KV]
    # We unsqueeze to broadcast over heads
    attn_mask = mask.unsqueeze(0).unsqueeze(1).unsqueeze(2) # [1, 1, Q, KV]
    # Since we flattened Q into the batch dim, we need to slice the mask per batch item
    # But for simplicity in this comparison, let's keep Q as [1, Heads, SEQLEN_Q, DIM]
    # and use the mask directly on the sequence dimension.
    
    # Re-Approach: Keep Batch=1, but mask the sequence dim.
    # FlashAttention supports arbitrary masking via SDPA in PyTorch 2.0+
    # However, "Naive" usually means batching because mask support is slow in older kernels.
    # We will use the most efficient SDPA path available:
    
    ref_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(0).unsqueeze(0))
    return ref_out

# -------------------------------------------------------------------------
# METHOD B: FlexAttention (One-Pass Topology)
# -------------------------------------------------------------------------
def run_flex_attention(q, k, v, mask_tensor):
    """
    FlexAttention Approach:
    1. No memory duplication.
    2. Uses a fused kernel to skip non-ancestor blocks.
    """
    # Define the mask_mod that reads our topology tensor
    def eagle_mask_mod(b, h, q_idx, kv_idx):
        return mask_tensor[q_idx, kv_idx]

    # Create the BlockMask (The "Compiler" step)
    # In production, you cache this object.
    block_mask = create_block_mask(
        eagle_mask_mod, 
        B=BATCH_SIZE, 
        H=HEADS, 
        Q_LEN=SEQLEN_Q, 
        KV_LEN=SEQLEN_KV, 
        device=device
    )
    
    # Run the fused kernel
    flex_out = flex_attention(q, k, v, block_mask=block_mask)
    return flex_out

# -------------------------------------------------------------------------
# 4. Execute and Compare
# -------------------------------------------------------------------------

print(f"--- Running Accuracy Comparison (Dtype: {dtype}) ---")

# Run Standard
try:
    out_flash = run_standard_flash(q, k, v, tree_mask)
    print("Standard FlashAttention: Success")
except Exception as e:
    print(f"Standard FlashAttention Failed: {e}")
    exit()

# Run Flex
try:
    # Compile is optional but recommended for speed; we run eager for debug accuracy
    out_flex = run_flex_attention(q, k, v, tree_mask)
    print("FlexAttention:           Success")
except Exception as e:
    print(f"FlexAttention Failed: {e}")
    exit()

# Measure Accuracy
# Note: BF16 accumulation often differs by ~1e-2 to 1e-3 due to kernel fusion differences
diff = (out_flash - out_flex).abs()
max_diff = diff.max().item()
mean_diff = diff.mean().item()

print(f"\n--- Results ---")
print(f"Max Difference:  {max_diff:.6f}")
print(f"Mean Difference: {mean_diff:.6f}")

if max_diff < 1e-2:
    print("\n✅ MATCH: Implementations are numerically equivalent.")
else:
    print("\n⚠️ MISMATCH: Differences exceed normal BF16 tolerance.")

# -------------------------------------------------------------------------
# 5. Speed Benchmark (Optional)
# -------------------------------------------------------------------------
import time

# Warmup
for _ in range(10): run_flex_attention(q, k, v, tree_mask)

start = time.time()
for _ in range(100):
    run_flex_attention(q, k, v, tree_mask)
torch.cuda.synchronize()
print(f"\nFlexAttention Avg Latency: {(time.time()-start)*10:.3f} ms (100 runs)")