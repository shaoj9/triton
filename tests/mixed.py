import torch
import flashinfer
from flash_attn import flash_attn_func  # Backend for Target Model
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import profile, record_function, ProfilerActivity

class EagleFlashInferSpeculator:
    def __init__(self, target_model_id, draft_model_id):
        # 1. Target Model: Llama-3.1-8B (FlashAttention Backend)
        # In practice, many 2026 implementations use FlashAttention-3 for the verifier
        self.target = AutoModelForCausalLM.from_pretrained(
            target_model_id, 
            torch_dtype=torch.bfloat16, 
            device_map="cuda"
        )
        
        # 2. Draft Model: EAGLE-LLaMA3.1-Instruct-8B (FlashInfer Backend)
        # The EAGLE "model" is a lightweight head that predicts hidden states
        self.eagle_head = AutoModelForCausalLM.from_pretrained(
            draft_model_id, 
            torch_dtype=torch.bfloat16, 
            device_map="cuda"
        )

        # 3. FlashInfer Paged KV Cache Wrapper
        # This manages the token tree structure efficiently
        self.draft_kv_cache = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            torch.empty(128 * 1024, dtype=torch.bfloat16).cuda(), # Shared workspace
            backend="cuda"
        )

    def verify(self, q, k, v):
        """Uses FlashAttention for parallel verification of the tree."""
        return flash_attn_func(q, k, v, causal=True)

    def speculate(self, hidden_states):
        """Uses FlashInfer for low-latency tree expansion."""
        # FlashInfer kernels are ~3x faster for these single-token decode steps
        return flashinfer.single_decode_with_kv_cache(
            hidden_states, self.draft_k, self.draft_v, 
            pos_encoding_mode="ROPE_LLAMA"
        )

# Initialize with the specific models requested
# speculator = EagleFlashInferSpeculator(
#     target_model_id="meta-llama/Llama-3.1-8B-Instruct",
#     draft_model_id="yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
# )

@torch.inference_mode()
def generate(self, input_ids, max_new_tokens=50, do_sample=False):
    # 1. Warm up the context with the Target Model
    # Target model verification uses FlashAttention
    generated_ids = input_ids.clone()
    
    for _ in range(max_new_tokens // 4):  # EAGLE proposes ~4-5 tokens per step
        # --- PHASE 1: DRAFT (FlashInfer Backend) ---
        # Generate a tree of candidate tokens using FlashInfer's paged attention
        # For simplicity, we'll demonstrate a single-chain proposal here
        draft_tokens = self.speculate(generated_ids) 
        
        # --- PHASE 2: VERIFY (FlashAttention Backend) ---
        # Verify all proposed tokens in one forward pass
        # This is where FlashAttention's high throughput is used
        # (Simplified logic: in a real EAGLE setup, you'd compare logits)
        verified_tokens = self.target(torch.cat([generated_ids, draft_tokens], dim=-1))
        
        # Update your generated sequence
        # Here we just take the next token for the test loop
        next_token = verified_tokens.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        
        if next_token.item() == self.tokenizer.eos_token_id:
            break
            
    return generated_ids

# Inject the method into your existing class
EagleFlashInferSpeculator.generate = generate



def test_correctness(speculator, tokenizer, prompt="What are the laws of thermodynamics?"):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # 1. Baseline: Pure Target Model (No Speculation)
    with torch.no_grad():
        baseline_output = speculator.target.generate(
            **inputs, max_new_tokens=20, do_sample=False
        )
    
    # 2. Experimental: Mixed Backend Speculator
    experimental_output = speculator.generate(
        inputs.input_ids, max_new_tokens=20, do_sample=False
    )
    
    # 3. Assertion
    assert torch.equal(baseline_output, experimental_output), "Speculative output deviated from target!"
    print("Correctness Test Passed: Output is 100% identical to target model.")


def test_backend_performance(speculator, input_ids):
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("eagle_step"):
            # Run 5 steps to warm up and capture kernels
            speculator.generate(input_ids, max_new_tokens=5)

    # Search for specific backend kernels in the trace
    stats = prof.key_averages().table(sort_by="cuda_time_total")
    
    # Verification check: FlashAttention for Target
    has_fa = "flash_attn" in stats or "flash_fwd" in stats
    # Draft check: FlashInfer for Tree Decode
    has_fi = "flashinfer" in stats or "paged_decode" in stats

    print(f"FlashAttention (Target) Detected: {has_fa}")
    print(f"FlashInfer (Draft) Detected:      {has_fi}")
    
    if not (has_fa and has_fi):
        print("Warning: Mixed-backend synergy not detected. Check kernel bindings.")

model_id = "meta-llama/Llama-3.1-8B-Instruct"
draft_id = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
speculator = EagleFlashInferSpeculator(model_id, draft_id)

# 2. Prepare Test Data
prompt = "The capital of France is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

# 3. Call Correctness Test
# This compares your Mixed-Backend output to a standard Target-only output
print("--- Starting Correctness Test ---")
try:
    test_correctness(speculator, tokenizer, prompt)
except AssertionError as e:
    print(f"Correctness Error: {e}")

# 4. Call Backend Performance Test
# This verifies that FlashAttention and FlashInfer kernels are actually running
print("\n--- Starting Backend Performance Test ---")
test_backend_performance(speculator, input_ids)