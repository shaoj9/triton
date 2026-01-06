from flashinfer.logits_processor import LogitsProcessor, Op, LogitsPipe, Softmax, TensorType
from typing import List, Any
import torch

import time
import flashinfer


class EntropyAndExpProcessor(LogitsProcessor):
    """Custom processor to calculate entropy and apply exp transformations."""
    def legalize(self, input_type: TensorType) -> List[Op]:
        # This operator expects probabilities (from a preceding Softmax)
        return [EntropyExpOp(**self.params)]

class EntropyExpOp(Op):
    # This Op consumes PROBS and outputs modified LOGITS/PROBS for further sampling
    IN = TensorType.PROBS
    OUT = TensorType.PROBS
    
    def __call__(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        # 1. log_softmax is effectively log(tensor) if input is already Softmax probs
        log_probs = torch.log(tensor + 1e-9) 
        
        # 2. sum(softmax * log_softmax) -> Negative Entropy
        # We use a reduction along the vocab dimension (dim=-1)
        entropy = -torch.sum(tensor * log_probs, dim=-1, keepdim=True)
        
        # 3. exp transformation
        # Applying exp to the entropy or specific probabilities as needed
        transformed_probs = torch.exp(tensor * entropy) 
        
        return transformed_probs
    
# pipe = LogitsPipe(
#     processors=[
#         Softmax(),                 # Step 1: Raw Logits -> Probs
#         EntropyAndExpProcessor(),  # Step 2: Custom Entropy & Exp logic
#         # You could add TopP() or Sample() here to complete the pass
#     ],
#     compile=True  # Fuses everything into one CUDA kernel launch
# )

# # Execution
# logits = torch.randn(batch_size, vocab_size, device="cuda")
# result_probs = pipe(logits)


# 1. Setup Custom Fused Processor (As defined in previous step)
# This processor handles entropy calculation internally
class FusedEntropyExpPipe(LogitsPipe):
    def __init__(self):
        super().__init__(
            processors=[Softmax(), EntropyProcessor()], 
            compile=True
        )

# 2. Benchmark Configuration
batch_size = 128
vocab_size = 128000 # Large vocab (e.g., Llama 3) highlights memory bottlenecks
device = "cuda"
dtype = torch.float16
logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype)

def benchmark_torch(logits):
    torch.cuda.synchronize()
    start = time.perf_counter()
    # Native PyTorch: 4 separate kernel launches
    p = torch.softmax(logits, dim=-1)
    log_p = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(p * log_p, dim=-1, keepdim=True)
    res = p * torch.exp(entropy)
    torch.cuda.synchronize()
    return time.perf_counter() - start

def benchmark_flashinfer(pipe, logits):
    torch.cuda.synchronize()
    start = time.perf_counter()
    # FlashInfer: 1 fused kernel launch
    res = pipe(logits)
    torch.cuda.synchronize()
    return time.perf_counter() - start

# 3. Execution
pipe = FusedEntropyExpPipe()
torch_time = sum(benchmark_torch(logits) for _ in range(100)) / 100
flash_time = sum(benchmark_flashinfer(pipe, logits) for _ in range(100)) / 100

print(f"PyTorch Native: {torch_time*1000:.3f} ms")
print(f"FlashInfer Fused: {flash_time*1000:.3f} ms")
print(f"Speedup: {torch_time/flash_time:.2f}x")