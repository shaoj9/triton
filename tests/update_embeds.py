"""
Updating Embeddings from Sampled Tokens in Tree Generation
==========================================================

This shows how to update embeddings after sampling tokens from tree generation.

Key concept:
1. Start with ZERO embeddings for tree positions
2. Generate logits in ONE pass
3. Sample token IDs from logits
4. UPDATE embeddings using sampled token IDs
5. Use updated embeddings for next iteration or verification

Usage:
    python update_embeddings_demo.py
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Tuple
from dataclasses import dataclass


# ============================================================================
# Tree Structure
# ============================================================================

@dataclass
class TreeNode:
    node_id: int
    depth: int
    parent_id: Optional[int]
    token_id: int = -1
    confidence: float = 0.0


def build_tree_structure(width: int, depth: int) -> Tuple[List[TreeNode], List[Optional[int]]]:
    """Build tree structure"""
    nodes = []
    parent_ids = []
    
    nodes.append(TreeNode(node_id=0, depth=0, parent_id=None))
    parent_ids.append(None)
    
    current_level = [0]
    next_id = 1
    
    for d in range(1, depth + 1):
        next_level = []
        for parent_id in current_level:
            for _ in range(width):
                node = TreeNode(node_id=next_id, depth=d, parent_id=parent_id)
                nodes.append(node)
                parent_ids.append(parent_id)
                next_level.append(next_id)
                next_id += 1
        current_level = next_level
    
    return nodes, parent_ids


# ============================================================================
# Embedding Update Pipeline
# ============================================================================

class EmbeddingUpdater:
    """
    Demonstrates how to update embeddings after sampling tokens
    """
    
    def __init__(
        self,
        model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        print(f"\n{'='*70}")
        print(f"INITIALIZING EMBEDDING UPDATER")
        print(f"{'='*70}")
        
        self.device = device
        self.dtype = dtype
        
        # Load model
        print(f"Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True
        )
        self.model.eval()
        print(f"  ✓ Model loaded")
        
        # Load tokenizer
        print(f"Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"  ✓ Tokenizer loaded")
        
        # Get embedding layer
        self.embedding_layer = self.model.model.embed_tokens
        print(f"  ✓ Embedding layer: {self.embedding_layer}")
        print(f"{'='*70}\n")
    
    def demonstrate_embedding_update(self):
        """
        Complete demonstration of embedding update process
        """
        print(f"\n{'='*70}")
        print(f"DEMONSTRATION: EMBEDDING UPDATE PIPELINE")
        print(f"{'='*70}\n")
        
        # ====================================================================
        # STEP 1: Initial Setup
        # ====================================================================
        print(f"STEP 1: INITIAL SETUP")
        print(f"-" * 70)
        
        prompt = "The future of AI is"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prefix_len = input_ids.shape[1]
        
        print(f"Prompt: '{prompt}'")
        print(f"Input IDs: {input_ids}")
        print(f"Prefix length: {prefix_len}")
        
        # Build tree
        nodes, parent_ids = build_tree_structure(width=2, depth=2)
        num_nodes = len(nodes)
        
        print(f"Tree: {num_nodes} nodes")
        print()
        
        # ====================================================================
        # STEP 2: Initial Embeddings (ZEROS for tree)
        # ====================================================================
        print(f"STEP 2: CREATE INITIAL EMBEDDINGS")
        print(f"-" * 70)
        
        # Get prefix embeddings from input_ids
        with torch.no_grad():
            prefix_embeds = self.embedding_layer(input_ids)
            hidden_dim = prefix_embeds.shape[-1]
        
        print(f"Prefix embeddings shape: {prefix_embeds.shape}")
        print(f"  = [batch=1, prefix_len={prefix_len}, hidden_dim={hidden_dim}]")
        
        # Initialize tree embeddings as ZEROS
        tree_embeds_zeros = torch.zeros(
            1, num_nodes, hidden_dim,
            dtype=self.dtype,
            device=self.device
        )
        
        print(f"\nTree embeddings (ZEROS) shape: {tree_embeds_zeros.shape}")
        print(f"  = [batch=1, num_nodes={num_nodes}, hidden_dim={hidden_dim}]")
        print(f"  All values are 0.0")
        
        # Concatenate
        initial_embeds = torch.cat([prefix_embeds, tree_embeds_zeros], dim=1)
        
        print(f"\nInitial embeddings shape: {initial_embeds.shape}")
        print(f"  = [prefix: {prefix_len}] + [tree: {num_nodes}]")
        print(f"  = [{prefix_len + num_nodes} total]")
        print()
        
        # ====================================================================
        # STEP 3: Generate Logits (ONE PASS)
        # ====================================================================
        print(f"STEP 3: GENERATE LOGITS (ONE PASS WITH ZEROS)")
        print(f"-" * 70)
        
        # Create simple attention mask (for demo)
        attention_mask = self._create_simple_mask(parent_ids, prefix_len)
        position_ids = torch.arange(prefix_len + num_nodes, device=self.device).unsqueeze(0)
        
        print(f"Running forward pass with ZERO tree embeddings...")
        
        with torch.no_grad():
            outputs = self.model.model(
                inputs_embeds=initial_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True
            )
            
            logits = self.model.lm_head(outputs.last_hidden_state)
        
        tree_logits = logits[0, prefix_len:, :]
        
        print(f"  ✓ Generated logits")
        print(f"  Tree logits shape: {tree_logits.shape}")
        print(f"  = [num_nodes={num_nodes}, vocab_size={tree_logits.shape[1]}]")
        print()
        
        # ====================================================================
        # STEP 4: Sample Token IDs
        # ====================================================================
        print(f"STEP 4: SAMPLE TOKEN IDS FROM LOGITS")
        print(f"-" * 70)
        
        sampled_token_ids = []
        
        print(f"Sampling tokens for each node:")
        for node_idx, node in enumerate(nodes):
            node_logits = tree_logits[node_idx]
            
            # Top-k sampling
            top_k = 4
            temperature = 0.8
            
            scaled = node_logits / temperature
            top_k_vals, top_k_idx = torch.topk(scaled, k=top_k)
            probs = F.softmax(top_k_vals, dim=-1)
            sampled = torch.multinomial(probs, 1).item()
            token_id = top_k_idx[sampled].item()
            
            sampled_token_ids.append(token_id)
            node.token_id = token_id
            node.confidence = probs[sampled].item()
            
            token_text = self.tokenizer.decode([token_id])
            print(f"  Node {node_idx}: token_id={token_id:5d} → '{token_text}'")
        
        # Convert to tensor
        tree_token_ids = torch.tensor(
            [sampled_token_ids], 
            dtype=torch.long, 
            device=self.device
        )
        
        print(f"\nTree token IDs tensor: {tree_token_ids.shape}")
        print(f"  = [batch=1, num_nodes={num_nodes}]")
        print(f"  Values: {tree_token_ids[0].tolist()}")
        print()
        
        # ====================================================================
        # STEP 5: UPDATE EMBEDDINGS (KEY STEP!)
        # ====================================================================
        print(f"STEP 5: UPDATE EMBEDDINGS FROM TOKEN IDS")
        print(f"-" * 70)
        print(f"This is the KEY step!")
        print()
        
        print(f"Method 1: Use embedding layer directly")
        print(f"-" * 40)
        
        # Convert token IDs to embeddings
        with torch.no_grad():
            tree_embeds_updated = self.embedding_layer(tree_token_ids)
        
        print(f"Input:  tree_token_ids = {tree_token_ids.shape}")
        print(f"        = [batch=1, num_nodes={num_nodes}]")
        print(f"        Values: {tree_token_ids[0, :5].tolist()} ...")
        print()
        print(f"Output: tree_embeds_updated = {tree_embeds_updated.shape}")
        print(f"        = [batch=1, num_nodes={num_nodes}, hidden_dim={hidden_dim}]")
        print(f"        First embedding (node 0):")
        print(f"        {tree_embeds_updated[0, 0, :5]} ...")
        print()
        
        print(f"Method 2: Build complete input_ids and get embeddings")
        print(f"-" * 40)
        
        # Concatenate prefix + tree token IDs
        complete_input_ids = torch.cat([input_ids, tree_token_ids], dim=1)
        
        print(f"Complete input_ids: {complete_input_ids.shape}")
        print(f"  = [prefix: {prefix_len}] + [tree: {num_nodes}]")
        print(f"  = [{complete_input_ids.shape[1]} total]")
        print(f"  Values: {complete_input_ids[0].tolist()}")
        print()
        
        # Get embeddings for complete sequence
        with torch.no_grad():
            complete_embeds = self.embedding_layer(complete_input_ids)
        
        print(f"Complete embeddings: {complete_embeds.shape}")
        print(f"  = [batch=1, total_len={complete_input_ids.shape[1]}, hidden_dim={hidden_dim}]")
        print()
        
        # ====================================================================
        # STEP 6: Comparison - ZEROS vs UPDATED
        # ====================================================================
        print(f"STEP 6: COMPARISON - ZEROS vs UPDATED EMBEDDINGS")
        print(f"-" * 70)
        
        print(f"BEFORE (zeros):")
        print(f"  tree_embeds_zeros[0, 0, :5] = {tree_embeds_zeros[0, 0, :5]}")
        print(f"  All zeros! ✗")
        print()
        
        print(f"AFTER (updated with sampled tokens):")
        print(f"  tree_embeds_updated[0, 0, :5] = {tree_embeds_updated[0, 0, :5]}")
        print(f"  Real embeddings! ✓")
        print()
        
        # ====================================================================
        # STEP 7: Use Updated Embeddings
        # ====================================================================
        print(f"STEP 7: USING UPDATED EMBEDDINGS")
        print(f"-" * 70)
        
        # Concatenate prefix + updated tree embeddings
        final_embeds = torch.cat([prefix_embeds, tree_embeds_updated], dim=1)
        
        print(f"Final embeddings: {final_embeds.shape}")
        print(f"  = [prefix embeds] + [updated tree embeds]")
        print()
        
        print(f"Use cases for updated embeddings:")
        print(f"  1. Next iteration of tree generation")
        print(f"  2. Verification with target model")
        print(f"  3. Building complete sequence")
        print(f"  4. Computing acceptance probabilities")
        print()
        
        # Example: Use for verification
        print(f"Example: Verification with target model")
        print(f"-" * 40)
        
        with torch.no_grad():
            verify_outputs = self.model.model(
                inputs_embeds=final_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True
            )
            
            verify_logits = self.model.lm_head(verify_outputs.last_hidden_state)
        
        print(f"Verification logits: {verify_logits.shape}")
        print(f"  Now using REAL embeddings instead of zeros")
        print(f"  Can compute target model probabilities")
        print()
        
        # ====================================================================
        # Summary
        # ====================================================================
        print(f"{'='*70}")
        print(f"SUMMARY: EMBEDDING UPDATE PROCESS")
        print(f"{'='*70}")
        print(f"""
1. Start with ZEROS for tree embeddings
   → tree_embeds = zeros([1, num_nodes, hidden_dim])

2. Generate logits with zeros
   → logits = model(concat([prefix_embeds, tree_embeds_zeros]))

3. Sample token IDs from logits
   → tree_token_ids = sample(logits)

4. UPDATE embeddings using embedding layer
   → tree_embeds_updated = embedding_layer(tree_token_ids)

5. Use updated embeddings for next iteration/verification
   → final_embeds = concat([prefix_embeds, tree_embeds_updated])

Key insight: The embedding layer converts token IDs → embeddings
        """)
        print(f"{'='*70}\n")
    
    def _create_simple_mask(self, parent_ids: List[Optional[int]], prefix_len: int):
        """Create simple attention mask for demo"""
        num_nodes = len(parent_ids)
        total_len = prefix_len + num_nodes
        
        mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=self.device)
        
        # Prefix causal
        for i in range(prefix_len):
            mask[i, :i+1] = True
        
        # Tree sees prefix + ancestors
        mask[prefix_len:, :prefix_len] = True
        for node_idx in range(num_nodes):
            q_pos = prefix_len + node_idx
            mask[q_pos, q_pos] = True
            parent_idx = parent_ids[node_idx]
            while parent_idx is not None:
                mask[q_pos, prefix_len + parent_idx] = True
                parent_idx = parent_ids[parent_idx]
        
        # Convert to additive
        attention_mask = torch.where(
            mask,
            torch.zeros(total_len, total_len, dtype=self.dtype, device=self.device),
            torch.full((total_len, total_len), float('-inf'), dtype=self.dtype, device=self.device)
        )
        
        return attention_mask.unsqueeze(0).unsqueeze(0)


# ============================================================================
# Complete Usage Example
# ============================================================================

def complete_usage_example():
    """Show complete usage in a practical scenario"""
    
    print(f"\n{'='*70}")
    print(f"COMPLETE USAGE EXAMPLE")
    print(f"{'='*70}\n")
    
    # Initialize
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    embedding_layer = model.model.embed_tokens
    
    # Setup
    prompt = "The future of AI is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    
    print(f"SCENARIO: Multi-iteration tree generation")
    print(f"-" * 70)
    print(f"Prompt: '{prompt}'")
    print()
    
    # Iteration 1: Generate first tree
    print(f"Iteration 1:")
    print(f"  1. Start: prefix_embeds + tree_embeds_ZEROS")
    
    prefix_embeds = embedding_layer(input_ids)
    tree_embeds_1 = torch.zeros(1, 7, 4096, dtype=torch.float16, device="cuda")
    
    print(f"     prefix_embeds: {prefix_embeds.shape}")
    print(f"     tree_embeds_1 (zeros): {tree_embeds_1.shape}")
    
    # Simulate sampling (in real code, this would be from model output)
    sampled_tokens_1 = torch.randint(0, 32000, (1, 7), device="cuda")
    
    print(f"  2. Sample tokens: {sampled_tokens_1[0].tolist()}")
    
    # UPDATE embeddings
    with torch.no_grad():
        tree_embeds_1_updated = embedding_layer(sampled_tokens_1)
    
    print(f"  3. UPDATE embeddings: {tree_embeds_1_updated.shape}")
    print(f"     BEFORE: all zeros")
    print(f"     AFTER:  real embeddings from token IDs")
    print()
    
    # Iteration 2: Use updated embeddings as new prefix
    print(f"Iteration 2:")
    print(f"  1. Use previous output as new prefix")
    
    # Concatenate prefix + previous tree
    new_prefix_ids = torch.cat([input_ids, sampled_tokens_1], dim=1)
    new_prefix_embeds = torch.cat([prefix_embeds, tree_embeds_1_updated], dim=1)
    
    print(f"     new_prefix_ids: {new_prefix_ids.shape}")
    print(f"     new_prefix_embeds: {new_prefix_embeds.shape}")
    
    # New tree
    tree_embeds_2 = torch.zeros(1, 7, 4096, dtype=torch.float16, device="cuda")
    
    print(f"  2. Generate new tree with zeros: {tree_embeds_2.shape}")
    print(f"  3. Repeat update process...")
    print()
    
    print(f"KEY INSIGHT:")
    print(f"  embedding_layer(token_ids) converts IDs → embeddings")
    print(f"  This allows us to build up the sequence iteratively")
    print(f"{'='*70}\n")


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("EMBEDDING UPDATE DEMONSTRATION")
    print("="*70)
    
    # Run demonstration
    updater = EmbeddingUpdater()
    updater.demonstrate_embedding_update()
    
    # Show complete usage
    complete_usage_example()
    
    print("="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
1. START: tree_embeds = zeros (placeholder)

2. GENERATE: logits = model(prefix_embeds + tree_embeds_zeros)

3. SAMPLE: token_ids = sample(logits)

4. UPDATE: tree_embeds = embedding_layer(token_ids)

5. USE: For next iteration, verification, or final output

The embedding_layer is the KEY:
  Input:  token_ids [batch, num_tokens]
  Output: embeddings [batch, num_tokens, hidden_dim]
    """)
    print("="*70 + "\n")


if __name__ == "__main__":
    main()