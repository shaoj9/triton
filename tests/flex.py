"""
FlexAttention for Draft Token Tree Generation in ONE Pass
=========================================================

This shows the EXACT code where FlexAttention helps generate
a draft token tree in one pass.

Key: FlexAttention uses score_mod functions to define custom
attention patterns without materializing full masks.

Usage:
    python flex_attention_draft_tree.py
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Callable


# ============================================================================
# THE KEY: FlexAttention score_mod Function
# ============================================================================

def create_tree_score_mod(
    parent_ids: List[Optional[int]],
    prefix_len: int
) -> Callable:
    """
    This is THE CORE of FlexAttention for tree generation!
    
    Creates a score_mod function that defines tree attention pattern:
    - Each node can see: prefix + ancestors
    - Each node CANNOT see: siblings, cousins, descendants
    
    Args:
        parent_ids: [None, 0, 0, 1, 1, 2, 2, ...]
                    Parent ID for each tree node
        prefix_len: Length of prefix (prompt)
    
    Returns:
        score_mod: Function for flex_attention
    """
    num_nodes = len(parent_ids)
    
    # Pre-compute ancestor chains for efficiency
    print(f"\n{'='*70}")
    print(f"CREATING FLEX ATTENTION SCORE_MOD")
    print(f"{'='*70}")
    print(f"Prefix length: {prefix_len}")
    print(f"Tree nodes: {num_nodes}")
    
    ancestor_chains = []
    for node_idx in range(num_nodes):
        ancestors = set([node_idx])  # Include self
        parent_idx = parent_ids[node_idx]
        
        # Walk up to root
        while parent_idx is not None:
            ancestors.add(parent_idx)
            parent_idx = parent_ids[parent_idx]
        
        ancestor_chains.append(ancestors)
    
    print(f"\nAncestor chains computed:")
    for i in range(min(5, num_nodes)):
        print(f"  Node {i}: ancestors = {sorted(ancestor_chains[i])}")
    
    # THIS IS THE CORE FLEXATTENTION FUNCTION
    def tree_score_mod(score, b, h, q_idx, kv_idx):
        """
        FlexAttention score modifier
        
        This function is called for EVERY attention computation
        to determine if query at q_idx can attend to key at kv_idx.
        
        Args:
            score: Raw attention score
            b: Batch index
            h: Head index
            q_idx: Query position (0 to prefix_len + num_nodes - 1)
            kv_idx: Key position (0 to prefix_len + num_nodes - 1)
        
        Returns:
            score: If can attend
            -inf: If should mask
        """
        # ================================================================
        # CASE 1: Prefix Region (Causal Attention)
        # ================================================================
        if q_idx < prefix_len:
            # Prefix tokens use standard causal attention
            if kv_idx <= q_idx:
                return score  # Can attend
            else:
                return float('-inf')  # Mask future
        
        # ================================================================
        # CASE 2: Tree Region (Custom Tree Attention)
        # ================================================================
        else:
            tree_q_idx = q_idx - prefix_len
            
            # Sub-case 2a: Tree attending to prefix
            if kv_idx < prefix_len:
                return score  # All tree nodes see entire prefix
            
            # Sub-case 2b: Tree attending to tree
            tree_kv_idx = kv_idx - prefix_len
            
            # Check if kv is ancestor of q
            if tree_kv_idx in ancestor_chains[tree_q_idx]:
                return score  # Can attend to ancestor
            else:
                return float('-inf')  # MASK sibling/cousin
    
    print(f"\n✓ score_mod function created")
    print(f"  This function will be called for every attention computation")
    print(f"  Total attention ops: {(prefix_len + num_nodes) ** 2}")
    print(f"{'='*70}\n")
    
    return tree_score_mod


# ============================================================================
# Draft Tree Generation with FlexAttention
# ============================================================================

def generate_draft_tree_with_flex_attention(
    model,
    tokenizer,
    prompt: str,
    tree_width: int = 2,
    tree_depth: int = 3,
    device: str = "cuda"
):
    """
    Generate draft token tree in ONE PASS using FlexAttention
    
    This is the complete workflow showing where FlexAttention helps.
    
    Returns:
        draft_tokens: [num_nodes] sampled token IDs
        draft_logits: [num_nodes, vocab_size] logits for all nodes
    """
    print(f"\n{'='*70}")
    print(f"GENERATING DRAFT TREE WITH FLEXATTENTION")
    print(f"{'='*70}")
    print(f"Prompt: '{prompt}'")
    print(f"Tree: width={tree_width}, depth={tree_depth}")
    
    # ========================================================================
    # Step 1: Build Tree Structure
    # ========================================================================
    print(f"\nStep 1: Build tree structure...")
    
    # Calculate total nodes
    num_nodes = sum(tree_width**d for d in range(tree_depth + 1))
    
    # Build parent_ids
    parent_ids = [None]  # Root has no parent
    current_level = [0]
    next_id = 1
    
    for d in range(1, tree_depth + 1):
        next_level = []
        for parent in current_level:
            for _ in range(tree_width):
                parent_ids.append(parent)
                next_level.append(next_id)
                next_id += 1
        current_level = next_level
    
    print(f"  Total nodes: {num_nodes}")
    print(f"  Parent IDs: {parent_ids[:10]}...")
    
    # ========================================================================
    # Step 2: Tokenize Prefix
    # ========================================================================
    print(f"\nStep 2: Tokenize prefix...")
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prefix_len = input_ids.shape[1]
    
    print(f"  Prefix length: {prefix_len}")
    print(f"  Input IDs: {input_ids[0].tolist()}")
    
    # ========================================================================
    # Step 3: Create FlexAttention score_mod (THE KEY!)
    # ========================================================================
    print(f"\nStep 3: Create FlexAttention score_mod...")
    
    score_mod = create_tree_score_mod(parent_ids, prefix_len)
    
    print(f"  ✓ score_mod function ready")
    print(f"  This defines the tree attention pattern!")
    
    # ========================================================================
    # Step 4: Prepare Embeddings
    # ========================================================================
    print(f"\nStep 4: Prepare embeddings...")
    
    embedding_layer = model.model.embed_tokens
    
    # Prefix embeddings
    prefix_embeds = embedding_layer(input_ids)
    hidden_dim = prefix_embeds.shape[-1]
    
    # Tree embeddings (zeros initially)
    tree_embeds = torch.zeros(
        1, num_nodes, hidden_dim,
        dtype=torch.float16,
        device=device
    )
    
    # Concatenate
    full_embeds = torch.cat([prefix_embeds, tree_embeds], dim=1)
    
    print(f"  Prefix embeds: {prefix_embeds.shape}")
    print(f"  Tree embeds (zeros): {tree_embeds.shape}")
    print(f"  Full embeds: {full_embeds.shape}")
    
    # ========================================================================
    # Step 5: Position IDs
    # ========================================================================
    print(f"\nStep 5: Create position IDs...")
    
    position_ids = torch.arange(
        prefix_len + num_nodes,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)
    
    print(f"  Position IDs: {position_ids.shape}")
    print(f"  Values: {position_ids[0, :10].tolist()}...")
    
    # ========================================================================
    # Step 6: ONE FORWARD PASS with FlexAttention
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"Step 6: FORWARD PASS WITH FLEXATTENTION")
    print(f"{'='*70}")
    
    # NOTE: In practice, you'd need to patch the model's attention layers
    # to use flex_attention with the score_mod function.
    # For demonstration, we show the conceptual usage:
    
    print(f"\nConceptual FlexAttention usage:")
    print(f"```python")
    print(f"from torch.nn.attention.flex_attention import flex_attention")
    print(f"")
    print(f"# Inside attention layer:")
    print(f"attn_output = flex_attention(")
    print(f"    query,")
    print(f"    key,")
    print(f"    value,")
    print(f"    score_mod=tree_score_mod,  # Our custom function!")
    print(f"    enable_gqa=True")
    print(f")")
    print(f"```")
    
    # For this demo, we'll use standard attention with a mask
    # (FlexAttention would be more memory efficient)
    print(f"\nUsing standard attention for demonstration...")
    
    # Build mask from score_mod
    total_len = prefix_len + num_nodes
    mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
    
    for q in range(total_len):
        for kv in range(total_len):
            score = score_mod(0.0, 0, 0, q, kv)
            mask[q, kv] = (score != float('-inf'))
    
    # Convert to additive
    attention_mask = torch.where(
        mask,
        torch.zeros(total_len, total_len, dtype=torch.float16, device=device),
        torch.full((total_len, total_len), float('-inf'), dtype=torch.float16, device=device)
    ).unsqueeze(0).unsqueeze(0)
    
    print(f"  Attention mask: {attention_mask.shape}")
    
    # Forward through model
    print(f"  Running forward pass...")
    
    with torch.no_grad():
        outputs = model.model(
            inputs_embeds=full_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True
        )
        
        logits = model.lm_head(outputs.last_hidden_state)
    
    # Extract tree logits
    draft_logits = logits[0, prefix_len:, :]
    
    print(f"  ✓ Forward complete!")
    print(f"  Draft logits: {draft_logits.shape}")
    print(f"  = [{num_nodes} nodes, {draft_logits.shape[1]} vocab]")
    
    # ========================================================================
    # Step 7: Sample Draft Tokens
    # ========================================================================
    print(f"\nStep 7: Sample draft tokens...")
    
    draft_tokens = []
    
    for node_idx in range(num_nodes):
        node_logits = draft_logits[node_idx]
        probs = F.softmax(node_logits, dim=-1)
        token_id = torch.multinomial(probs, 1).item()
        draft_tokens.append(token_id)
    
    draft_tokens = torch.tensor(draft_tokens, device=device)
    
    print(f"  ✓ Sampled {num_nodes} tokens")
    print(f"  Draft tokens: {draft_tokens[:10].tolist()}...")
    
    print(f"\n{'='*70}")
    print(f"SUCCESS: GENERATED {num_nodes} DRAFT TOKENS IN ONE PASS!")
    print(f"{'='*70}\n")
    
    return draft_tokens, draft_logits


# ============================================================================
# Comparison: FlexAttention vs Standard Attention
# ============================================================================

def compare_flex_vs_standard():
    """Show the difference between FlexAttention and standard attention"""
    
    print(f"\n{'='*70}")
    print(f"FLEXATTENTION vs STANDARD ATTENTION")
    print(f"{'='*70}\n")
    
    prefix_len = 10
    num_nodes = 15
    total_len = prefix_len + num_nodes  # 25
    
    print(f"Setup:")
    print(f"  Prefix length: {prefix_len}")
    print(f"  Tree nodes: {num_nodes}")
    print(f"  Total length: {total_len}")
    print()
    
    # Standard attention
    print(f"Standard Attention:")
    print(f"  Needs full mask: [{total_len}, {total_len}]")
    print(f"  Memory: {total_len * total_len * 2} bytes (float16)")
    print(f"  = {total_len * total_len * 2 / 1024:.1f} KB")
    print()
    
    # FlexAttention
    print(f"FlexAttention:")
    print(f"  Needs score_mod function (no mask storage!)")
    print(f"  Memory: ~0 bytes for mask")
    print(f"  Computation: On-the-fly per attention op")
    print()
    
    # Larger example
    prefix_len = 100
    num_nodes = 121  # width=3, depth=4
    total_len = prefix_len + num_nodes  # 221
    
    print(f"Larger Tree Example:")
    print(f"  Prefix: {prefix_len}, Nodes: {num_nodes}, Total: {total_len}")
    print()
    
    standard_memory = total_len * total_len * 2
    print(f"Standard Attention:")
    print(f"  Mask size: [{total_len}, {total_len}]")
    print(f"  Memory: {standard_memory:,} bytes = {standard_memory / 1024 / 1024:.2f} MB")
    print()
    
    print(f"FlexAttention:")
    print(f"  Mask size: None (computed on-the-fly)")
    print(f"  Memory: ~0 bytes")
    print(f"  Savings: {standard_memory / 1024 / 1024:.2f} MB!")
    print()
    
    print(f"{'='*70}\n")


# ============================================================================
# Main Demo
# ============================================================================

def main():
    """Main demonstration"""
    
    print("\n" + "="*70)
    print("FLEXATTENTION FOR DRAFT TOKEN TREE GENERATION")
    print("="*70)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load model
    print(f"\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        torch_dtype=torch.float16,
        device_map="cuda",
        low_cpu_mem_usage=True
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    print(f"  ✓ Model loaded")
    
    # Generate draft tree
    prompt = "The future of AI is"
    
    draft_tokens, draft_logits = generate_draft_tree_with_flex_attention(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        tree_width=2,
        tree_depth=3,
        device="cuda"
    )
    
    # Decode tokens
    print(f"Draft tokens decoded:")
    for i in range(min(10, len(draft_tokens))):
        token_text = tokenizer.decode([draft_tokens[i]])
        print(f"  [{i}] {draft_tokens[i]:5d} → '{token_text}'")
    
    # Show comparison
    compare_flex_vs_standard()
    
    print(f"{'='*70}")
    print(f"KEY TAKEAWAY")
    print(f"{'='*70}")
    print(f"""
FlexAttention helps by:

1. MEMORY: No need to store full attention mask
   - Standard: O(n²) memory
   - Flex: O(1) memory

2. FLEXIBILITY: Define attention pattern via function
   - Standard: Build explicit mask tensor
   - Flex: score_mod function called per operation

3. EFFICIENCY: Sparse patterns computed on-the-fly
   - Standard: Materialize full mask
   - Flex: Only compute what's needed

For tree generation:
- score_mod defines "node sees ancestors only"
- Called automatically during attention
- No explicit mask storage needed!
    """)
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Check for FlexAttention
    try:
        from torch.nn.attention.flex_attention import flex_attention
        print("✓ FlexAttention available!")
    except ImportError:
        print("⚠ FlexAttention not available (requires PyTorch 2.5+)")
        print("  Demonstration will use standard attention with mask")
    
    main()
    
    print("="*70)
    print("FLEXATTENTION CODE SUMMARY")
    print("="*70)
    print("""
The key code for FlexAttention in tree generation:

1. DEFINE score_mod function:
   def tree_score_mod(score, b, h, q_idx, kv_idx):
       if q_idx < prefix_len:
           return score if kv_idx <= q_idx else -inf
       else:
           tree_q = q_idx - prefix_len
           if kv_idx < prefix_len:
               return score
           tree_kv = kv_idx - prefix_len
           return score if tree_kv in ancestors[tree_q] else -inf

2. USE in attention:
   from torch.nn.attention.flex_attention import flex_attention
   
   attn_output = flex_attention(
       query, key, value,
       score_mod=tree_score_mod  # Custom tree pattern!
   )

3. RESULT: Tree structure enforced without explicit masks!
    """)
    print("="*70 + "\n")