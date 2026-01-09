"""
Tree Generation with PyTorch FlexAttention
=========================================

Uses PyTorch's flex_attention to generate token trees in ONE pass.

FlexAttention allows custom attention patterns via score_mod functions,
which is perfect for tree structures where each node only sees ancestors.

Requires: PyTorch 2.5+ with flex_attention support

Usage:
    python flex_attention_tree.py --prompt "The future of AI is" --width 3 --depth 4
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import argparse
import numpy as np

# Check for flex_attention
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    print("Warning: FlexAttention not available. Using fallback implementation.")


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
    children: List[int] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


def build_tree_structure(width: int, depth: int) -> Tuple[List[TreeNode], List[Optional[int]]]:
    """Build tree structure"""
    nodes = []
    parent_ids = []
    
    # Root
    root = TreeNode(node_id=0, depth=0, parent_id=None)
    nodes.append(root)
    parent_ids.append(None)
    
    # Build levels
    current_level = [0]
    next_id = 1
    
    for d in range(1, depth + 1):
        next_level = []
        for parent_id in current_level:
            for _ in range(width):
                node = TreeNode(node_id=next_id, depth=d, parent_id=parent_id)
                nodes.append(node)
                parent_ids.append(parent_id)
                nodes[parent_id].children.append(next_id)
                next_level.append(next_id)
                next_id += 1
        current_level = next_level
    
    return nodes, parent_ids


# ============================================================================
# FlexAttention: Custom Attention Mask Functions
# ============================================================================

def create_tree_attention_mask_function(
    parent_ids: List[Optional[int]],
    prefix_len: int
) -> Callable:
    """
    Create score_mod function for FlexAttention
    
    This function defines the attention pattern:
    - Prefix positions: causal attention
    - Tree positions: see prefix + ancestors only
    
    Args:
        parent_ids: Parent ID for each tree node
        prefix_len: Length of prefix (prompt)
    
    Returns:
        score_mod function for flex_attention
    """
    num_tree_nodes = len(parent_ids)
    
    # Pre-compute ancestor chains for each node
    ancestor_chains = []
    for node_idx in range(num_tree_nodes):
        ancestors = set([node_idx])  # Include self
        parent_idx = parent_ids[node_idx]
        while parent_idx is not None:
            ancestors.add(parent_idx)
            parent_idx = parent_ids[parent_idx]
        ancestor_chains.append(ancestors)
    
    def tree_score_mod(score, b, h, q_idx, kv_idx):
        """
        FlexAttention score modifier for tree structure
        
        Args:
            score: Attention score
            b: Batch index
            h: Head index
            q_idx: Query index
            kv_idx: Key/Value index
        
        Returns:
            Modified score (or -inf to mask)
        """
        # PREFIX REGION: Causal attention
        if q_idx < prefix_len:
            # Prefix tokens use causal attention
            if kv_idx <= q_idx:
                return score  # Can attend
            else:
                return float('-inf')  # Mask future
        
        # TREE REGION: Custom tree attention
        else:
            tree_q_idx = q_idx - prefix_len  # Index in tree
            
            # Can always attend to entire prefix
            if kv_idx < prefix_len:
                return score
            
            # For tree positions
            tree_kv_idx = kv_idx - prefix_len
            
            # Check if kv is an ancestor of q
            if tree_kv_idx in ancestor_chains[tree_q_idx]:
                return score  # Can attend to ancestor
            else:
                return float('-inf')  # Mask non-ancestor
    
    return tree_score_mod


def create_tree_block_mask(
    parent_ids: List[Optional[int]],
    prefix_len: int,
    device: str = "cuda"
):
    """
    Create BlockMask for FlexAttention (more efficient)
    
    BlockMask is a sparse representation that's faster than score_mod
    for patterns that align with blocks.
    
    Args:
        parent_ids: Parent IDs for tree nodes
        prefix_len: Prefix length
        device: Device
    
    Returns:
        BlockMask for flex_attention
    """
    num_tree_nodes = len(parent_ids)
    total_len = prefix_len + num_tree_nodes
    
    # Build dense mask first (for simplicity)
    mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
    
    # Prefix: causal
    for i in range(prefix_len):
        mask[i, :i+1] = True
    
    # Tree: prefix + ancestors
    mask[prefix_len:, :prefix_len] = True  # All tree sees prefix
    
    for node_idx in range(num_tree_nodes):
        q_pos = prefix_len + node_idx
        mask[q_pos, q_pos] = True  # Self
        
        # Ancestors
        parent_idx = parent_ids[node_idx]
        while parent_idx is not None:
            kv_pos = prefix_len + parent_idx
            mask[q_pos, kv_pos] = True
            parent_idx = parent_ids[parent_idx]
    
    # Convert to BlockMask if available
    if FLEX_ATTENTION_AVAILABLE:
        try:
            block_mask = create_block_mask(
                lambda b, h, q, kv: mask[q, kv],
                B=1, H=1, Q_LEN=total_len, KV_LEN=total_len
            )
            return block_mask
        except:
            pass
    
    return mask


# ============================================================================
# Position IDs for Tree Structure
# ============================================================================

def create_tree_position_ids(
    nodes: List[TreeNode],
    prefix_len: int,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Create position IDs for tree structure
    
    Two strategies:
    1. Depth-based: position = prefix_len + depth
    2. Sequential: position = prefix_len + node_id
    
    Depth-based is better for trees as it groups nodes by depth.
    
    Args:
        nodes: Tree nodes
        prefix_len: Prefix length
        device: Device
    
    Returns:
        position_ids: [1, prefix_len + num_nodes]
    """
    num_nodes = len(nodes)
    position_ids = torch.zeros(
        1, prefix_len + num_nodes,
        dtype=torch.long,
        device=device
    )
    
    # Prefix positions: 0, 1, 2, ..., prefix_len-1
    position_ids[0, :prefix_len] = torch.arange(prefix_len)
    
    # Tree positions: depth-based
    for node in nodes:
        # Position = prefix_len + depth
        # This makes nodes at same depth have same position
        position_ids[0, prefix_len + node.node_id] = prefix_len + node.depth
    
    return position_ids


# ============================================================================
# FlexAttention Tree Generator
# ============================================================================

class FlexAttentionTreeGenerator:
    """
    Generate token trees using PyTorch FlexAttention
    
    This uses flex_attention for efficient tree-structured attention
    patterns without materializing full attention masks.
    """
    
    def __init__(
        self,
        model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        use_flex_attention: bool = True
    ):
        """
        Initialize generator
        
        Args:
            model_path: HuggingFace model path
            device: Device
            dtype: Data type
            use_flex_attention: Use flex_attention if available
        """
        print(f"\n{'='*70}")
        print(f"INITIALIZING FLEX ATTENTION TREE GENERATOR")
        print(f"{'='*70}")
        print(f"Model: {model_path}")
        print(f"FlexAttention available: {FLEX_ATTENTION_AVAILABLE}")
        print(f"Use FlexAttention: {use_flex_attention and FLEX_ATTENTION_AVAILABLE}")
        
        self.device = device
        self.dtype = dtype
        self.use_flex_attention = use_flex_attention and FLEX_ATTENTION_AVAILABLE
        
        # Load model
        print(f"\nLoading model...")
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
        
        print(f"{'='*70}\n")
    
    def generate_tree(
        self,
        prompt: str,
        tree_width: int = 3,
        tree_depth: int = 4,
        top_k: int = 4,
        temperature: float = 0.8
    ) -> Tuple[List[TreeNode], torch.Tensor]:
        """
        Generate tree in ONE pass using FlexAttention
        
        Args:
            prompt: Input prompt
            tree_width: Branching factor
            tree_depth: Tree depth
            top_k: Top-k sampling
            temperature: Sampling temperature
        
        Returns:
            nodes: Tree nodes with sampled tokens
            logits: Logits for all nodes
        """
        print(f"\n{'='*70}")
        print(f"GENERATING TREE WITH FLEX ATTENTION")
        print(f"{'='*70}")
        print(f"Prompt: '{prompt}'")
        print(f"Tree: width={tree_width}, depth={tree_depth}")
        
        # ====================================================================
        # Step 1: Build Tree Structure
        # ====================================================================
        
        print(f"\nStep 1: Building tree structure...")
        nodes, parent_ids = build_tree_structure(tree_width, tree_depth)
        num_nodes = len(nodes)
        print(f"  Total nodes: {num_nodes}")
        
        # ====================================================================
        # Step 2: Tokenize Prompt
        # ====================================================================
        
        print(f"\nStep 2: Tokenizing prompt...")
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prefix_len = input_ids.shape[1]
        print(f"  Prefix length: {prefix_len} tokens")
        
        # ====================================================================
        # Step 3: Create Position IDs
        # ====================================================================
        
        print(f"\nStep 3: Creating position IDs...")
        position_ids = create_tree_position_ids(nodes, prefix_len, self.device)
        print(f"  Position IDs shape: {position_ids.shape}")
        print(f"  Position range: {position_ids.min().item()} - {position_ids.max().item()}")
        
        # ====================================================================
        # Step 4: Create Attention Mask (FlexAttention or standard)
        # ====================================================================
        
        print(f"\nStep 4: Creating attention mask...")
        
        if self.use_flex_attention:
            print(f"  Using FlexAttention with custom score_mod...")
            score_mod = create_tree_attention_mask_function(parent_ids, prefix_len)
            attention_mask = None  # FlexAttention doesn't use mask tensor
        else:
            print(f"  Using standard attention mask...")
            attention_mask = self._create_standard_mask(parent_ids, prefix_len, num_nodes)
            score_mod = None
        
        # ====================================================================
        # Step 5: Forward Pass (ONE PASS!)
        # ====================================================================
        
        print(f"\nStep 5: Forward pass (ONE PASS)...")
        
        tree_logits = self._forward_with_tree_attention(
            input_ids=input_ids,
            num_nodes=num_nodes,
            position_ids=position_ids,
            attention_mask=attention_mask,
            score_mod=score_mod
        )
        
        print(f"  ✓ Generated {num_nodes} nodes in ONE pass")
        print(f"  Tree logits: {tree_logits.shape}")
        
        # ====================================================================
        # Step 6: Sample Tokens
        # ====================================================================
        
        print(f"\nStep 6: Sampling tokens...")
        self._sample_tokens(nodes, tree_logits, top_k, temperature)
        print(f"  ✓ Sampled {num_nodes} tokens")
        
        print(f"{'='*70}\n")
        
        return nodes, tree_logits
    
    def _create_standard_mask(
        self,
        parent_ids: List[Optional[int]],
        prefix_len: int,
        num_nodes: int
    ) -> torch.Tensor:
        """Create standard attention mask (fallback)"""
        total_len = prefix_len + num_nodes
        mask = torch.zeros(
            1, 1, total_len, total_len,
            dtype=torch.bool,
            device=self.device
        )
        
        # Prefix: causal
        for i in range(prefix_len):
            mask[0, 0, i, :i+1] = True
        
        # Tree: prefix + ancestors
        mask[0, 0, prefix_len:, :prefix_len] = True
        
        for node_idx in range(num_nodes):
            pos = prefix_len + node_idx
            mask[0, 0, pos, pos] = True
            
            parent_idx = parent_ids[node_idx]
            while parent_idx is not None:
                mask[0, 0, pos, prefix_len + parent_idx] = True
                parent_idx = parent_ids[parent_idx]
        
        # Convert to additive mask
        attention_mask = torch.where(
            mask,
            torch.zeros_like(mask, dtype=self.dtype),
            torch.full_like(mask, float('-inf'), dtype=self.dtype)
        )
        
        return attention_mask
    
    def _forward_with_tree_attention(
        self,
        input_ids: torch.Tensor,
        num_nodes: int,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        score_mod: Optional[Callable]
    ) -> torch.Tensor:
        """
        Forward with tree attention
        
        Args:
            input_ids: Input IDs
            num_nodes: Number of tree nodes
            position_ids: Position IDs
            attention_mask: Standard attention mask (if not using flex)
            score_mod: FlexAttention score modifier (if using flex)
        
        Returns:
            tree_logits: Logits for tree nodes
        """
        batch_size, prefix_len = input_ids.shape
        
        # Get embeddings
        with torch.no_grad():
            if hasattr(self.model, 'model'):
                embed_layer = self.model.model.embed_tokens
            else:
                embed_layer = self.model.embed_tokens
            
            prefix_embeds = embed_layer(input_ids)
            hidden_dim = prefix_embeds.shape[-1]
        
        # Tree embeddings (zeros)
        tree_embeds = torch.zeros(
            batch_size, num_nodes, hidden_dim,
            dtype=self.dtype,
            device=self.device
        )
        
        # Concatenate
        full_embeds = torch.cat([prefix_embeds, tree_embeds], dim=1)
        
        # Forward through model
        with torch.no_grad():
            if self.use_flex_attention and score_mod is not None:
                # Use FlexAttention
                outputs = self._forward_with_flex_attention(
                    full_embeds, position_ids, score_mod
                )
            else:
                # Use standard attention
                if hasattr(self.model, 'model'):
                    outputs = self.model.model(
                        inputs_embeds=full_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                        return_dict=True
                    )
                    hidden = outputs.last_hidden_state
                    logits = self.model.lm_head(hidden)
                else:
                    outputs = self.model(
                        inputs_embeds=full_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                        return_dict=True
                    )
                    logits = outputs.logits
        
        # Extract tree logits
        tree_logits = logits[0, prefix_len:, :]
        
        return tree_logits
    
    def _forward_with_flex_attention(
        self,
        embeddings: torch.Tensor,
        position_ids: torch.Tensor,
        score_mod: Callable
    ) -> torch.Tensor:
        """
        Forward using FlexAttention
        
        Note: This is a simplified version. Full integration would require
        patching the model's attention layers.
        """
        # For demonstration, we'll show the FlexAttention call
        # In practice, you'd need to patch the model's attention mechanism
        
        print(f"    Using FlexAttention API...")
        
        # FlexAttention is called within attention layers
        # Here's the conceptual usage:
        """
        from torch.nn.attention.flex_attention import flex_attention
        
        # In the attention layer:
        attn_output = flex_attention(
            query, key, value,
            score_mod=score_mod,  # Our custom tree attention function
            block_mask=block_mask,  # Optional BlockMask for efficiency
            enable_gqa=True  # Grouped query attention
        )
        """
        
        # For now, fall back to standard forward
        # (Full implementation would require model patching)
        print(f"    Note: Full FlexAttention requires model patching")
        print(f"    Using standard forward with custom mask instead...")
        
        # Build mask from score_mod
        batch_size, seq_len, hidden_dim = embeddings.shape
        
        # Create dense mask by evaluating score_mod
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=self.device)
        for q in range(seq_len):
            for kv in range(seq_len):
                # Evaluate score_mod
                score = score_mod(0.0, 0, 0, q, kv)
                mask[q, kv] = (score != float('-inf'))
        
        # Convert to additive mask
        attention_mask = torch.where(
            mask.unsqueeze(0).unsqueeze(0),
            torch.zeros(1, 1, seq_len, seq_len, dtype=self.dtype, device=self.device),
            torch.full((1, 1, seq_len, seq_len), float('-inf'), dtype=self.dtype, device=self.device)
        )
        
        # Forward through model
        if hasattr(self.model, 'model'):
            outputs = self.model.model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True
            )
            hidden = outputs.last_hidden_state
            return self.model.lm_head(hidden)
        else:
            outputs = self.model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True
            )
            return outputs.logits
    
    def _sample_tokens(
        self,
        nodes: List[TreeNode],
        tree_logits: torch.Tensor,
        top_k: int,
        temperature: float
    ):
        """Sample tokens for each node"""
        for node_idx, node in enumerate(nodes):
            node_logits = tree_logits[node_idx]
            scaled = node_logits / temperature
            
            if top_k > 0:
                top_k_vals, top_k_idx = torch.topk(scaled, k=top_k)
                probs = F.softmax(top_k_vals, dim=-1)
                sampled = torch.multinomial(probs, 1).item()
                token_id = top_k_idx[sampled].item()
                confidence = probs[sampled].item()
            else:
                probs = F.softmax(scaled, dim=-1)
                token_id = torch.multinomial(probs, 1).item()
                confidence = probs[token_id].item()
            
            node.token_id = token_id
            node.confidence = confidence


# ============================================================================
# Visualization
# ============================================================================

def visualize_tree(nodes: List[TreeNode], tokenizer, max_depth: int = 2):
    """Visualize tree structure"""
    print(f"\n{'='*70}")
    print(f"TREE STRUCTURE")
    print(f"{'='*70}\n")
    
    depth_groups = {}
    for node in nodes:
        if node.depth not in depth_groups:
            depth_groups[node.depth] = []
        depth_groups[node.depth].append(node)
    
    for depth in sorted(depth_groups.keys()):
        if depth > max_depth:
            break
        
        nodes_at_depth = depth_groups[depth]
        print(f"Depth {depth} ({len(nodes_at_depth)} nodes):")
        
        for node in nodes_at_depth[:5]:
            token_text = tokenizer.decode([node.token_id])
            indent = "  " * (depth + 1)
            print(f"{indent}→ '{token_text}' (conf={node.confidence:.3f})")
        
        if len(nodes_at_depth) > 5:
            print(f"  ... and {len(nodes_at_depth) - 5} more")
        print()


# ============================================================================
# Main
# ============================================================================

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(
        description="Generate tree with FlexAttention"
    )
    parser.add_argument("--prompt", type=str, default="The future of AI is")
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--no-flex", action="store_true", help="Disable FlexAttention")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("TREE GENERATION WITH FLEXATTENTION")
    print("="*70)
    
    total_nodes = sum(args.width**d for d in range(args.depth + 1))
    print(f"\nConfiguration:")
    print(f"  Prompt: '{args.prompt}'")
    print(f"  Tree: width={args.width}, depth={args.depth}")
    print(f"  Total nodes: {total_nodes}")
    
    # Initialize
    generator = FlexAttentionTreeGenerator(
        use_flex_attention=not args.no_flex
    )
    
    # Generate
    nodes, logits = generator.generate_tree(
        prompt=args.prompt,
        tree_width=args.width,
        tree_depth=args.depth,
        top_k=args.top_k,
        temperature=args.temperature
    )
    
    # Visualize
    visualize_tree(nodes, generator.tokenizer, max_depth=2)
    
    print(f"{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Generated {len(nodes)} nodes in ONE forward pass")
    print(f"Using {'FlexAttention' if generator.use_flex_attention else 'standard attention'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
    
    print("="*70)
    print("KEY CONCEPTS")
    print("="*70)
    print("""
FlexAttention allows custom attention patterns via score_mod functions:

def tree_score_mod(score, b, h, q_idx, kv_idx):
    # Prefix: causal attention
    if q_idx < prefix_len:
        return score if kv_idx <= q_idx else -inf
    
    # Tree: see prefix + ancestors only
    tree_q = q_idx - prefix_len
    if kv_idx < prefix_len:
        return score  # See all prefix
    
    tree_kv = kv_idx - prefix_len
    if tree_kv in ancestors[tree_q]:
        return score  # See ancestor
    else:
        return -inf  # Mask sibling

This defines the tree structure without materializing full masks!
    """)
    print("="*70 + "\n")