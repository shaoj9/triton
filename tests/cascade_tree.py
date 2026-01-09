"""
vLLM Tree-Structured Output Generation with Cascade Attention
============================================================

Generates outputs following a tree structure (tree_width, tree_depth) using
cascade attention for efficient computation.

Each node in the tree represents a different continuation of the text.

Usage:
    python vllm_tree_outputs.py \
        --prompt "The future of AI" \
        --width 3 \
        --depth 3
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import argparse
import json


# ============================================================================
# Tree Output Structure
# ============================================================================

@dataclass
class TreeOutput:
    """Single output node in the tree"""
    node_id: int
    depth: int
    parent_id: Optional[int]
    token_id: int
    token_text: str
    confidence: float
    full_text: str  # Complete text from root to this node
    children: List[int] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'node_id': self.node_id,
            'depth': self.depth,
            'parent_id': self.parent_id,
            'token_id': self.token_id,
            'token_text': self.token_text,
            'confidence': self.confidence,
            'full_text': self.full_text,
            'children': self.children
        }


@dataclass
class TreeStructure:
    """Complete tree structure of outputs"""
    outputs: List[TreeOutput]
    width: int
    depth: int
    prompt: str
    
    def get_outputs_at_depth(self, depth: int) -> List[TreeOutput]:
        """Get all outputs at a specific depth"""
        return [o for o in self.outputs if o.depth == depth]
    
    def get_leaf_outputs(self) -> List[TreeOutput]:
        """Get all leaf nodes (no children)"""
        return [o for o in self.outputs if len(o.children) == 0]
    
    def get_path_to_node(self, node_id: int) -> List[TreeOutput]:
        """Get path from root to node"""
        path = []
        current = self.outputs[node_id]
        
        while current is not None:
            path.insert(0, current)
            if current.parent_id is None:
                break
            current = self.outputs[current.parent_id]
        
        return path
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'prompt': self.prompt,
            'width': self.width,
            'depth': self.depth,
            'num_outputs': len(self.outputs),
            'outputs': [o.to_dict() for o in self.outputs]
        }


# ============================================================================
# Cascade Attention for Tree
# ============================================================================

def build_tree_cascade_metadata(
    num_nodes: int,
    parent_ids: List[Optional[int]],
    prefix_len: int,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Build cascade attention metadata for tree structure
    
    Args:
        num_nodes: Number of nodes in tree
        parent_ids: Parent ID for each node
        prefix_len: Length of prefix (prompt)
        device: Device
    
    Returns:
        Dictionary with cascade components
    """
    use_cascade = prefix_len >= 128
    
    if use_cascade:
        # Cascade: split into prefix, tree-to-prefix, tree-to-tree
        prefix_mask = torch.tril(
            torch.ones(prefix_len, prefix_len, dtype=torch.bool, device=device)
        )
        
        tree_to_prefix = torch.ones(
            num_nodes, prefix_len, dtype=torch.bool, device=device
        )
        
        tree_to_tree = torch.zeros(
            num_nodes, num_nodes, dtype=torch.bool, device=device
        )
        
        # Each node sees itself and ancestors only
        for node_idx in range(num_nodes):
            tree_to_tree[node_idx, node_idx] = True
            
            parent_idx = parent_ids[node_idx]
            while parent_idx is not None:
                tree_to_tree[node_idx, parent_idx] = True
                parent_idx = parent_ids[parent_idx]
        
        return {
            'use_cascade': True,
            'prefix_len': prefix_len,
            'prefix_mask': prefix_mask,
            'tree_to_prefix': tree_to_prefix,
            'tree_to_tree': tree_to_tree
        }
    else:
        # Standard: single full mask
        total_len = prefix_len + num_nodes
        full_mask = torch.zeros(
            total_len, total_len, dtype=torch.bool, device=device
        )
        
        # Prefix causal
        for i in range(prefix_len):
            full_mask[i, :i+1] = True
        
        # Tree sees prefix + ancestors
        full_mask[prefix_len:, :prefix_len] = True
        for node_idx in range(num_nodes):
            pos = prefix_len + node_idx
            full_mask[pos, pos] = True
            
            parent_idx = parent_ids[node_idx]
            while parent_idx is not None:
                full_mask[pos, prefix_len + parent_idx] = True
                parent_idx = parent_ids[parent_idx]
        
        return {
            'use_cascade': False,
            'prefix_len': prefix_len,
            'full_mask': full_mask
        }


# ============================================================================
# vLLM Tree Output Generator
# ============================================================================

class VLLMTreeOutputGenerator:
    """
    Generate tree-structured outputs using vLLM-style cascade attention
    
    This creates a tree of possible continuations where:
    - Each node represents a different token choice
    - Siblings represent alternative continuations
    - Children represent further continuations
    """
    
    def __init__(
        self,
        model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize generator
        
        Args:
            model_path: HuggingFace model path
            device: Device
            dtype: Data type
        """
        print(f"\n{'='*70}")
        print(f"INITIALIZING vLLM TREE OUTPUT GENERATOR")
        print(f"{'='*70}")
        print(f"Model: {model_path}")
        
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
        
        print(f"{'='*70}\n")
    
    def generate_tree_outputs(
        self,
        prompt: str,
        tree_width: int = 3,
        tree_depth: int = 3,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> TreeStructure:
        """
        Generate tree-structured outputs with cascade attention
        
        Args:
            prompt: Input prompt
            tree_width: Number of children per node (branching factor)
            tree_depth: Maximum depth of tree
            temperature: Sampling temperature (higher = more diverse)
            top_k: Top-k sampling
        
        Returns:
            TreeStructure with all outputs
        """
        print(f"\n{'='*70}")
        print(f"GENERATING TREE-STRUCTURED OUTPUTS")
        print(f"{'='*70}")
        print(f"Prompt: '{prompt}'")
        print(f"Tree structure: width={tree_width}, depth={tree_depth}")
        
        # ====================================================================
        # Step 1: Build Tree Topology
        # ====================================================================
        
        print(f"\nStep 1: Building tree topology...")
        
        # Build tree structure
        node_infos = []
        parent_ids = []
        
        # Root node (depth 0)
        node_infos.append({'node_id': 0, 'depth': 0, 'parent_id': None})
        parent_ids.append(None)
        
        # Build levels
        current_level = [0]
        next_id = 1
        
        for d in range(1, tree_depth + 1):
            next_level = []
            for parent_id in current_level:
                for _ in range(tree_width):
                    node_infos.append({
                        'node_id': next_id,
                        'depth': d,
                        'parent_id': parent_id
                    })
                    parent_ids.append(parent_id)
                    next_level.append(next_id)
                    next_id += 1
            current_level = next_level
        
        num_nodes = len(node_infos)
        print(f"  Total nodes: {num_nodes}")
        print(f"  Nodes per depth:")
        for d in range(tree_depth + 1):
            count = sum(1 for n in node_infos if n['depth'] == d)
            print(f"    Depth {d}: {count} nodes")
        
        # ====================================================================
        # Step 2: Tokenize Prompt
        # ====================================================================
        
        print(f"\nStep 2: Tokenizing prompt...")
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prefix_len = input_ids.shape[1]
        print(f"  Prefix length: {prefix_len} tokens")
        
        # ====================================================================
        # Step 3: Build Cascade Attention Metadata
        # ====================================================================
        
        print(f"\nStep 3: Building cascade attention metadata...")
        cascade_meta = build_tree_cascade_metadata(
            num_nodes, parent_ids, prefix_len, self.device
        )
        
        print(f"  Cascade enabled: {cascade_meta['use_cascade']}")
        if cascade_meta['use_cascade']:
            print(f"  Prefix mask: {cascade_meta['prefix_mask'].shape}")
            print(f"  Tree-to-prefix: {cascade_meta['tree_to_prefix'].shape}")
            print(f"  Tree-to-tree: {cascade_meta['tree_to_tree'].shape}")
        
        # ====================================================================
        # Step 4: Generate ALL Outputs in ONE Pass
        # ====================================================================
        
        print(f"\nStep 4: Generating all outputs (ONE PASS with cascade)...")
        
        tree_logits = self._forward_with_cascade(
            input_ids, num_nodes, cascade_meta
        )
        
        print(f"  ✓ Generated logits for {num_nodes} outputs")
        print(f"  Logits shape: {tree_logits.shape}")
        
        # ====================================================================
        # Step 5: Sample Tokens for Each Node
        # ====================================================================
        
        print(f"\nStep 5: Sampling tokens for each output...")
        
        outputs = []
        
        for node_idx, node_info in enumerate(node_infos):
            # Get logits for this node
            node_logits = tree_logits[node_idx]
            
            # Apply temperature
            scaled_logits = node_logits / temperature
            
            # Top-k sampling
            if top_k > 0:
                top_k_vals, top_k_indices = torch.topk(scaled_logits, k=top_k)
                probs = F.softmax(top_k_vals, dim=-1)
                sampled_idx = torch.multinomial(probs, 1).item()
                token_id = top_k_indices[sampled_idx].item()
                confidence = probs[sampled_idx].item()
            else:
                probs = F.softmax(scaled_logits, dim=-1)
                token_id = torch.multinomial(probs, 1).item()
                confidence = probs[token_id].item()
            
            # Decode token
            token_text = self.tokenizer.decode([token_id])
            
            # Build full text (path from root)
            if node_info['parent_id'] is None:
                full_text = prompt + token_text
            else:
                parent_full_text = outputs[node_info['parent_id']].full_text
                full_text = parent_full_text + token_text
            
            # Create output
            output = TreeOutput(
                node_id=node_info['node_id'],
                depth=node_info['depth'],
                parent_id=node_info['parent_id'],
                token_id=token_id,
                token_text=token_text,
                confidence=confidence,
                full_text=full_text,
                children=[]
            )
            
            outputs.append(output)
            
            # Add to parent's children
            if output.parent_id is not None:
                outputs[output.parent_id].children.append(output.node_id)
        
        print(f"  ✓ Sampled {num_nodes} outputs")
        
        # ====================================================================
        # Create Tree Structure
        # ====================================================================
        
        tree = TreeStructure(
            outputs=outputs,
            width=tree_width,
            depth=tree_depth,
            prompt=prompt
        )
        
        print(f"{'='*70}\n")
        
        return tree
    
    def _forward_with_cascade(
        self,
        input_ids: torch.Tensor,
        num_nodes: int,
        cascade_meta: Dict
    ) -> torch.Tensor:
        """
        Forward through model with cascade attention
        
        Args:
            input_ids: Input token IDs [1, prefix_len]
            num_nodes: Number of tree nodes
            cascade_meta: Cascade metadata
        
        Returns:
            tree_logits: Logits for all nodes [num_nodes, vocab_size]
        """
        batch_size, prefix_len = input_ids.shape
        
        # Get embeddings for prefix
        with torch.no_grad():
            if hasattr(self.model, 'model'):
                embed_layer = self.model.model.embed_tokens
            else:
                embed_layer = self.model.embed_tokens
            
            prefix_embeds = embed_layer(input_ids)
            hidden_dim = prefix_embeds.shape[-1]
        
        # Initialize tree embeddings (zeros)
        tree_embeds = torch.zeros(
            batch_size, num_nodes, hidden_dim,
            dtype=self.dtype,
            device=self.device
        )
        
        # Concatenate
        full_embeds = torch.cat([prefix_embeds, tree_embeds], dim=1)
        
        # Build position IDs
        position_ids = torch.arange(
            prefix_len + num_nodes,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)
        
        # Build attention mask
        if cascade_meta['use_cascade']:
            total_len = prefix_len + num_nodes
            full_mask = torch.zeros(
                batch_size, 1, total_len, total_len,
                dtype=torch.bool,
                device=self.device
            )
            
            full_mask[0, 0, :prefix_len, :prefix_len] = cascade_meta['prefix_mask']
            full_mask[0, 0, prefix_len:, :prefix_len] = cascade_meta['tree_to_prefix']
            full_mask[0, 0, prefix_len:, prefix_len:] = cascade_meta['tree_to_tree']
        else:
            full_mask = cascade_meta['full_mask'].unsqueeze(0).unsqueeze(0)
        
        # Convert to additive mask
        attention_mask = torch.where(
            full_mask,
            torch.zeros_like(full_mask, dtype=self.dtype),
            torch.full_like(full_mask, float('-inf'), dtype=self.dtype)
        )
        
        # Forward
        with torch.no_grad():
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


# ============================================================================
# Visualization
# ============================================================================

def visualize_tree(tree: TreeStructure, max_depth: int = None):
    """
    Visualize tree structure
    
    Args:
        tree: TreeStructure to visualize
        max_depth: Maximum depth to show (None = all)
    """
    print(f"\n{'='*70}")
    print(f"TREE VISUALIZATION")
    print(f"{'='*70}")
    print(f"Prompt: '{tree.prompt}'")
    print(f"Tree: {tree.width} children per node, {tree.depth} levels deep")
    print(f"Total outputs: {len(tree.outputs)}")
    print(f"{'='*70}\n")
    
    if max_depth is None:
        max_depth = tree.depth
    
    for depth in range(min(max_depth + 1, tree.depth + 1)):
        outputs_at_depth = tree.get_outputs_at_depth(depth)
        print(f"Depth {depth} ({len(outputs_at_depth)} outputs):")
        
        for output in outputs_at_depth[:10]:  # Show first 10
            indent = "  " * (depth + 1)
            parent_info = ""
            if output.parent_id is not None:
                parent = tree.outputs[output.parent_id]
                parent_info = f" (from '{parent.token_text}')"
            
            print(f"{indent}→ '{output.token_text}' "
                  f"(conf={output.confidence:.3f}){parent_info}")
        
        if len(outputs_at_depth) > 10:
            print(f"  ... and {len(outputs_at_depth) - 10} more")
        print()


def print_all_paths(tree: TreeStructure, max_paths: int = 10):
    """Print all paths from root to leaves"""
    print(f"\n{'='*70}")
    print(f"ALL PATHS (Top {max_paths})")
    print(f"{'='*70}\n")
    
    leaf_outputs = tree.get_leaf_outputs()
    
    # Sort by confidence (product of path confidences)
    def path_score(output):
        path = tree.get_path_to_node(output.node_id)
        return sum(o.confidence for o in path) / len(path)
    
    sorted_leaves = sorted(leaf_outputs, key=path_score, reverse=True)
    
    for i, leaf in enumerate(sorted_leaves[:max_paths], 1):
        path = tree.get_path_to_node(leaf.node_id)
        
        print(f"Path {i} (avg conf={path_score(leaf):.3f}):")
        print(f"  Text: '{leaf.full_text}'")
        print(f"  Tokens: {' → '.join([o.token_text for o in path])}")
        print()


def save_tree_json(tree: TreeStructure, filename: str):
    """Save tree structure to JSON"""
    with open(filename, 'w') as f:
        json.dump(tree.to_dict(), f, indent=2)
    print(f"✓ Saved tree to {filename}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(
        description="Generate tree-structured outputs with cascade attention"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of AI is",
        help="Input prompt"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=3,
        help="Tree width (children per node)"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Tree depth"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model path"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Save tree to JSON file"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("vLLM TREE-STRUCTURED OUTPUT GENERATION")
    print("="*70)
    
    # Calculate tree size
    total_outputs = sum(args.width**d for d in range(args.depth + 1))
    
    print(f"\nConfiguration:")
    print(f"  Prompt: '{args.prompt}'")
    print(f"  Tree width: {args.width}")
    print(f"  Tree depth: {args.depth}")
    print(f"  Total outputs: {total_outputs}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k: {args.top_k}")
    
    # Initialize generator
    generator = VLLMTreeOutputGenerator(
        model_path=args.model
    )
    
    # Generate tree outputs
    tree = generator.generate_tree_outputs(
        prompt=args.prompt,
        tree_width=args.width,
        tree_depth=args.depth,
        temperature=args.temperature,
        top_k=args.top_k
    )
    
    # Visualize
    visualize_tree(tree, max_depth=2)
    
    # Print paths
    print_all_paths(tree, max_paths=5)
    
    # Save to JSON
    if args.output_json:
        save_tree_json(tree, args.output_json)
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Generated {len(tree.outputs)} outputs in tree structure")
    print(f"Using cascade attention for efficient computation")
    print(f"Tree structure: {args.width} branches × {args.depth} levels")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
    
    print("="*70)
    print("KEY FEATURES")
    print("="*70)
    print("✅ Generates tree-structured outputs (width × depth)")
    print("✅ Uses cascade attention for efficiency")
    print("✅ All outputs generated in ONE forward pass")
    print("✅ Each node = different continuation")
    print("✅ Visualizes tree structure")
    print("✅ Exports to JSON")
    print("="*70 + "\n")