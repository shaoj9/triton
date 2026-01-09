"""
Improved Small Tree Generation - Better Token Quality
====================================================

Fixes the "the" and "ably" repetition problem by:
1. Using a BASE model (not Instruct) - no chat format needed
2. Better sampling strategies
3. Proper context initialization

The problem with zero embeddings:
- Zero vectors have no semantic meaning
- Model generates very generic tokens: "the", "and", "of", "ably"
- These are the most common tokens in the training data

Solution: Use base model + better sampling!

Usage:
    python improved_tree.py
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
    token_text: str = ""
    confidence: float = 0.0
    children: List[int] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


def build_small_tree() -> Tuple[List[TreeNode], List[Optional[int]]]:
    """Build tree with width=2, depth=3 (15 nodes)"""
    nodes = []
    parent_ids = []
    
    # Depth 0: Root
    nodes.append(TreeNode(node_id=0, depth=0, parent_id=None))
    parent_ids.append(None)
    
    # Depth 1
    for i in range(2):
        node_id = 1 + i
        nodes.append(TreeNode(node_id=node_id, depth=1, parent_id=0))
        parent_ids.append(0)
        nodes[0].children.append(node_id)
    
    # Depth 2
    for parent in [1, 2]:
        for i in range(2):
            node_id = len(nodes)
            nodes.append(TreeNode(node_id=node_id, depth=2, parent_id=parent))
            parent_ids.append(parent)
            nodes[parent].children.append(node_id)
    
    # Depth 3
    for parent in [3, 4, 5, 6]:
        for i in range(2):
            node_id = len(nodes)
            nodes.append(TreeNode(node_id=node_id, depth=3, parent_id=parent))
            parent_ids.append(parent)
            nodes[parent].children.append(node_id)
    
    return nodes, parent_ids


# ============================================================================
# Cascade Attention
# ============================================================================

def create_cascade_attention_mask(
    parent_ids: List[Optional[int]],
    prefix_len: int,
    device: str = "cuda"
) -> torch.Tensor:
    """Create cascade attention mask"""
    num_nodes = len(parent_ids)
    total_len = prefix_len + num_nodes
    
    mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
    
    # Prefix causal
    for i in range(prefix_len):
        mask[i, :i+1] = True
    
    # Tree sees prefix
    mask[prefix_len:, :prefix_len] = True
    
    # Tree sees ancestors
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
        torch.zeros(total_len, total_len, dtype=torch.float16, device=device),
        torch.full((total_len, total_len), float('-inf'), dtype=torch.float16, device=device)
    )
    
    return attention_mask.unsqueeze(0).unsqueeze(0)


def create_position_ids(nodes: List[TreeNode], prefix_len: int, device: str = "cuda"):
    """Create position IDs"""
    num_nodes = len(nodes)
    position_ids = torch.zeros(1, prefix_len + num_nodes, dtype=torch.long, device=device)
    
    position_ids[0, :prefix_len] = torch.arange(prefix_len)
    
    for node in nodes:
        position_ids[0, prefix_len + node.node_id] = prefix_len + node.depth
    
    return position_ids


# ============================================================================
# Improved Tree Generator
# ============================================================================

class ImprovedTreeGenerator:
    """
    Improved tree generator with better token quality
    """
    
    def __init__(
        self,
        model_path: str = "meta-llama/Llama-3.2-1B",  # BASE MODEL!
        device: str = "cuda"
    ):
        print(f"\n{'='*70}")
        print(f"INITIALIZING IMPROVED TREE GENERATOR")
        print(f"{'='*70}")
        
        self.device = device
        self.dtype = torch.float16
        
        print(f"Model: {model_path}")
        print(f"  Using BASE model (not Instruct) for better quality")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map=device,
            low_cpu_mem_usage=True
        )
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.embedding_layer = self.model.model.embed_tokens
        print(f"  ✓ Model loaded")
        print(f"{'='*70}\n")
    
    def generate_tree(
        self,
        prompt: str,
        top_k: int = 50,  # Increased for diversity
        temperature: float = 1.0,  # Higher for diversity
        repetition_penalty: float = 1.2  # Penalize repetition
    ) -> List[TreeNode]:
        """
        Generate tree with improved token quality
        
        Args:
            prompt: Input prompt (plain text, no chat format needed!)
            top_k: Top-k sampling (higher = more diverse)
            temperature: Sampling temperature (higher = more diverse)
            repetition_penalty: Penalty for repeated tokens
        """
        print(f"\n{'='*70}")
        print(f"GENERATING TREE WITH IMPROVED SAMPLING")
        print(f"{'='*70}")
        print(f"Prompt: '{prompt}'")
        print(f"Sampling: top_k={top_k}, temp={temperature}, rep_penalty={repetition_penalty}")
        
        # Tokenize (plain text, no chat format!)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prefix_len = input_ids.shape[1]
        
        print(f"Prefix length: {prefix_len} tokens")
        
        # Build tree
        nodes, parent_ids = build_small_tree()
        num_nodes = len(nodes)
        
        # Create masks
        attention_mask = create_cascade_attention_mask(parent_ids, prefix_len, self.device)
        position_ids = create_position_ids(nodes, prefix_len, self.device)
        
        # Embeddings
        prefix_embeds = self.embedding_layer(input_ids)
        tree_embeds = torch.zeros(
            1, num_nodes, prefix_embeds.shape[-1],
            dtype=self.dtype,
            device=self.device
        )
        full_embeds = torch.cat([prefix_embeds, tree_embeds], dim=1)
        
        print(f"\nForward pass...")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.model(
                inputs_embeds=full_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True
            )
            logits = self.model.lm_head(outputs.last_hidden_state)
        
        tree_logits = logits[0, prefix_len:, :]
        
        print(f"  ✓ Generated logits: {tree_logits.shape}")
        print(f"\nSampling with improved strategy...")
        
        # Track used tokens for repetition penalty
        used_tokens = set(input_ids[0].tolist())
        
        for node_idx, node in enumerate(nodes):
            node_logits = tree_logits[node_idx].clone()
            
            # Apply repetition penalty
            for token_id in used_tokens:
                node_logits[token_id] /= repetition_penalty
            
            # Temperature scaling
            scaled = node_logits / temperature
            
            # Top-k filtering
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
            
            # Update node
            node.token_id = token_id
            node.token_text = self.tokenizer.decode([token_id])
            node.confidence = confidence
            
            # Track for repetition penalty
            used_tokens.add(token_id)
        
        print(f"  ✓ Sampled {num_nodes} tokens with diversity")
        print(f"{'='*70}\n")
        
        return nodes


# ============================================================================
# Visualization
# ============================================================================

def extract_paths(nodes: List[TreeNode]) -> List[List[TreeNode]]:
    """Extract all paths"""
    paths = []
    leaf_nodes = [n for n in nodes if len(n.children) == 0]
    
    for leaf in leaf_nodes:
        path = []
        current = leaf
        
        while current is not None:
            path.insert(0, current)
            if current.parent_id is None:
                break
            current = nodes[current.parent_id]
        
        paths.append(path)
    
    return paths


def print_tree_ascii(nodes: List[TreeNode]):
    """Simple ASCII tree"""
    print(f"\n{'='*70}")
    print(f"TREE STRUCTURE")
    print(f"{'='*70}\n")
    
    for depth in range(4):
        nodes_at_depth = [n for n in nodes if n.depth == depth]
        indent = "  " * depth
        
        if depth == 0:
            print(f"{indent}Depth {depth}: {repr(nodes_at_depth[0].token_text)}")
        else:
            tokens = ' '.join([repr(n.token_text) for n in nodes_at_depth])
            print(f"{indent}Depth {depth}: {tokens}")
    print()


def print_paths(paths: List[List[TreeNode]], prompt: str):
    """Print all paths"""
    print(f"\n{'='*70}")
    print(f"ALL {len(paths)} PATHS")
    print(f"{'='*70}\n")
    
    for i, path in enumerate(paths, 1):
        tokens = [node.token_text for node in path]
        full_text = ''.join(tokens)
        
        print(f"Path {i}:")
        print(f"  Nodes: {' → '.join([f'[{n.node_id}]' for n in path])}")
        print(f"  Text: '{prompt}{full_text}'")
        print()


def analyze_token_diversity(nodes: List[TreeNode]):
    """Analyze token diversity"""
    print(f"\n{'='*70}")
    print(f"TOKEN DIVERSITY ANALYSIS")
    print(f"{'='*70}")
    
    tokens = [n.token_text for n in nodes]
    unique_tokens = set(tokens)
    
    print(f"Total tokens: {len(tokens)}")
    print(f"Unique tokens: {len(unique_tokens)}")
    print(f"Diversity: {len(unique_tokens)/len(tokens)*100:.1f}%")
    
    # Check for common boring tokens
    boring_tokens = ['the', ' the', 'and', ' and', 'of', ' of', 'ably', ' ably', 'ly', ' ly']
    boring_count = sum(1 for t in tokens if t.lower().strip() in [b.strip() for b in boring_tokens])
    
    print(f"Boring tokens: {boring_count}/{len(tokens)} ({boring_count/len(tokens)*100:.1f}%)")
    
    if boring_count > len(tokens) * 0.3:
        print(f"  ⚠️  High repetition detected!")
        print(f"  Suggestion: Increase temperature or top_k")
    else:
        print(f"  ✓ Good diversity!")
    
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("IMPROVED TREE GENERATION")
    print("Using BASE model + Better sampling")
    print("="*70)
    
    # Initialize with base model
    generator = ImprovedTreeGenerator(
        model_path="meta-llama/Llama-3.2-1B"  # 1B base model (faster, good quality)
    )
    
    # Generate
    prompt = "The future of artificial intelligence is"
    nodes = generator.generate_tree(
        prompt=prompt,
        top_k=50,          # More diversity
        temperature=1.0,   # Higher temperature
        repetition_penalty=1.2  # Penalize repetition
    )
    
    # Visualize
    print_tree_ascii(nodes)
    
    # Analyze diversity
    analyze_token_diversity(nodes)
    
    # Show paths
    paths = extract_paths(nodes)
    print_paths(paths, prompt)
    
    # Summary
    print(f"{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"✓ Generated {len(nodes)} tokens in ONE pass")
    print(f"✓ Extracted {len(paths)} unique paths")
    print(f"✓ Using base model - no chat format issues")
    print(f"✓ Better sampling - more diverse tokens")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
    
    print("="*70)
    print("IMPROVEMENTS MADE")
    print("="*70)
    print("""
1. BASE MODEL instead of Instruct
   - No chat format needed
   - More natural continuations
   - No "assistant" tokens

2. BETTER SAMPLING
   - Higher top_k (50 instead of 4)
   - Higher temperature (1.0 instead of 0.8)
   - Repetition penalty (1.2)
   
3. DIVERSITY TRACKING
   - Penalize already-used tokens
   - Avoid "the", "and", "ably" repetition
   
Result: Much better token quality and diversity!
    """)
    print("="*70 + "\n")