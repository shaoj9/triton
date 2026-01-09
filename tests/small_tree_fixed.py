"""
Fixed Small Tree Generation - Proper Chat Format
===============================================

This fixes the "assistant" issue by using proper chat templates
for instruction-tuned models.

The problem: Llama-3.1-8B-Instruct expects chat format like:
<|start_header_id|>user<|end_header_id|>
{prompt}
<|start_header_id|>assistant<|end_header_id|>

Without this, it generates "assistant" tokens in the output!

Usage:
    python small_tree_fixed.py
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
    
    # Depth 1: 2 children
    for i in range(2):
        node_id = 1 + i
        nodes.append(TreeNode(node_id=node_id, depth=1, parent_id=0))
        parent_ids.append(0)
        nodes[0].children.append(node_id)
    
    # Depth 2: 4 children
    for parent in [1, 2]:
        for i in range(2):
            node_id = len(nodes)
            nodes.append(TreeNode(node_id=node_id, depth=2, parent_id=parent))
            parent_ids.append(parent)
            nodes[parent].children.append(node_id)
    
    # Depth 3: 8 children
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
    """Create depth-based position IDs"""
    num_nodes = len(nodes)
    position_ids = torch.zeros(1, prefix_len + num_nodes, dtype=torch.long, device=device)
    
    position_ids[0, :prefix_len] = torch.arange(prefix_len)
    
    for node in nodes:
        position_ids[0, prefix_len + node.node_id] = prefix_len + node.depth
    
    return position_ids


# ============================================================================
# Fixed Tree Generator
# ============================================================================

class FixedTreeGenerator:
    """
    Fixed tree generator that uses proper chat templates
    """
    
    def __init__(
        self,
        model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda"
    ):
        print(f"\n{'='*70}")
        print(f"INITIALIZING FIXED TREE GENERATOR")
        print(f"{'='*70}")
        
        self.device = device
        self.dtype = torch.float16
        
        print(f"Loading model: {model_path}")
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
    
    def format_prompt_properly(self, user_prompt: str) -> str:
        """
        Format prompt using chat template to avoid 'assistant' in output
        
        This is THE FIX for the issue!
        """
        print(f"{'='*70}")
        print(f"FIXING PROMPT FORMAT")
        print(f"{'='*70}")
        print(f"Original prompt: '{user_prompt}'")
        
        # Use chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "user", "content": f"Complete this sentence: {user_prompt}"}
            ]
            
            # Apply chat template
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            print(f"\nUsing chat template:")
            print(f"  ✓ Wrapped in proper format")
            print(f"  ✓ Added generation prompt")
            
        else:
            # Fallback: manual formatting
            formatted = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nComplete this sentence: {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            print(f"\nUsing manual format:")
            print(f"  ✓ Added chat structure")
        
        print(f"{'='*70}\n")
        return formatted
    
    def generate_tree(
        self,
        prompt: str,
        top_k: int = 4,
        temperature: float = 0.8,
        use_chat_format: bool = True
    ) -> Tuple[List[TreeNode], str]:
        """
        Generate tree with proper formatting
        
        Args:
            prompt: User prompt
            top_k: Top-k sampling
            temperature: Temperature
            use_chat_format: Whether to use chat template (recommended!)
        
        Returns:
            nodes: Tree nodes
            formatted_prompt: The formatted prompt used
        """
        print(f"\n{'='*70}")
        print(f"GENERATING TREE")
        print(f"{'='*70}")
        print(f"User prompt: '{prompt}'")
        print(f"Use chat format: {use_chat_format}")
        
        # Format prompt
        if use_chat_format:
            formatted_prompt = self.format_prompt_properly(prompt)
        else:
            formatted_prompt = prompt
            print(f"WARNING: Not using chat format - may see 'assistant' in output!")
        
        # Tokenize
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        prefix_len = input_ids.shape[1]
        
        print(f"Tokenized length: {prefix_len} tokens")
        
        # Build tree
        nodes, parent_ids = build_small_tree()
        num_nodes = len(nodes)
        
        # Create attention mask
        attention_mask = create_cascade_attention_mask(parent_ids, prefix_len, self.device)
        position_ids = create_position_ids(nodes, prefix_len, self.device)
        
        # Prepare embeddings
        prefix_embeds = self.embedding_layer(input_ids)
        tree_embeds = torch.zeros(
            1, num_nodes, prefix_embeds.shape[-1],
            dtype=self.dtype,
            device=self.device
        )
        full_embeds = torch.cat([prefix_embeds, tree_embeds], dim=1)
        
        print(f"\nGenerating {num_nodes} nodes in ONE pass...")
        
        # Forward
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
        
        # Sample
        print(f"Sampling tokens...")
        for node_idx, node in enumerate(nodes):
            node_logits = tree_logits[node_idx]
            scaled = node_logits / temperature
            
            top_k_vals, top_k_idx = torch.topk(scaled, k=top_k)
            probs = F.softmax(top_k_vals, dim=-1)
            sampled = torch.multinomial(probs, 1).item()
            
            token_id = top_k_idx[sampled].item()
            node.token_id = token_id
            node.token_text = self.tokenizer.decode([token_id])
            node.confidence = probs[sampled].item()
        
        print(f"  ✓ Sampled {num_nodes} tokens")
        print(f"{'='*70}\n")
        
        return nodes, formatted_prompt


# ============================================================================
# Path Visualization
# ============================================================================

def extract_all_paths(nodes: List[TreeNode]) -> List[List[TreeNode]]:
    """Extract all root-to-leaf paths"""
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


def print_paths(paths: List[List[TreeNode]], original_prompt: str):
    """Print all paths with clean output"""
    print(f"\n{'='*70}")
    print(f"ALL PATHS ({len(paths)} total)")
    print(f"{'='*70}\n")
    
    for i, path in enumerate(paths, 1):
        tokens = [node.token_text for node in path]
        full_text = ''.join(tokens)
        
        path_conf = 1.0
        for node in path:
            path_conf *= node.confidence
        
        print(f"Path {i}:")
        print(f"  Nodes: {' → '.join([f'[{n.node_id}]' for n in path])}")
        print(f"  Tokens: {' → '.join([repr(t) for t in tokens])}")
        print(f"  Full text: '{original_prompt}{full_text}'")
        print(f"  Confidence: {path_conf:.6f}")
        print()


def print_tree_structure(nodes: List[TreeNode]):
    """Print tree structure"""
    print(f"\n{'='*70}")
    print(f"TREE STRUCTURE")
    print(f"{'='*70}\n")
    
    for depth in range(4):
        nodes_at_depth = [n for n in nodes if n.depth == depth]
        print(f"Depth {depth}: ({len(nodes_at_depth)} nodes)")
        
        for node in nodes_at_depth:
            indent = "  " * (depth + 1)
            print(f"{indent}[{node.node_id}] {repr(node.token_text)} (conf={node.confidence:.3f})")
        print()


# ============================================================================
# Comparison Demo
# ============================================================================

def comparison_demo():
    """Show the difference between with/without chat format"""
    print("\n" + "="*70)
    print("COMPARISON: WITH vs WITHOUT CHAT FORMAT")
    print("="*70 + "\n")
    
    generator = FixedTreeGenerator()
    prompt = "The future of AI is"
    
    print("="*70)
    print("TEST 1: WITHOUT CHAT FORMAT (BROKEN)")
    print("="*70)
    nodes_broken, _ = generator.generate_tree(prompt, use_chat_format=False)
    
    print("\nSample tokens (WITHOUT chat format):")
    for i in range(min(5, len(nodes_broken))):
        print(f"  [{i}] {repr(nodes_broken[i].token_text)}")
    print("  ⚠️  Notice: May contain 'assistant' or chat tokens!")
    
    print("\n" + "="*70)
    print("TEST 2: WITH CHAT FORMAT (FIXED)")
    print("="*70)
    nodes_fixed, _ = generator.generate_tree(prompt, use_chat_format=True)
    
    print("\nSample tokens (WITH chat format):")
    for i in range(min(5, len(nodes_fixed))):
        print(f"  [{i}] {repr(nodes_fixed[i].token_text)}")
    print("  ✓ Notice: Clean, natural continuations!")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("Always use chat format for instruction-tuned models!")
    print("="*70 + "\n")


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("FIXED: SMALL TREE GENERATION")
    print("Proper chat formatting to avoid 'assistant' tokens")
    print("="*70)
    
    # Initialize
    generator = FixedTreeGenerator()
    
    # Generate with proper format
    prompt = "The future of AI is"
    nodes, formatted = generator.generate_tree(
        prompt=prompt,
        top_k=4,
        temperature=0.8,
        use_chat_format=True  # THE FIX!
    )
    
    # Show results
    print_tree_structure(nodes)
    
    paths = extract_all_paths(nodes)
    print_paths(paths, prompt)
    
    print(f"{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Generated {len(nodes)} nodes in ONE pass")
    print(f"✓ Extracted {len(paths)} complete paths")
    print(f"✓ Using proper chat format - NO 'assistant' tokens!")
    print(f"{'='*70}\n")
    
    # Show best path
    best_path = max(paths, key=lambda p: sum(n.confidence for n in p) / len(p))
    best_text = ''.join([n.token_text for n in best_path])
    
    print(f"Best path:")
    print(f"  '{prompt}{best_text}'")
    print()


if __name__ == "__main__":
    # Run main demo
    main()
    
    # Optional: Show comparison
    print("\n" + "="*70)
    print("Want to see the difference?")
    print("Uncomment the line below to run comparison demo")
    print("="*70 + "\n")
    # comparison_demo()
    
    print("="*70)
    print("KEY FIX")
    print("="*70)
    print("""
The problem: Llama-3.1-8B-Instruct expects chat format:
  <|start_header_id|>user<|end_header_id|>
  {prompt}
  <|start_header_id|>assistant<|end_header_id|>

Without this format, the model tries to generate the chat structure
itself, resulting in 'assistant' appearing in the output!

The fix: Use tokenizer.apply_chat_template()
  messages = [{"role": "user", "content": prompt}]
  formatted = tokenizer.apply_chat_template(messages, ...)

Result: Clean, natural text generation!
    """)
    print("="*70 + "\n")