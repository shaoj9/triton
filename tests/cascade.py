"""
EAGLE Tree Generation with Cascade Attention and Verification
=============================================================

Complete implementation for:
1. Loading yuhuili/EAGLE-LLaMA3.1-Instruct-8B
2. Generating token trees with cascade attention
3. Verifying trees with target model
4. Rejection sampling and path extraction

Usage:
    python eagle_cascade_tree_complete.py --width 3 --depth 4
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import time


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TreeNode:
    """Single node in draft tree"""
    node_id: int
    depth: int
    parent_id: Optional[int]
    token_id: int = -1
    confidence: float = 0.0
    children: List[int] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


@dataclass
class TreeStructure:
    """Complete tree structure"""
    nodes: List[TreeNode]
    num_nodes: int
    max_depth: int
    width: int
    
    def get_nodes_at_depth(self, depth: int) -> List[TreeNode]:
        """Get all nodes at a specific depth"""
        return [n for n in self.nodes if n.depth == depth]


@dataclass
class VerificationResult:
    """Result of tree verification"""
    accepted_tokens: List[int]
    accepted_nodes: List[int]
    num_accepted: int
    acceptance_rate: float
    best_path: List[int]


# ============================================================================
# Tree Structure Building
# ============================================================================

def build_tree_structure(width: int, depth: int) -> TreeStructure:
    """
    Build tree structure with given width and depth
    
    Args:
        width: Branching factor (children per node)
        depth: Maximum depth
    
    Returns:
        TreeStructure object
    """
    nodes = []
    
    # Root node
    root = TreeNode(node_id=0, depth=0, parent_id=None)
    nodes.append(root)
    
    # Build level by level
    current_level = [0]
    next_node_id = 1
    
    for d in range(1, depth + 1):
        next_level = []
        for parent_id in current_level:
            for _ in range(width):
                node = TreeNode(
                    node_id=next_node_id,
                    depth=d,
                    parent_id=parent_id
                )
                nodes.append(node)
                nodes[parent_id].children.append(next_node_id)
                next_level.append(next_node_id)
                next_node_id += 1
        current_level = next_level
    
    return TreeStructure(
        nodes=nodes,
        num_nodes=len(nodes),
        max_depth=depth,
        width=width
    )


# ============================================================================
# Cascade Attention Implementation
# ============================================================================

class CascadeAttentionManager:
    """
    Manages cascade attention for efficient tree generation
    
    Cascade attention splits computation into:
    1. Prefix attention (shared across all tree nodes)
    2. Tree-specific attention
    """
    
    def __init__(self, use_cascade: bool = True, threshold: int = 128):
        """
        Initialize cascade attention manager
        
        Args:
            use_cascade: Whether to use cascade attention
            threshold: Prefix length threshold for using cascade
        """
        self.use_cascade = use_cascade
        self.threshold = threshold
    
    def should_use_cascade(self, prefix_len: int) -> bool:
        """
        Determine if cascade attention should be used
        
        Cascade is beneficial when prefix is large
        """
        return self.use_cascade and prefix_len >= self.threshold
    
    def build_cascade_mask(
        self,
        tree_structure: TreeStructure,
        prefix_len: int,
        device: str = "cuda"
    ) -> Dict[str, torch.Tensor]:
        """
        Build cascade attention masks
        
        Returns:
            Dictionary with:
            - prefix_mask: Attention mask for prefix [prefix_len, prefix_len]
            - tree_to_prefix: Tree nodes attending to prefix [num_nodes, prefix_len]
            - tree_to_tree: Tree nodes attending to ancestors [num_nodes, num_nodes]
        """
        num_nodes = tree_structure.num_nodes
        
        # 1. Prefix self-attention (causal)
        prefix_mask = torch.tril(
            torch.ones(prefix_len, prefix_len, dtype=torch.bool, device=device)
        )
        
        # 2. Tree-to-prefix attention (all tree nodes see all prefix)
        tree_to_prefix = torch.ones(
            num_nodes, prefix_len, dtype=torch.bool, device=device
        )
        
        # 3. Tree-to-tree attention (ancestors only)
        tree_to_tree = torch.zeros(
            num_nodes, num_nodes, dtype=torch.bool, device=device
        )
        
        for node in tree_structure.nodes:
            node_idx = node.node_id
            
            # See self
            tree_to_tree[node_idx, node_idx] = True
            
            # See ancestors
            current_id = node.parent_id
            while current_id is not None:
                tree_to_tree[node_idx, current_id] = True
                current_id = tree_structure.nodes[current_id].parent_id
        
        return {
            'prefix_mask': prefix_mask,
            'tree_to_prefix': tree_to_prefix,
            'tree_to_tree': tree_to_tree,
            'use_cascade': True
        }
    
    def build_standard_mask(
        self,
        tree_structure: TreeStructure,
        prefix_len: int,
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        Build standard (non-cascade) attention mask
        
        Returns full attention mask [total_len, total_len]
        """
        num_nodes = tree_structure.num_nodes
        total_len = prefix_len + num_nodes
        
        mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
        
        # Prefix: causal
        for i in range(prefix_len):
            mask[i, :i+1] = True
        
        # Tree nodes: see prefix
        mask[prefix_len:, :prefix_len] = True
        
        # Tree nodes: see ancestors
        for node in tree_structure.nodes:
            pos = prefix_len + node.node_id
            mask[pos, pos] = True
            
            current_id = node.parent_id
            while current_id is not None:
                mask[pos, prefix_len + current_id] = True
                current_id = tree_structure.nodes[current_id].parent_id
        
        return mask


# ============================================================================
# EAGLE Tree Generator with Cascade Attention
# ============================================================================

class EAGLETreeGenerator:
    """
    Generate draft trees using EAGLE model with cascade attention
    """
    
    def __init__(
        self,
        eagle_model_path: str = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        target_model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        use_cascade: bool = True
    ):
        """
        Initialize EAGLE tree generator
        
        Args:
            eagle_model_path: Path to EAGLE model
            target_model_path: Path to target model
            device: Device to use
            dtype: Data type
            use_cascade: Whether to use cascade attention
        """
        print(f"\n{'='*70}")
        print(f"INITIALIZING EAGLE TREE GENERATOR")
        print(f"{'='*70}")
        print(f"EAGLE model: {eagle_model_path}")
        print(f"Target model: {target_model_path}")
        print(f"Cascade attention: {use_cascade}")
        
        self.device = device
        self.dtype = dtype
        
        # Load target model
        print(f"\nLoading target model...")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_path,
            torch_dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True
        )
        self.target_model.eval()
        print(f"  ✓ Target model loaded")
        
        # Load EAGLE model
        print(f"\nLoading EAGLE model...")
        self.eagle_model = AutoModelForCausalLM.from_pretrained(
            eagle_model_path,
            torch_dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True
        )
        self.eagle_model.eval()
        print(f"  ✓ EAGLE model loaded")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize cascade attention manager
        self.cascade_manager = CascadeAttentionManager(use_cascade=use_cascade)
        
        print(f"{'='*70}\n")
    
    def generate_tree_draft(
        self,
        prompt: str,
        tree_width: int = 3,
        tree_depth: int = 4,
        top_k: int = 4,
        temperature: float = 0.8
    ) -> Tuple[TreeStructure, torch.Tensor]:
        """
        Generate draft tree using EAGLE with cascade attention
        
        Args:
            prompt: Input prompt
            tree_width: Branching factor
            tree_depth: Maximum depth
            top_k: Top-k sampling
            temperature: Sampling temperature
        
        Returns:
            tree_structure: TreeStructure with tokens filled
            draft_logits: Logits for all nodes [num_nodes, vocab_size]
        """
        print(f"\n{'='*70}")
        print(f"GENERATING DRAFT TREE")
        print(f"{'='*70}")
        print(f"Prompt: '{prompt}'")
        print(f"Tree: width={tree_width}, depth={tree_depth}")
        
        # Build tree structure
        tree_structure = build_tree_structure(tree_width, tree_depth)
        print(f"Total nodes: {tree_structure.num_nodes}")
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prefix_len = input_ids.shape[1]
        print(f"Prefix length: {prefix_len}")
        
        # Get hidden states from target model
        print(f"\nStep 1: Extracting features from target model...")
        with torch.no_grad():
            target_outputs = self.target_model(
                input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            # EAGLE uses features from second-to-last layer
            hidden_states = target_outputs.hidden_states[-1]
        
        print(f"  Features extracted: {hidden_states.shape}")
        
        # Determine if we should use cascade
        use_cascade = self.cascade_manager.should_use_cascade(prefix_len)
        print(f"\nStep 2: Building attention masks...")
        print(f"  Using cascade attention: {use_cascade}")
        
        if use_cascade:
            attention_masks = self.cascade_manager.build_cascade_mask(
                tree_structure, prefix_len, self.device
            )
            print(f"  Cascade masks built:")
            print(f"    - Prefix: {attention_masks['prefix_mask'].shape}")
            print(f"    - Tree-to-prefix: {attention_masks['tree_to_prefix'].shape}")
            print(f"    - Tree-to-tree: {attention_masks['tree_to_tree'].shape}")
        else:
            attention_mask = self.cascade_manager.build_standard_mask(
                tree_structure, prefix_len, self.device
            )
            attention_masks = {'standard': attention_mask, 'use_cascade': False}
            print(f"  Standard mask built: {attention_mask.shape}")
        
        # Generate tree in ONE pass
        print(f"\nStep 3: Forward pass through EAGLE (ONE PASS)...")
        start_time = time.time()
        
        draft_logits = self._forward_eagle_tree(
            hidden_states=hidden_states,
            input_ids=input_ids,
            tree_structure=tree_structure,
            attention_masks=attention_masks
        )
        
        elapsed = time.time() - start_time
        print(f"  ✓ Generated {tree_structure.num_nodes} nodes in {elapsed:.3f}s")
        print(f"  Draft logits: {draft_logits.shape}")
        
        # Sample tokens
        print(f"\nStep 4: Sampling tokens...")
        self._sample_tree_tokens(
            tree_structure=tree_structure,
            tree_logits=draft_logits,
            top_k=top_k,
            temperature=temperature
        )
        print(f"  ✓ Sampled {tree_structure.num_nodes} tokens")
        
        # Display samples
        print(f"\nFirst 10 sampled tokens:")
        for i in range(min(10, tree_structure.num_nodes)):
            node = tree_structure.nodes[i]
            token_text = self.tokenizer.decode([node.token_id])
            print(f"  Node {i} (depth={node.depth}): "
                  f"'{token_text}' (conf={node.confidence:.3f})")
        
        print(f"{'='*70}\n")
        
        return tree_structure, draft_logits
    
    def _forward_eagle_tree(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        tree_structure: TreeStructure,
        attention_masks: Dict
    ) -> torch.Tensor:
        """
        Forward through EAGLE with cascade attention
        
        Args:
            hidden_states: Features from target model
            input_ids: Input token IDs
            tree_structure: Tree structure
            attention_masks: Attention masks (cascade or standard)
        
        Returns:
            tree_logits: Logits for all tree nodes [num_nodes, vocab_size]
        """
        batch_size, prefix_len, hidden_dim = hidden_states.shape
        num_nodes = tree_structure.num_nodes
        
        # Initialize tree node embeddings (zeros)
        tree_embeddings = torch.zeros(
            batch_size, num_nodes, hidden_dim,
            dtype=self.dtype,
            device=self.device
        )
        
        # Concatenate prefix + tree
        full_embeddings = torch.cat([hidden_states, tree_embeddings], dim=1)
        
        # Build position IDs (depth-based)
        position_ids = torch.zeros(
            batch_size, prefix_len + num_nodes,
            dtype=torch.long,
            device=self.device
        )
        position_ids[0, :prefix_len] = torch.arange(prefix_len)
        for node in tree_structure.nodes:
            position_ids[0, prefix_len + node.node_id] = prefix_len + node.depth
        
        # Apply attention based on cascade or standard
        if attention_masks['use_cascade']:
            # Cascade attention
            logits = self._forward_with_cascade(
                full_embeddings,
                position_ids,
                attention_masks,
                prefix_len,
                num_nodes
            )
        else:
            # Standard attention
            logits = self._forward_with_standard(
                full_embeddings,
                position_ids,
                attention_masks['standard']
            )
        
        # Extract tree logits
        tree_logits = logits[0, prefix_len:, :]
        
        return tree_logits
    
    def _forward_with_cascade(
        self,
        embeddings: torch.Tensor,
        position_ids: torch.Tensor,
        cascade_masks: Dict,
        prefix_len: int,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Forward with cascade attention
        
        This is more efficient for large prefixes
        """
        # For simplicity, we'll use standard forward with full mask
        # In production, you'd compute prefix attention once and reuse it
        
        # Build full mask from cascade components
        total_len = prefix_len + num_nodes
        full_mask = torch.zeros(
            1, 1, total_len, total_len,
            dtype=torch.bool,
            device=self.device
        )
        
        # Prefix block
        full_mask[0, 0, :prefix_len, :prefix_len] = cascade_masks['prefix_mask']
        
        # Tree-to-prefix block
        full_mask[0, 0, prefix_len:, :prefix_len] = cascade_masks['tree_to_prefix']
        
        # Tree-to-tree block
        full_mask[0, 0, prefix_len:, prefix_len:] = cascade_masks['tree_to_tree']
        
        # Convert to additive mask
        attention_mask = torch.where(
            full_mask,
            torch.zeros_like(full_mask, dtype=self.dtype),
            torch.full_like(full_mask, float('-inf'), dtype=self.dtype)
        )
        
        # Forward
        with torch.no_grad():
            outputs = self.eagle_model.model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True
            )
            hidden = outputs.last_hidden_state
            logits = self.eagle_model.lm_head(hidden)
        
        return logits
    
    def _forward_with_standard(
        self,
        embeddings: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward with standard attention mask"""
        # Add batch and head dimensions
        attention_mask_4d = attention_mask.unsqueeze(0).unsqueeze(0)
        
        # Convert to additive mask
        attention_mask_4d = torch.where(
            attention_mask_4d,
            torch.zeros_like(attention_mask_4d, dtype=self.dtype),
            torch.full_like(attention_mask_4d, float('-inf'), dtype=self.dtype)
        )
        
        # Forward
        with torch.no_grad():
            outputs = self.eagle_model.model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask_4d,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True
            )
            hidden = outputs.last_hidden_state
            logits = self.eagle_model.lm_head(hidden)
        
        return logits
    
    def _sample_tree_tokens(
        self,
        tree_structure: TreeStructure,
        tree_logits: torch.Tensor,
        top_k: int,
        temperature: float
    ):
        """Sample tokens for each tree node"""
        for node_idx, node in enumerate(tree_structure.nodes):
            node_logits = tree_logits[node_idx]
            
            # Temperature scaling
            scaled = node_logits / temperature
            
            # Top-k sampling
            if top_k > 0:
                top_k_vals, top_k_idx = torch.topk(scaled, k=top_k)
                probs = F.softmax(top_k_vals, dim=-1)
                
                if temperature == 0:
                    sampled_idx = 0
                else:
                    sampled_idx = torch.multinomial(probs, 1).item()
                
                token_id = top_k_idx[sampled_idx].item()
                confidence = probs[sampled_idx].item()
            else:
                probs = F.softmax(scaled, dim=-1)
                token_id = torch.multinomial(probs, 1).item()
                confidence = probs[token_id].item()
            
            node.token_id = token_id
            node.confidence = confidence
    
    def verify_tree(
        self,
        prompt: str,
        tree_structure: TreeStructure,
        temperature: float = 0.8
    ) -> VerificationResult:
        """
        Verify draft tree with target model
        
        Args:
            prompt: Original prompt
            tree_structure: Draft tree to verify
            temperature: Sampling temperature
        
        Returns:
            VerificationResult with accepted tokens
        """
        print(f"\n{'='*70}")
        print(f"VERIFYING DRAFT TREE")
        print(f"{'='*70}")
        print(f"Total draft nodes: {tree_structure.num_nodes}")
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prefix_len = input_ids.shape[1]
        
        # Flatten tree for verification
        print(f"\nStep 1: Flattening tree for verification...")
        draft_tokens = [node.token_id for node in tree_structure.nodes]
        draft_confidences = [node.confidence for node in tree_structure.nodes]
        
        # Build tree attention mask for verification
        verification_mask = self.cascade_manager.build_standard_mask(
            tree_structure, prefix_len, self.device
        )
        
        # Concatenate input + draft
        draft_tensor = torch.tensor(
            draft_tokens, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        full_input = torch.cat([input_ids, draft_tensor], dim=1)
        
        print(f"  Verification input: {full_input.shape}")
        
        # Forward through target model
        print(f"\nStep 2: Forward through target model...")
        with torch.no_grad():
            # Prepare attention mask
            attn_mask_4d = verification_mask.unsqueeze(0).unsqueeze(0)
            attn_mask_4d = torch.where(
                attn_mask_4d,
                torch.zeros_like(attn_mask_4d, dtype=self.dtype),
                torch.full_like(attn_mask_4d, float('-inf'), dtype=self.dtype)
            )
            
            outputs = self.target_model(
                input_ids=full_input,
                attention_mask=attn_mask_4d,
                return_dict=True
            )
            target_logits = outputs.logits
        
        print(f"  Target logits: {target_logits.shape}")
        
        # Verify each token using rejection sampling
        print(f"\nStep 3: Rejection sampling...")
        accepted_nodes = []
        accepted_tokens = []
        
        for node_idx, node in enumerate(tree_structure.nodes):
            pos = prefix_len + node_idx
            
            # Get target probability
            target_probs = F.softmax(
                target_logits[0, pos - 1] / temperature, dim=-1
            )
            p_target = target_probs[node.token_id].item()
            
            # Get draft probability
            p_draft = node.confidence
            
            # Rejection sampling
            acceptance_prob = min(1.0, p_target / (p_draft + 1e-10))
            
            if np.random.random() < acceptance_prob:
                accepted_nodes.append(node.node_id)
                accepted_tokens.append(node.token_id)
            else:
                # Rejection - stop verification
                break
        
        num_accepted = len(accepted_tokens)
        acceptance_rate = num_accepted / tree_structure.num_nodes
        
        print(f"  ✓ Accepted {num_accepted}/{tree_structure.num_nodes} tokens")
        print(f"  Acceptance rate: {acceptance_rate:.2%}")
        
        # Find best path (longest accepted path)
        best_path = self._find_best_path(tree_structure, accepted_nodes)
        
        print(f"\nStep 4: Extracting best path...")
        print(f"  Best path length: {len(best_path)}")
        if best_path:
            path_tokens = [tree_structure.nodes[nid].token_id for nid in best_path]
            path_text = self.tokenizer.decode(path_tokens)
            print(f"  Best path text: '{path_text}'")
        
        print(f"{'='*70}\n")
        
        return VerificationResult(
            accepted_tokens=accepted_tokens,
            accepted_nodes=accepted_nodes,
            num_accepted=num_accepted,
            acceptance_rate=acceptance_rate,
            best_path=best_path
        )
    
    def _find_best_path(
        self,
        tree_structure: TreeStructure,
        accepted_nodes: List[int]
    ) -> List[int]:
        """Find best path through accepted nodes"""
        if not accepted_nodes:
            return []
        
        # Build accepted set
        accepted_set = set(accepted_nodes)
        
        # Find longest path from root through accepted nodes
        def dfs(node_id: int) -> List[int]:
            if node_id not in accepted_set:
                return []
            
            node = tree_structure.nodes[node_id]
            if not node.children:
                return [node_id]
            
            # Try each child
            best = [node_id]
            for child_id in node.children:
                child_path = dfs(child_id)
                if len(child_path) > len(best) - 1:
                    best = [node_id] + child_path
            
            return best
        
        return dfs(0)


# ============================================================================
# Utility Functions
# ============================================================================

def visualize_tree(tree_structure: TreeStructure, tokenizer, max_depth: int = 3):
    """Visualize tree structure"""
    print(f"\n{'='*70}")
    print(f"TREE VISUALIZATION")
    print(f"{'='*70}\n")
    
    for depth in range(min(max_depth + 1, tree_structure.max_depth + 1)):
        nodes_at_depth = tree_structure.get_nodes_at_depth(depth)
        print(f"Depth {depth} ({len(nodes_at_depth)} nodes):")
        
        for node in nodes_at_depth[:5]:  # Show first 5
            token_text = tokenizer.decode([node.token_id])
            parent_text = ""
            if node.parent_id is not None:
                parent_token = tokenizer.decode(
                    [tree_structure.nodes[node.parent_id].token_id]
                )
                parent_text = f" (parent: '{parent_token}')"
            
            print(f"  Node {node.node_id}: '{token_text}' "
                  f"(conf={node.confidence:.3f}){parent_text}")
        
        if len(nodes_at_depth) > 5:
            print(f"  ... and {len(nodes_at_depth) - 5} more")
        print()


# ============================================================================
# Main Example
# ============================================================================

def main():
    """Complete example with EAGLE tree generation and verification"""
    
    print("\n" + "="*70)
    print("EAGLE TREE GENERATION WITH CASCADE ATTENTION")
    print("Complete Pipeline: Generation → Verification")
    print("="*70 + "\n")
    
    # Configuration
    prompt = "The future of artificial intelligence is"
    tree_width = 3
    tree_depth = 4
    top_k = 4
    temperature = 0.8
    
    print("Configuration:")
    print(f"  Prompt: '{prompt}'")
    print(f"  Tree: width={tree_width}, depth={tree_depth}")
    print(f"  Total nodes: {sum(tree_width**d for d in range(tree_depth+1))}")
    print(f"  Top-k: {top_k}")
    print(f"  Temperature: {temperature}")
    
    # Initialize generator
    generator = EAGLETreeGenerator(
        eagle_model_path="yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        target_model_path="meta-llama/Llama-3.1-8B-Instruct",
        use_cascade=True
    )
    
    # Generate draft tree
    tree_structure, draft_logits = generator.generate_tree_draft(
        prompt=prompt,
        tree_width=tree_width,
        tree_depth=tree_depth,
        top_k=top_k,
        temperature=temperature
    )
    
    # Visualize tree
    visualize_tree(tree_structure, generator.tokenizer, max_depth=2)
    
    # Verify tree with target model
    verification_result = generator.verify_tree(
        prompt=prompt,
        tree_structure=tree_structure,
        temperature=temperature
    )
    
    # Print final results
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Draft nodes: {tree_structure.num_nodes}")
    print(f"Accepted tokens: {verification_result.num_accepted}")
    print(f"Acceptance rate: {verification_result.acceptance_rate:.2%}")
    print(f"Best path length: {len(verification_result.best_path)}")
    
    if verification_result.best_path:
        path_tokens = [
            tree_structure.nodes[nid].token_id
            for nid in verification_result.best_path
        ]
        full_text = prompt + generator.tokenizer.decode(path_tokens)
        print(f"\nGenerated text:")
        print(f"  '{full_text}'")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is")
    parser.add_argument("--no-cascade", action="store_true", help="Disable cascade attention")
    
    args = parser.parse_args()
    
    generator = EAGLETreeGenerator(use_cascade=not args.no_cascade)
    tree, logits = generator.generate_tree_draft(
        prompt=args.prompt,
        tree_width=args.width,
        tree_depth=args.depth
    )
    
    visualize_tree(tree, generator.tokenizer)
    
    result = generator.verify_tree(args.prompt, tree)
    print(f"\nAccepted: {result.num_accepted} tokens ({result.acceptance_rate:.1%})")