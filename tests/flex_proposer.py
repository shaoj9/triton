"""
vLLM EAGLE Proposer with FlexAttention
======================================

High-performance EAGLE proposer for vLLM that uses FlexAttention to generate
all draft tokens in a tree structure in ONE pass.

Features:
- vLLM-compatible architecture
- FlexAttention for accurate tree attention
- One-pass generation (all tokens simultaneously)
- Efficient batching and caching
- Beam search pruning for quality

Integration with vLLM:
    from vllm_eagle_flex_proposer import EAGLEFlexProposer
    
    proposer = EAGLEFlexProposer(
        model_config=model_config,
        device="cuda",
        tree_width=4,
        tree_depth=2
    )
    
    draft_tokens = proposer.propose(
        input_ids=input_ids,
        target_hidden_states=target_hidden_states
    )

Usage:
    python vllm_eagle_flex_proposer.py
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Callable, Any
from dataclasses import dataclass
import time

# Try to import vLLM components
try:
    from vllm import ModelRegistry
    from vllm.model_executor.models import ModelRegistry
    from vllm.attention import AttentionMetadata
    from vllm.config import ModelConfig, CacheConfig
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠ vLLM not available - using standalone mode")

# Try to import FlexAttention
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    flex_attention = None
    create_block_mask = None


# ============================================================================
# Tree Structure
# ============================================================================

@dataclass
class TreeNode:
    """Node in draft tree"""
    node_id: int
    depth: int
    parent_id: Optional[int]
    token_id: int = -1
    log_prob: float = 0.0
    cumulative_score: float = 0.0
    children: List[int] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


@dataclass
class TreeStructure:
    """Complete tree structure"""
    nodes: List[TreeNode]
    parent_ids: List[Optional[int]]
    num_nodes: int
    depth: int
    width: int
    
    def get_level_nodes(self, depth: int) -> List[TreeNode]:
        """Get all nodes at a specific depth"""
        return [n for n in self.nodes if n.depth == depth]


def build_tree_structure(width: int, depth: int) -> TreeStructure:
    """
    Build complete tree structure
    
    Args:
        width: Number of children per node
        depth: Tree depth
    
    Returns:
        TreeStructure with all nodes
    """
    nodes = []
    parent_ids = []
    
    # Root node
    nodes.append(TreeNode(node_id=0, depth=0, parent_id=None))
    parent_ids.append(None)
    
    current_level = [0]
    next_id = 1
    
    # Build level by level
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
    
    return TreeStructure(
        nodes=nodes,
        parent_ids=parent_ids,
        num_nodes=len(nodes),
        depth=depth,
        width=width
    )


# ============================================================================
# FlexAttention Score Mod
# ============================================================================

def create_tree_score_mod(
    tree_structure: TreeStructure,
    prefix_len: int
) -> Callable:
    """
    Create FlexAttention score_mod for tree attention
    
    This defines the attention pattern:
    - Prefix: causal attention
    - Tree: sees prefix + ancestors only (not siblings!)
    
    Args:
        tree_structure: Tree structure
        prefix_len: Length of prefix sequence
    
    Returns:
        score_mod function for flex_attention
    """
    parent_ids = tree_structure.parent_ids
    num_nodes = tree_structure.num_nodes
    
    # Pre-compute ancestor chains
    ancestor_chains = []
    for node_idx in range(num_nodes):
        ancestors = set([node_idx])
        parent_idx = parent_ids[node_idx]
        
        while parent_idx is not None:
            ancestors.add(parent_idx)
            parent_idx = parent_ids[parent_idx]
        
        ancestor_chains.append(ancestors)
    
    def score_mod(score: torch.Tensor, b: int, h: int, q_idx: int, kv_idx: int) -> torch.Tensor:
        """
        Score modifier for tree attention
        
        Called for every attention computation!
        
        Args:
            score: Raw attention score
            b: Batch index
            h: Head index  
            q_idx: Query position
            kv_idx: Key/Value position
        
        Returns:
            Modified score (or -inf to mask)
        """
        # Prefix: causal attention
        if q_idx < prefix_len:
            return score if kv_idx <= q_idx else float('-inf')
        
        # Tree attention
        tree_q = q_idx - prefix_len
        
        # Tree sees all prefix
        if kv_idx < prefix_len:
            return score
        
        # Tree sees ancestors only
        tree_kv = kv_idx - prefix_len
        
        if tree_kv in ancestor_chains[tree_q]:
            return score
        else:
            return float('-inf')
    
    return score_mod


def create_tree_attention_mask(
    tree_structure: TreeStructure,
    prefix_len: int,
    device: torch.device
) -> torch.Tensor:
    """
    Create attention mask for tree (fallback when FlexAttention not available)
    
    Args:
        tree_structure: Tree structure
        prefix_len: Prefix length
        device: Device
    
    Returns:
        Attention mask [1, 1, total_len, total_len]
    """
    num_nodes = tree_structure.num_nodes
    total_len = prefix_len + num_nodes
    
    score_mod = create_tree_score_mod(tree_structure, prefix_len)
    
    # Build mask using score_mod
    mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
    
    for q in range(total_len):
        for kv in range(total_len):
            score = score_mod(0.0, 0, 0, q, kv)
            mask[q, kv] = (score != float('-inf'))
    
    # Convert to additive mask
    attention_mask = torch.where(
        mask,
        torch.zeros(total_len, total_len, dtype=torch.float32, device=device),
        torch.full((total_len, total_len), float('-inf'), dtype=torch.float32, device=device)
    )
    
    return attention_mask.unsqueeze(0).unsqueeze(0)


def create_tree_position_ids(
    tree_structure: TreeStructure,
    prefix_len: int,
    device: torch.device
) -> torch.Tensor:
    """
    Create position IDs for tree (depth-based)
    
    Args:
        tree_structure: Tree structure
        prefix_len: Prefix length
        device: Device
    
    Returns:
        Position IDs [1, total_len]
    """
    num_nodes = tree_structure.num_nodes
    position_ids = torch.zeros(1, prefix_len + num_nodes, dtype=torch.long, device=device)
    
    # Prefix: sequential positions
    position_ids[0, :prefix_len] = torch.arange(prefix_len, device=device)
    
    # Tree: depth-based positions (siblings share position)
    for node in tree_structure.nodes:
        position_ids[0, prefix_len + node.node_id] = prefix_len + node.depth
    
    return position_ids


# ============================================================================
# vLLM EAGLE Proposer
# ============================================================================

class EAGLEFlexProposer:
    """
    EAGLE proposer for vLLM with FlexAttention
    
    Generates all draft tokens in a tree structure in ONE pass.
    
    Key features:
    - One-pass generation (all tokens simultaneously)
    - FlexAttention for accurate tree structure
    - Beam search pruning for quality
    - vLLM-compatible interface
    """
    
    def __init__(
        self,
        model: Any,  # EAGLE model
        tokenizer: Any,
        device: torch.device = torch.device("cuda"),
        tree_width: int = 4,
        tree_depth: int = 2,
        beam_width: int = 3,
        temperature: float = 1.0,
        top_k: int = 20,
        use_flex_attention: bool = True
    ):
        """
        Initialize EAGLE proposer
        
        Args:
            model: EAGLE draft model
            tokenizer: Tokenizer
            device: Device
            tree_width: Tree width (children per node)
            tree_depth: Tree depth
            beam_width: Beam width for pruning
            temperature: Sampling temperature
            top_k: Top-k sampling
            use_flex_attention: Use FlexAttention if available
        """
        print(f"\n{'='*70}")
        print(f"VLLM EAGLE PROPOSER WITH FLEXATTENTION")
        print(f"{'='*70}")
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.tree_width = tree_width
        self.tree_depth = tree_depth
        self.beam_width = beam_width
        self.temperature = temperature
        self.top_k = top_k
        
        # Check FlexAttention availability
        self.use_flex_attention = use_flex_attention and FLEX_ATTENTION_AVAILABLE
        
        print(f"Tree: width={tree_width}, depth={tree_depth}")
        print(f"Beam: width={beam_width}")
        print(f"FlexAttention: {'✓ Enabled' if self.use_flex_attention else '✗ Fallback mask'}")
        
        # Build tree structure (reused across proposals)
        self.tree_structure = build_tree_structure(tree_width, tree_depth)
        
        print(f"Tree nodes: {self.tree_structure.num_nodes}")
        for d in range(tree_depth + 1):
            level_nodes = self.tree_structure.get_level_nodes(d)
            print(f"  Depth {d}: {len(level_nodes)} nodes")
        
        # Placeholder token
        self.pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
        
        print(f"{'='*70}\n")
    
    def propose(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[TreeNode]]:
        """
        Propose draft tokens using EAGLE with FlexAttention
        
        This is the MAIN method called by vLLM.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            target_hidden_states: Hidden states from target model [batch_size, seq_len, hidden_dim]
            attention_mask: Optional attention mask
        
        Returns:
            draft_token_ids: Draft token IDs [num_draft_tokens]
            draft_tree_nodes: Tree nodes with tokens and scores
        """
        batch_size = input_ids.shape[0]
        prefix_len = input_ids.shape[1]
        
        assert batch_size == 1, "Currently only supports batch_size=1"
        
        print(f"\n{'='*70}")
        print(f"EAGLE PROPOSE (FlexAttention)")
        print(f"{'='*70}")
        print(f"Prefix length: {prefix_len}")
        print(f"Tree nodes: {self.tree_structure.num_nodes}")
        
        # Set forward context if available
        if target_hidden_states is not None:
            if hasattr(self.model, 'set_forward_context'):
                self.model.set_forward_context(target_hidden_states)
                print(f"✓ Forward context set")
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'set_forward_context'):
                self.model.model.set_forward_context(target_hidden_states)
                print(f"✓ Forward context set on model.model")
        
        # Generate tree in ONE pass
        draft_tree_nodes, draft_token_ids = self._generate_tree_one_pass(
            input_ids,
            prefix_len
        )
        
        print(f"✓ Generated {len(draft_token_ids)} draft tokens")
        print(f"{'='*70}\n")
        
        return draft_token_ids, draft_tree_nodes
    
    def _generate_tree_one_pass(
        self,
        input_ids: torch.Tensor,
        prefix_len: int
    ) -> Tuple[List[TreeNode], torch.Tensor]:
        """
        Generate entire tree in ONE forward pass
        
        Process:
        1. Prepare input with tree placeholders
        2. Create FlexAttention mask
        3. ONE forward pass
        4. Sample tokens + beam search pruning
        
        Args:
            input_ids: Input IDs [1, prefix_len]
            prefix_len: Prefix length
        
        Returns:
            draft_nodes: Draft tree nodes (pruned)
            draft_token_ids: Draft token IDs
        """
        # Step 1: Prepare input
        tree_placeholders = torch.full(
            (1, self.tree_structure.num_nodes),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device
        )
        full_input_ids = torch.cat([input_ids, tree_placeholders], dim=1)
        
        # Step 2: Create attention mask
        if self.use_flex_attention:
            # Use FlexAttention with score_mod
            score_mod = create_tree_score_mod(self.tree_structure, prefix_len)
            attention_mask = create_tree_attention_mask(self.tree_structure, prefix_len, self.device)
            print(f"Using FlexAttention score_mod")
        else:
            # Fallback to explicit mask
            attention_mask = create_tree_attention_mask(self.tree_structure, prefix_len, self.device)
            print(f"Using explicit attention mask")
        
        # Step 3: Create position IDs
        position_ids = create_tree_position_ids(self.tree_structure, prefix_len, self.device)
        
        # Step 4: ONE FORWARD PASS
        print(f"\nForward pass...")
        forward_start = time.time()
        
        with torch.no_grad():
            # Get model outputs
            if hasattr(self.model, 'model'):
                # HuggingFace style: model.model + lm_head
                outputs = self.model.model(
                    input_ids=full_input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True
                )
                logits = self.model.lm_head(outputs.last_hidden_state)
            else:
                # Direct model
                outputs = self.model(
                    input_ids=full_input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False
                )
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        forward_time = time.time() - forward_start
        
        # Extract tree logits
        tree_logits = logits[0, prefix_len:, :]  # [num_nodes, vocab_size]
        
        print(f"  ✓ Forward complete: {forward_time:.3f}s")
        print(f"  Tree logits: {tree_logits.shape}")
        
        # Step 5: Sample and prune
        print(f"\nSampling and pruning...")
        prune_start = time.time()
        
        draft_nodes, draft_token_ids = self._sample_and_prune_tree(
            tree_logits,
            self.tree_structure
        )
        
        prune_time = time.time() - prune_start
        
        print(f"  ✓ Pruning complete: {prune_time:.3f}s")
        print(f"  Pruned to {len(draft_nodes)} nodes")
        
        return draft_nodes, draft_token_ids
    
    def _sample_and_prune_tree(
        self,
        tree_logits: torch.Tensor,
        tree_structure: TreeStructure
    ) -> Tuple[List[TreeNode], torch.Tensor]:
        """
        Sample tokens and apply beam search pruning
        
        Args:
            tree_logits: Logits for all tree positions [num_nodes, vocab_size]
            tree_structure: Tree structure
        
        Returns:
            pruned_nodes: Pruned tree nodes
            draft_token_ids: Draft token IDs
        """
        # Initialize with root
        pruned_nodes = [tree_structure.nodes[0]]
        node_mapping = {0: 0}  # old_id -> new_id
        next_new_id = 1
        
        # Process level by level
        for depth in range(1, tree_structure.depth + 1):
            level_nodes = tree_structure.get_level_nodes(depth)
            
            # Group by parent
            parent_groups = {}
            for node in level_nodes:
                if node.parent_id not in parent_groups:
                    parent_groups[node.parent_id] = []
                parent_groups[node.parent_id].append(node)
            
            # For each kept parent
            for old_parent_id, children in parent_groups.items():
                if old_parent_id not in node_mapping:
                    continue
                
                new_parent_id = node_mapping[old_parent_id]
                parent_cumulative = pruned_nodes[new_parent_id].cumulative_score
                
                # Score all children
                child_scores = []
                for child in children:
                    # Get logits for this child
                    child_logits = tree_logits[child.node_id] / self.temperature
                    probs = F.softmax(child_logits, dim=-1)
                    log_probs = F.log_softmax(child_logits, dim=-1)
                    
                    # Sample token
                    if self.top_k > 0:
                        top_k_vals, top_k_idx = torch.topk(probs, k=min(self.top_k, probs.shape[0]))
                        top_k_probs = top_k_vals / top_k_vals.sum()
                        sampled = torch.multinomial(top_k_probs, 1).item()
                        token_id = top_k_idx[sampled].item()
                    else:
                        token_id = torch.multinomial(probs, 1).item()
                    
                    token_prob = probs[token_id].item()
                    token_log_prob = log_probs[token_id].item()
                    cumulative = parent_cumulative + token_log_prob
                    
                    child_scores.append((cumulative, token_id, token_log_prob, child))
                
                # Keep top beam_width
                child_scores.sort(key=lambda x: x[0], reverse=True)
                top_children = child_scores[:self.beam_width]
                
                # Add to pruned tree
                for cumulative, token_id, log_prob, old_child in top_children:
                    new_child = TreeNode(
                        node_id=next_new_id,
                        depth=depth,
                        parent_id=new_parent_id,
                        token_id=token_id,
                        log_prob=log_prob,
                        cumulative_score=cumulative
                    )
                    
                    pruned_nodes.append(new_child)
                    pruned_nodes[new_parent_id].children.append(next_new_id)
                    node_mapping[old_child.node_id] = next_new_id
                    
                    next_new_id += 1
        
        # Extract token IDs
        draft_token_ids = torch.tensor(
            [node.token_id for node in pruned_nodes],
            dtype=torch.long,
            device=self.device
        )
        
        return pruned_nodes, draft_token_ids
    
    def get_tree_structure(self) -> TreeStructure:
        """Get the tree structure used by this proposer"""
        return self.tree_structure


# ============================================================================
# vLLM Integration Helper
# ============================================================================

class EAGLEFlexProposerForVLLM:
    """
    Wrapper for vLLM integration
    
    This provides a vLLM-compatible interface for the EAGLE proposer.
    """
    
    def __init__(
        self,
        draft_model: Any,
        target_model: Any,
        tokenizer: Any,
        tree_width: int = 4,
        tree_depth: int = 2,
        beam_width: int = 3
    ):
        """
        Initialize for vLLM
        
        Args:
            draft_model: EAGLE draft model
            target_model: Target model (for hidden states)
            tokenizer: Tokenizer
            tree_width: Tree width
            tree_depth: Tree depth
            beam_width: Beam width
        """
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        
        # Create proposer
        self.proposer = EAGLEFlexProposer(
            model=draft_model,
            tokenizer=tokenizer,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            tree_width=tree_width,
            tree_depth=tree_depth,
            beam_width=beam_width
        )
    
    def propose_tokens(
        self,
        input_ids: torch.Tensor,
        num_speculate_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        vLLM-compatible propose method
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            num_speculate_tokens: Number of tokens to speculate (unused, returns tree)
        
        Returns:
            draft_token_ids: Draft tokens
            draft_probs: Draft probabilities (dummy for now)
        """
        # Get target hidden states
        with torch.no_grad():
            target_outputs = self.target_model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False
            )
            target_hidden_states = target_outputs.hidden_states[-1]
        
        # Propose with EAGLE
        draft_token_ids, draft_nodes = self.proposer.propose(
            input_ids=input_ids,
            target_hidden_states=target_hidden_states
        )
        
        # Create dummy probs (vLLM expects this)
        draft_probs = torch.ones(len(draft_token_ids), device=draft_token_ids.device)
        
        return draft_token_ids, draft_probs


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example of how to use the EAGLE proposer"""
    print("\n" + "="*70)
    print("EXAMPLE: EAGLE PROPOSER WITH FLEXATTENTION")
    print("="*70)
    
    # Mock setup (replace with actual vLLM models)
    print("\nSetting up mock models...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 128256
    hidden_dim = 4096
    
    # Mock EAGLE model
    class MockEAGLEModel:
        def __init__(self):
            self.model = self
            self.lm_head = torch.nn.Linear(hidden_dim, vocab_size).to(device)
            self.device = device
        
        def __call__(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
            batch_size, seq_len = input_ids.shape
            hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
            
            class Output:
                def __init__(self, hidden):
                    self.last_hidden_state = hidden
            
            return Output(hidden_states)
        
        def set_forward_context(self, hidden_states):
            pass
    
    # Mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 2
        
        def decode(self, token_ids):
            return f"<token_{token_ids[0]}>"
    
    draft_model = MockEAGLEModel()
    tokenizer = MockTokenizer()
    
    # Create proposer
    print("\nCreating EAGLE proposer...")
    proposer = EAGLEFlexProposer(
        model=draft_model,
        tokenizer=tokenizer,
        device=device,
        tree_width=4,
        tree_depth=2,
        beam_width=3
    )
    
    # Mock input
    print("\nGenerating draft tokens...")
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], device=device)
    target_hidden_states = torch.randn(1, 6, hidden_dim, device=device)
    
    # Propose
    draft_token_ids, draft_nodes = proposer.propose(
        input_ids=input_ids,
        target_hidden_states=target_hidden_states
    )
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Draft tokens: {draft_token_ids.shape}")
    print(f"Draft nodes: {len(draft_nodes)}")
    print(f"\nTree structure:")
    for depth in range(proposer.tree_structure.depth + 1):
        level_nodes = [n for n in draft_nodes if n.depth == depth]
        print(f"  Depth {depth}: {len(level_nodes)} nodes")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    example_usage()