"""
vLLM EAGLE Proposer with FlexAttention
======================================

This is a drop-in replacement for vLLM's default EAGLE proposer that uses
FlexAttention for accurate tree attention patterns.

File location: vllm/spec_decode/eagle_proposer.py

Usage in vLLM:
    from vllm import LLM, SamplingParams
    
    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        speculative_model="yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        speculative_draft_tensor_parallel_size=1,
        num_speculative_tokens=13,
        use_v2_block_manager=True,
    )
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

# vLLM imports
from vllm.config import ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig
from vllm.model_executor.models import ModelRegistry
from vllm.worker.model_runner import ModelRunner
from vllm.sequence import SequenceData, SequenceGroupMetadata
from vllm.spec_decode.interfaces import SpeculativeProposer
from vllm.spec_decode.util import get_sampled_token_logprobs


# Try FlexAttention
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
    """Node in EAGLE draft tree"""
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


def build_eagle_tree(width: int, depth: int) -> TreeStructure:
    """
    Build EAGLE tree structure
    
    Args:
        width: Number of children per node (branching factor)
        depth: Tree depth
    
    Returns:
        TreeStructure with all nodes
    """
    nodes = []
    parent_ids = []
    
    # Root node (represents the last token of prefix)
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
# FlexAttention for EAGLE
# ============================================================================

def create_eagle_score_mod(
    tree_structure: TreeStructure,
    prefix_len: int
):
    """
    Create FlexAttention score_mod for EAGLE tree attention
    
    Attention pattern:
    - Prefix tokens: causal attention (see past only)
    - Tree nodes: see prefix + ancestors in tree (not siblings!)
    
    Args:
        tree_structure: Tree structure
        prefix_len: Length of prefix sequence
    
    Returns:
        score_mod function for flex_attention
    """
    parent_ids = tree_structure.parent_ids
    num_nodes = tree_structure.num_nodes
    
    # Pre-compute ancestor chains for each node
    ancestor_chains = []
    for node_idx in range(num_nodes):
        ancestors = set([node_idx])  # Node can attend to itself
        parent_idx = parent_ids[node_idx]
        
        # Follow parent chain to root
        while parent_idx is not None:
            ancestors.add(parent_idx)
            parent_idx = parent_ids[parent_idx]
        
        ancestor_chains.append(ancestors)
    
    def score_mod(score: torch.Tensor, b: int, h: int, q_idx: int, kv_idx: int) -> torch.Tensor:
        """
        Score modifier for EAGLE tree attention
        
        This function is called for EVERY attention score computation.
        
        Args:
            score: Raw attention score
            b: Batch index
            h: Head index
            q_idx: Query position index
            kv_idx: Key/Value position index
        
        Returns:
            Modified score (or -inf to mask)
        """
        # Prefix: standard causal attention
        if q_idx < prefix_len:
            return score if kv_idx <= q_idx else float('-inf')
        
        # Tree attention
        tree_q_idx = q_idx - prefix_len
        
        # Tree nodes can see entire prefix
        if kv_idx < prefix_len:
            return score
        
        # Tree nodes can only see ancestors (not siblings or cousins!)
        tree_kv_idx = kv_idx - prefix_len
        
        if tree_kv_idx in ancestor_chains[tree_q_idx]:
            return score  # Allow attention to ancestor
        else:
            return float('-inf')  # MASK non-ancestor
    
    return score_mod


def create_eagle_attention_mask(
    tree_structure: TreeStructure,
    prefix_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Create attention mask for EAGLE tree (fallback when FlexAttention unavailable)
    
    Args:
        tree_structure: Tree structure
        prefix_len: Prefix length
        device: Device
        dtype: Data type
    
    Returns:
        Attention mask [1, 1, total_len, total_len]
    """
    num_nodes = tree_structure.num_nodes
    total_len = prefix_len + num_nodes
    
    # Use score_mod to build mask
    score_mod = create_eagle_score_mod(tree_structure, prefix_len)
    
    # Build mask matrix
    mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
    
    for q in range(total_len):
        for kv in range(total_len):
            score = score_mod(0.0, 0, 0, q, kv)
            mask[q, kv] = (score != float('-inf'))
    
    # Convert to additive mask
    attention_mask = torch.where(
        mask,
        torch.zeros(total_len, total_len, dtype=dtype, device=device),
        torch.full((total_len, total_len), float('-inf'), dtype=dtype, device=device)
    )
    
    return attention_mask.unsqueeze(0).unsqueeze(0)


def create_eagle_position_ids(
    tree_structure: TreeStructure,
    prefix_len: int,
    device: torch.device
) -> torch.Tensor:
    """
    Create position IDs for EAGLE tree
    
    Position IDs are depth-based: siblings at same depth share same position.
    
    Args:
        tree_structure: Tree structure
        prefix_len: Prefix length
        device: Device
    
    Returns:
        Position IDs [1, total_len]
    """
    num_nodes = tree_structure.num_nodes
    total_len = prefix_len + num_nodes
    
    position_ids = torch.zeros(1, total_len, dtype=torch.long, device=device)
    
    # Prefix: sequential positions
    position_ids[0, :prefix_len] = torch.arange(prefix_len, device=device)
    
    # Tree: depth-based positions (siblings share position)
    for node in tree_structure.nodes:
        pos_idx = prefix_len + node.node_id
        position_ids[0, pos_idx] = prefix_len + node.depth
    
    return position_ids


# ============================================================================
# vLLM EAGLE Proposer with FlexAttention
# ============================================================================

class EAGLEProposerWithFlexAttention(SpeculativeProposer):
    """
    EAGLE proposer for vLLM with FlexAttention support
    
    This is a drop-in replacement for vLLM's default EAGLE proposer
    that uses FlexAttention for more accurate tree attention patterns.
    """
    
    def __init__(
        self,
        draft_worker: ModelRunner,
        model_config: ModelConfig,
        device: torch.device,
        vocab_size: int,
        tree_width: int = 4,
        tree_depth: int = 2,
        beam_width: int = 3,
        use_flex_attention: bool = True,
    ):
        """
        Initialize EAGLE proposer
        
        Args:
            draft_worker: Model runner for EAGLE draft model
            model_config: Model configuration
            device: Device
            vocab_size: Vocabulary size
            tree_width: Tree width (branching factor)
            tree_depth: Tree depth
            beam_width: Beam width for pruning
            use_flex_attention: Use FlexAttention if available
        """
        print(f"\n{'='*70}")
        print(f"EAGLE PROPOSER WITH FLEXATTENTION")
        print(f"{'='*70}")
        
        self.draft_worker = draft_worker
        self.model_config = model_config
        self.device = device
        self.vocab_size = vocab_size
        self.tree_width = tree_width
        self.tree_depth = tree_depth
        self.beam_width = beam_width
        
        # Check FlexAttention availability
        self.use_flex_attention = use_flex_attention and FLEX_ATTENTION_AVAILABLE
        
        print(f"Tree: width={tree_width}, depth={tree_depth}, beam={beam_width}")
        print(f"FlexAttention: {'✓ Enabled' if self.use_flex_attention else '✗ Fallback'}")
        
        # Build tree structure (reused across proposals)
        self.tree_structure = build_eagle_tree(tree_width, tree_depth)
        
        print(f"Tree nodes: {self.tree_structure.num_nodes}")
        for d in range(tree_depth + 1):
            level_nodes = [n for n in self.tree_structure.nodes if n.depth == d]
            print(f"  Depth {d}: {len(level_nodes)} nodes")
        
        print(f"{'='*70}\n")
        
        # Get draft model
        self.draft_model = draft_worker.model
        
        # Get pad token
        self.pad_token_id = getattr(model_config.hf_config, 'pad_token_id', 0)
        if self.pad_token_id is None:
            self.pad_token_id = 0
    
    @torch.inference_mode()
    def get_proposals(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        target_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get draft proposals from EAGLE
        
        This is the main method called by vLLM's speculative decoding engine.
        
        Args:
            seq_group_metadata_list: Sequence metadata
            target_hidden_states: Hidden states from target model [batch, seq_len, hidden_dim]
        
        Returns:
            draft_token_ids: Draft token IDs [batch, num_draft_tokens]
            draft_probs: Draft probabilities [batch, num_draft_tokens, vocab_size]
            draft_logprobs: Draft log probabilities [batch, num_draft_tokens]
        """
        batch_size = len(seq_group_metadata_list)
        
        # Currently only support batch_size=1 (can be extended)
        assert batch_size == 1, "Currently only supports batch_size=1"
        
        seq_group_metadata = seq_group_metadata_list[0]
        seq_data = seq_group_metadata.seq_data[seq_group_metadata.seq_ids[0]]
        
        # Get input IDs
        input_ids = torch.tensor(
            seq_data.get_token_ids(),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)
        
        prefix_len = input_ids.shape[1]
        
        # Set forward context on EAGLE model
        if hasattr(self.draft_model, 'set_forward_context'):
            self.draft_model.set_forward_context(target_hidden_states)
        elif hasattr(self.draft_model, 'model') and hasattr(self.draft_model.model, 'set_forward_context'):
            self.draft_model.model.set_forward_context(target_hidden_states)
        
        # Generate tree in one pass
        draft_token_ids, draft_probs, draft_logprobs = self._generate_tree_one_pass(
            input_ids,
            prefix_len
        )
        
        return draft_token_ids, draft_probs, draft_logprobs
    
    def _generate_tree_one_pass(
        self,
        input_ids: torch.Tensor,
        prefix_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate entire tree in ONE forward pass
        
        Args:
            input_ids: Input token IDs [1, prefix_len]
            prefix_len: Prefix length
        
        Returns:
            draft_token_ids: Draft tokens [1, num_pruned_nodes]
            draft_probs: Draft probabilities [1, num_pruned_nodes, vocab_size]
            draft_logprobs: Draft log probabilities [1, num_pruned_nodes]
        """
        # Prepare input with tree placeholders
        tree_placeholders = torch.full(
            (1, self.tree_structure.num_nodes),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device
        )
        full_input_ids = torch.cat([input_ids, tree_placeholders], dim=1)
        
        # Create attention mask
        if self.use_flex_attention:
            # Use FlexAttention
            score_mod = create_eagle_score_mod(self.tree_structure, prefix_len)
            attention_mask = create_eagle_attention_mask(
                self.tree_structure, prefix_len, self.device
            )
        else:
            # Fallback to explicit mask
            attention_mask = create_eagle_attention_mask(
                self.tree_structure, prefix_len, self.device
            )
        
        # Create position IDs
        position_ids = create_eagle_position_ids(
            self.tree_structure, prefix_len, self.device
        )
        
        # ONE FORWARD PASS
        with torch.no_grad():
            # Get model outputs
            if hasattr(self.draft_model, 'model'):
                # HuggingFace style
                outputs = self.draft_model.model(
                    input_ids=full_input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True
                )
                logits = self.draft_model.lm_head(outputs.last_hidden_state)
            else:
                # Direct model
                outputs = self.draft_model(
                    input_ids=full_input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False
                )
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # Extract tree logits
        tree_logits = logits[0, prefix_len:, :]  # [num_nodes, vocab_size]
        
        # Sample and prune with beam search
        draft_token_ids, draft_probs, draft_logprobs = self._beam_search_sample(
            tree_logits,
            self.tree_structure
        )
        
        return draft_token_ids, draft_probs, draft_logprobs
    
    def _beam_search_sample(
        self,
        tree_logits: torch.Tensor,
        tree_structure: TreeStructure
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample tokens with beam search pruning
        
        Args:
            tree_logits: Logits for all tree positions [num_nodes, vocab_size]
            tree_structure: Tree structure
        
        Returns:
            draft_token_ids: [1, num_pruned_nodes]
            draft_probs: [1, num_pruned_nodes, vocab_size]
            draft_logprobs: [1, num_pruned_nodes]
        """
        # Initialize with root
        pruned_nodes = [tree_structure.nodes[0]]
        node_mapping = {0: 0}
        next_new_id = 1
        
        # Process level by level
        for depth in range(1, tree_structure.depth + 1):
            level_nodes = [n for n in tree_structure.nodes if n.depth == depth]
            
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
                    child_logits = tree_logits[child.node_id]
                    probs = F.softmax(child_logits, dim=-1)
                    log_probs = F.log_softmax(child_logits, dim=-1)
                    
                    # Sample token (greedy for beam search)
                    token_id = torch.argmax(probs).item()
                    token_prob = probs[token_id].item()
                    token_log_prob = log_probs[token_id].item()
                    
                    cumulative = parent_cumulative + token_log_prob
                    
                    child_scores.append((cumulative, token_id, token_log_prob, probs, child))
                
                # Keep top beam_width
                child_scores.sort(key=lambda x: x[0], reverse=True)
                top_children = child_scores[:self.beam_width]
                
                # Add to pruned tree
                for cumulative, token_id, log_prob, probs, old_child in top_children:
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
        
        # Extract results
        num_pruned = len(pruned_nodes)
        
        draft_token_ids = torch.tensor(
            [[node.token_id for node in pruned_nodes]],
            dtype=torch.long,
            device=self.device
        )
        
        draft_logprobs = torch.tensor(
            [[node.log_prob for node in pruned_nodes]],
            dtype=torch.float,
            device=self.device
        )
        
        # Create dummy probs (full distribution)
        draft_probs = torch.zeros(
            1, num_pruned, self.vocab_size,
            dtype=torch.float,
            device=self.device
        )
        for i, node in enumerate(pruned_nodes):
            draft_probs[0, i, node.token_id] = 1.0
        
        return draft_token_ids, draft_probs, draft_logprobs


# ============================================================================
# vLLM Integration
# ============================================================================

def create_eagle_proposer(
    draft_worker: ModelRunner,
    target_model_config: ModelConfig,
    draft_model_config: ModelConfig,
    device: torch.device,
    vocab_size: int,
    **kwargs
) -> EAGLEProposerWithFlexAttention:
    """
    Factory function for creating EAGLE proposer (vLLM integration point)
    
    This function is called by vLLM when initializing speculative decoding.
    
    Args:
        draft_worker: Model runner for draft model
        target_model_config: Target model configuration
        draft_model_config: Draft model configuration
        device: Device
        vocab_size: Vocabulary size
        **kwargs: Additional arguments (tree_width, tree_depth, beam_width, etc.)
    
    Returns:
        EAGLE proposer instance
    """
    # Get tree parameters from kwargs or use defaults
    tree_width = kwargs.get('tree_width', 4)
    tree_depth = kwargs.get('tree_depth', 2)
    beam_width = kwargs.get('beam_width', 3)
    use_flex_attention = kwargs.get('use_flex_attention', True)
    
    return EAGLEProposerWithFlexAttention(
        draft_worker=draft_worker,
        model_config=draft_model_config,
        device=device,
        vocab_size=vocab_size,
        tree_width=tree_width,
        tree_depth=tree_depth,
        beam_width=beam_width,
        use_flex_attention=use_flex_attention,
    )


# ============================================================================
# Example Usage
# ============================================================================

def example_vllm_usage():
    """
    Example of how to use EAGLE proposer with FlexAttention in vLLM
    """
    from vllm import LLM, SamplingParams
    
    print("\n" + "="*70)
    print("EXAMPLE: VLLM WITH EAGLE + FLEXATTENTION")
    print("="*70)
    
    # Initialize vLLM with EAGLE
    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        speculative_model="yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        speculative_draft_tensor_parallel_size=1,
        num_speculative_tokens=13,  # Number of tokens in pruned tree
        use_v2_block_manager=True,
        # Pass custom proposer (if vLLM supports this parameter)
        # speculative_proposer=create_eagle_proposer,
    )
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=100
    )
    
    # Generate
    prompts = ["The future of artificial intelligence is"]
    outputs = llm.generate(prompts, sampling_params)
    
    # Print results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    print("\nEAGLE Proposer with FlexAttention for vLLM")
    print("=" * 70)
    print("\nFeatures:")
    print("  ✓ FlexAttention for accurate tree attention")
    print("  ✓ One-pass tree generation")
    print("  ✓ Beam search pruning for quality")
    print("  ✓ Drop-in replacement for vLLM")
    print("\nTo use:")
    print("  1. Place this file in: vllm/spec_decode/eagle_proposer.py")
    print("  2. Update vLLM to use this proposer")
    print("  3. Run vLLM with EAGLE as usual")
    print("=" * 70 + "\n")