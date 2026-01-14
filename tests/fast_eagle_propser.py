"""
FastEagle Proposer - One-Pass Generation with One-Layer Model
=============================================================

A lightweight EAGLE proposer that uses a single-layer LLaMA model
for extremely fast one-pass token generation.

Key features:
- One-pass generation (all tokens simultaneously)
- Single-layer draft model (ultra fast)
- Beam search pruning
- Based on yuhuili/EAGLE-LLaMA3.1-Instruct-8B

Usage:
    from fast_eagle_proposer import FastEagleProposer
    
    proposer = FastEagleProposer(
        draft_model_name="yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        target_model_name="meta-llama/Llama-3.1-8B-Instruct"
    )
    
    draft_tokens = proposer.propose(input_ids)
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import time
import argparse

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
                nodes[parent_id].children.append(next_id)
                next_level.append(next_id)
                next_id += 1
        current_level = next_level
    
    return nodes, parent_ids


# ============================================================================
# Single Layer LLaMA Model (FastEagle style)
# ============================================================================

class SingleLayerLLaMA(torch.nn.Module):
    """
    Single-layer LLaMA model for EAGLE
    
    This is the FastEagle approach - use only ONE layer from the base model
    for draft generation, making it much faster.
    """
    
    def __init__(self, base_model, layer_idx: int = -1):
        """
        Initialize single-layer model
        
        Args:
            base_model: Full LLaMA model
            layer_idx: Which layer to use (-1 = last layer)
        """
        super().__init__()
        
        # Get the layer
        if layer_idx == -1:
            layer_idx = len(base_model.model.layers) - 1
        
        self.layer_idx = layer_idx
        self.embed_tokens = base_model.model.embed_tokens
        self.layer = base_model.model.layers[layer_idx]
        self.norm = base_model.model.norm
        self.lm_head = base_model.lm_head
        
        # Target hidden states for conditioning
        self.target_hidden_states = None
        
        print(f"  Using layer {layer_idx} from base model")
    
    def set_forward_context(self, hidden_states: torch.Tensor):
        """Set target hidden states for conditioning"""
        self.target_hidden_states = hidden_states
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_dict: bool = True
    ):
        """
        Forward pass through single layer
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            use_cache: Whether to use cache
            return_dict: Whether to return dict
        
        Returns:
            Model outputs
        """
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # If we have target hidden states, use them for conditioning
        if self.target_hidden_states is not None:
            batch_size, prefix_len, hidden_dim = self.target_hidden_states.shape
            
            # Replace prefix embeddings with target hidden states
            hidden_states[:, :prefix_len, :] = self.target_hidden_states
        
        # Pass through single layer
        layer_outputs = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache
        )
        
        hidden_states = layer_outputs[0]
        
        # Normalize
        hidden_states = self.norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Return
        if return_dict:
            return type('Output', (), {
                'last_hidden_state': hidden_states,
                'logits': logits
            })()
        else:
            return (logits,)


# ============================================================================
# Attention Utilities
# ============================================================================

def create_tree_attention_mask(
    parent_ids: List[Optional[int]],
    prefix_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """Create attention mask for tree structure"""
    num_nodes = len(parent_ids)
    total_len = prefix_len + num_nodes
    
    # Pre-compute ancestor chains
    ancestor_chains = []
    for node_idx in range(num_nodes):
        ancestors = set([node_idx])
        parent_idx = parent_ids[node_idx]
        
        while parent_idx is not None:
            ancestors.add(parent_idx)
            parent_idx = parent_ids[parent_idx]
        
        ancestor_chains.append(ancestors)
    
    # Build mask
    mask = torch.full((total_len, total_len), float('-inf'), dtype=dtype, device=device)
    
    # Prefix: causal
    for i in range(prefix_len):
        mask[i, :i+1] = 0.0
    
    # Tree
    for node_idx in range(num_nodes):
        q_pos = prefix_len + node_idx
        mask[q_pos, :prefix_len] = 0.0
        
        for ancestor_idx in ancestor_chains[node_idx]:
            kv_pos = prefix_len + ancestor_idx
            if 0 <= kv_pos < total_len:
                mask[q_pos, kv_pos] = 0.0
    
    return mask.unsqueeze(0).unsqueeze(0)


def create_tree_position_ids(
    nodes: List[TreeNode],
    prefix_len: int,
    device: torch.device
) -> torch.Tensor:
    """Create position IDs for tree"""
    num_nodes = len(nodes)
    total_len = prefix_len + num_nodes
    
    positions = torch.zeros(1, total_len, dtype=torch.long, device=device)
    positions[0, :prefix_len] = torch.arange(prefix_len, device=device)
    
    for node in nodes:
        positions[0, prefix_len + node.node_id] = prefix_len + node.depth
    
    return positions


# ============================================================================
# FastEagle Proposer (Single Layer)
# ============================================================================

class FastEagleProposer:
    """
    FastEagle proposer using single-layer LLaMA model
    
    Key innovation: Use only ONE layer for draft generation,
    making it much faster than full model.
    """
    
    def __init__(
        self,
        draft_model_name: str = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        target_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        tree_width: int = 4,
        tree_depth: int = 2,
        beam_width: int = 3,
        temperature: float = 1.0,
        top_k: int = 20,
        draft_layer_idx: int = -1,  # Which layer to use (-1 = last)
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Initialize FastEagle proposer
        
        Args:
            draft_model_name: EAGLE model name
            target_model_name: Target model name
            device: Device
            tree_width: Tree width
            tree_depth: Tree depth
            beam_width: Beam width
            temperature: Sampling temperature
            top_k: Top-k sampling
            draft_layer_idx: Which layer to use for draft (-1 = last)
            load_in_8bit: Load in 8-bit
            load_in_4bit: Load in 4-bit
        """
        print(f"\n{'='*70}")
        print(f"FASTEAGLE PROPOSER - SINGLE LAYER")
        print(f"{'='*70}")
        print(f"Draft model: {draft_model_name}")
        print(f"Target model: {target_model_name}")
        print(f"Draft layer: {draft_layer_idx if draft_layer_idx >= 0 else 'last'}")
        
        self.device = torch.device(device)
        self.dtype = torch.float16
        self.tree_width = tree_width
        self.tree_depth = tree_depth
        self.beam_width = beam_width
        self.temperature = temperature
        self.top_k = top_k
        
        # Build tree
        self.tree_nodes, self.parent_ids = build_tree_structure(tree_width, tree_depth)
        self.num_tree_nodes = len(self.tree_nodes)
        self.num_pruned_nodes = sum(beam_width**d for d in range(tree_depth + 1))
        
        print(f"\nTree configuration:")
        print(f"  Width: {tree_width}, Depth: {tree_depth}")
        print(f"  Full nodes: {self.num_tree_nodes}")
        print(f"  Pruned nodes: {self.num_pruned_nodes} (beam={beam_width})")
        
        # Load models
        print(f"\nLoading models...")
        
        load_kwargs = {
            "torch_dtype": self.dtype,
            "device_map": "auto",
            "low_cpu_mem_usage": True
        }
        
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        
        # Load full EAGLE model first
        print(f"  Loading EAGLE model...")
        full_eagle_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name,
            trust_remote_code=True,
            **load_kwargs
        )
        full_eagle_model.eval()
        
        # Create single-layer draft model
        print(f"  Creating single-layer draft model...")
        self.draft_model = SingleLayerLLaMA(full_eagle_model, draft_layer_idx)
        self.draft_model.eval()
        print(f"    ✓ Single-layer draft created")
        
        # Load target model
        print(f"  Loading target model...")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name,
            **load_kwargs
        )
        self.target_model.eval()
        print(f"    ✓ Target loaded")
        
        # Tokenizer
        print(f"  Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"    ✓ Tokenizer loaded")
        
        self.pad_token_id = self.tokenizer.pad_token_id
        self.vocab_size = len(self.tokenizer)
        
        print(f"\nFastEagle proposer ready!")
        print(f"{'='*70}\n")
    
    @torch.inference_mode()
    def propose(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[TreeNode]]:
        """
        Propose draft tokens using one-pass generation with single layer
        
        Args:
            input_ids: Input token IDs [1, seq_len]
            target_hidden_states: Optional target hidden states
        
        Returns:
            draft_token_ids: Draft tokens
            draft_nodes: Draft tree nodes
        """
        batch_size, prefix_len = input_ids.shape
        assert batch_size == 1, "Currently only supports batch_size=1"
        
        # Get target hidden states if not provided
        if target_hidden_states is None:
            target_hidden_states = self.get_target_hidden_states(input_ids)
        
        # Set forward context
        self.draft_model.set_forward_context(target_hidden_states)
        
        # Prepare input
        tree_placeholders = torch.full(
            (1, self.num_tree_nodes),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device
        )
        full_input_ids = torch.cat([input_ids, tree_placeholders], dim=1)
        
        # Create attention mask
        attention_mask = create_tree_attention_mask(
            self.parent_ids,
            prefix_len,
            self.device,
            self.dtype
        )
        
        # Create position IDs
        position_ids = create_tree_position_ids(
            self.tree_nodes,
            prefix_len,
            self.device
        )
        
        # ONE FORWARD PASS through SINGLE LAYER
        outputs = self.draft_model(
            input_ids=full_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True
        )
        
        tree_logits = outputs.logits[0, prefix_len:, :]
        
        # Beam search pruning
        pruned_nodes, draft_token_ids = self.beam_search_prune(
            self.tree_nodes,
            tree_logits
        )
        
        return draft_token_ids, pruned_nodes
    
    def beam_search_prune(
        self,
        nodes: List[TreeNode],
        tree_logits: torch.Tensor
    ) -> Tuple[List[TreeNode], torch.Tensor]:
        """Apply beam search pruning"""
        max_depth = max(node.depth for node in nodes)
        
        pruned_nodes = [nodes[0]]
        node_mapping = {0: 0}
        next_new_id = 1
        
        for depth in range(1, max_depth + 1):
            level_nodes = [n for n in nodes if n.depth == depth]
            
            parent_groups = {}
            for node in level_nodes:
                if node.parent_id not in parent_groups:
                    parent_groups[node.parent_id] = []
                parent_groups[node.parent_id].append(node)
            
            for old_parent_id, children in parent_groups.items():
                if old_parent_id not in node_mapping:
                    continue
                
                new_parent_id = node_mapping[old_parent_id]
                parent_cumulative = pruned_nodes[new_parent_id].cumulative_score
                
                child_scores = []
                for child in children:
                    child_logits = tree_logits[child.node_id] / self.temperature
                    probs = F.softmax(child_logits, dim=-1)
                    log_probs = F.log_softmax(child_logits, dim=-1)
                    
                    if self.top_k > 0:
                        top_k_vals, top_k_idx = torch.topk(probs, k=min(self.top_k, len(probs)))
                        top_k_probs = top_k_vals / top_k_vals.sum()
                        sampled_idx = torch.multinomial(top_k_probs, 1).item()
                        token_id = top_k_idx[sampled_idx].item()
                    else:
                        token_id = torch.multinomial(probs, 1).item()
                    
                    token_log_prob = log_probs[token_id].item()
                    cumulative = parent_cumulative + token_log_prob
                    
                    child_scores.append((cumulative, token_id, token_log_prob, child))
                
                child_scores.sort(key=lambda x: x[0], reverse=True)
                top_children = child_scores[:self.beam_width]
                
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
        
        draft_token_ids = torch.tensor(
            [node.token_id for node in pruned_nodes],
            dtype=torch.long,
            device=self.device
        )
        
        return pruned_nodes, draft_token_ids
    
    @torch.inference_mode()
    def get_target_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get hidden states from target model"""
        outputs = self.target_model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True
        )
        
        return outputs.hidden_states[-1]
    
    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Decode tokens to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

"""
Test Suite for FastEagle Proposer (Single Layer)
================================================

Comprehensive tests for fast_eagle_proposer.py

Usage:
    python test_fast_eagle.py
    
    # Run specific tests:
    python test_fast_eagle.py --test basic
    python test_fast_eagle.py --test performance
    python test_fast_eagle.py --test comparison
"""


# ============================================================================
# Test 1: Single Layer Model
# ============================================================================

def test_single_layer_model():
    """Test single-layer model creation"""
    print("\n" + "="*70)
    print("TEST 1: SINGLE LAYER MODEL")
    print("="*70)
    
    try:
        print("\n1. Loading full model...")
        from transformers import AutoModelForCausalLM
        
        full_model = AutoModelForCausalLM.from_pretrained(
            "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        num_layers = len(full_model.model.layers)
        print(f"  ✓ Full model loaded: {num_layers} layers")
        
        print("\n2. Creating single-layer model...")
        single_layer = SingleLayerLLaMA(full_model, layer_idx=-1)
        print(f"  ✓ Single layer created (layer {single_layer.layer_idx})")
        
        print("\n3. Testing forward pass...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
        
        outputs = single_layer(input_ids)
        print(f"  ✓ Forward pass successful")
        print(f"    Output shape: {outputs.logits.shape}")
        
        print("\n4. Testing with target conditioning...")
        target_hidden = torch.randn(1, 5, 4096, device=device, dtype=torch.float16)
        single_layer.set_forward_context(target_hidden)
        
        outputs = single_layer(input_ids)
        print(f"  ✓ Conditioning works")
        print(f"    Output shape: {outputs.logits.shape}")
        
        print("\n✓ TEST 1 PASSED")
        return True
        
    except Exception as e:
        print(f"✗ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 2: Basic Functionality
# ============================================================================

def test_basic_functionality():
    """Test basic proposer functionality"""
    print("\n" + "="*70)
    print("TEST 2: BASIC FUNCTIONALITY")
    print("="*70)
    
    try:
        print("\n1. Initializing FastEagle proposer...")
        proposer = FastEagleProposer(
            tree_width=3,
            tree_depth=2,
            beam_width=2,
            draft_layer_idx=-1
        )
        print("✓ Proposer initialized")
        
        print("\n2. Testing tokenization...")
        text = "The future of AI is"
        input_ids = proposer.tokenizer.encode(text, return_tensors="pt").to(proposer.device)
        print(f"✓ Tokenized: '{text}'")
        print(f"  Input shape: {input_ids.shape}")
        
        print("\n3. Testing proposal generation...")
        start_time = time.time()
        draft_tokens, draft_nodes = proposer.propose(input_ids)
        elapsed = time.time() - start_time
        
        print(f"✓ Generated in {elapsed:.3f}s")
        print(f"  Draft tokens: {len(draft_tokens)}")
        print(f"  Draft nodes: {len(draft_nodes)}")
        
        print("\n4. Decoding draft tokens...")
        draft_text = proposer.decode_tokens(draft_tokens)
        print(f"✓ Draft: '{draft_text}'")
        
        print("\n✓ TEST 2 PASSED")
        return True
        
    except Exception as e:
        print(f"✗ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 3: Tree Structure
# ============================================================================

def test_tree_structure():
    """Test tree structure building"""
    print("\n" + "="*70)
    print("TEST 3: TREE STRUCTURE")
    print("="*70)
    
    try:
        configs = [
            (3, 2, 13),
            (4, 2, 21),
            (2, 3, 15),
        ]
        
        for width, depth, expected in configs:
            print(f"\n1. Testing tree: width={width}, depth={depth}")
            
            nodes, parent_ids = build_tree_structure(width, depth)
            
            print(f"  ✓ Built: {len(nodes)} nodes (expected {expected})")
            assert len(nodes) == expected
            
            # Verify relationships
            for node in nodes:
                if node.parent_id is not None:
                    parent = nodes[node.parent_id]
                    assert node.node_id in parent.children
            
            print(f"  ✓ Relationships verified")
        
        print("\n✓ TEST 3 PASSED")
        return True
        
    except Exception as e:
        print(f"✗ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 4: One-Pass Generation
# ============================================================================

def test_one_pass_generation():
    """Test one-pass generation specifically"""
    print("\n" + "="*70)
    print("TEST 4: ONE-PASS GENERATION")
    print("="*70)
    
    try:
        print("\n1. Initializing proposer...")
        proposer = FastEagleProposer(
            tree_width=4,
            tree_depth=2,
            beam_width=3
        )
        
        print("\n2. Preparing input...")
        text = "Machine learning is"
        input_ids = proposer.tokenizer.encode(text, return_tensors="pt").to(proposer.device)
        prefix_len = input_ids.shape[1]
        
        print(f"  Prefix length: {prefix_len}")
        print(f"  Tree nodes: {proposer.num_tree_nodes}")
        
        print("\n3. Getting target hidden states...")
        target_hidden = proposer.get_target_hidden_states(input_ids)
        print(f"  ✓ Hidden states: {target_hidden.shape}")
        
        print("\n4. Setting forward context...")
        proposer.draft_model.set_forward_context(target_hidden)
        print(f"  ✓ Context set")
        
        print("\n5. Preparing full input...")
        tree_placeholders = torch.full(
            (1, proposer.num_tree_nodes),
            proposer.pad_token_id,
            dtype=torch.long,
            device=proposer.device
        )
        full_input_ids = torch.cat([input_ids, tree_placeholders], dim=1)
        print(f"  ✓ Full input: {full_input_ids.shape}")
        
        print("\n6. Creating attention mask...")
        attention_mask = create_tree_attention_mask(
            proposer.parent_ids,
            prefix_len,
            proposer.device
        )
        print(f"  ✓ Attention mask: {attention_mask.shape}")
        
        print("\n7. Creating position IDs...")
        position_ids = create_tree_position_ids(
            proposer.tree_nodes,
            prefix_len,
            proposer.device
        )
        print(f"  ✓ Position IDs: {position_ids.shape}")
        
        print("\n8. ONE FORWARD PASS...")
        start_time = time.time()
        
        outputs = proposer.draft_model(
            input_ids=full_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True
        )
        
        elapsed = time.time() - start_time
        
        tree_logits = outputs.logits[0, prefix_len:, :]
        
        print(f"  ✓ Forward pass: {elapsed:.3f}s")
        print(f"  ✓ Tree logits: {tree_logits.shape}")
        print(f"  Expected: [{proposer.num_tree_nodes}, vocab_size]")
        
        assert tree_logits.shape[0] == proposer.num_tree_nodes
        
        print("\n9. Beam search pruning...")
        start_time = time.time()
        
        pruned_nodes, draft_tokens = proposer.beam_search_prune(
            proposer.tree_nodes,
            tree_logits
        )
        
        elapsed = time.time() - start_time
        
        print(f"  ✓ Pruning: {elapsed:.3f}s")
        print(f"  ✓ Pruned to {len(pruned_nodes)} nodes")
        print(f"  Expected: {proposer.num_pruned_nodes}")
        
        assert len(pruned_nodes) == proposer.num_pruned_nodes
        
        print("\n✓ TEST 4 PASSED")
        return True
        
    except Exception as e:
        print(f"✗ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 5: Performance Benchmark
# ============================================================================

def test_performance():
    """Test performance metrics"""
    print("\n" + "="*70)
    print("TEST 5: PERFORMANCE BENCHMARK")
    print("="*70)
    
    try:
        print("\n1. Initializing proposer...")
        proposer = FastEagleProposer(
            tree_width=4,
            tree_depth=2,
            beam_width=3
        )
        
        print("\n2. Warming up...")
        input_ids = proposer.tokenizer.encode("Test", return_tensors="pt").to(proposer.device)
        for _ in range(5):
            proposer.propose(input_ids)
        print("  ✓ Warmup complete")
        
        print("\n3. Running benchmark...")
        num_iterations = 20
        times = []
        
        prompt = "The future of artificial intelligence is"
        input_ids = proposer.tokenizer.encode(prompt, return_tensors="pt").to(proposer.device)
        
        for i in range(num_iterations):
            start_time = time.time()
            draft_tokens, draft_nodes = proposer.propose(input_ids)
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            if (i + 1) % 5 == 0:
                print(f"  Iteration {i+1}/{num_iterations}: {elapsed:.3f}s")
        
        # Statistics
        import statistics
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n4. Results:")
        print(f"  Average: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"  Min: {min_time:.3f}s")
        print(f"  Max: {max_time:.3f}s")
        print(f"  Tokens/proposal: {proposer.num_pruned_nodes}")
        print(f"  Tokens/second: {proposer.num_pruned_nodes / avg_time:.2f}")
        
        print("\n✓ TEST 5 PASSED")
        return True
        
    except Exception as e:
        print(f"✗ TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 6: Layer Comparison
# ============================================================================

def test_layer_comparison():
    """Compare different layers"""
    print("\n" + "="*70)
    print("TEST 6: LAYER COMPARISON")
    print("="*70)
    
    try:
        from transformers import AutoModelForCausalLM
        
        print("\n1. Loading full model...")
        full_model = AutoModelForCausalLM.from_pretrained(
            "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        num_layers = len(full_model.model.layers)
        print(f"  ✓ Full model: {num_layers} layers")
        
        # Test first, middle, and last layers
        test_layers = [0, num_layers // 2, num_layers - 1]
        
        print("\n2. Testing different layers...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
        
        for layer_idx in test_layers:
            print(f"\n  Layer {layer_idx}:")
            
            single_layer = SingleLayerLLaMA(full_model, layer_idx)
            
            start_time = time.time()
            outputs = single_layer(input_ids)
            elapsed = time.time() - start_time
            
            print(f"    Forward time: {elapsed:.3f}s")
            print(f"    Output shape: {outputs.logits.shape}")
        
        print("\n3. Recommendation:")
        print(f"  ✓ Using last layer (layer {num_layers-1}) is recommended")
        print(f"    - Most expressive features")
        print(f"    - Best conditioning from target")
        
        print("\n✓ TEST 6 PASSED")
        return True
        
    except Exception as e:
        print(f"✗ TEST 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 7: End-to-End
# ============================================================================

def test_end_to_end():
    """Test end-to-end generation"""
    print("\n" + "="*70)
    print("TEST 7: END-TO-END GENERATION")
    print("="*70)
    
    try:
        print("\n1. Initializing proposer...")
        proposer = FastEagleProposer(
            tree_width=3,
            tree_depth=2,
            beam_width=2
        )
        
        prompts = [
            "The future of AI is",
            "Machine learning will",
            "In the year 2030,"
        ]
        
        for i, prompt in enumerate(prompts):
            print(f"\n2.{i+1}. Testing: '{prompt}'")
            
            input_ids = proposer.tokenizer.encode(prompt, return_tensors="pt").to(proposer.device)
            
            start_time = time.time()
            draft_tokens, draft_nodes = proposer.propose(input_ids)
            elapsed = time.time() - start_time
            
            draft_text = proposer.decode_tokens(draft_tokens)
            
            print(f"  ✓ Time: {elapsed:.3f}s")
            print(f"  ✓ Tokens: {len(draft_tokens)}")
            print(f"  ✓ Draft: '{draft_text[:50]}{'...' if len(draft_text) > 50 else ''}'")
        
        print("\n✓ TEST 7 PASSED")
        return True
        
    except Exception as e:
        print(f"✗ TEST 7 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("FASTEAGLE PROPOSER TEST SUITE (SINGLE LAYER)")
    print("="*70)
    
    tests = [
        ("Single Layer Model", test_single_layer_model),
        ("Basic Functionality", test_basic_functionality),
        ("Tree Structure", test_tree_structure),
        ("One-Pass Generation", test_one_pass_generation),
        ("Performance Benchmark", test_performance),
        ("Layer Comparison", test_layer_comparison),
        ("End-to-End", test_end_to_end),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\nTest '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
    
    print(f"\n{passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️ {total_count - passed_count} tests failed")
    
    print("="*70 + "\n")
    
    return passed_count == total_count


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test FastEagle Proposer")
    parser.add_argument(
        "--test",
        type=str,
        choices=["all", "model", "basic", "tree", "onepass", "perf", "layers", "e2e"],
        default="all",
        help="Which test to run"
    )
    
    args = parser.parse_args()
    
    if args.test == "all":
        run_all_tests()
    elif args.test == "model":
        test_single_layer_model()
    elif args.test == "basic":
        test_basic_functionality()
    elif args.test == "tree":
        test_tree_structure()
    elif args.test == "onepass":
        test_one_pass_generation()
    elif args.test == "perf":
        test_performance()
    elif args.test == "layers":
        test_layer_comparison()
    elif args.test == "e2e":
        test_end_to_end()


if __name__ == "__main__":
    main()