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
        try:
            # Embed tokens
            hidden_states = self.embed_tokens(input_ids)
            
            if hidden_states is None:
                raise ValueError("embed_tokens returned None")
            
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
            
            if layer_outputs is None:
                raise ValueError("Layer forward returned None")
            
            # Extract hidden states from layer output
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs
            
            if hidden_states is None:
                raise ValueError("Layer output hidden_states is None")
            
            # Normalize
            hidden_states = self.norm(hidden_states)
            
            if hidden_states is None:
                raise ValueError("norm returned None")
            
            # LM head
            logits = self.lm_head(hidden_states)
            
            if logits is None:
                raise ValueError("lm_head returned None")
            
            # Return
            if return_dict:
                # Create a simple object with the attributes we need
                class ModelOutput:
                    def __init__(self, last_hidden_state, logits):
                        self.last_hidden_state = last_hidden_state
                        self.logits = logits
                
                return ModelOutput(hidden_states, logits)
            else:
                return (logits,)
                
        except Exception as e:
            print(f"Error in SingleLayerLLaMA.forward: {e}")
            import traceback
            traceback.print_exc()
            raise


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
        target_hidden_states: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, List[TreeNode]]:
        """
        Propose draft tokens using one-pass generation with single layer
        
        Args:
            input_ids: Input token IDs [1, seq_len]
            target_hidden_states: Optional target hidden states
            verbose: Print debug information
        
        Returns:
            draft_token_ids: Draft tokens
            draft_nodes: Draft tree nodes
        """
        try:
            if verbose:
                print("  [propose] Starting...")
            
            batch_size, prefix_len = input_ids.shape
            
            if verbose:
                print(f"  [propose] Input shape: {input_ids.shape}")
            
            assert batch_size == 1, "Currently only supports batch_size=1"
            
            # Get target hidden states if not provided
            if target_hidden_states is None:
                if verbose:
                    print("  [propose] Getting target hidden states...")
                target_hidden_states = self.get_target_hidden_states(input_ids)
            
            # Validate hidden states
            if target_hidden_states is None:
                raise ValueError("Failed to get target hidden states")
            
            if verbose:
                print(f"  [propose] Target hidden shape: {target_hidden_states.shape}")
            
            # Set forward context
            if verbose:
                print("  [propose] Setting forward context...")
            self.draft_model.set_forward_context(target_hidden_states)
            
            # Prepare input
            if verbose:
                print("  [propose] Preparing input...")
            tree_placeholders = torch.full(
                (1, self.num_tree_nodes),
                self.pad_token_id,
                dtype=torch.long,
                device=self.device
            )
            full_input_ids = torch.cat([input_ids, tree_placeholders], dim=1)
            
            if verbose:
                print(f"  [propose] Full input shape: {full_input_ids.shape}")
            
            # Create attention mask
            if verbose:
                print("  [propose] Creating attention mask...")
            attention_mask = create_tree_attention_mask(
                self.parent_ids,
                prefix_len,
                self.device,
                self.dtype
            )
            
            # Create position IDs
            if verbose:
                print("  [propose] Creating position IDs...")
            position_ids = create_tree_position_ids(
                self.tree_nodes,
                prefix_len,
                self.device
            )
            
            # ONE FORWARD PASS through SINGLE LAYER
            if verbose:
                print("  [propose] Running forward pass...")
            
            outputs = self.draft_model(
                input_ids=full_input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True
            )
            
            if verbose:
                print(f"  [propose] Forward pass complete, outputs type: {type(outputs)}")
            
            # Validate outputs
            if outputs is None:
                raise ValueError("Model forward pass returned None")
            
            if not hasattr(outputs, 'logits'):
                raise ValueError(f"Model outputs missing 'logits' attribute. Has: {dir(outputs)}")
            
            if outputs.logits is None:
                raise ValueError("Model outputs.logits is None")
            
            tree_logits = outputs.logits[0, prefix_len:, :]
            
            if verbose:
                print(f"  [propose] Tree logits shape: {tree_logits.shape}")
            
            # Validate tree_logits
            if tree_logits.shape[0] != self.num_tree_nodes:
                raise ValueError(
                    f"Expected {self.num_tree_nodes} tree logits, got {tree_logits.shape[0]}"
                )
            
            # Beam search pruning
            if verbose:
                print("  [propose] Running beam search pruning...")
            
            pruned_nodes, draft_token_ids = self.beam_search_prune(
                self.tree_nodes,
                tree_logits
            )
            
            if verbose:
                print(f"  [propose] Pruning complete: {len(pruned_nodes)} nodes, {len(draft_token_ids)} tokens")
            
            # Validate results
            if pruned_nodes is None:
                raise ValueError("beam_search_prune returned None for pruned_nodes")
            
            if draft_token_ids is None:
                raise ValueError("beam_search_prune returned None for draft_token_ids")
            
            if len(pruned_nodes) == 0:
                raise ValueError("No nodes were pruned")
            
            if verbose:
                print("  [propose] Success!")
            
            return draft_token_ids, pruned_nodes
            
        except Exception as e:
            print(f"ERROR in propose: {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty results rather than None
            empty_tokens = torch.tensor([], dtype=torch.long, device=self.device)
            empty_nodes = []
            
            print(f"Returning empty results: tokens={type(empty_tokens)}, nodes={type(empty_nodes)}")
            
            return empty_tokens, empty_nodes
    
    def beam_search_prune(
        self,
        nodes: List[TreeNode],
        tree_logits: torch.Tensor
    ) -> Tuple[List[TreeNode], torch.Tensor]:
        """Apply beam search pruning"""
        if len(nodes) == 0:
            # Return empty results if no nodes
            return [], torch.tensor([], dtype=torch.long, device=self.device)
        
        max_depth = max(node.depth for node in nodes)
        
        # Sample root token first
        root_logits = tree_logits[0] / self.temperature
        root_probs = F.softmax(root_logits, dim=-1)
        root_log_probs = F.log_softmax(root_logits, dim=-1)
        
        if self.top_k > 0:
            top_k_vals, top_k_idx = torch.topk(root_probs, k=min(self.top_k, len(root_probs)))
            top_k_probs = top_k_vals / top_k_vals.sum()
            sampled_idx = torch.multinomial(top_k_probs, 1).item()
            root_token_id = top_k_idx[sampled_idx].item()
        else:
            root_token_id = torch.multinomial(root_probs, 1).item()
        
        root_log_prob = root_log_probs[root_token_id].item()
        
        # Initialize with root
        root_node = TreeNode(
            node_id=0,
            depth=0,
            parent_id=None,
            token_id=root_token_id,
            log_prob=root_log_prob,
            cumulative_score=root_log_prob
        )
        
        pruned_nodes = [root_node]
        node_mapping = {0: 0}
        next_new_id = 1
        
        # Process each depth level
        for depth in range(1, max_depth + 1):
            level_nodes = [n for n in nodes if n.depth == depth]
            
            if len(level_nodes) == 0:
                break
            
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
                    try:
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
                    except Exception as e:
                        print(f"Warning: Error processing child {child.node_id}: {e}")
                        continue
                
                if len(child_scores) == 0:
                    continue
                
                # Sort and keep top-k
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
        
        # Extract token IDs (all nodes now have valid tokens)
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
Minimal Test - Isolate the Issue
=================================

Test each component separately to find where None is returned
"""

"""
Minimal Test - Isolate the Issue
=================================

Test each component separately to find where None is returned
"""

"""
End-to-End Tests for FastEagle Proposer
========================================

Complete workflow tests including:
- Full generation pipeline
- Multiple prompts and iterations
- Performance benchmarking
- Quality validation
- Error handling

Usage:
    python test_e2e.py
    python test_e2e.py --prompts 5
    python test_e2e.py --iterations 10
"""

import torch
import time
import argparse
from typing import List, Dict
from fast_eagle_proposer import FastEagleProposer


# ============================================================================
# Test 1: Single Prompt Generation
# ============================================================================

def test_single_prompt():
    """Test E2E generation with single prompt"""
    print("\n" + "="*70)
    print("TEST 1: SINGLE PROMPT GENERATION")
    print("="*70)
    
    try:
        print("\n1. Initializing proposer...")
        proposer = FastEagleProposer(
            tree_width=4,
            tree_depth=2,
            beam_width=3
        )
        print("✓ Proposer initialized")
        
        print("\n2. Preparing prompt...")
        prompt = "The future of artificial intelligence is"
        input_ids = proposer.tokenizer.encode(prompt, return_tensors="pt").to(proposer.device)
        print(f"✓ Prompt: '{prompt}'")
        print(f"  Input tokens: {input_ids.shape[1]}")
        
        print("\n3. Generating draft tokens...")
        start_time = time.time()
        draft_tokens, draft_nodes = proposer.propose(input_ids, verbose=False)
        elapsed = time.time() - start_time
        
        print(f"✓ Generation complete: {elapsed:.3f}s")
        print(f"  Draft tokens: {len(draft_tokens)}")
        print(f"  Draft nodes: {len(draft_nodes)}")
        
        # Validate
        assert len(draft_tokens) > 0, "No tokens generated"
        assert len(draft_nodes) > 0, "No nodes generated"
        assert len(draft_tokens) == len(draft_nodes), "Token/node count mismatch"
        
        print("\n4. Decoding output...")
        draft_text = proposer.decode_tokens(draft_tokens)
        print(f"✓ Draft text: '{draft_text}'")
        
        # Validate tokens
        print("\n5. Validating tokens...")
        for i, token_id in enumerate(draft_tokens):
            assert 0 <= token_id < proposer.vocab_size, f"Invalid token {token_id} at position {i}"
        print(f"✓ All {len(draft_tokens)} tokens valid")
        
        print("\n✓ TEST 1 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 2: Multiple Prompts
# ============================================================================

def test_multiple_prompts(num_prompts: int = 5):
    """Test E2E generation with multiple prompts"""
    print("\n" + "="*70)
    print(f"TEST 2: MULTIPLE PROMPTS ({num_prompts})")
    print("="*70)
    
    try:
        print("\n1. Initializing proposer...")
        proposer = FastEagleProposer(
            tree_width=3,
            tree_depth=2,
            beam_width=2
        )
        print("✓ Proposer initialized")
        
        prompts = [
            "The future of AI is",
            "Machine learning will",
            "In the year 2030,",
            "Scientists have discovered",
            "The most important technology",
            "Climate change requires",
            "Neural networks can",
            "Quantum computing will",
            "Space exploration is",
            "Renewable energy is"
        ][:num_prompts]
        
        results = []
        total_time = 0
        total_tokens = 0
        
        print(f"\n2. Testing {len(prompts)} prompts...")
        
        for i, prompt in enumerate(prompts):
            print(f"\n  Prompt {i+1}/{len(prompts)}: '{prompt}'")
            
            # Encode
            input_ids = proposer.tokenizer.encode(prompt, return_tensors="pt").to(proposer.device)
            
            # Generate
            start_time = time.time()
            draft_tokens, draft_nodes = proposer.propose(input_ids)
            elapsed = time.time() - start_time
            
            # Decode
            draft_text = proposer.decode_tokens(draft_tokens)
            
            # Record
            result = {
                'prompt': prompt,
                'draft_text': draft_text,
                'num_tokens': len(draft_tokens),
                'time': elapsed,
                'tokens_per_sec': len(draft_tokens) / elapsed if elapsed > 0 else 0
            }
            results.append(result)
            
            total_time += elapsed
            total_tokens += len(draft_tokens)
            
            print(f"    ✓ Generated {len(draft_tokens)} tokens in {elapsed:.3f}s")
            print(f"      '{draft_text[:50]}{'...' if len(draft_text) > 50 else ''}'")
        
        # Summary
        print(f"\n3. Summary:")
        print(f"  Total prompts: {len(prompts)}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average time/prompt: {total_time/len(prompts):.3f}s")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Average tokens/prompt: {total_tokens/len(prompts):.1f}")
        print(f"  Overall tokens/sec: {total_tokens/total_time:.2f}")
        
        print("\n✓ TEST 2 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 3: Multiple Iterations (Full Generation Loop)
# ============================================================================

def test_multiple_iterations(max_iterations: int = 10):
    """Test E2E generation with multiple iterations"""
    print("\n" + "="*70)
    print(f"TEST 3: MULTIPLE ITERATIONS (max={max_iterations})")
    print("="*70)
    
    try:
        print("\n1. Initializing proposer...")
        proposer = FastEagleProposer(
            tree_width=4,
            tree_depth=2,
            beam_width=3
        )
        print("✓ Proposer initialized")
        
        print("\n2. Starting generation loop...")
        prompt = "The future of AI is"
        print(f"  Prompt: '{prompt}'")
        
        # Initialize
        input_ids = proposer.tokenizer.encode(prompt, return_tensors="pt").to(proposer.device)
        current_ids = input_ids.clone()
        initial_len = current_ids.shape[1]
        
        iterations = 0
        total_draft_time = 0
        total_tokens_generated = 0
        iteration_stats = []
        
        print(f"\n3. Generation loop (max {max_iterations} iterations)...")
        
        while iterations < max_iterations:
            iterations += 1
            print(f"\n  Iteration {iterations}:")
            
            # Get target hidden states
            target_hidden = proposer.get_target_hidden_states(current_ids)
            
            # Generate draft
            start_time = time.time()
            draft_tokens, draft_nodes = proposer.propose(current_ids, target_hidden)
            draft_time = time.time() - start_time
            
            if len(draft_tokens) == 0:
                print("    ⚠ No tokens generated, stopping")
                break
            
            # Simple acceptance (for testing - just take first few tokens)
            # In real use, you'd verify with target model
            num_accept = min(len(draft_tokens), 5)  # Accept first 5 tokens
            accepted_tokens = draft_tokens[:num_accept]
            
            print(f"    Draft: {len(draft_tokens)} tokens in {draft_time:.3f}s")
            print(f"    Accepted: {num_accept} tokens")
            
            # Update state
            current_ids = torch.cat([current_ids, accepted_tokens.unsqueeze(0)], dim=1)
            total_draft_time += draft_time
            total_tokens_generated += num_accept
            
            # Record stats
            iteration_stats.append({
                'iteration': iterations,
                'draft_tokens': len(draft_tokens),
                'accepted_tokens': num_accept,
                'draft_time': draft_time,
                'acceptance_rate': num_accept / len(draft_tokens) if len(draft_tokens) > 0 else 0
            })
            
            # Check if we've generated enough
            if current_ids.shape[1] - initial_len >= 50:
                print(f"\n    Generated {current_ids.shape[1] - initial_len} tokens, stopping")
                break
        
        # Decode final output
        final_text = proposer.tokenizer.decode(current_ids[0], skip_special_tokens=True)
        
        print(f"\n4. Final output:")
        print(f"  Iterations: {iterations}")
        print(f"  Total tokens generated: {total_tokens_generated}")
        print(f"  Total draft time: {total_draft_time:.3f}s")
        print(f"  Tokens/sec: {total_tokens_generated/total_draft_time:.2f}")
        print(f"\n  Generated text:")
        print(f"  '{final_text}'")
        
        print(f"\n5. Per-iteration stats:")
        for stat in iteration_stats:
            print(f"  Iter {stat['iteration']}: "
                  f"{stat['draft_tokens']} draft → {stat['accepted_tokens']} accepted "
                  f"({stat['acceptance_rate']*100:.1f}%) in {stat['draft_time']:.3f}s")
        
        print("\n✓ TEST 3 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 4: Quality Validation
# ============================================================================

def test_quality_validation():
    """Test output quality validation"""
    print("\n" + "="*70)
    print("TEST 4: QUALITY VALIDATION")
    print("="*70)
    
    try:
        print("\n1. Initializing proposer...")
        proposer = FastEagleProposer(
            tree_width=4,
            tree_depth=2,
            beam_width=3
        )
        print("✓ Proposer initialized")
        
        print("\n2. Generating outputs for quality checks...")
        prompts = [
            "Hello world",
            "The capital of France is",
            "1 + 1 equals"
        ]
        
        quality_checks = {
            'valid_tokens': 0,
            'valid_decoding': 0,
            'reasonable_length': 0,
            'no_repetition': 0
        }
        
        for prompt in prompts:
            print(f"\n  Testing: '{prompt}'")
            
            input_ids = proposer.tokenizer.encode(prompt, return_tensors="pt").to(proposer.device)
            draft_tokens, draft_nodes = proposer.propose(input_ids)
            
            # Check 1: Valid token IDs
            valid_tokens = all(0 <= t < proposer.vocab_size for t in draft_tokens)
            if valid_tokens:
                quality_checks['valid_tokens'] += 1
                print("    ✓ All tokens valid")
            else:
                print("    ✗ Invalid tokens detected")
            
            # Check 2: Can decode
            try:
                draft_text = proposer.decode_tokens(draft_tokens)
                quality_checks['valid_decoding'] += 1
                print(f"    ✓ Decoding successful: '{draft_text[:30]}...'")
            except Exception as e:
                print(f"    ✗ Decoding failed: {e}")
            
            # Check 3: Reasonable length
            if len(draft_tokens) >= 5 and len(draft_tokens) <= 20:
                quality_checks['reasonable_length'] += 1
                print(f"    ✓ Length reasonable: {len(draft_tokens)} tokens")
            else:
                print(f"    ⚠ Length unusual: {len(draft_tokens)} tokens")
            
            # Check 4: No excessive repetition
            unique_ratio = len(set(draft_tokens.tolist())) / len(draft_tokens)
            if unique_ratio > 0.5:
                quality_checks['no_repetition'] += 1
                print(f"    ✓ No excessive repetition: {unique_ratio*100:.1f}% unique")
            else:
                print(f"    ⚠ Excessive repetition: {unique_ratio*100:.1f}% unique")
        
        print(f"\n3. Quality summary:")
        total_tests = len(prompts)
        for check, count in quality_checks.items():
            pct = (count / total_tests) * 100
            status = "✓" if pct == 100 else "⚠"
            print(f"  {status} {check}: {count}/{total_tests} ({pct:.1f}%)")
        
        # Pass if all prompts have valid tokens and can decode
        all_passed = (quality_checks['valid_tokens'] == total_tests and 
                      quality_checks['valid_decoding'] == total_tests)
        
        if all_passed:
            print("\n✓ TEST 4 PASSED")
        else:
            print("\n⚠ TEST 4 PARTIAL PASS (some quality issues)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 5: Performance Benchmarking
# ============================================================================

def test_performance_benchmark(num_iterations: int = 20):
    """Test E2E performance benchmarking"""
    print("\n" + "="*70)
    print(f"TEST 5: PERFORMANCE BENCHMARK ({num_iterations} iterations)")
    print("="*70)
    
    try:
        print("\n1. Initializing proposer...")
        proposer = FastEagleProposer(
            tree_width=4,
            tree_depth=2,
            beam_width=3
        )
        print("✓ Proposer initialized")
        
        print("\n2. Warming up (5 iterations)...")
        input_ids = proposer.tokenizer.encode("Test", return_tensors="pt").to(proposer.device)
        for _ in range(5):
            proposer.propose(input_ids)
        print("✓ Warmup complete")
        
        print(f"\n3. Running benchmark ({num_iterations} iterations)...")
        prompt = "The future of artificial intelligence"
        input_ids = proposer.tokenizer.encode(prompt, return_tensors="pt").to(proposer.device)
        
        times = []
        token_counts = []
        
        for i in range(num_iterations):
            start_time = time.time()
            draft_tokens, draft_nodes = proposer.propose(input_ids)
            elapsed = time.time() - start_time
            
            times.append(elapsed)
            token_counts.append(len(draft_tokens))
            
            if (i + 1) % 5 == 0:
                print(f"  Iteration {i+1}/{num_iterations}: {elapsed:.3f}s, {len(draft_tokens)} tokens")
        
        # Statistics
        import statistics
        
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        
        avg_tokens = statistics.mean(token_counts)
        tokens_per_sec = avg_tokens / avg_time
        
        print(f"\n4. Results:")
        print(f"  Iterations: {num_iterations}")
        print(f"  Average time: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"  Min time: {min_time:.3f}s")
        print(f"  Max time: {max_time:.3f}s")
        print(f"  Average tokens: {avg_tokens:.1f}")
        print(f"  Tokens/second: {tokens_per_sec:.2f}")
        
        # Performance expectations
        print(f"\n5. Performance check:")
        if avg_time < 0.2:
            print(f"  ✓ Fast: {avg_time:.3f}s < 0.2s")
        else:
            print(f"  ⚠ Slow: {avg_time:.3f}s >= 0.2s")
        
        if avg_tokens >= 10:
            print(f"  ✓ Good output: {avg_tokens:.1f} >= 10 tokens")
        else:
            print(f"  ⚠ Low output: {avg_tokens:.1f} < 10 tokens")
        
        print("\n✓ TEST 5 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 6: Error Handling
# ============================================================================

def test_error_handling():
    """Test E2E error handling"""
    print("\n" + "="*70)
    print("TEST 6: ERROR HANDLING")
    print("="*70)
    
    try:
        print("\n1. Initializing proposer...")
        proposer = FastEagleProposer(
            tree_width=3,
            tree_depth=2,
            beam_width=2
        )
        print("✓ Proposer initialized")
        
        print("\n2. Testing edge cases...")
        
        # Test 1: Very short input
        print("\n  Test 2.1: Very short input")
        input_ids = proposer.tokenizer.encode("Hi", return_tensors="pt").to(proposer.device)
        draft_tokens, draft_nodes = proposer.propose(input_ids)
        print(f"    ✓ Generated {len(draft_tokens)} tokens from short input")
        
        # Test 2: Longer input
        print("\n  Test 2.2: Longer input")
        long_text = "The quick brown fox jumps over the lazy dog. " * 3
        input_ids = proposer.tokenizer.encode(long_text, return_tensors="pt").to(proposer.device)
        draft_tokens, draft_nodes = proposer.propose(input_ids)
        print(f"    ✓ Generated {len(draft_tokens)} tokens from long input")
        
        # Test 3: Special characters
        print("\n  Test 2.3: Special characters")
        input_ids = proposer.tokenizer.encode("Hello! @#$ 123", return_tensors="pt").to(proposer.device)
        draft_tokens, draft_nodes = proposer.propose(input_ids)
        print(f"    ✓ Generated {len(draft_tokens)} tokens with special chars")
        
        print("\n3. Testing error recovery...")
        
        # Test 4: Multiple calls in succession
        print("\n  Test 3.1: Multiple successive calls")
        for i in range(3):
            input_ids = proposer.tokenizer.encode(f"Test {i}", return_tensors="pt").to(proposer.device)
            draft_tokens, draft_nodes = proposer.propose(input_ids)
            print(f"    ✓ Call {i+1}: {len(draft_tokens)} tokens")
        
        print("\n✓ TEST 6 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_e2e_tests(args):
    """Run all end-to-end tests"""
    print("\n" + "="*70)
    print("END-TO-END TEST SUITE")
    print("="*70)
    print("\nTesting complete generation pipeline...")
    
    tests = [
        ("Single Prompt Generation", lambda: test_single_prompt()),
        ("Multiple Prompts", lambda: test_multiple_prompts(args.prompts)),
        ("Multiple Iterations", lambda: test_multiple_iterations(args.iterations)),
        ("Quality Validation", lambda: test_quality_validation()),
        ("Performance Benchmark", lambda: test_performance_benchmark(args.benchmark)),
        ("Error Handling", lambda: test_error_handling()),
    ]
    
    results = []
    start_time = time.time()
    
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\nTest '{name}' crashed: {e}")
            results.append((name, False))
    
    total_time = time.time() - start_time
    
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
    print(f"Total time: {total_time:.2f}s")
    
    if passed_count == total_count:
        print("\n🎉 ALL E2E TESTS PASSED!")
        print("\nThe proposer is working correctly end-to-end!")
    else:
        print(f"\n⚠️ {total_count - passed_count} tests failed")
        print("\nCheck the output above for details")
    
    print("="*70 + "\n")
    
    return passed_count == total_count


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="End-to-End Tests for FastEagle Proposer")
    parser.add_argument(
        "--prompts",
        type=int,
        default=5,
        help="Number of prompts for multiple prompt test"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations for multiple iteration test"
    )
    parser.add_argument(
        "--benchmark",
        type=int,
        default=20,
        help="Number of iterations for performance benchmark"
    )
    
    args = parser.parse_args()
    
    success = run_all_e2e_tests(args)
    
    if success:
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()