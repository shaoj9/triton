"""
Alternative Single Layer Implementation - Simpler Approach
==========================================================

This version uses the full EAGLE model but only computes one forward pass,
avoiding the complexity of extracting a single layer.

This is simpler and more robust than the single-layer extraction approach.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
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
# Fast EAGLE Proposer (Simplified)
# ============================================================================

class FastEagleProposerSimple:
    """
    Simplified FastEagle proposer using full EAGLE model
    
    Instead of extracting a single layer (which can be complex),
    we just use the full EAGLE model with one forward pass.
    This is simpler and more robust.
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
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """Initialize simplified FastEagle proposer"""
        print(f"\n{'='*70}")
        print(f"FASTEAGLE PROPOSER - SIMPLIFIED (FULL MODEL)")
        print(f"{'='*70}")
        print(f"Draft model: {draft_model_name}")
        print(f"Target model: {target_model_name}")
        
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
        
        # Load EAGLE draft model (full model, not single layer)
        print(f"  Loading EAGLE draft model (full)...")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name,
            trust_remote_code=True,
            **load_kwargs
        )
        self.draft_model.eval()
        print(f"    ✓ EAGLE loaded (full model)")
        
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
        
        print(f"\nSimplified FastEagle proposer ready!")
        print(f"{'='*70}\n")
    
    @torch.inference_mode()
    def propose(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, List[TreeNode]]:
        """Propose draft tokens using one-pass generation"""
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
            
            if target_hidden_states is None:
                raise ValueError("Failed to get target hidden states")
            
            if verbose:
                print(f"  [propose] Target hidden shape: {target_hidden_states.shape}")
            
            # Set forward context on EAGLE model
            if verbose:
                print("  [propose] Setting forward context...")
            
            if hasattr(self.draft_model, 'set_forward_context'):
                self.draft_model.set_forward_context(target_hidden_states)
            elif hasattr(self.draft_model.model, 'set_forward_context'):
                self.draft_model.model.set_forward_context(target_hidden_states)
            else:
                print("  WARNING: EAGLE model doesn't have set_forward_context method")
            
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
            
            # ONE FORWARD PASS through FULL MODEL
            if verbose:
                print("  [propose] Running forward pass (full model)...")
            
            # Use the model's forward method directly
            outputs = self.draft_model(
                input_ids=full_input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True
            )
            
            if verbose:
                print(f"  [propose] Forward pass complete")
            
            if outputs is None:
                raise ValueError("Model forward pass returned None")
            
            if not hasattr(outputs, 'logits'):
                raise ValueError(f"Model outputs missing 'logits' attribute")
            
            if outputs.logits is None:
                raise ValueError("Model outputs.logits is None")
            
            tree_logits = outputs.logits[0, prefix_len:, :]
            
            if verbose:
                print(f"  [propose] Tree logits shape: {tree_logits.shape}")
            
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
                print(f"  [propose] Pruning complete: {len(pruned_nodes)} nodes")
            
            if pruned_nodes is None or draft_token_ids is None:
                raise ValueError("beam_search_prune returned None")
            
            if len(pruned_nodes) == 0:
                raise ValueError("No nodes were pruned")
            
            if verbose:
                print("  [propose] Success!")
            
            return draft_token_ids, pruned_nodes
            
        except Exception as e:
            print(f"ERROR in propose: {e}")
            import traceback
            traceback.print_exc()
            
            empty_tokens = torch.tensor([], dtype=torch.long, device=self.device)
            empty_nodes = []
            
            return empty_tokens, empty_nodes
    
    def beam_search_prune(
        self,
        nodes: List[TreeNode],
        tree_logits: torch.Tensor
    ) -> Tuple[List[TreeNode], torch.Tensor]:
        """Apply beam search pruning"""
        if len(nodes) == 0:
            return [], torch.tensor([], dtype=torch.long, device=self.device)
        
        max_depth = max(node.depth for node in nodes)
        
        # Sample root token
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
        
        for depth in range(1, max_depth + 1):
            level_nodes = [n for n in nodes if n.depth == depth]
            
            if len(level_nodes) == 0:
                break
            
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
                        print(f"Warning: Error processing child: {e}")
                        continue
                
                if len(child_scores) == 0:
                    continue
                
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
Debug Test for Simplified FastEagle Proposer
=============================================

Test the simpler version that uses full EAGLE model
"""

def test_simple_debug():
    """Debug test for simple proposer"""
    print("\n" + "="*70)
    print("DEBUG TEST - SIMPLIFIED VERSION")
    print("="*70)
    
    try:
        print("\n1. Checking CUDA availability...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Device: {device}")
        
        print("\n2. Initializing proposer (simplified - full model)...")
        print("   This will load models (may take a minute)...")
        
        proposer = FastEagleProposerSimple(
            tree_width=3,
            tree_depth=2,
            beam_width=2
        )
        
        print("   ✓ Proposer initialized")
        print(f"   Device: {proposer.device}")
        print(f"   Tree nodes (full): {proposer.num_tree_nodes}")
        print(f"   Tree nodes (pruned): {proposer.num_pruned_nodes}")
        
        print("\n3. Preparing input...")
        text = "Hello world"
        print(f"   Text: '{text}'")
        
        input_ids = proposer.tokenizer.encode(text, return_tensors="pt")
        print(f"   Input IDs shape: {input_ids.shape}")
        
        input_ids = input_ids.to(proposer.device)
        print(f"   Moved to device: {input_ids.device}")
        
        print("\n4. Getting target hidden states...")
        target_hidden = proposer.get_target_hidden_states(input_ids)
        
        if target_hidden is None:
            print("   ✗ ERROR: get_target_hidden_states returned None!")
            return False
        
        print(f"   Target hidden shape: {target_hidden.shape}")
        
        print("\n5. Calling propose method with verbose mode...")
        result = proposer.propose(input_ids, verbose=True)
        
        print(f"\n6. Checking result...")
        print(f"   Result type: {type(result)}")
        
        if result is None:
            print("   ✗ ERROR: propose() returned None!")
            return False
        
        if not isinstance(result, tuple) or len(result) != 2:
            print(f"   ✗ ERROR: Expected tuple of length 2, got {type(result)}")
            return False
        
        print(f"   ✓ Result is a tuple of length 2")
        
        print("\n7. Unpacking result...")
        draft_tokens, draft_nodes = result
        
        print(f"   Draft tokens type: {type(draft_tokens)}")
        print(f"   Draft tokens shape: {draft_tokens.shape}")
        print(f"   Draft nodes type: {type(draft_nodes)}")
        print(f"   Draft nodes length: {len(draft_nodes)}")
        
        if len(draft_tokens) == 0:
            print("   ⚠ WARNING: No tokens generated!")
            return False
        
        print("\n8. Decoding tokens...")
        draft_text = proposer.decode_tokens(draft_tokens)
        print(f"   Draft text: '{draft_text}'")
        
        print("\n✓ DEBUG TEST PASSED (Simplified Version)")
        return True
        
    except TypeError as e:
        print(f"\n✗ DEBUG TEST FAILED - TypeError")
        print(f"Error: {e}")
        
        if "cannot unpack non-iterable NoneType" in str(e):
            print("\n⚠ This is the unpacking error!")
        
        import traceback
        traceback.print_exc()
        return False
        
    except Exception as e:
        print(f"\n✗ DEBUG TEST FAILED")
        print(f"Error: {e}")
        
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_simple_debug()
    
    if success:
        print("\n" + "="*70)
        print("SUCCESS - Simplified proposer is working!")
        print("="*70)
        print("\nThe simplified version uses the full EAGLE model")
        print("which is more robust than extracting a single layer.")
    else:
        print("\n" + "="*70)
        print("FAILURE - Check error messages above")
        print("="*70)