"""
EAGLE One-Pass Generation with Beam Search Pruning
===================================================

Key innovation: Generate ALL tree tokens in ONE forward pass, then apply
beam search pruning to select best candidates.

Process:
1. Single forward pass → logits for ALL tree positions
2. Apply beam search pruning on logits → select top-k at each level
3. Verify selected tokens with target
4. Accept/reject with speculative sampling

Much faster than level-by-level generation!

Usage:
    python eagle_one_pass_beam.py --prompt "The future of AI is" --beam-width 3
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Tuple, Dict, Callable
from dataclasses import dataclass
import argparse
import time


# ============================================================================
# Tree Node with Beam Score
# ============================================================================

@dataclass
class BeamTreeNode:
    """Tree node with beam search score"""
    node_id: int
    depth: int
    parent_id: Optional[int]
    token_id: int = -1
    log_prob: float = 0.0
    cumulative_score: float = 0.0
    draft_prob: float = 0.0
    target_prob: float = 0.0
    children: List[int] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


# ============================================================================
# FlexAttention Score Mod
# ============================================================================

def create_tree_score_mod(
    parent_ids: List[Optional[int]],
    prefix_len: int
) -> Callable:
    """Create FlexAttention score_mod for tree structure"""
    num_nodes = len(parent_ids)
    
    # Pre-compute ancestor chains
    ancestor_chains = []
    for node_idx in range(num_nodes):
        ancestors = set([node_idx])
        parent_idx = parent_ids[node_idx]
        
        while parent_idx is not None:
            ancestors.add(parent_idx)
            parent_idx = parent_ids[parent_idx]
        
        ancestor_chains.append(ancestors)
    
    def tree_score_mod(score, b, h, q_idx, kv_idx):
        """Tree attention pattern"""
        # Prefix: causal
        if q_idx < prefix_len:
            return score if kv_idx <= q_idx else float('-inf')
        
        # Tree
        tree_q_idx = q_idx - prefix_len
        
        # Tree sees all prefix
        if kv_idx < prefix_len:
            return score
        
        # Tree sees ancestors only
        tree_kv_idx = kv_idx - prefix_len
        
        if tree_kv_idx in ancestor_chains[tree_q_idx]:
            return score
        else:
            return float('-inf')
    
    return tree_score_mod


def create_attention_mask_from_score_mod(
    score_mod: Callable,
    total_len: int,
    device: str = "cuda"
) -> torch.Tensor:
    """Create attention mask from score_mod"""
    mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
    
    for q in range(total_len):
        for kv in range(total_len):
            score = score_mod(0.0, 0, 0, q, kv)
            mask[q, kv] = (score != float('-inf'))
    
    attention_mask = torch.where(
        mask,
        torch.zeros(total_len, total_len, dtype=torch.float16, device=device),
        torch.full((total_len, total_len), float('-inf'), dtype=torch.float16, device=device)
    )
    
    return attention_mask.unsqueeze(0).unsqueeze(0)


def create_positions(nodes: List[BeamTreeNode], prefix_len: int, device: str = "cuda"):
    """Create positions tensor"""
    num_nodes = len(nodes)
    positions = torch.zeros(1, prefix_len + num_nodes, dtype=torch.long, device=device)
    positions[0, :prefix_len] = torch.arange(prefix_len)
    
    for node in nodes:
        positions[0, prefix_len + node.node_id] = prefix_len + node.depth
    
    return positions


def build_full_tree_structure(width: int, depth: int) -> Tuple[List[BeamTreeNode], List[Optional[int]]]:
    """Build FULL tree structure (all width^depth nodes)"""
    nodes = []
    parent_ids = []
    
    nodes.append(BeamTreeNode(node_id=0, depth=0, parent_id=None))
    parent_ids.append(None)
    
    current_level = [0]
    next_id = 1
    
    for d in range(1, depth + 1):
        next_level = []
        for parent_id in current_level:
            for _ in range(width):
                node = BeamTreeNode(node_id=next_id, depth=d, parent_id=parent_id)
                nodes.append(node)
                parent_ids.append(parent_id)
                nodes[parent_id].children.append(next_id)
                next_level.append(next_id)
                next_id += 1
        current_level = next_level
    
    return nodes, parent_ids


# ============================================================================
# One-Pass Beam Search Pruner
# ============================================================================

class OnePassBeamSearchPruner:
    """
    Apply beam search pruning AFTER one-pass generation
    
    Key idea:
    1. Generate logits for ALL tree positions in one pass
    2. Select top-k candidates at each level based on cumulative scores
    3. Keep only selected nodes
    """
    
    def __init__(self, beam_width: int):
        self.beam_width = beam_width
    
    def prune_tree(
        self,
        nodes: List[BeamTreeNode],
        tree_logits: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[List[BeamTreeNode], List[Optional[int]], torch.Tensor]:
        """
        Prune tree using beam search on generated logits
        
        Args:
            nodes: Full tree structure
            tree_logits: [num_nodes, vocab_size] logits from one pass
            temperature: Sampling temperature
        
        Returns:
            pruned_nodes: Pruned tree nodes
            pruned_parent_ids: Parent IDs for pruned nodes
            selected_logits: Logits for selected positions
        """
        print(f"\n{'='*70}")
        print(f"BEAM SEARCH PRUNING (Post-hoc)")
        print(f"{'='*70}")
        print(f"Input nodes: {len(nodes)}")
        print(f"Beam width: {self.beam_width}")
        
        # Get max depth
        max_depth = max(node.depth for node in nodes)
        
        # Initialize pruned tree with root
        pruned_nodes = [nodes[0]]  # Keep root
        pruned_parent_ids = [None]
        node_mapping = {0: 0}  # old_id -> new_id
        next_new_id = 1
        
        # Process level by level
        for depth in range(1, max_depth + 1):
            print(f"\nLevel {depth}:")
            
            # Get all nodes at this depth in original tree
            level_nodes = [n for n in nodes if n.depth == depth]
            print(f"  Original nodes: {len(level_nodes)}")
            
            # Group by parent
            parent_groups = {}
            for node in level_nodes:
                parent_id = node.parent_id
                if parent_id not in parent_groups:
                    parent_groups[parent_id] = []
                parent_groups[parent_id].append(node)
            
            # For each parent that was kept
            selected_count = 0
            for old_parent_id, children in parent_groups.items():
                # Check if this parent was kept
                if old_parent_id not in node_mapping:
                    continue
                
                new_parent_id = node_mapping[old_parent_id]
                
                # Get logits for all children
                children_logits = []
                for child in children:
                    children_logits.append(tree_logits[child.node_id])
                
                children_logits = torch.stack(children_logits)
                
                # Apply temperature and get probabilities
                scaled_logits = children_logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)
                log_probs = F.log_softmax(scaled_logits, dim=-1)
                
                # Get parent's cumulative score
                parent_cumulative = pruned_nodes[new_parent_id].cumulative_score
                
                # Calculate scores for each child
                child_scores = []
                for i, child in enumerate(children):
                    # Get top-1 token for this child
                    top_prob, top_token = torch.max(probs[i], dim=-1)
                    top_log_prob = log_probs[i, top_token]
                    
                    cumulative = parent_cumulative + top_log_prob.item()
                    child_scores.append((cumulative, top_token.item(), top_prob.item(), top_log_prob.item(), child))
                
                # Sort by cumulative score and keep top-k
                child_scores.sort(key=lambda x: x[0], reverse=True)
                top_k = child_scores[:self.beam_width]
                
                # Add selected children to pruned tree
                for cumulative, token_id, prob, log_prob, old_child in top_k:
                    new_child = BeamTreeNode(
                        node_id=next_new_id,
                        depth=depth,
                        parent_id=new_parent_id,
                        token_id=token_id,
                        log_prob=log_prob,
                        cumulative_score=cumulative,
                        draft_prob=prob
                    )
                    
                    pruned_nodes.append(new_child)
                    pruned_parent_ids.append(new_parent_id)
                    pruned_nodes[new_parent_id].children.append(next_new_id)
                    
                    # Update mapping
                    node_mapping[old_child.node_id] = next_new_id
                    
                    next_new_id += 1
                    selected_count += 1
            
            print(f"  Selected nodes: {selected_count}")
        
        print(f"\n{'='*70}")
        print(f"PRUNING COMPLETE")
        print(f"{'='*70}")
        print(f"Original nodes: {len(nodes)}")
        print(f"Pruned nodes: {len(pruned_nodes)}")
        print(f"Reduction: {(1 - len(pruned_nodes)/len(nodes))*100:.1f}%")
        
        # Show tree structure
        print(f"\nPruned tree structure:")
        for d in range(max_depth + 1):
            nodes_at_depth = [n for n in pruned_nodes if n.depth == d]
            print(f"  Depth {d}: {len(nodes_at_depth)} nodes")
        print(f"{'='*70}\n")
        
        # Extract logits for selected positions (not needed, we already sampled)
        selected_logits = torch.zeros(len(pruned_nodes), tree_logits.shape[1], device=tree_logits.device)
        
        return pruned_nodes, pruned_parent_ids, selected_logits


# ============================================================================
# EAGLE One-Pass with Beam Search
# ============================================================================

class EAGLEOnePassBeamSearch:
    """
    EAGLE with:
    1. One-pass tree generation (all tokens simultaneously)
    2. Post-hoc beam search pruning
    3. FlexAttention for accurate structure
    """
    
    def __init__(
        self,
        draft_model_path: str = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        target_model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        beam_width: int = 3,
        tree_width: int = 4,  # Generate more, then prune to beam_width
        tree_depth: int = 2
    ):
        print(f"\n{'='*70}")
        print(f"INITIALIZING EAGLE ONE-PASS + BEAM SEARCH")
        print(f"{'='*70}")
        
        self.device = device
        self.dtype = torch.float16
        self.beam_width = beam_width
        self.tree_width = tree_width
        self.tree_depth = tree_depth
        
        # Full tree size (before pruning)
        self.num_full_nodes = sum(tree_width**d for d in range(tree_depth + 1))
        # Pruned tree size (after pruning)
        self.num_pruned_nodes = sum(beam_width**d for d in range(tree_depth + 1))
        
        print(f"Full tree: width={tree_width}, depth={tree_depth}, nodes={self.num_full_nodes}")
        print(f"After pruning: beam={beam_width}, nodes={self.num_pruned_nodes}")
        print(f"Strategy: Generate {self.num_full_nodes} → Prune to {self.num_pruned_nodes}")
        
        # Load models
        print(f"\nLoading EAGLE draft model...")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_path,
            torch_dtype=self.dtype,
            device_map=device,
            low_cpu_mem_usage=True
        )
        self.draft_model.eval()
        print(f"  ✓ EAGLE loaded")
        
        print(f"\nLoading target model...")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_path,
            torch_dtype=self.dtype,
            device_map=device,
            low_cpu_mem_usage=True
        )
        self.target_model.eval()
        print(f"  ✓ Target loaded")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.placeholder_token_id = self.tokenizer.pad_token_id
        
        # Beam search pruner
        self.pruner = OnePassBeamSearchPruner(beam_width)
        
        print(f"{'='*70}\n")
    
    def get_target_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get hidden states from target model"""
        with torch.no_grad():
            outputs = self.target_model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True
            )
            hidden_states = outputs.hidden_states[-1]
        
        return hidden_states
    
    def generate_and_prune_tree(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[List[BeamTreeNode], torch.Tensor, torch.Tensor]:
        """
        ONE-PASS generation + beam search pruning
        
        Process:
        1. Build full tree structure
        2. ONE forward pass → logits for ALL positions
        3. Apply beam search pruning → select top-k
        4. Sample tokens for selected nodes
        
        Returns:
            pruned_nodes: Pruned tree (only top-k at each level)
            draft_token_ids: Token IDs for pruned nodes
            draft_probs: Probabilities for pruned nodes
        """
        prefix_len = input_ids.shape[1]
        
        print(f"\n{'='*70}")
        print(f"ONE-PASS GENERATION + BEAM SEARCH PRUNING")
        print(f"{'='*70}")
        
        # Step 1: Build FULL tree structure
        print(f"\nStep 1: Building full tree structure...")
        nodes, parent_ids = build_full_tree_structure(self.tree_width, self.tree_depth)
        print(f"  ✓ Full tree: {len(nodes)} nodes")
        
        # Step 2: Set forward context
        print(f"\nStep 2: Setting forward context from target...")
        if hasattr(self.draft_model.model, 'set_forward_context'):
            self.draft_model.model.set_forward_context(target_hidden_states)
            print(f"  ✓ Context set on draft_model.model")
        else:
            print(f"  ⚠ Warning: set_forward_context not available")
        
        # Step 3: Create attention mask for full tree
        print(f"\nStep 3: Creating attention mask...")
        score_mod = create_tree_score_mod(parent_ids, prefix_len)
        total_len = prefix_len + len(nodes)
        attention_mask = create_attention_mask_from_score_mod(score_mod, total_len, self.device)
        positions = create_positions(nodes, prefix_len, self.device)
        print(f"  ✓ Attention mask: {attention_mask.shape}")
        
        # Step 4: Prepare input_ids with placeholders
        print(f"\nStep 4: Preparing input_ids...")
        tree_token_ids = torch.full(
            (1, len(nodes)),
            self.placeholder_token_id,
            dtype=torch.long,
            device=self.device
        )
        full_input_ids = torch.cat([input_ids, tree_token_ids], dim=1)
        print(f"  ✓ Full input_ids: {full_input_ids.shape}")
        
        # Step 5: ONE FORWARD PASS - Generate ALL logits!
        print(f"\nStep 5: ONE forward pass through EAGLE...")
        forward_start = time.time()
        
        with torch.no_grad():
            outputs = self.draft_model.model(
                input_ids=full_input_ids,
                attention_mask=attention_mask,
                position_ids=positions,
                use_cache=False,
                return_dict=True
            )
            logits = self.draft_model.lm_head(outputs.last_hidden_state)
        
        forward_time = time.time() - forward_start
        
        tree_logits = logits[0, prefix_len:, :]
        print(f"  ✓ Forward complete in {forward_time:.3f}s")
        print(f"  ✓ Generated logits for {len(nodes)} positions")
        print(f"  Tree logits: {tree_logits.shape}")
        
        # Step 6: Apply beam search pruning
        print(f"\nStep 6: Applying beam search pruning...")
        prune_start = time.time()
        
        pruned_nodes, pruned_parent_ids, _ = self.pruner.prune_tree(
            nodes,
            tree_logits,
            temperature=temperature
        )
        
        prune_time = time.time() - prune_start
        print(f"  ✓ Pruning complete in {prune_time:.3f}s")
        
        # Step 7: Extract token IDs and probabilities
        draft_token_ids = torch.tensor(
            [node.token_id for node in pruned_nodes],
            dtype=torch.long,
            device=self.device
        )
        
        draft_probs_list = []
        for node in pruned_nodes:
            probs = torch.zeros(self.tokenizer.vocab_size, device=self.device)
            probs[node.token_id] = node.draft_prob
            draft_probs_list.append(probs)
        
        draft_probs = torch.stack(draft_probs_list)
        
        # Show pruned tree tokens
        print(f"\nPRUNED TREE TOKENS:")
        for depth in range(self.tree_depth + 1):
            nodes_at_depth = [n for n in pruned_nodes if n.depth == depth]
            if nodes_at_depth:
                tokens = [self.tokenizer.decode([n.token_id]) for n in nodes_at_depth[:5]]
                print(f"  Depth {depth}: {tokens}{'...' if len(nodes_at_depth) > 5 else ''}")
        
        print(f"\n{'='*70}")
        print(f"GENERATION SUMMARY")
        print(f"{'='*70}")
        print(f"Forward pass time: {forward_time:.3f}s")
        print(f"Pruning time: {prune_time:.3f}s")
        print(f"Total time: {forward_time + prune_time:.3f}s")
        print(f"Draft tokens: {len(draft_token_ids)}")
        print(f"{'='*70}\n")
        
        return pruned_nodes, draft_token_ids, draft_probs
    
    def verify_and_select_path(
        self,
        input_ids: torch.Tensor,
        draft_nodes: List[BeamTreeNode],
        draft_token_ids: torch.Tensor,
        draft_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[int]]:
        """Verify pruned tree with target"""
        prefix_len = input_ids.shape[1]
        parent_ids = [node.parent_id for node in draft_nodes]
        
        print(f"\n{'='*70}")
        print(f"VERIFICATION")
        print(f"{'='*70}")
        
        # Create attention mask for pruned tree
        score_mod = create_tree_score_mod(parent_ids, prefix_len)
        total_len = prefix_len + len(draft_nodes)
        attention_mask = create_attention_mask_from_score_mod(score_mod, total_len, self.device)
        positions = create_positions(draft_nodes, prefix_len, self.device)
        
        # Concatenate
        full_input_ids = torch.cat([input_ids, draft_token_ids.unsqueeze(0)], dim=1)
        
        print(f"Verifying {len(draft_token_ids)} pruned tokens...")
        
        # Forward through target
        verify_start = time.time()
        
        with torch.no_grad():
            outputs = self.target_model.model(
                input_ids=full_input_ids,
                attention_mask=attention_mask,
                position_ids=positions,
                use_cache=False,
                return_dict=True
            )
            target_logits = self.target_model.lm_head(outputs.last_hidden_state)
        
        verify_time = time.time() - verify_start
        
        tree_target_logits = target_logits[0, prefix_len:, :]
        target_probs = F.softmax(tree_target_logits, dim=-1)
        
        print(f"  ✓ Verification complete in {verify_time:.3f}s")
        
        # Store target probabilities
        for node_idx, node in enumerate(draft_nodes):
            node.target_prob = target_probs[node_idx, node.token_id].item()
        
        # Find best path
        print(f"\nFinding best path...")
        accepted_path = self._find_best_path_speculative(
            draft_nodes, target_probs, draft_probs
        )
        
        print(f"  ✓ Accepted {len(accepted_path)} tokens")
        
        if accepted_path:
            accepted_tokens_text = [self.tokenizer.decode([draft_nodes[nid].token_id]) for nid in accepted_path[:8]]
            print(f"  Tokens: {accepted_tokens_text}{'...' if len(accepted_path) > 8 else ''}")
        
        print(f"{'='*70}\n")
        
        if len(accepted_path) == 0:
            new_token = torch.multinomial(target_probs[0], 1).item()
            return torch.tensor([new_token], device=self.device), []
        
        accepted_tokens = torch.tensor(
            [draft_nodes[node_id].token_id for node_id in accepted_path],
            dtype=torch.long,
            device=self.device
        )
        
        return accepted_tokens, accepted_path
    
    def _find_best_path_speculative(
        self,
        nodes: List[BeamTreeNode],
        target_probs: torch.Tensor,
        draft_probs: torch.Tensor
    ) -> List[int]:
        """Find best path using speculative sampling"""
        accepted_path = []
        current_node_id = 0
        
        while True:
            node = nodes[current_node_id]
            
            # Speculative sampling
            p_target = target_probs[node.node_id, node.token_id].item()
            p_draft = node.draft_prob
            
            accept_prob = min(1.0, p_target / (p_draft + 1e-10))
            
            if torch.rand(1).item() < accept_prob:
                accepted_path.append(node.node_id)
                
                if len(node.children) == 0:
                    break
                
                # Select best child by cumulative score
                best_child_id = None
                best_score = float('-inf')
                
                for child_id in node.children:
                    child = nodes[child_id]
                    if child.cumulative_score > best_score:
                        best_score = child.cumulative_score
                        best_child_id = child_id
                
                if best_child_id is None:
                    break
                
                current_node_id = best_child_id
            else:
                break
        
        return accepted_path
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        verbose: bool = True
    ) -> Tuple[str, Dict]:
        """Generate with one-pass + beam search pruning"""
        print(f"\n{'='*70}")
        print(f"EAGLE ONE-PASS + BEAM SEARCH")
        print(f"{'='*70}")
        print(f"Prompt: '{prompt}'")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"Strategy: Generate {self.tree_width}^{self.tree_depth}={self.num_full_nodes} → Prune to beam={self.beam_width} ({self.num_pruned_nodes} nodes)")
        print(f"{'='*70}\n")
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()
        initial_len = current_ids.shape[1]
        
        # Stats
        stats = {
            'iterations': 0,
            'total_draft': 0,
            'total_accepted': 0,
            'acceptance_rates': []
        }
        
        start_time = time.time()
        
        # Generation loop
        while (current_ids.shape[1] - initial_len) < max_new_tokens:
            iteration = stats['iterations']
            current_generated = current_ids.shape[1] - initial_len
            
            if verbose:
                print(f"\n{'='*70}")
                print(f"ITERATION {iteration + 1}")
                print(f"{'='*70}")
                print(f"Generated: {current_generated}/{max_new_tokens} tokens")
            
            # Get target hidden states
            if verbose:
                print(f"\nGetting target hidden states...")
            
            target_hidden_states = self.get_target_hidden_states(current_ids)
            
            if verbose:
                print(f"  ✓ Hidden states: {target_hidden_states.shape}")
            
            # Generate and prune tree
            nodes, draft_ids, draft_probs = self.generate_and_prune_tree(
                current_ids,
                target_hidden_states,
                temperature=temperature
            )
            
            stats['total_draft'] += len(draft_ids)
            
            # Verify
            accepted_ids, accepted_path = self.verify_and_select_path(
                current_ids,
                nodes,
                draft_ids,
                draft_probs
            )
            
            num_accepted = len(accepted_ids)
            stats['total_accepted'] += num_accepted
            
            acceptance_rate = num_accepted / len(draft_ids) * 100
            stats['acceptance_rates'].append(acceptance_rate)
            
            if verbose:
                print(f"ITERATION SUMMARY:")
                print(f"  Generated: {self.num_full_nodes} → Pruned to: {len(draft_ids)}")
                print(f"  Accepted: {num_accepted}/{len(draft_ids)} ({acceptance_rate:.1f}%)")
            
            # Update
            current_ids = torch.cat([current_ids, accepted_ids.unsqueeze(0)], dim=1)
            
            current_text = self.tokenizer.decode(
                current_ids[0, initial_len:],
                skip_special_tokens=True
            )
            
            if verbose:
                print(f"\nCurrent: '{current_text}'")
            
            # Stopping conditions
            if self.tokenizer.eos_token_id in accepted_ids:
                if verbose:
                    print(f"\n✓ EOS token")
                break
            
            if num_accepted == 0:
                if verbose:
                    print(f"\n⚠ No tokens accepted")
                break
            
            stats['iterations'] += 1
            
            if stats['iterations'] >= 20:
                break
        
        total_time = time.time() - start_time
        
        # Final text
        final_text = self.tokenizer.decode(current_ids[0], skip_special_tokens=True)
        
        # Stats
        stats['total_time'] = total_time
        stats['tokens_generated'] = current_ids.shape[1] - initial_len
        stats['tokens_per_second'] = stats['tokens_generated'] / total_time if total_time > 0 else 0
        stats['avg_acceptance'] = sum(stats['acceptance_rates']) / len(stats['acceptance_rates']) if stats['acceptance_rates'] else 0
        
        # Summary
        print(f"\n{'='*70}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"Iterations: {stats['iterations']}")
        print(f"Tokens generated: {stats['tokens_generated']}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Tokens/second: {stats['tokens_per_second']:.2f}")
        print(f"Avg acceptance rate: {stats['avg_acceptance']:.1f}%")
        print(f"Efficiency: {stats['total_accepted']}/{stats['total_draft']} = {stats['total_accepted']/max(stats['total_draft'],1)*100:.1f}%")
        print(f"\nFinal text:")
        print(f"  {final_text}")
        print(f"{'='*70}\n")
        
        return final_text, stats


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="EAGLE One-Pass + Beam Search")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--beam-width", type=int, default=3,
                       help="Number of candidates to keep after pruning")
    parser.add_argument("--tree-width", type=int, default=4,
                       help="Width of full tree before pruning")
    parser.add_argument("--tree-depth", type=int, default=2,
                       help="Tree depth")
    parser.add_argument("--temperature", type=float, default=1.0)
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("EAGLE ONE-PASS GENERATION + BEAM SEARCH PRUNING")
    print("="*70)
    print("Features:")
    print("  ✓ ONE forward pass generates ALL tree positions")
    print("  ✓ Post-hoc beam search pruning")
    print("  ✓ FlexAttention for accurate structure")
    print("  ✓ MUCH faster than level-by-level")
    print("="*70)
    
    # Initialize
    generator = EAGLEOnePassBeamSearch(
        draft_model_path="yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        target_model_path="meta-llama/Llama-3.1-8B-Instruct",
        beam_width=args.beam_width,
        tree_width=args.tree_width,
        tree_depth=args.tree_depth
    )
    
    # Generate
    final_text, stats = generator.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("FINAL OUTPUT")
    print("="*70)
    print(final_text)
    print("="*70 + "\n")


if __name__ == "__main__":
    main()