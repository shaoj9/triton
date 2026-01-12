"""
EAGLE Tree Generation in ONE Pass - Proper Implementation
=========================================================

Combines:
1. EAGLE's proper architecture (set_forward_context with target hidden states)
2. Tree generation in ONE pass (using FlexAttention-style tree attention)

Architecture:
1. Target generates hidden states for current sequence
2. EAGLE.set_forward_context(hidden_states) - conditions EAGLE on target
3. EAGLE generates ENTIRE tree in ONE pass (not sequential!)
4. Target verifies all tree tokens
5. Select best path through tree using speculative sampling
6. Repeat

Usage:
    python eagle_tree_one_pass.py --prompt "The future of AI is"
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import argparse
import time


# ============================================================================
# Tree Structure
# ============================================================================

@dataclass
class TreeNode:
    node_id: int
    depth: int
    parent_id: Optional[int]
    token_id: int = -1
    draft_prob: float = 0.0
    target_prob: float = 0.0
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
# FlexAttention Tree Masks
# ============================================================================

def create_tree_attention_mask(
    parent_ids: List[Optional[int]],
    prefix_len: int,
    device: str = "cuda"
) -> torch.Tensor:
    """Create tree attention mask - each node sees prefix + ancestors only"""
    num_nodes = len(parent_ids)
    total_len = prefix_len + num_nodes
    
    mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
    
    # Prefix: causal
    for i in range(prefix_len):
        mask[i, :i+1] = True
    
    # Tree sees prefix
    mask[prefix_len:, :prefix_len] = True
    
    # Tree sees ancestors (not siblings!)
    for node_idx in range(num_nodes):
        q_pos = prefix_len + node_idx
        mask[q_pos, q_pos] = True
        
        parent_idx = parent_ids[node_idx]
        while parent_idx is not None:
            mask[q_pos, prefix_len + parent_idx] = True
            parent_idx = parent_ids[parent_idx]
    
    # Convert to additive mask
    attention_mask = torch.where(
        mask,
        torch.zeros(total_len, total_len, dtype=torch.float16, device=device),
        torch.full((total_len, total_len), float('-inf'), dtype=torch.float16, device=device)
    )
    
    return attention_mask.unsqueeze(0).unsqueeze(0)


def create_position_ids(nodes: List[TreeNode], prefix_len: int, device: str = "cuda"):
    """Create position IDs for tree"""
    num_nodes = len(nodes)
    position_ids = torch.zeros(1, prefix_len + num_nodes, dtype=torch.long, device=device)
    position_ids[0, :prefix_len] = torch.arange(prefix_len)
    
    for node in nodes:
        position_ids[0, prefix_len + node.node_id] = prefix_len + node.depth
    
    return position_ids


# ============================================================================
# EAGLE Tree Generator - ONE PASS
# ============================================================================

class EAGLETreeGenerator:
    """
    EAGLE with proper architecture that generates entire tree in ONE pass
    """
    
    def __init__(
        self,
        draft_model_path: str = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        target_model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        tree_width: int = 3,
        tree_depth: int = 2
    ):
        print(f"\n{'='*70}")
        print(f"INITIALIZING EAGLE TREE GENERATOR (ONE PASS)")
        print(f"{'='*70}")
        
        self.device = device
        self.dtype = torch.float16
        self.tree_width = tree_width
        self.tree_depth = tree_depth
        self.num_tree_nodes = sum(tree_width**d for d in range(tree_depth + 1))
        
        print(f"Tree: width={tree_width}, depth={tree_depth}, nodes={self.num_tree_nodes}")
        
        # Load EAGLE
        print(f"\nLoading EAGLE draft model...")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_path,
            torch_dtype=self.dtype,
            device_map=device,
            low_cpu_mem_usage=True
        )
        self.draft_model.eval()
        print(f"  ✓ EAGLE loaded")
        
        # Load target
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
        
        self.draft_embedding = self.draft_model.model.embed_tokens
        
        print(f"{'='*70}\n")
    
    def get_target_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        STEP 1: Get hidden states from target model
        """
        with torch.no_grad():
            outputs = self.target_model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True
            )
            hidden_states = outputs.hidden_states[-1]
        
        return hidden_states
    
    def generate_tree_one_pass(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 20
    ) -> Tuple[List[TreeNode], torch.Tensor, torch.Tensor]:
        """
        STEP 2: Generate ENTIRE tree in ONE pass using EAGLE
        
        Key differences from previous approach:
        1. Uses target_hidden_states via set_forward_context
        2. Generates ALL tree tokens in single forward pass
        3. Much more efficient than sequential generation
        
        Args:
            input_ids: Current sequence [1, seq_len]
            target_hidden_states: Target's hidden states [1, seq_len, hidden_dim]
            temperature: Sampling temperature
            top_k: Top-k sampling
        
        Returns:
            nodes: Tree nodes with sampled tokens
            draft_token_ids: [num_nodes] sampled token IDs
            draft_probs: [num_nodes, vocab_size] probability distributions
        """
        prefix_len = input_ids.shape[1]
        
        # Build tree structure
        nodes, parent_ids = build_tree_structure(self.tree_width, self.tree_depth)
        
        print(f"\n{'='*70}")
        print(f"TREE GENERATION IN ONE PASS")
        print(f"{'='*70}")
        print(f"Tree nodes: {len(nodes)}")
        print(f"Prefix length: {prefix_len}")
        
        # Create tree attention mask
        attention_mask = create_tree_attention_mask(parent_ids, prefix_len, self.device)
        position_ids = create_position_ids(nodes, prefix_len, self.device)
        
        print(f"Attention mask: {attention_mask.shape}")
        print(f"Position IDs: {position_ids.shape}")
        
        # THE KEY: Set forward context from target hidden states
        print(f"\nSetting forward context from target...")
        
        # EAGLE architecture: set_forward_context is on the model (not the wrapper)
        if hasattr(self.draft_model.model, 'set_forward_context'):
            self.draft_model.model.set_forward_context(target_hidden_states)
            print(f"  ✓ Context set: draft_model.model.set_forward_context()")
            print(f"  ✓ EAGLE now conditioned on target's hidden states")
        else:
            print(f"  ⚠ Warning: draft_model.model doesn't have set_forward_context")
            print(f"  This model may not be a proper EAGLE model")
            print(f"  Continuing without context (results may be poor)")
        
        # Prepare embeddings for tree
        prefix_embeds = self.draft_embedding(input_ids)
        
        # Initialize tree embeddings to zeros (will be filled by EAGLE)
        tree_embeds = torch.zeros(
            1, len(nodes), prefix_embeds.shape[-1],
            dtype=self.dtype,
            device=self.device
        )
        
        full_embeds = torch.cat([prefix_embeds, tree_embeds], dim=1)
        
        print(f"Full embeddings: {full_embeds.shape}")
        
        # ONE FORWARD PASS - Generate all tree tokens!
        print(f"\nForward pass through EAGLE...")
        forward_start = time.time()
        
        with torch.no_grad():
            # Forward through EAGLE's model (with set_forward_context already applied)
            outputs = self.draft_model.model(
                inputs_embeds=full_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True
            )
            
            # Get logits from LM head
            # Note: draft_model.lm_head is separate from draft_model.model
            logits = self.draft_model.lm_head(outputs.last_hidden_state)
        
        forward_time = time.time() - forward_start
        
        # Extract tree logits
        tree_logits = logits[0, prefix_len:, :]
        
        print(f"  ✓ Forward complete in {forward_time:.3f}s")
        print(f"  Tree logits: {tree_logits.shape}")
        print(f"{'='*70}\n")
        
        # Sample tokens from tree
        print(f"Sampling {len(nodes)} tokens from tree...")
        
        draft_token_ids = []
        draft_probs_list = []
        used_tokens = set(input_ids[0].tolist())
        
        for node_idx, node in enumerate(nodes):
            node_logits = tree_logits[node_idx].clone()
            
            # Light repetition penalty
            for token_id in used_tokens:
                if token_id < node_logits.shape[0]:
                    node_logits[token_id] /= 1.1
            
            # Sample
            scaled = node_logits / temperature
            probs = F.softmax(scaled, dim=-1)
            
            if top_k > 0:
                top_k_vals, top_k_idx = torch.topk(probs, k=min(top_k, probs.shape[0]))
                top_k_probs = top_k_vals / top_k_vals.sum()
                sampled = torch.multinomial(top_k_probs, 1).item()
                token_id = top_k_idx[sampled].item()
            else:
                token_id = torch.multinomial(probs, 1).item()
            
            node.token_id = token_id
            node.draft_prob = probs[token_id].item()
            draft_token_ids.append(token_id)
            draft_probs_list.append(probs)
            used_tokens.add(token_id)
        
        draft_token_ids = torch.tensor(draft_token_ids, dtype=torch.long, device=self.device)
        draft_probs = torch.stack(draft_probs_list)
        
        print(f"  ✓ Sampled {len(draft_token_ids)} tokens")
        
        # Show sample tokens
        sample_tokens = [self.tokenizer.decode([t]) for t in draft_token_ids[:5]]
        print(f"  Sample: {sample_tokens}...")
        
        return nodes, draft_token_ids, draft_probs
    
    def verify_and_select_path(
        self,
        input_ids: torch.Tensor,
        draft_nodes: List[TreeNode],
        draft_token_ids: torch.Tensor,
        draft_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        STEP 3: Verify entire tree with target and select best path
        
        Args:
            input_ids: Current sequence [1, seq_len]
            draft_nodes: Tree nodes with draft tokens
            draft_token_ids: [num_nodes] draft token IDs
            draft_probs: [num_nodes, vocab_size] draft probabilities
        
        Returns:
            accepted_tokens: Accepted token IDs from best path
            accepted_path: Node IDs in accepted path
        """
        prefix_len = input_ids.shape[1]
        parent_ids = [node.parent_id for node in draft_nodes]
        
        print(f"\n{'='*70}")
        print(f"VERIFICATION WITH TARGET")
        print(f"{'='*70}")
        
        # Create same tree attention mask for verification
        attention_mask = create_tree_attention_mask(parent_ids, prefix_len, self.device)
        position_ids = create_position_ids(draft_nodes, prefix_len, self.device)
        
        # Prepare embeddings with draft tokens
        prefix_embeds = self.target_model.model.embed_tokens(input_ids)
        draft_embeds = self.target_model.model.embed_tokens(draft_token_ids.unsqueeze(0))
        full_embeds = torch.cat([prefix_embeds, draft_embeds], dim=1)
        
        print(f"Verifying {len(draft_token_ids)} draft tokens...")
        
        # Forward through target
        verify_start = time.time()
        
        with torch.no_grad():
            outputs = self.target_model.model(
                inputs_embeds=full_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True
            )
            target_logits = self.target_model.lm_head(outputs.last_hidden_state)
        
        verify_time = time.time() - verify_start
        
        tree_target_logits = target_logits[0, prefix_len:, :]
        target_probs = F.softmax(tree_target_logits, dim=-1)
        
        print(f"  ✓ Verification complete in {verify_time:.3f}s")
        
        # Store target probabilities in nodes
        for node_idx, node in enumerate(draft_nodes):
            node.target_prob = target_probs[node_idx, node.token_id].item()
        
        # Find best path through tree using speculative sampling
        print(f"\nFinding best path through tree...")
        accepted_path = self._find_best_path_speculative(
            draft_nodes, target_probs, draft_probs
        )
        
        print(f"  ✓ Found path with {len(accepted_path)} tokens")
        print(f"  Path: {accepted_path}")
        print(f"{'='*70}\n")
        
        if len(accepted_path) == 0:
            # Fallback: sample from target at root
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
        nodes: List[TreeNode],
        target_probs: torch.Tensor,
        draft_probs: torch.Tensor
    ) -> List[int]:
        """
        Find best path through tree using speculative sampling
        """
        accepted_path = []
        current_node_id = 0
        
        while True:
            node = nodes[current_node_id]
            
            # Speculative sampling
            p_target = target_probs[node.node_id, node.token_id].item()
            p_draft = draft_probs[node.node_id, node.token_id].item()
            
            accept_prob = min(1.0, p_target / (p_draft + 1e-10))
            
            if torch.rand(1).item() < accept_prob:
                accepted_path.append(node.node_id)
                
                # If leaf, done
                if len(node.children) == 0:
                    break
                
                # Select best child based on target probability
                best_child_id = None
                best_prob = -1.0
                
                for child_id in node.children:
                    child = nodes[child_id]
                    child_target_prob = target_probs[child_id, child.token_id].item()
                    
                    if child_target_prob > best_prob:
                        best_prob = child_target_prob
                        best_child_id = child_id
                
                if best_child_id is None:
                    break
                
                current_node_id = best_child_id
            else:
                # Rejected
                break
        
        return accepted_path
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 20,
        verbose: bool = True
    ) -> Tuple[str, Dict]:
        """
        Generate text using EAGLE with tree generation in one pass
        """
        print(f"\n{'='*70}")
        print(f"EAGLE TREE GENERATION - ONE PASS")
        print(f"{'='*70}")
        print(f"Prompt: '{prompt}'")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"Tree: width={self.tree_width}, depth={self.tree_depth}, nodes={self.num_tree_nodes}")
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
            'acceptance_rates': [],
            'forward_times': [],
            'verify_times': []
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
            
            # STEP 1: Get target hidden states
            if verbose:
                print(f"\nStep 1: Getting target hidden states...")
            
            hidden_start = time.time()
            target_hidden_states = self.get_target_hidden_states(current_ids)
            hidden_time = time.time() - hidden_start
            
            if verbose:
                print(f"  ✓ Hidden states: {target_hidden_states.shape}")
                print(f"  Time: {hidden_time:.3f}s")
            
            # STEP 2: Generate tree in ONE pass
            if verbose:
                print(f"\nStep 2: Generating tree ({self.num_tree_nodes} tokens) in ONE pass...")
            
            nodes, draft_ids, draft_probs = self.generate_tree_one_pass(
                current_ids,
                target_hidden_states,
                temperature=temperature,
                top_k=top_k
            )
            
            stats['total_draft'] += len(draft_ids)
            
            # STEP 3: Verify and select path
            if verbose:
                print(f"\nStep 3: Verifying tree and selecting best path...")
            
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
                print(f"Accepted: {num_accepted}/{len(draft_ids)} ({acceptance_rate:.1f}%)")
                accepted_text = [self.tokenizer.decode([t]) for t in accepted_ids]
                print(f"Tokens: {accepted_text}")
            
            # Update sequence
            current_ids = torch.cat([current_ids, accepted_ids.unsqueeze(0)], dim=1)
            
            # Current text
            current_text = self.tokenizer.decode(
                current_ids[0, initial_len:],
                skip_special_tokens=True
            )
            
            if verbose:
                print(f"\nCurrent output: '{current_text}'")
            
            # Stopping conditions
            if self.tokenizer.eos_token_id in accepted_ids:
                if verbose:
                    print(f"\n✓ EOS token, stopping")
                break
            
            if num_accepted == 0:
                if verbose:
                    print(f"\n⚠ No tokens accepted, stopping")
                break
            
            stats['iterations'] += 1
            
            if stats['iterations'] >= 20:
                if verbose:
                    print(f"\n⚠ Max iterations")
                break
        
        total_time = time.time() - start_time
        
        # Final text
        final_text = self.tokenizer.decode(
            current_ids[0],
            skip_special_tokens=True
        )
        
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
        print(f"Avg acceptance: {stats['avg_acceptance']:.1f}%")
        print(f"Draft efficiency: {stats['total_accepted']}/{stats['total_draft']} = {stats['total_accepted']/max(stats['total_draft'],1)*100:.1f}%")
        print(f"\nFinal text:")
        print(f"  {final_text}")
        print(f"{'='*70}\n")
        
        return final_text, stats


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="EAGLE Tree Generation - One Pass")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is",
                       help="Input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=50,
                       help="Maximum new tokens")
    parser.add_argument("--tree-width", type=int, default=3,
                       help="Tree width")
    parser.add_argument("--tree-depth", type=int, default=2,
                       help="Tree depth")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature")
    parser.add_argument("--top-k", type=int, default=20,
                       help="Top-k")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("EAGLE TREE GENERATION IN ONE PASS")
    print("="*70)
    print("Features:")
    print("  ✓ Proper EAGLE architecture (set_forward_context)")
    print("  ✓ Tree generation in ONE pass (not sequential)")
    print("  ✓ FlexAttention-style tree masks")
    print("  ✓ Speculative sampling for path selection")
    print("="*70)
    
    # Initialize
    generator = EAGLETreeGenerator(
        draft_model_path="yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        target_model_path="meta-llama/Llama-3.1-8B-Instruct",
        tree_width=args.tree_width,
        tree_depth=args.tree_depth
    )
    
    # Generate
    final_text, stats = generator.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("FINAL OUTPUT")
    print("="*70)
    print(final_text)
    print("="*70 + "\n")


if __name__ == "__main__":
    main()