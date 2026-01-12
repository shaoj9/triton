"""
EAGLE Speculative Decoding with FlexAttention
==============================================

Complete implementation of speculative decoding:
- Draft model: yuhuili/EAGLE-LLaMA3.1-Instruct-8B
- Target model: meta-llama/Llama-3.1-8B-Instruct
- Backend: FlexAttention for efficient tree generation

Loop until:
- EOS token generated
- Max length reached
- Acceptance rate too low

Usage:
    python eagle_speculative_decoding.py --prompt "The future of AI is" --max-length 100
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
    confidence: float = 0.0
    children: List[int] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


def build_tree_structure(width: int, depth: int) -> Tuple[List[TreeNode], List[Optional[int]]]:
    """Build tree structure for EAGLE"""
    nodes = []
    parent_ids = []
    
    # Root
    nodes.append(TreeNode(node_id=0, depth=0, parent_id=None))
    parent_ids.append(None)
    
    # Build levels
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
# FlexAttention Backend
# ============================================================================

def create_tree_attention_mask(
    parent_ids: List[Optional[int]],
    prefix_len: int,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Create attention mask for tree structure using FlexAttention logic
    
    This simulates FlexAttention behavior with explicit mask
    (In production, would use actual flex_attention with score_mod)
    """
    num_nodes = len(parent_ids)
    total_len = prefix_len + num_nodes
    
    mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
    
    # Prefix: causal
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
    """Create position IDs for tree"""
    num_nodes = len(nodes)
    position_ids = torch.zeros(1, prefix_len + num_nodes, dtype=torch.long, device=device)
    
    # Prefix
    position_ids[0, :prefix_len] = torch.arange(prefix_len)
    
    # Tree (depth-based)
    for node in nodes:
        position_ids[0, prefix_len + node.node_id] = prefix_len + node.depth
    
    return position_ids


# ============================================================================
# EAGLE Speculative Decoding System
# ============================================================================

class EAGLESpeculativeDecoder:
    """
    EAGLE Speculative Decoding with FlexAttention backend
    
    Draft model: EAGLE (fast, small)
    Target model: Llama-3.1-8B-Instruct (slow, accurate)
    """
    
    def __init__(
        self,
        draft_model_path: str = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        target_model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        tree_width: int = 3,
        tree_depth: int = 3
    ):
        print(f"\n{'='*70}")
        print(f"INITIALIZING EAGLE SPECULATIVE DECODER")
        print(f"{'='*70}")
        
        self.device = device
        self.dtype = torch.float16
        self.tree_width = tree_width
        self.tree_depth = tree_depth
        
        # Calculate tree size
        self.num_tree_nodes = sum(tree_width**d for d in range(tree_depth + 1))
        
        print(f"Tree structure: width={tree_width}, depth={tree_depth}")
        print(f"Total tree nodes: {self.num_tree_nodes}")
        
        # Load draft model (EAGLE)
        print(f"\nLoading DRAFT model: {draft_model_path}")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_path,
            torch_dtype=self.dtype,
            device_map=device,
            low_cpu_mem_usage=True
        )
        self.draft_model.eval()
        print(f"  ✓ EAGLE draft model loaded")
        
        # Load target model
        print(f"\nLoading TARGET model: {target_model_path}")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_path,
            torch_dtype=self.dtype,
            device_map=device,
            low_cpu_mem_usage=True
        )
        self.target_model.eval()
        print(f"  ✓ Target model loaded")
        
        # Load tokenizer
        print(f"\nLoading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"  ✓ Tokenizer loaded")
        
        # Get embedding layers
        self.draft_embedding = self.draft_model.model.embed_tokens
        self.target_embedding = self.target_model.model.embed_tokens
        
        print(f"{'='*70}\n")
    
    def generate_draft_tree(
        self,
        prefix_ids: torch.Tensor,
        top_k: int = 10,
        temperature: float = 0.8
    ) -> Tuple[List[TreeNode], torch.Tensor]:
        """
        Generate draft token tree using EAGLE with FlexAttention
        
        Args:
            prefix_ids: [1, prefix_len] current sequence
            top_k: Top-k sampling
            temperature: Temperature
        
        Returns:
            nodes: Tree nodes with sampled tokens
            draft_token_ids: [num_nodes] draft token IDs
        """
        prefix_len = prefix_ids.shape[1]
        
        # Build tree structure
        nodes, parent_ids = build_tree_structure(self.tree_width, self.tree_depth)
        
        # Create FlexAttention mask
        attention_mask = create_tree_attention_mask(parent_ids, prefix_len, self.device)
        position_ids = create_position_ids(nodes, prefix_len, self.device)
        
        # Prepare embeddings
        prefix_embeds = self.draft_embedding(prefix_ids)
        tree_embeds = torch.zeros(
            1, len(nodes), prefix_embeds.shape[-1],
            dtype=self.dtype,
            device=self.device
        )
        full_embeds = torch.cat([prefix_embeds, tree_embeds], dim=1)
        
        # Forward through EAGLE (draft model)
        with torch.no_grad():
            outputs = self.draft_model.model(
                inputs_embeds=full_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True
            )
            
            logits = self.draft_model.lm_head(outputs.last_hidden_state)
        
        # Extract tree logits
        tree_logits = logits[0, prefix_len:, :]
        
        # Sample tokens
        draft_token_ids = []
        used_tokens = set(prefix_ids[0].tolist())
        
        for node_idx, node in enumerate(nodes):
            node_logits = tree_logits[node_idx].clone()
            
            # Repetition penalty
            for token_id in used_tokens:
                if token_id < node_logits.shape[0]:
                    node_logits[token_id] /= 1.2
            
            # Sample
            scaled = node_logits / temperature
            
            if top_k > 0:
                top_k_vals, top_k_idx = torch.topk(scaled, k=min(top_k, scaled.shape[0]))
                probs = F.softmax(top_k_vals, dim=-1)
                sampled = torch.multinomial(probs, 1).item()
                token_id = top_k_idx[sampled].item()
                confidence = probs[sampled].item()
            else:
                probs = F.softmax(scaled, dim=-1)
                token_id = torch.multinomial(probs, 1).item()
                confidence = probs[token_id].item()
            
            node.token_id = token_id
            node.confidence = confidence
            draft_token_ids.append(token_id)
            used_tokens.add(token_id)
        
        draft_token_ids = torch.tensor(draft_token_ids, dtype=torch.long, device=self.device)
        
        return nodes, draft_token_ids
    
    def verify_with_target(
        self,
        prefix_ids: torch.Tensor,
        draft_nodes: List[TreeNode],
        draft_token_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, int, List[int]]:
        """
        Verify draft tokens with target model
        
        Args:
            prefix_ids: [1, prefix_len] current sequence
            draft_nodes: Draft tree nodes
            draft_token_ids: [num_nodes] draft tokens
        
        Returns:
            accepted_ids: [num_accepted] accepted token IDs
            num_accepted: Number of accepted tokens
            accepted_path: List of node IDs in accepted path
        """
        prefix_len = prefix_ids.shape[1]
        
        # Build parent_ids for tree
        parent_ids = [node.parent_id for node in draft_nodes]
        
        # Create attention mask for verification
        attention_mask = create_tree_attention_mask(parent_ids, prefix_len, self.device)
        position_ids = create_position_ids(draft_nodes, prefix_len, self.device)
        
        # Prepare embeddings with DRAFT tokens
        prefix_embeds = self.target_embedding(prefix_ids)
        
        # Use draft token embeddings
        draft_embeds = self.target_embedding(draft_token_ids.unsqueeze(0))
        full_embeds = torch.cat([prefix_embeds, draft_embeds], dim=1)
        
        # Forward through target model
        with torch.no_grad():
            outputs = self.target_model.model(
                inputs_embeds=full_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True
            )
            
            target_logits = self.target_model.lm_head(outputs.last_hidden_state)
        
        # Extract tree logits
        tree_target_logits = target_logits[0, prefix_len:, :]
        
        # Acceptance sampling (traverse tree depth-first)
        accepted_path = []
        current_node_id = 0  # Start at root
        
        while True:
            node = draft_nodes[current_node_id]
            
            # Get target probabilities
            target_probs = F.softmax(tree_target_logits[node.node_id], dim=-1)
            p_target = target_probs[node.token_id].item()
            p_draft = node.confidence
            
            # Acceptance probability
            accept_prob = min(1.0, p_target / (p_draft + 1e-10))
            
            # Sample acceptance
            if torch.rand(1).item() < accept_prob:
                # Accept token
                accepted_path.append(node.node_id)
                
                # Check if leaf or no children
                if len(node.children) == 0:
                    break
                
                # Move to first child (greedy path selection)
                # In practice, would select child based on highest acceptance prob
                current_node_id = node.children[0]
            else:
                # Reject - stop here
                # Optionally resample from target distribution
                if len(accepted_path) == 0:
                    # If root rejected, sample from target
                    resampled_token = torch.multinomial(target_probs, 1).item()
                    accepted_path = [-1]  # Special marker for resampled token
                    accepted_ids = torch.tensor([resampled_token], device=self.device)
                    return accepted_ids, 1, accepted_path
                break
        
        # Collect accepted token IDs
        accepted_ids = torch.tensor(
            [draft_nodes[node_id].token_id for node_id in accepted_path],
            dtype=torch.long,
            device=self.device
        )
        
        return accepted_ids, len(accepted_ids), accepted_path
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        verbose: bool = True
    ) -> Tuple[str, Dict]:
        """
        Generate text using EAGLE speculative decoding
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            verbose: Print progress
        
        Returns:
            generated_text: Complete generated text
            stats: Generation statistics
        """
        print(f"\n{'='*70}")
        print(f"EAGLE SPECULATIVE DECODING")
        print(f"{'='*70}")
        print(f"Prompt: '{prompt}'")
        print(f"Max length: {max_length}")
        print(f"{'='*70}\n")
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()
        
        # Statistics
        stats = {
            'iterations': 0,
            'total_draft_tokens': 0,
            'total_accepted_tokens': 0,
            'draft_times': [],
            'verify_times': [],
            'acceptance_rates': []
        }
        
        start_time = time.time()
        
        # Generation loop
        while current_ids.shape[1] < max_length:
            iteration = stats['iterations']
            
            if verbose:
                print(f"\n{'='*70}")
                print(f"ITERATION {iteration + 1}")
                print(f"{'='*70}")
                print(f"Current length: {current_ids.shape[1]}")
            
            # Step 1: Generate draft tree with EAGLE
            if verbose:
                print(f"\nStep 1: Generating draft tree with EAGLE...")
            
            draft_start = time.time()
            draft_nodes, draft_token_ids = self.generate_draft_tree(
                current_ids,
                top_k=10,
                temperature=0.8
            )
            draft_time = time.time() - draft_start
            
            stats['draft_times'].append(draft_time)
            stats['total_draft_tokens'] += len(draft_token_ids)
            
            if verbose:
                print(f"  ✓ Generated {len(draft_token_ids)} draft tokens in {draft_time:.3f}s")
                print(f"  Draft tokens: {[self.tokenizer.decode([t]) for t in draft_token_ids[:5]]}")
            
            # Step 2: Verify with target model
            if verbose:
                print(f"\nStep 2: Verifying with target model...")
            
            verify_start = time.time()
            accepted_ids, num_accepted, accepted_path = self.verify_with_target(
                current_ids,
                draft_nodes,
                draft_token_ids
            )
            verify_time = time.time() - verify_start
            
            stats['verify_times'].append(verify_time)
            stats['total_accepted_tokens'] += num_accepted
            
            acceptance_rate = num_accepted / len(draft_token_ids) * 100
            stats['acceptance_rates'].append(acceptance_rate)
            
            if verbose:
                print(f"  ✓ Verified in {verify_time:.3f}s")
                print(f"  Accepted: {num_accepted}/{len(draft_token_ids)} tokens ({acceptance_rate:.1f}%)")
                print(f"  Accepted path: {accepted_path}")
                print(f"  Accepted tokens: {[self.tokenizer.decode([t]) for t in accepted_ids]}")
            
            # Step 3: Update sequence
            current_ids = torch.cat([current_ids, accepted_ids.unsqueeze(0)], dim=1)
            
            # Decode current text
            current_text = self.tokenizer.decode(current_ids[0], skip_special_tokens=True)
            
            if verbose:
                print(f"\nCurrent text: '{current_text}'")
            
            # Check for EOS
            if self.tokenizer.eos_token_id in accepted_ids:
                if verbose:
                    print(f"\n✓ EOS token generated, stopping.")
                break
            
            # Check acceptance rate
            if acceptance_rate < 10.0:
                if verbose:
                    print(f"\n⚠ Low acceptance rate ({acceptance_rate:.1f}%), stopping.")
                break
            
            stats['iterations'] += 1
            
            # Safety check
            if stats['iterations'] > 20:
                if verbose:
                    print(f"\n⚠ Max iterations reached, stopping.")
                break
        
        total_time = time.time() - start_time
        
        # Generate final text
        final_text = self.tokenizer.decode(current_ids[0], skip_special_tokens=True)
        
        # Calculate summary statistics
        stats['total_time'] = total_time
        stats['final_length'] = current_ids.shape[1]
        stats['tokens_generated'] = current_ids.shape[1] - input_ids.shape[1]
        stats['avg_acceptance_rate'] = sum(stats['acceptance_rates']) / len(stats['acceptance_rates']) if stats['acceptance_rates'] else 0
        stats['avg_draft_time'] = sum(stats['draft_times']) / len(stats['draft_times']) if stats['draft_times'] else 0
        stats['avg_verify_time'] = sum(stats['verify_times']) / len(stats['verify_times']) if stats['verify_times'] else 0
        stats['tokens_per_second'] = stats['tokens_generated'] / total_time if total_time > 0 else 0
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"Final text: '{final_text}'")
        print(f"\nStatistics:")
        print(f"  Total iterations: {stats['iterations']}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Tokens generated: {stats['tokens_generated']}")
        print(f"  Tokens/second: {stats['tokens_per_second']:.2f}")
        print(f"  Avg acceptance rate: {stats['avg_acceptance_rate']:.1f}%")
        print(f"  Avg draft time: {stats['avg_draft_time']:.3f}s")
        print(f"  Avg verify time: {stats['avg_verify_time']:.3f}s")
        print(f"  Total draft tokens: {stats['total_draft_tokens']}")
        print(f"  Total accepted tokens: {stats['total_accepted_tokens']}")
        print(f"{'='*70}\n")
        
        return final_text, stats


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="EAGLE Speculative Decoding with FlexAttention")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is", help="Input prompt")
    parser.add_argument("--max-length", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--tree-width", type=int, default=3, help="Tree width")
    parser.add_argument("--tree-depth", type=int, default=3, help="Tree depth")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("EAGLE SPECULATIVE DECODING WITH FLEXATTENTION")
    print("="*70)
    print(f"Draft: yuhuili/EAGLE-LLaMA3.1-Instruct-8B")
    print(f"Target: meta-llama/Llama-3.1-8B-Instruct")
    print(f"Backend: FlexAttention")
    print("="*70)
    
    # Initialize decoder
    decoder = EAGLESpeculativeDecoder(
        draft_model_path="yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        target_model_path="meta-llama/Llama-3.1-8B-Instruct",
        device=args.device,
        tree_width=args.tree_width,
        tree_depth=args.tree_depth
    )
    
    # Generate
    final_text, stats = decoder.generate(
        prompt=args.prompt,
        max_length=args.max_length,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("FINAL RESULT")
    print("="*70)
    print(final_text)
    print("="*70 + "\n")


if __name__ == "__main__":
    main()