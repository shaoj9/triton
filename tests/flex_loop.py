"""
EAGLE Speculative Decoding - CORRECTED VERSION
==============================================

Fixes:
1. Proper chat template formatting (no "assistant" tokens)
2. Better acceptance logic with speculative sampling
3. Greedy path selection based on target probabilities
4. Realistic stopping conditions
5. Better token quality and diversity

Usage:
    python eagle_speculative_corrected.py --prompt "The future of AI is"
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
# FlexAttention Backend
# ============================================================================

def create_tree_attention_mask(
    parent_ids: List[Optional[int]],
    prefix_len: int,
    device: str = "cuda"
) -> torch.Tensor:
    """Create FlexAttention-style tree attention mask"""
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
    
    attention_mask = torch.where(
        mask,
        torch.zeros(total_len, total_len, dtype=torch.float16, device=device),
        torch.full((total_len, total_len), float('-inf'), dtype=torch.float16, device=device)
    )
    
    return attention_mask.unsqueeze(0).unsqueeze(0)


def create_position_ids(nodes: List[TreeNode], prefix_len: int, device: str = "cuda"):
    """Create position IDs"""
    num_nodes = len(nodes)
    position_ids = torch.zeros(1, prefix_len + num_nodes, dtype=torch.long, device=device)
    position_ids[0, :prefix_len] = torch.arange(prefix_len)
    
    for node in nodes:
        position_ids[0, prefix_len + node.node_id] = prefix_len + node.depth
    
    return position_ids


# ============================================================================
# EAGLE Speculative Decoder - CORRECTED
# ============================================================================

class EAGLESpeculativeDecoderCorrected:
    """
    CORRECTED EAGLE Speculative Decoder with proper formatting and logic
    """
    
    def __init__(
        self,
        draft_model_path: str = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        target_model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        tree_width: int = 3,
        tree_depth: int = 2  # Smaller tree for better quality
    ):
        print(f"\n{'='*70}")
        print(f"INITIALIZING CORRECTED EAGLE DECODER")
        print(f"{'='*70}")
        
        self.device = device
        self.dtype = torch.float16
        self.tree_width = tree_width
        self.tree_depth = tree_depth
        self.num_tree_nodes = sum(tree_width**d for d in range(tree_depth + 1))
        
        print(f"Tree: width={tree_width}, depth={tree_depth}, nodes={self.num_tree_nodes}")
        
        # Load models
        print(f"\nLoading DRAFT model...")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_path,
            torch_dtype=self.dtype,
            device_map=device,
            low_cpu_mem_usage=True
        )
        self.draft_model.eval()
        print(f"  ✓ EAGLE loaded")
        
        print(f"\nLoading TARGET model...")
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
        self.target_embedding = self.target_model.model.embed_tokens
        
        print(f"{'='*70}\n")
    
    def format_prompt_for_instruct(self, prompt: str) -> str:
        """
        FIX 1: Use proper chat template for instruction-tuned models
        This prevents "assistant" tokens in output
        """
        messages = [
            {"role": "user", "content": f"Continue this text naturally: {prompt}"}
        ]
        
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return formatted
    
    def generate_draft_tree(
        self,
        prefix_ids: torch.Tensor,
        top_k: int = 20,  # Increased for diversity
        temperature: float = 1.0  # Higher for diversity
    ) -> Tuple[List[TreeNode], torch.Tensor, torch.Tensor]:
        """
        Generate draft tree with EAGLE
        
        Returns:
            nodes: Tree nodes
            draft_token_ids: [num_nodes] sampled token IDs
            draft_probs: [num_nodes, vocab_size] probability distributions
        """
        prefix_len = prefix_ids.shape[1]
        
        # Build tree
        nodes, parent_ids = build_tree_structure(self.tree_width, self.tree_depth)
        
        # Create attention mask
        attention_mask = create_tree_attention_mask(parent_ids, prefix_len, self.device)
        position_ids = create_position_ids(nodes, prefix_len, self.device)
        
        # Embeddings
        prefix_embeds = self.draft_embedding(prefix_ids)
        tree_embeds = torch.zeros(
            1, len(nodes), prefix_embeds.shape[-1],
            dtype=self.dtype,
            device=self.device
        )
        full_embeds = torch.cat([prefix_embeds, tree_embeds], dim=1)
        
        # Forward through draft
        with torch.no_grad():
            outputs = self.draft_model.model(
                inputs_embeds=full_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True
            )
            logits = self.draft_model.lm_head(outputs.last_hidden_state)
        
        tree_logits = logits[0, prefix_len:, :]
        
        # Sample tokens with better diversity
        draft_token_ids = []
        draft_probs_list = []
        used_tokens = set(prefix_ids[0].tolist())
        
        for node_idx, node in enumerate(nodes):
            node_logits = tree_logits[node_idx].clone()
            
            # Light repetition penalty
            for token_id in used_tokens:
                if token_id < node_logits.shape[0]:
                    node_logits[token_id] /= 1.1
            
            # Sample
            scaled = node_logits / temperature
            probs = F.softmax(scaled, dim=-1)
            
            # Top-k sampling
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
        
        return nodes, draft_token_ids, draft_probs
    
    def verify_and_accept(
        self,
        prefix_ids: torch.Tensor,
        draft_nodes: List[TreeNode],
        draft_token_ids: torch.Tensor,
        draft_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        FIX 2: Proper speculative sampling with rejection
        """
        prefix_len = prefix_ids.shape[1]
        parent_ids = [node.parent_id for node in draft_nodes]
        
        # Create masks
        attention_mask = create_tree_attention_mask(parent_ids, prefix_len, self.device)
        position_ids = create_position_ids(draft_nodes, prefix_len, self.device)
        
        # Embeddings with draft tokens
        prefix_embeds = self.target_embedding(prefix_ids)
        draft_embeds = self.target_embedding(draft_token_ids.unsqueeze(0))
        full_embeds = torch.cat([prefix_embeds, draft_embeds], dim=1)
        
        # Forward through target
        with torch.no_grad():
            outputs = self.target_model.model(
                inputs_embeds=full_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True
            )
            target_logits = self.target_model.lm_head(outputs.last_hidden_state)
        
        tree_target_logits = target_logits[0, prefix_len:, :]
        target_probs = F.softmax(tree_target_logits, dim=-1)
        
        # Store target probs in nodes
        for node_idx, node in enumerate(draft_nodes):
            node.target_prob = target_probs[node_idx, node.token_id].item()
        
        # FIX 3: Find best path through tree based on target probabilities
        accepted_path = self._find_best_path(draft_nodes, target_probs, draft_probs)
        
        if len(accepted_path) == 0:
            # If no tokens accepted, sample from target at root
            root_target_probs = target_probs[0]
            new_token = torch.multinomial(root_target_probs, 1).item()
            return torch.tensor([new_token], device=self.device), []
        
        accepted_ids = torch.tensor(
            [draft_nodes[node_id].token_id for node_id in accepted_path],
            dtype=torch.long,
            device=self.device
        )
        
        return accepted_ids, accepted_path
    
    def _find_best_path(
        self,
        nodes: List[TreeNode],
        target_probs: torch.Tensor,
        draft_probs: torch.Tensor
    ) -> List[int]:
        """
        FIX 3: Find best path through tree using speculative sampling
        """
        accepted_path = []
        current_node_id = 0
        
        while True:
            node = nodes[current_node_id]
            
            # Acceptance probability
            p_target = target_probs[node.node_id, node.token_id].item()
            p_draft = draft_probs[node.node_id, node.token_id].item()
            
            accept_prob = min(1.0, p_target / (p_draft + 1e-10))
            
            # Accept/reject
            if torch.rand(1).item() < accept_prob:
                accepted_path.append(node.node_id)
                
                # If leaf, stop
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
                # Rejected - stop
                break
        
        return accepted_path
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        verbose: bool = True
    ) -> Tuple[str, Dict]:
        """
        Generate with EAGLE speculative decoding
        """
        print(f"\n{'='*70}")
        print(f"EAGLE SPECULATIVE DECODING - CORRECTED")
        print(f"{'='*70}")
        print(f"Prompt: '{prompt}'")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"{'='*70}\n")
        
        # FIX 1: Format with chat template
        formatted_prompt = self.format_prompt_for_instruct(prompt)
        
        if verbose:
            print(f"Formatted prompt (first 100 chars):")
            print(f"  {formatted_prompt[:100]}...")
            print()
        
        # Tokenize
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()
        
        initial_len = current_ids.shape[1]
        
        # Stats
        stats = {
            'iterations': 0,
            'total_draft': 0,
            'total_accepted': 0,
            'acceptance_rates': [],
            'tokens_per_iteration': []
        }
        
        start_time = time.time()
        
        # Generation loop
        while (current_ids.shape[1] - initial_len) < max_new_tokens:
            iteration = stats['iterations']
            
            if verbose:
                print(f"\n{'='*70}")
                print(f"ITERATION {iteration + 1}")
                print(f"{'='*70}")
                current_generated = current_ids.shape[1] - initial_len
                print(f"Generated so far: {current_generated}/{max_new_tokens} tokens")
            
            # Draft
            if verbose:
                print(f"\nDrafting {self.num_tree_nodes} tokens with EAGLE...")
            
            draft_start = time.time()
            nodes, draft_ids, draft_probs = self.generate_draft_tree(current_ids)
            draft_time = time.time() - draft_start
            
            if verbose:
                print(f"  ✓ Draft complete ({draft_time:.3f}s)")
                draft_tokens_text = [self.tokenizer.decode([t]) for t in draft_ids[:5]]
                print(f"  Sample: {draft_tokens_text}...")
            
            # Verify
            if verbose:
                print(f"\nVerifying with target...")
            
            verify_start = time.time()
            accepted_ids, accepted_path = self.verify_and_accept(
                current_ids, nodes, draft_ids, draft_probs
            )
            verify_time = time.time() - verify_start
            
            num_accepted = len(accepted_ids)
            acceptance_rate = num_accepted / len(draft_ids) * 100
            
            stats['total_draft'] += len(draft_ids)
            stats['total_accepted'] += num_accepted
            stats['acceptance_rates'].append(acceptance_rate)
            stats['tokens_per_iteration'].append(num_accepted)
            
            if verbose:
                print(f"  ✓ Verify complete ({verify_time:.3f}s)")
                print(f"  Accepted: {num_accepted}/{len(draft_ids)} ({acceptance_rate:.1f}%)")
                accepted_text = [self.tokenizer.decode([t]) for t in accepted_ids]
                print(f"  Tokens: {accepted_text}")
            
            # Update
            current_ids = torch.cat([current_ids, accepted_ids.unsqueeze(0)], dim=1)
            
            # Decode
            current_text = self.tokenizer.decode(
                current_ids[0, initial_len:],
                skip_special_tokens=True
            )
            
            if verbose:
                print(f"\nGenerated text: '{current_text}'")
            
            # Check EOS
            if self.tokenizer.eos_token_id in accepted_ids:
                if verbose:
                    print(f"\n✓ EOS token, stopping")
                break
            
            # Check if we accepted nothing
            if num_accepted == 0:
                if verbose:
                    print(f"\n⚠ No tokens accepted, stopping")
                break
            
            stats['iterations'] += 1
            
            if stats['iterations'] >= 30:
                if verbose:
                    print(f"\n⚠ Max iterations, stopping")
                break
        
        total_time = time.time() - start_time
        
        # Final text
        final_text = self.tokenizer.decode(
            current_ids[0, initial_len:],
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
        print(f"Time: {total_time:.2f}s")
        print(f"Tokens/sec: {stats['tokens_per_second']:.2f}")
        print(f"Avg acceptance: {stats['avg_acceptance']:.1f}%")
        print(f"Draft efficiency: {stats['total_accepted']}/{stats['total_draft']} = {stats['total_accepted']/stats['total_draft']*100:.1f}%")
        print(f"\nFinal text:")
        print(f"  {prompt}{final_text}")
        print(f"{'='*70}\n")
        
        return prompt + final_text, stats


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="EAGLE Speculative Decoding - Corrected")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is",
                       help="Input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=50,
                       help="Maximum new tokens to generate")
    parser.add_argument("--tree-width", type=int, default=3,
                       help="Tree width")
    parser.add_argument("--tree-depth", type=int, default=2,
                       help="Tree depth (2-3 recommended)")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("EAGLE SPECULATIVE DECODING - CORRECTED VERSION")
    print("="*70)
    print("Fixes:")
    print("  1. Proper chat template (no 'assistant' tokens)")
    print("  2. Better acceptance logic (speculative sampling)")
    print("  3. Best path selection (target probabilities)")
    print("  4. Improved token quality and diversity")
    print("="*70)
    
    # Initialize
    decoder = EAGLESpeculativeDecoderCorrected(
        draft_model_path="yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        target_model_path="meta-llama/Llama-3.1-8B-Instruct",
        tree_width=args.tree_width,
        tree_depth=args.tree_depth
    )
    
    # Generate
    final_text, stats = decoder.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("FINAL OUTPUT")
    print("="*70)
    print(final_text)
    print("="*70 + "\n")


if __name__ == "__main__":
    main()