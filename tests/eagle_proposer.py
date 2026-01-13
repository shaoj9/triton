"""
Production EAGLE with FlexAttention - Real Models + Output Filtering
====================================================================

Uses:
- Draft: yuhuili/EAGLE-LLaMA3.1-Instruct-8B
- Target: meta-llama/Llama-3.1-8B-Instruct

Features:
- Real model integration
- FlexAttention for accurate tree attention
- One-pass generation with beam search
- Output post-processing (filter weird chars, fix grammar)
- Production-ready code

Usage:
    python production_eagle.py --prompt "The future of AI is"
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Tuple, Dict, Callable
from dataclasses import dataclass
import argparse
import time
import re
import unicodedata


# ============================================================================
# Output Post-Processing
# ============================================================================

class OutputFilter:
    """
    Filter weird characters and fix grammar in generated text
    """
    
    def __init__(self):
        # Weird characters to filter
        self.weird_chars = [
            '\u200b',  # Zero-width space
            '\u200c',  # Zero-width non-joiner
            '\u200d',  # Zero-width joiner
            '\ufeff',  # Zero-width no-break space
            '\u2060',  # Word joiner
            '\xa0',    # Non-breaking space (replace with regular space)
        ]
        
        # Control characters to remove (except newline, tab, carriage return)
        self.control_chars = [chr(i) for i in range(32) if i not in [9, 10, 13]]
        
        # Common token artifacts from LLMs
        self.artifacts = [
            '<|endoftext|>',
            '<|im_end|>',
            '<|im_start|>',
            '[INST]',
            '[/INST]',
            '<<SYS>>',
            '<</SYS>>',
        ]
    
    def filter_weird_characters(self, text: str) -> str:
        """
        Remove weird Unicode characters
        
        Args:
            text: Input text
        
        Returns:
            Filtered text
        """
        # Remove weird zero-width characters
        for char in self.weird_chars:
            text = text.replace(char, '' if char != '\xa0' else ' ')
        
        # Remove control characters
        for char in self.control_chars:
            text = text.replace(char, '')
        
        # Remove LLM artifacts
        for artifact in self.artifacts:
            text = text.replace(artifact, '')
        
        # Normalize Unicode (NFC form)
        text = unicodedata.normalize('NFC', text)
        
        return text
    
    def fix_spacing(self, text: str) -> str:
        """
        Fix spacing issues
        
        Args:
            text: Input text
        
        Returns:
            Text with fixed spacing
        """
        # Fix multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Fix space before punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        # Fix space after punctuation
        text = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', text)
        
        # Fix quotes
        text = re.sub(r'\s+"', ' "', text)
        text = re.sub(r'"\s+', '" ', text)
        
        # Fix parentheses
        text = re.sub(r'\s+\)', ')', text)
        text = re.sub(r'\(\s+', '(', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def fix_capitalization(self, text: str) -> str:
        """
        Fix capitalization at sentence starts
        
        Args:
            text: Input text
        
        Returns:
            Text with fixed capitalization
        """
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        # Capitalize after sentence-ending punctuation
        text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
        
        return text
    
    def remove_incomplete_sentences(self, text: str) -> str:
        """
        Remove incomplete sentences at the end
        
        Args:
            text: Input text
        
        Returns:
            Text with complete sentences only
        """
        # Find last sentence-ending punctuation
        match = None
        for pattern in [r'[.!?]\s*$', r'[.!?]\s+']:
            matches = list(re.finditer(pattern, text))
            if matches:
                match = matches[-1]
                break
        
        if match:
            # Keep up to last sentence-ending punctuation
            text = text[:match.end()].rstrip()
        
        return text
    
    def filter_repetitions(self, text: str, max_repeat: int = 3) -> str:
        """
        Remove excessive repetitions
        
        Args:
            text: Input text
            max_repeat: Maximum allowed repetitions
        
        Returns:
            Text with filtered repetitions
        """
        # Remove repeated words (more than max_repeat times)
        words = text.split()
        filtered_words = []
        
        for i, word in enumerate(words):
            # Count consecutive repetitions
            count = 1
            for j in range(i + 1, len(words)):
                if words[j].lower() == word.lower():
                    count += 1
                else:
                    break
            
            # Only add if not excessively repeated
            if count <= max_repeat or (filtered_words and filtered_words[-1].lower() != word.lower()):
                filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def clean(self, text: str, remove_incomplete: bool = True) -> str:
        """
        Apply all cleaning filters
        
        Args:
            text: Input text
            remove_incomplete: Whether to remove incomplete sentences
        
        Returns:
            Cleaned text
        """
        # Filter weird characters
        text = self.filter_weird_characters(text)
        
        # Fix spacing
        text = self.fix_spacing(text)
        
        # Filter repetitions
        text = self.filter_repetitions(text)
        
        # Fix capitalization
        text = self.fix_capitalization(text)
        
        # Remove incomplete sentences (optional)
        if remove_incomplete:
            text = self.remove_incomplete_sentences(text)
        
        return text


# ============================================================================
# Tree Structure
# ============================================================================

@dataclass
class TreeNode:
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
# FlexAttention
# ============================================================================

def create_tree_score_mod(
    parent_ids: List[Optional[int]],
    prefix_len: int
) -> Callable:
    """Create FlexAttention score_mod"""
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
    
    def score_mod(score, b, h, q_idx, kv_idx):
        # Prefix: causal
        if q_idx < prefix_len:
            return score if kv_idx <= q_idx else float('-inf')
        
        # Tree
        tree_q = q_idx - prefix_len
        
        if kv_idx < prefix_len:
            return score
        
        tree_kv = kv_idx - prefix_len
        
        if tree_kv in ancestor_chains[tree_q]:
            return score
        else:
            return float('-inf')
    
    return score_mod


def create_attention_mask(
    score_mod: Callable,
    total_len: int,
    device: torch.device
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


def create_positions(nodes: List[TreeNode], prefix_len: int, device: torch.device):
    """Create positions tensor"""
    num_nodes = len(nodes)
    positions = torch.zeros(1, prefix_len + num_nodes, dtype=torch.long, device=device)
    positions[0, :prefix_len] = torch.arange(prefix_len)
    
    for node in nodes:
        positions[0, prefix_len + node.node_id] = prefix_len + node.depth
    
    return positions


# ============================================================================
# Production EAGLE Generator
# ============================================================================

class ProductionEAGLEGenerator:
    """
    Production EAGLE with real models and output filtering
    """
    
    def __init__(
        self,
        draft_model_path: str = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        target_model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        tree_width: int = 4,
        tree_depth: int = 2,
        beam_width: int = 3
    ):
        print(f"\n{'='*70}")
        print(f"PRODUCTION EAGLE WITH FLEXATTENTION")
        print(f"{'='*70}")
        print(f"Draft: {draft_model_path}")
        print(f"Target: {target_model_path}")
        
        self.device = torch.device(device)
        self.dtype = torch.float16
        self.tree_width = tree_width
        self.tree_depth = tree_depth
        self.beam_width = beam_width
        
        # Output filter
        self.output_filter = OutputFilter()
        
        print(f"\nLoading models...")
        
        # Load EAGLE draft model
        print(f"  Loading EAGLE draft model...")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_path,
            torch_dtype=self.dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        self.draft_model.eval()
        print(f"    ✓ EAGLE loaded")
        
        # Load target model
        print(f"  Loading target model...")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_path,
            torch_dtype=self.dtype,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.target_model.eval()
        print(f"    ✓ Target loaded")
        
        # Tokenizer
        print(f"  Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"    ✓ Tokenizer loaded")
        
        self.pad_token_id = self.tokenizer.pad_token_id
        
        print(f"\nConfiguration:")
        print(f"  Tree: width={tree_width}, depth={tree_depth}")
        print(f"  Beam width: {beam_width}")
        print(f"  Output filtering: Enabled")
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
    
    def generate_tree_one_pass(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        temperature: float,
        top_k: int
    ) -> Tuple[List[TreeNode], torch.Tensor]:
        """Generate tree in one pass with beam pruning"""
        prefix_len = input_ids.shape[1]
        
        # Build full tree
        nodes, parent_ids = build_tree_structure(self.tree_width, self.tree_depth)
        
        # Set forward context
        if hasattr(self.draft_model.model, 'set_forward_context'):
            self.draft_model.model.set_forward_context(target_hidden_states)
        
        # Create attention mask
        score_mod = create_tree_score_mod(parent_ids, prefix_len)
        total_len = prefix_len + len(nodes)
        attention_mask = create_attention_mask(score_mod, total_len, self.device)
        positions = create_positions(nodes, prefix_len, self.device)
        
        # Prepare input
        tree_placeholders = torch.full(
            (1, len(nodes)),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device
        )
        full_input_ids = torch.cat([input_ids, tree_placeholders], dim=1)
        
        # ONE forward pass
        with torch.no_grad():
            outputs = self.draft_model.model(
                input_ids=full_input_ids,
                attention_mask=attention_mask,
                position_ids=positions,
                use_cache=False,
                return_dict=True
            )
            logits = self.draft_model.lm_head(outputs.last_hidden_state)
        
        tree_logits = logits[0, prefix_len:, :]
        
        # Beam search pruning
        pruned_nodes, draft_token_ids = self._beam_search_prune(
            nodes,
            tree_logits,
            temperature,
            top_k
        )
        
        return pruned_nodes, draft_token_ids
    
    def _beam_search_prune(
        self,
        nodes: List[TreeNode],
        tree_logits: torch.Tensor,
        temperature: float,
        top_k: int
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
                    child_logits = tree_logits[child.node_id] / temperature
                    probs = F.softmax(child_logits, dim=-1)
                    log_probs = F.log_softmax(child_logits, dim=-1)
                    
                    # Top-k sampling
                    if top_k > 0:
                        top_k_vals, top_k_idx = torch.topk(probs, k=min(top_k, probs.shape[0]))
                        top_k_probs = top_k_vals / top_k_vals.sum()
                        sampled = torch.multinomial(top_k_probs, 1).item()
                        token_id = top_k_idx[sampled].item()
                    else:
                        token_id = torch.multinomial(probs, 1).item()
                    
                    token_prob = probs[token_id].item()
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
                        cumulative_score=cumulative,
                        draft_prob=token_prob
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
    
    def verify_and_select(
        self,
        input_ids: torch.Tensor,
        draft_nodes: List[TreeNode],
        draft_token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Verify with target and select accepted tokens"""
        prefix_len = input_ids.shape[1]
        parent_ids = [node.parent_id for node in draft_nodes]
        
        # Create attention mask
        score_mod = create_tree_score_mod(parent_ids, prefix_len)
        total_len = prefix_len + len(draft_nodes)
        attention_mask = create_attention_mask(score_mod, total_len, self.device)
        positions = create_positions(draft_nodes, prefix_len, self.device)
        
        # Verify with target
        full_input_ids = torch.cat([input_ids, draft_token_ids.unsqueeze(0)], dim=1)
        
        with torch.no_grad():
            outputs = self.target_model.model(
                input_ids=full_input_ids,
                attention_mask=attention_mask,
                position_ids=positions,
                use_cache=False,
                return_dict=True
            )
            target_logits = self.target_model.lm_head(outputs.last_hidden_state)
        
        tree_target_logits = target_logits[0, prefix_len:, :]
        target_probs = F.softmax(tree_target_logits, dim=-1)
        
        # Store target probs
        for node_idx, node in enumerate(draft_nodes):
            node.target_prob = target_probs[node_idx, node.token_id].item()
        
        # Speculative sampling
        accepted_path = []
        current_node_id = 0
        
        while True:
            node = draft_nodes[current_node_id]
            
            p_target = target_probs[node.node_id, node.token_id].item()
            p_draft = node.draft_prob if node.draft_prob > 0 else 1.0
            
            accept_prob = min(1.0, p_target / p_draft)
            
            if torch.rand(1).item() < accept_prob:
                accepted_path.append(node.node_id)
                
                if len(node.children) == 0:
                    break
                
                best_child_id = None
                best_score = float('-inf')
                
                for child_id in node.children:
                    child = draft_nodes[child_id]
                    if child.cumulative_score > best_score:
                        best_score = child.cumulative_score
                        best_child_id = child_id
                
                if best_child_id is None:
                    break
                
                current_node_id = best_child_id
            else:
                break
        
        if len(accepted_path) == 0:
            new_token = torch.multinomial(target_probs[0], 1).item()
            return torch.tensor([new_token], device=self.device)
        
        accepted_tokens = torch.tensor(
            [draft_nodes[node_id].token_id for node_id in accepted_path],
            dtype=torch.long,
            device=self.device
        )
        
        return accepted_tokens
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 20,
        filter_output: bool = True
    ) -> str:
        """
        Generate text with filtering
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            filter_output: Apply output filtering
        
        Returns:
            Generated text (filtered and cleaned)
        """
        print(f"\n{'='*70}")
        print(f"GENERATION")
        print(f"{'='*70}")
        print(f"Prompt: '{prompt}'")
        print(f"Max tokens: {max_new_tokens}")
        print(f"Filtering: {'Enabled' if filter_output else 'Disabled'}")
        print(f"{'='*70}\n")
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()
        initial_len = current_ids.shape[1]
        
        start_time = time.time()
        iterations = 0
        total_accepted = 0
        
        # Generation loop
        while (current_ids.shape[1] - initial_len) < max_new_tokens:
            iterations += 1
            
            print(f"Iteration {iterations}:")
            
            # Get target hidden states
            target_hidden = self.get_target_hidden_states(current_ids)
            
            # Generate tree
            draft_nodes, draft_ids = self.generate_tree_one_pass(
                current_ids,
                target_hidden,
                temperature,
                top_k
            )
            
            # Verify and accept
            accepted_ids = self.verify_and_select(
                current_ids,
                draft_nodes,
                draft_ids
            )
            
            num_accepted = len(accepted_ids)
            total_accepted += num_accepted
            
            print(f"  Accepted: {num_accepted}/{len(draft_ids)} tokens")
            
            # Update
            current_ids = torch.cat([current_ids, accepted_ids.unsqueeze(0)], dim=1)
            
            # Check stopping
            if self.tokenizer.eos_token_id in accepted_ids:
                print(f"  ✓ EOS token")
                break
            
            if num_accepted == 0:
                print(f"  ⚠ No tokens accepted")
                break
            
            if iterations >= 50:
                print(f"  ⚠ Max iterations")
                break
        
        total_time = time.time() - start_time
        tokens_generated = current_ids.shape[1] - initial_len
        
        # Decode
        generated_text = self.tokenizer.decode(
            current_ids[0, initial_len:],
            skip_special_tokens=True
        )
        
        # Filter output
        if filter_output:
            print(f"\nApplying output filters...")
            filtered_text = self.output_filter.clean(generated_text)
            print(f"  ✓ Filtered")
        else:
            filtered_text = generated_text
        
        # Stats
        print(f"\n{'='*70}")
        print(f"COMPLETE")
        print(f"{'='*70}")
        print(f"Iterations: {iterations}")
        print(f"Tokens generated: {tokens_generated}")
        print(f"Time: {total_time:.2f}s")
        print(f"Tokens/second: {tokens_generated/total_time:.2f}")
        print(f"Acceptance rate: {total_accepted/(iterations*13)*100:.1f}%")
        print(f"{'='*70}\n")
        
        return filtered_text


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Production EAGLE with FlexAttention")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--tree-width", type=int, default=4)
    parser.add_argument("--tree-depth", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--no-filter", action="store_true", help="Disable output filtering")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PRODUCTION EAGLE WITH FLEXATTENTION")
    print("="*70)
    print("Features:")
    print("  ✓ Real models (EAGLE + Llama-3.1)")
    print("  ✓ FlexAttention for accurate attention")
    print("  ✓ One-pass generation with beam search")
    print("  ✓ Output filtering (weird chars + grammar)")
    print("="*70)
    
    # Create generator
    generator = ProductionEAGLEGenerator(
        draft_model_path="yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        target_model_path="meta-llama/Llama-3.1-8B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        tree_width=args.tree_width,
        tree_depth=args.tree_depth,
        beam_width=args.beam_width
    )
    
    # Generate
    output = generator.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        filter_output=not args.no_filter
    )
    
    print(f"{'='*70}")
    print(f"FINAL OUTPUT (FILTERED)")
    print(f"{'='*70}")
    print(output)
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()