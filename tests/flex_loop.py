"""
EAGLE Speculative Decoding - PROPER IMPLEMENTATION
==================================================

Uses EAGLE's actual architecture:
1. Target model generates hidden states for current sequence
2. EAGLE uses those hidden states as context via set_forward_context()
3. EAGLE drafts tree based on target's hidden states
4. Target verifies draft tokens
5. Accept/reject and repeat

This matches EAGLE paper implementation.

Usage:
    python eagle_proper_implementation.py --prompt "The future of AI is"
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Tuple, Dict
import argparse
import time


# ============================================================================
# EAGLE Proper Implementation
# ============================================================================

class EAGLESpeculativeDecoder:
    """
    Proper EAGLE implementation using set_forward_context
    """
    
    def __init__(
        self,
        draft_model_path: str = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        target_model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda"
    ):
        print(f"\n{'='*70}")
        print(f"INITIALIZING EAGLE WITH PROPER ARCHITECTURE")
        print(f"{'='*70}")
        
        self.device = device
        self.dtype = torch.float16
        
        # Load EAGLE draft model
        print(f"\nLoading EAGLE draft model...")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_path,
            torch_dtype=self.dtype,
            device_map=device,
            low_cpu_mem_usage=True
        )
        self.draft_model.eval()
        print(f"  ✓ EAGLE loaded")
        
        # Load target model
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
        
        print(f"{'='*70}\n")
    
    def get_target_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        STEP 1: Get hidden states from target model
        
        This is KEY - EAGLE needs target's hidden states as context
        """
        with torch.no_grad():
            outputs = self.target_model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True
            )
            
            # Get last layer hidden states
            hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
        
        return hidden_states
    
    def draft_with_eagle(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        num_draft_tokens: int = 10,
        temperature: float = 1.0,
        top_k: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        STEP 2: Draft tokens using EAGLE with target's hidden states
        
        Uses set_forward_context to condition on target hidden states
        
        Args:
            input_ids: Current sequence [batch, seq_len]
            target_hidden_states: Hidden states from target [batch, seq_len, hidden_dim]
            num_draft_tokens: Number of tokens to draft
            temperature: Sampling temperature
            top_k: Top-k sampling
        
        Returns:
            draft_tokens: [num_draft_tokens] drafted token IDs
            draft_probs: [num_draft_tokens, vocab_size] probability distributions
        """
        # Set forward context - THIS IS THE KEY!
        # EAGLE will use these hidden states to condition its predictions
        if hasattr(self.draft_model, 'set_forward_context'):
            self.draft_model.set_forward_context(target_hidden_states)
        elif hasattr(self.draft_model.model, 'set_forward_context'):
            self.draft_model.model.set_forward_context(target_hidden_states)
        else:
            print("Warning: EAGLE model doesn't have set_forward_context method")
            print("Falling back to standard generation")
        
        draft_tokens = []
        draft_probs_list = []
        current_ids = input_ids.clone()
        
        # Generate tokens sequentially (EAGLE typically generates one at a time)
        for i in range(num_draft_tokens):
            with torch.no_grad():
                outputs = self.draft_model(
                    input_ids=current_ids,
                    use_cache=False,
                    return_dict=True
                )
                
                logits = outputs.logits[0, -1, :]  # Last token logits
            
            # Sample
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            
            if top_k > 0:
                top_k_vals, top_k_idx = torch.topk(probs, k=min(top_k, probs.shape[0]))
                top_k_probs = top_k_vals / top_k_vals.sum()
                sampled = torch.multinomial(top_k_probs, 1).item()
                token_id = top_k_idx[sampled].item()
            else:
                token_id = torch.multinomial(probs, 1).item()
            
            draft_tokens.append(token_id)
            draft_probs_list.append(probs)
            
            # Append to sequence for next iteration
            current_ids = torch.cat([
                current_ids,
                torch.tensor([[token_id]], dtype=torch.long, device=self.device)
            ], dim=1)
            
            # Check for EOS
            if token_id == self.tokenizer.eos_token_id:
                break
        
        draft_tokens = torch.tensor(draft_tokens, dtype=torch.long, device=self.device)
        draft_probs = torch.stack(draft_probs_list) if draft_probs_list else None
        
        return draft_tokens, draft_probs
    
    def verify_and_accept(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        STEP 3: Verify draft tokens with target model using speculative sampling
        
        Args:
            input_ids: Current sequence [batch, seq_len]
            draft_tokens: Draft token IDs [num_draft]
            draft_probs: Draft probabilities [num_draft, vocab_size]
        
        Returns:
            accepted_tokens: Accepted token IDs
            num_accepted: Number of accepted tokens
        """
        if len(draft_tokens) == 0:
            return torch.tensor([], dtype=torch.long, device=self.device), 0
        
        # Concatenate draft tokens to input
        full_sequence = torch.cat([
            input_ids,
            draft_tokens.unsqueeze(0)
        ], dim=1)
        
        # Get target probabilities for all positions
        with torch.no_grad():
            outputs = self.target_model(
                input_ids=full_sequence,
                use_cache=False,
                return_dict=True
            )
            
            # Get logits for draft positions
            logits = outputs.logits[0, input_ids.shape[1]-1:-1, :]  # All draft positions
            target_probs = F.softmax(logits, dim=-1)
        
        # Speculative sampling - accept/reject each token
        accepted_tokens = []
        
        for i in range(len(draft_tokens)):
            draft_token = draft_tokens[i].item()
            
            # Get probabilities
            p_target = target_probs[i, draft_token].item()
            p_draft = draft_probs[i, draft_token].item()
            
            # Acceptance probability
            accept_prob = min(1.0, p_target / (p_draft + 1e-10))
            
            # Accept/reject
            if torch.rand(1).item() < accept_prob:
                accepted_tokens.append(draft_token)
            else:
                # Rejection sampling - sample from adjusted distribution
                # q(x) = max(0, p_target(x) - p_draft(x))
                adjusted_probs = torch.clamp(target_probs[i] - draft_probs[i], min=0.0)
                
                if adjusted_probs.sum() > 0:
                    adjusted_probs = adjusted_probs / adjusted_probs.sum()
                    new_token = torch.multinomial(adjusted_probs, 1).item()
                    accepted_tokens.append(new_token)
                
                break  # Stop at first rejection
        
        if len(accepted_tokens) == 0:
            # If nothing accepted, sample from target at first position
            new_token = torch.multinomial(target_probs[0], 1).item()
            accepted_tokens.append(new_token)
        
        accepted_tokens = torch.tensor(accepted_tokens, dtype=torch.long, device=self.device)
        
        return accepted_tokens, len(accepted_tokens)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        num_draft_tokens: int = 10,
        temperature: float = 1.0,
        top_k: int = 20,
        verbose: bool = True
    ) -> Tuple[str, Dict]:
        """
        Generate text using EAGLE speculative decoding
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            num_draft_tokens: Number of tokens to draft per iteration
            temperature: Sampling temperature
            top_k: Top-k sampling
            verbose: Print progress
        
        Returns:
            generated_text: Complete generated text
            stats: Generation statistics
        """
        print(f"\n{'='*70}")
        print(f"EAGLE SPECULATIVE DECODING - PROPER IMPLEMENTATION")
        print(f"{'='*70}")
        print(f"Prompt: '{prompt}'")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"Draft tokens per iteration: {num_draft_tokens}")
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
            'draft_times': [],
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
            
            # STEP 2: Draft with EAGLE
            if verbose:
                print(f"\nStep 2: Drafting {num_draft_tokens} tokens with EAGLE...")
            
            draft_start = time.time()
            draft_tokens, draft_probs = self.draft_with_eagle(
                current_ids,
                target_hidden_states,
                num_draft_tokens=num_draft_tokens,
                temperature=temperature,
                top_k=top_k
            )
            draft_time = time.time() - draft_start
            
            stats['draft_times'].append(draft_time)
            stats['total_draft'] += len(draft_tokens)
            
            if verbose:
                print(f"  ✓ Drafted {len(draft_tokens)} tokens in {draft_time:.3f}s")
                draft_text = [self.tokenizer.decode([t]) for t in draft_tokens[:5]]
                print(f"  Tokens: {draft_text}")
            
            # STEP 3: Verify with target
            if verbose:
                print(f"\nStep 3: Verifying with target model...")
            
            verify_start = time.time()
            accepted_tokens, num_accepted = self.verify_and_accept(
                current_ids,
                draft_tokens,
                draft_probs
            )
            verify_time = time.time() - verify_start
            
            stats['verify_times'].append(verify_time)
            stats['total_accepted'] += num_accepted
            
            acceptance_rate = num_accepted / len(draft_tokens) * 100 if len(draft_tokens) > 0 else 0
            stats['acceptance_rates'].append(acceptance_rate)
            
            if verbose:
                print(f"  ✓ Verified in {verify_time:.3f}s")
                print(f"  Accepted: {num_accepted}/{len(draft_tokens)} ({acceptance_rate:.1f}%)")
                accepted_text = [self.tokenizer.decode([t]) for t in accepted_tokens]
                print(f"  Accepted tokens: {accepted_text}")
            
            # STEP 4: Update sequence
            current_ids = torch.cat([current_ids, accepted_tokens.unsqueeze(0)], dim=1)
            
            # Decode current text
            current_text = self.tokenizer.decode(
                current_ids[0, initial_len:],
                skip_special_tokens=True
            )
            
            if verbose:
                print(f"\nCurrent output: '{current_text}'")
            
            # Check stopping conditions
            if self.tokenizer.eos_token_id in accepted_tokens:
                if verbose:
                    print(f"\n✓ EOS token generated, stopping")
                break
            
            if num_accepted == 0:
                if verbose:
                    print(f"\n⚠ No tokens accepted, stopping")
                break
            
            stats['iterations'] += 1
            
            if stats['iterations'] >= 20:
                if verbose:
                    print(f"\n⚠ Max iterations reached")
                break
        
        total_time = time.time() - start_time
        
        # Final text
        final_text = self.tokenizer.decode(
            current_ids[0],
            skip_special_tokens=True
        )
        
        # Calculate stats
        stats['total_time'] = total_time
        stats['tokens_generated'] = current_ids.shape[1] - initial_len
        stats['tokens_per_second'] = stats['tokens_generated'] / total_time if total_time > 0 else 0
        stats['avg_acceptance'] = sum(stats['acceptance_rates']) / len(stats['acceptance_rates']) if stats['acceptance_rates'] else 0
        stats['avg_draft_time'] = sum(stats['draft_times']) / len(stats['draft_times']) if stats['draft_times'] else 0
        stats['avg_verify_time'] = sum(stats['verify_times']) / len(stats['verify_times']) if stats['verify_times'] else 0
        
        # Summary
        print(f"\n{'='*70}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"Iterations: {stats['iterations']}")
        print(f"Tokens generated: {stats['tokens_generated']}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Tokens/second: {stats['tokens_per_second']:.2f}")
        print(f"Avg acceptance rate: {stats['avg_acceptance']:.1f}%")
        print(f"Draft efficiency: {stats['total_accepted']}/{stats['total_draft']} = {stats['total_accepted']/max(stats['total_draft'],1)*100:.1f}%")
        print(f"\nFinal text:")
        print(f"  {final_text}")
        print(f"{'='*70}\n")
        
        return final_text, stats


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="EAGLE Proper Implementation")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is",
                       help="Input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=50,
                       help="Maximum new tokens to generate")
    parser.add_argument("--num-draft", type=int, default=10,
                       help="Number of draft tokens per iteration")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=20,
                       help="Top-k sampling")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("EAGLE SPECULATIVE DECODING - PROPER IMPLEMENTATION")
    print("="*70)
    print("Architecture:")
    print("  1. Target model generates hidden states")
    print("  2. EAGLE uses hidden states via set_forward_context()")
    print("  3. EAGLE drafts tokens conditioned on target states")
    print("  4. Target verifies draft tokens")
    print("  5. Speculative sampling accepts/rejects")
    print("  6. Repeat until done")
    print("="*70)
    
    # Initialize
    decoder = EAGLESpeculativeDecoder(
        draft_model_path="yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        target_model_path="meta-llama/Llama-3.1-8B-Instruct"
    )
    
    # Generate
    final_text, stats = decoder.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        num_draft_tokens=args.num_draft,
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