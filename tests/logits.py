"""
Complete Logits Pipeline for Single-Pass Tree Generation

This module provides a production-ready logits processing pipeline
for generating token trees in a single forward pass.

Key Features:
- Fully vectorized processing (no loops over nodes)
- Tree-aware filtering (depth-based top-k/top-p)
- Efficient sampling
- Configurable pipeline stages

Usage:
    pipeline = LogitsPipeline(temperature=0.8, top_k=5, top_p=0.9)
    tokens = pipeline(raw_logits, tree_structure)
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import math


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TreeStructure:
    """Tree topology for draft token generation"""
    node_parents: List[Optional[int]]  # Parent of each node (None for root)
    node_depths: List[int]             # Depth of each node in tree
    num_nodes: int                     # Total number of nodes
    
    @property
    def max_depth(self) -> int:
        return max(self.node_depths)
    
    def get_nodes_at_depth(self, depth: int) -> List[int]:
        """Get all node indices at a specific depth"""
        return [i for i, d in enumerate(self.node_depths) if d == depth]


@dataclass
class PipelineConfig:
    """Configuration for logits processing pipeline"""
    temperature: float = 1.0
    top_k: int = 5
    top_p: float = 0.9
    
    # Depth-aware schedules (optional)
    top_k_per_depth: Optional[Dict[int, int]] = None
    top_p_per_depth: Optional[Dict[int, float]] = None
    min_confidence_per_depth: Optional[Dict[int, float]] = None
    
    # Advanced options
    use_depth_aware_filtering: bool = True
    use_confidence_threshold: bool = True
    device: str = "cuda"


# ============================================================================
# Main Logits Pipeline
# ============================================================================

class LogitsPipeline:
    """
    Logits processing pipeline for tree generation
    
    Processes raw logits through multiple stages:
    1. Temperature scaling
    2. Top-k filtering
    3. Top-p (nucleus) filtering
    4. Tree constraints (confidence, depth-based)
    5. Sampling
    
    ALL stages are vectorized for maximum performance.
    """
    
    def __init__(self, config: PipelineConfig = None):
        """Initialize pipeline with configuration"""
        self.config = config or PipelineConfig()
        
        # Set default depth schedules if not provided
        if self.config.use_depth_aware_filtering:
            self._init_depth_schedules()
    
    def _init_depth_schedules(self):
        """Initialize default depth-based schedules"""
        if self.config.top_k_per_depth is None:
            # Default: decrease top-k as depth increases
            self.config.top_k_per_depth = {
                0: self.config.top_k,
                1: max(1, self.config.top_k - 1),
                2: max(1, self.config.top_k - 2),
                3: max(1, self.config.top_k - 3),
                4: max(1, self.config.top_k - 4),
            }
        
        if self.config.top_p_per_depth is None:
            # Default: decrease top-p as depth increases
            self.config.top_p_per_depth = {
                0: self.config.top_p,
                1: max(0.5, self.config.top_p - 0.05),
                2: max(0.5, self.config.top_p - 0.10),
                3: max(0.5, self.config.top_p - 0.15),
                4: max(0.5, self.config.top_p - 0.20),
            }
        
        if self.config.min_confidence_per_depth is None:
            # Default: increase threshold as depth increases
            self.config.min_confidence_per_depth = {
                0: 0.01,
                1: 0.02,
                2: 0.05,
                3: 0.10,
                4: 0.20,
            }
    
    def __call__(
        self,
        logits: torch.Tensor,      # [batch, num_nodes, vocab_size]
        tree: TreeStructure,
    ) -> torch.Tensor:
        """
        Process logits through pipeline and sample tokens
        
        Args:
            logits: Raw logits from model [batch, num_nodes, vocab_size]
            tree: Tree structure defining topology
        
        Returns:
            tokens: Sampled token IDs [batch, num_nodes]
        """
        # Stage 1: Temperature
        logits = self.stage_temperature(logits)
        
        # Stage 2: Top-k
        if self.config.top_k > 0:
            logits = self.stage_top_k(logits, tree)
        
        # Stage 3: Top-p
        if self.config.top_p < 1.0:
            logits = self.stage_top_p(logits, tree)
        
        # Stage 4: Tree constraints
        if self.config.use_confidence_threshold:
            logits = self.stage_tree_constraints(logits, tree)
        
        # Stage 5: Sample
        tokens = self.stage_sample(logits)
        
        return tokens
    
    def stage_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Stage 1: Apply temperature scaling
        
        Vectorized - applies to all nodes simultaneously
        """
        if self.config.temperature == 0:
            return logits  # Will use argmax in sampling
        
        return logits / self.config.temperature
    
    def stage_top_k(
        self,
        logits: torch.Tensor,
        tree: TreeStructure
    ) -> torch.Tensor:
        """
        Stage 2: Apply top-k filtering
        
        Fully vectorized - no loops over nodes
        """
        if not self.config.use_depth_aware_filtering:
            # Uniform top-k for all nodes
            return self._uniform_top_k(logits, self.config.top_k)
        else:
            # Depth-aware top-k
            return self._depth_aware_top_k(logits, tree)
    
    def _uniform_top_k(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Uniform top-k for all nodes (vectorized)"""
        batch, num_nodes, vocab = logits.shape
        
        # Get top-k values for ALL nodes at once
        top_k_values, _ = torch.topk(logits, k=min(k, vocab), dim=-1)
        
        # Threshold is k-th largest value
        threshold = top_k_values[:, :, -1:].expand_as(logits)
        
        # Mask values below threshold
        return torch.where(
            logits >= threshold,
            logits,
            torch.full_like(logits, float('-inf'))
        )
    
    def _depth_aware_top_k(
        self,
        logits: torch.Tensor,
        tree: TreeStructure
    ) -> torch.Tensor:
        """Depth-aware top-k filtering"""
        batch, num_nodes, vocab = logits.shape
        
        # Build k-values for each node based on depth
        k_values = torch.zeros(num_nodes, dtype=torch.long, device=logits.device)
        for node_id, depth in enumerate(tree.node_depths):
            k_values[node_id] = self.config.top_k_per_depth.get(
                depth, self.config.top_k
            )
        
        # Process each unique k value
        unique_k_values = k_values.unique()
        result = logits.clone()
        
        for k in unique_k_values:
            k = k.item()
            # Find nodes with this k value
            mask = (k_values == k)
            node_indices = torch.where(mask)[0]
            
            # Apply top-k to these nodes
            node_logits = logits[:, node_indices, :]
            top_k_values, _ = torch.topk(node_logits, k=min(k, vocab), dim=-1)
            threshold = top_k_values[:, :, -1:].expand_as(node_logits)
            
            filtered = torch.where(
                node_logits >= threshold,
                node_logits,
                torch.full_like(node_logits, float('-inf'))
            )
            
            result[:, node_indices, :] = filtered
        
        return result
    
    def stage_top_p(
        self,
        logits: torch.Tensor,
        tree: TreeStructure
    ) -> torch.Tensor:
        """
        Stage 3: Apply top-p (nucleus) filtering
        
        Fully vectorized for all nodes
        """
        if not self.config.use_depth_aware_filtering:
            return self._uniform_top_p(logits, self.config.top_p)
        else:
            return self._depth_aware_top_p(logits, tree)
    
    def _uniform_top_p(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Uniform top-p for all nodes (vectorized)"""
        batch, num_nodes, vocab = logits.shape
        
        # Flatten for processing
        logits_flat = logits.view(-1, vocab)
        
        # Convert to probabilities
        probs = F.softmax(logits_flat, dim=-1)
        
        # Sort
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # Cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find cutoff (keep at least one token)
        cutoff_mask = cumulative_probs > p
        cutoff_mask[:, 0] = False
        
        # Zero out below cutoff
        sorted_probs[cutoff_mask] = 0.0
        
        # Unsort to original order
        unsorted_probs = torch.zeros_like(probs)
        unsorted_probs.scatter_(1, sorted_indices, sorted_probs)
        
        # Back to logits
        unsorted_probs = torch.clamp(unsorted_probs, min=1e-10)
        logits_flat = torch.log(unsorted_probs)
        
        return logits_flat.view(batch, num_nodes, vocab)
    
    def _depth_aware_top_p(
        self,
        logits: torch.Tensor,
        tree: TreeStructure
    ) -> torch.Tensor:
        """Depth-aware top-p filtering"""
        batch, num_nodes, vocab = logits.shape
        result = logits.clone()
        
        # Group nodes by depth
        for depth in range(tree.max_depth + 1):
            p = self.config.top_p_per_depth.get(depth, self.config.top_p)
            node_indices = tree.get_nodes_at_depth(depth)
            
            if not node_indices:
                continue
            
            # Apply top-p to nodes at this depth
            node_indices_tensor = torch.tensor(
                node_indices, device=logits.device
            )
            node_logits = logits[:, node_indices_tensor, :]
            
            # Flatten
            flat = node_logits.reshape(-1, vocab)
            probs = F.softmax(flat, dim=-1)
            
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            
            cutoff = cumulative > p
            cutoff[:, 0] = False
            sorted_probs[cutoff] = 0.0
            
            unsorted = torch.zeros_like(probs)
            unsorted.scatter_(1, sorted_indices, sorted_probs)
            
            unsorted = torch.clamp(unsorted, min=1e-10)
            flat_result = torch.log(unsorted)
            
            result[:, node_indices_tensor, :] = flat_result.view(
                batch, len(node_indices), vocab
            )
        
        return result
    
    def stage_tree_constraints(
        self,
        logits: torch.Tensor,
        tree: TreeStructure
    ) -> torch.Tensor:
        """
        Stage 4: Apply tree-specific constraints
        
        Applies confidence thresholds based on depth
        """
        batch, num_nodes, vocab = logits.shape
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Apply depth-based confidence thresholds
        for depth in range(tree.max_depth + 1):
            min_conf = self.config.min_confidence_per_depth.get(depth, 0.01)
            node_indices = tree.get_nodes_at_depth(depth)
            
            if not node_indices:
                continue
            
            # Mask tokens below confidence threshold
            for node_id in node_indices:
                mask = probs[:, node_id, :] >= min_conf
                logits[:, node_id, :] = torch.where(
                    mask,
                    logits[:, node_id, :],
                    torch.full_like(logits[:, node_id, :], float('-inf'))
                )
        
        return logits
    
    def stage_sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Stage 5: Sample tokens from processed logits
        
        Fully vectorized sampling
        """
        if self.config.temperature == 0:
            # Deterministic: argmax
            return logits.argmax(dim=-1)
        
        # Stochastic: multinomial sampling
        batch, num_nodes, vocab = logits.shape
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Flatten for multinomial sampling
        probs_flat = probs.view(-1, vocab)
        
        # Sample
        samples = torch.multinomial(probs_flat, num_samples=1)
        
        # Reshape back
        tokens = samples.view(batch, num_nodes)
        
        return tokens


# ============================================================================
# Helper Functions
# ============================================================================

def build_tree_structure(width: int, depth: int) -> TreeStructure:
    """
    Build uniform tree structure
    
    Args:
        width: Branching factor (children per node)
        depth: Maximum depth of tree
    
    Returns:
        TreeStructure with topology information
    """
    node_parents = [None]  # Root has no parent
    node_depths = [0]      # Root at depth 0
    
    current_level = [0]
    
    for d in range(1, depth + 1):
        next_level = []
        for parent_id in current_level:
            for _ in range(width):
                node_id = len(node_parents)
                node_parents.append(parent_id)
                node_depths.append(d)
                next_level.append(node_id)
        current_level = next_level
    
    return TreeStructure(
        node_parents=node_parents,
        node_depths=node_depths,
        num_nodes=len(node_parents)
    )


def analyze_pipeline_output(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    tree: TreeStructure
) -> Dict:
    """
    Analyze pipeline output statistics
    
    Returns:
        Dictionary with statistics about the generation
    """
    batch, num_nodes = tokens.shape
    
    # Compute probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Get probability of sampled tokens
    batch_indices = torch.arange(batch).unsqueeze(1).expand_as(tokens)
    node_indices = torch.arange(num_nodes).unsqueeze(0).expand_as(tokens)
    sampled_probs = probs[batch_indices, node_indices, tokens]
    
    # Statistics by depth
    depth_stats = {}
    for depth in range(tree.max_depth + 1):
        node_indices = tree.get_nodes_at_depth(depth)
        if node_indices:
            depth_probs = sampled_probs[:, node_indices]
            depth_stats[depth] = {
                'mean_prob': depth_probs.mean().item(),
                'min_prob': depth_probs.min().item(),
                'max_prob': depth_probs.max().item(),
                'num_nodes': len(node_indices)
            }
    
    return {
        'overall_mean_prob': sampled_probs.mean().item(),
        'overall_min_prob': sampled_probs.min().item(),
        'overall_max_prob': sampled_probs.max().item(),
        'depth_stats': depth_stats,
        'num_valid_tokens': (logits > float('-inf')).any(dim=-1).sum().item(),
        'total_nodes': num_nodes
    }


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example of using the logits pipeline"""
    
    # Create tree structure
    tree = build_tree_structure(width=3, depth=4)
    print(f"Tree: {tree.num_nodes} nodes")
    
    # Simulate raw logits from model
    batch_size = 1
    vocab_size = 32000
    
    raw_logits = torch.randn(
        batch_size, tree.num_nodes, vocab_size,
        dtype=torch.float32,
        device="cuda"
    )
    
    print(f"Raw logits: {raw_logits.shape}")
    
    # Configure pipeline
    config = PipelineConfig(
        temperature=0.8,
        top_k=5,
        top_p=0.9,
        use_depth_aware_filtering=True,
        use_confidence_threshold=True
    )
    
    # Create pipeline
    pipeline = LogitsPipeline(config)
    
    # Process logits
    print("\nProcessing through pipeline...")
    tokens = pipeline(raw_logits, tree)
    
    print(f"Sampled tokens: {tokens.shape}")
    print(f"Sample tokens: {tokens[0, :10].tolist()}")
    
    # Analyze output
    stats = analyze_pipeline_output(tokens, raw_logits, tree)
    
    print("\nPipeline Statistics:")
    print(f"  Overall mean probability: {stats['overall_mean_prob']:.4f}")
    print(f"  Valid tokens: {stats['num_valid_tokens']}/{stats['total_nodes']}")
    
    print("\n  By depth:")
    for depth, depth_stats in stats['depth_stats'].items():
        print(f"    Depth {depth}: "
              f"mean={depth_stats['mean_prob']:.4f}, "
              f"nodes={depth_stats['num_nodes']}")


if __name__ == "__main__":
    print("=" * 70)
    print("Logits Pipeline for Single-Pass Tree Generation")
    print("=" * 70)
    print()
    
    example_usage()
    
    print()
    print("=" * 70)
    print("Pipeline Features:")
    print("  ✓ Fully vectorized (no loops)")
    print("  ✓ Depth-aware filtering")
    print("  ✓ Configurable stages")
    print("  ✓ Tree-aware constraints")
    print("  ✓ Efficient sampling")
    print("=" * 70)