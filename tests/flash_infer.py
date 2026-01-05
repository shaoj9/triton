import torch
from flashinfer import batch_decode_with_paged_kv_cache
from transformers import AutoTokenizer

class FlashEagleDrafter:
    def __init__(self, head_layer, tokenizer):
        self.head = head_layer  # The 1-layer EAGLE Transformer head
        self.tokenizer = tokenizer

    def generate_tree_one_pass(self, base_hidden_states, tree_structure):
        """
        Generates a tree in one pass using FlashInfer kernels.
        tree_structure: list of parent indices [None, 0, 0, 1, 1] 
                        representing root, 2 children, 2 grandchildren.
        """
        # 1. Prepare FlashInfer Metadata for the tree
        # FlashInfer allows the draft head to attend to multiple prefixes at once
        # Using BatchDCP (Decoupled KV Cache) for the tree
        
        # 2. Structural Forward Pass (One GPU Kernel)
        # Instead of a loop, we pass the flattened tree candidates
        # The FlashInfer kernel manages the sparse attention across the branches
        tree_logits = self.head(
            base_hidden_states, 
            use_flashinfer=True,
            tree_mask=self.create_tree_mask(tree_structure)
        )
        
        # 3. Sample Top-K for each node in the tree structure
        # Logits shape: [num_nodes, vocab_size]
        draft_tokens = torch.argmax(tree_logits, dim=-1)
        
        return draft_tokens, tree_structure

    def create_tree_mask(self, parents):
        # Generates a causal mask where each node attends only to its ancestors
        n = len(parents)
        mask = torch.zeros((n, n), dtype=torch.bool)
        for i, p in enumerate(parents):
            if p is not None:
                mask[i, p] = True
                mask[i] = mask[i] | mask[p] # Inherit ancestor mask
            mask[i, i] = True
        return mask

def print_eagle_tree(tokens, parents, tokenizer):
    print("\n--- Generated Draft Tree (One Pass) ---")
    for i, (tid, p) in enumerate(zip(tokens, parents)):
        depth = 0
        curr = p
        while curr is not None:
            curr = parents[curr]
            depth += 1
        indent = "  " * depth
        print(f"{indent}└── [{tokenizer.decode([tid])}] (Node: {i}, Parent: {p})")