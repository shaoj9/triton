import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
import math

def load_eagle3_model(model_path="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"):
    """
    Load EAGLE-3 model and extract components.
    
    EAGLE-3 has only ONE transformer decoder layer!
    """
    
    # Load the model
    eagle_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True
    )
    
    # Extract components
    components = {
        'eagle_layer': eagle_model.model.layers[0],  # Single layer!
        'embed_tokens': eagle_model.model.embed_tokens,
        'lm_head': eagle_model.lm_head,
        'fc': eagle_model.model.fc,  # Feature fusion layer (EAGLE-3)
        'config': eagle_model.config
    }
    
    return components

# Load model
eagle_components = load_eagle3_model()

### **Step 2: Build Tree Structure**

class TreeStructure:
    """
    Tree structure for single-pass generation.
    """
    
    def __init__(self, width=3, depth=4):
        """
        Build uniform tree.
        
        Args:
            width: Branching factor (number of children per node)
            depth: Maximum depth of tree
        """
        self.width = width
        self.depth = depth
        
        # Build tree
        self.node_parents = [None]  # Root has no parent
        self.node_depths = [0]      # Root at depth 0
        
        current_level = [0]  # Start with root
        
        for d in range(1, depth + 1):
            next_level = []
            for parent_idx in current_level:
                for _ in range(width):
                    node_idx = len(self.node_parents)
                    self.node_parents.append(parent_idx)
                    self.node_depths.append(d)
                    next_level.append(node_idx)
            current_level = next_level
        
        self.num_nodes = len(self.node_parents)
        
        print(f"Tree built: {self.num_nodes} nodes, depth {depth}, width {width}")
    
    def build_attention_mask(self, prefix_len, device='cuda'):
        """
        Build tree attention mask.
        
        Each node can attend to:
        - All prefix tokens
        - Itself
        - All ancestors in the tree
        
        Returns: [total_len, total_len] mask
        """
        total_len = prefix_len + self.num_nodes
        
        # Initialize mask (True = can attend)
        mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
        
        # Prefix: causal attention (each token attends to previous)
        for i in range(prefix_len):
            mask[i, :i+1] = True
        
        # Tree nodes attend to all prefix tokens
        mask[prefix_len:, :prefix_len] = True
        
        # Tree nodes attend to ancestors
        for i in range(self.num_nodes):
            # Start from current node
            current = i
            # Walk up to root
            while current is not None:
                mask[prefix_len + i, prefix_len + current] = True
                current = self.node_parents[current]
        
        # Convert to attention format (0 for attend, -inf for mask)
        attention_mask = torch.zeros(
            1, 1, total_len, total_len,
            dtype=torch.float16,
            device=device
        )
        attention_mask.masked_fill_(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        return attention_mask
    
    def build_position_ids(self, prefix_len, device='cuda'):
        """
        Build position IDs.
        
        Prefix: sequential (0, 1, 2, ...)
        Tree: prefix_len + depth for each node
        """
        # Prefix positions
        prefix_pos = torch.arange(prefix_len, device=device)
        
        # Tree positions (based on depth)
        tree_pos = torch.tensor(
            [prefix_len + d for d in self.node_depths],
            dtype=torch.long,
            device=device
        )
        
        return torch.cat([prefix_pos, tree_pos])


# Create tree structure
tree = TreeStructure(width=3, depth=4)  # 121 nodes
# Or: tree = TreeStructure(width=4, depth=3)  # 85 nodes

### **Step 3: Single-Pass Tree Generation**

class EAGLETreeGenerator:
    """
    Generate entire tree in ONE forward pass using EAGLE-3 single layer.
    """
    
    def __init__(self, eagle_components, tree_structure):
        """
        Initialize with EAGLE components and tree structure.
        """
        self.eagle_layer = eagle_components['eagle_layer']
        self.embed_tokens = eagle_components['embed_tokens']
        self.lm_head = eagle_components['lm_head']
        self.fc = eagle_components['fc']
        
        self.tree = tree_structure
        
        self.hidden_size = eagle_components['config'].hidden_size
        self.num_heads = self.eagle_layer.self_attn.num_heads
        self.head_dim = self.hidden_size // self.num_heads
    
    def generate_tree_single_pass(
        self,
        fused_features,  # From target model: [batch, prefix_len, hidden]
        input_ids,       # Input tokens: [batch, prefix_len]
    ):
        """
        Generate entire tree in ONE forward pass!
        
        Args:
            fused_features: Multi-level features from target model
            input_ids: Input token IDs
        
        Returns:
            tree_tokens: [batch, num_nodes] - all draft tokens!
        """
        
        batch_size, prefix_len, hidden_dim = fused_features.shape
        num_nodes = self.tree.num_nodes
        device = fused_features.device
        
        # Step 1: Project fused features (EAGLE-3 feature fusion)
        projected_features = self.fc(fused_features)  # [batch, prefix_len, hidden]
        
        # Step 2: Get embeddings for last token
        last_token_embed = self.embed_tokens(input_ids[:, -1:])  # [batch, 1, hidden]
        
        # Step 3: Combine features and embeddings
        prefix_input = projected_features + last_token_embed  # [batch, prefix_len, hidden]
        
        # Step 4: Initialize tree node embeddings (will be computed)
        tree_input = torch.zeros(
            batch_size, num_nodes, hidden_dim,
            dtype=fused_features.dtype,
            device=device
        )
        
        # Step 5: Concatenate prefix and tree
        full_input = torch.cat([prefix_input, tree_input], dim=1)  # [batch, total_len, hidden]
        
        # Step 6: Build attention mask and position IDs
        attention_mask = self.tree.build_attention_mask(prefix_len, device)
        position_ids = self.tree.build_position_ids(prefix_len, device)
        
        # Step 7: ONE FORWARD PASS through single EAGLE layer!
        hidden_states = self._single_layer_forward(
            full_input,
            attention_mask,
            position_ids
        )
        
        # Step 8: Extract tree node hidden states
        tree_hidden = hidden_states[:, prefix_len:, :]  # [batch, num_nodes, hidden]
        
        # Step 9: Compute logits for all tree nodes
        tree_logits = self.lm_head(tree_hidden)  # [batch, num_nodes, vocab]
        
        # Step 10: Sample tokens
        tree_tokens = tree_logits.argmax(dim=-1)  # [batch, num_nodes]
        
        return tree_tokens, tree_logits
    
    def _single_layer_forward(
        self,
        hidden_states,   # [batch, total_len, hidden]
        attention_mask,  # [1, 1, total_len, total_len]
        position_ids,    # [total_len]
    ):
        """
        Forward through the SINGLE EAGLE layer with tree attention.
        """
        
        # Residual connection
        residual = hidden_states
        
        # Input layer norm
        if hasattr(self.eagle_layer, 'input_layernorm'):
            hidden_states = self.eagle_layer.input_layernorm(hidden_states)
        
        # Self-attention with tree mask
        attn_output = self._tree_attention(
            hidden_states,
            attention_mask,
            position_ids
        )
        
        # Residual
        hidden_states = residual + attn_output
        
        # Post-attention norm and MLP
        residual = hidden_states
        if hasattr(self.eagle_layer, 'post_attention_layernorm'):
            hidden_states = self.eagle_layer.post_attention_layernorm(hidden_states)
        
        hidden_states = self.eagle_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def _tree_attention(
        self,
        hidden_states,   # [batch, total_len, hidden]
        attention_mask,  # [1, 1, total_len, total_len]
        position_ids,    # [total_len]
    ):
        """
        Custom attention with tree masking.
        
        This is the KEY - it allows tree-structured attention!
        """
        
        bsz, seq_len, _ = hidden_states.shape
        attn = self.eagle_layer.self_attn
        
        # Q, K, V projections
        query = attn.q_proj(hidden_states)
        key = attn.k_proj(hidden_states)
        value = attn.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if available
        if hasattr(attn, 'rotary_emb'):
            cos, sin = attn.rotary_emb(value, seq_len=seq_len)
            query, key = self._apply_rotary_pos_emb(query, key, cos, sin, position_ids)
        
        # Compute attention scores
        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Apply tree attention mask (THE KEY PART!)
        attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        
        # Apply to values
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, seq_len, self.hidden_size)
        attn_output = attn.o_proj(attn_output)
        
        return attn_output
    
    def _apply_rotary_pos_emb(self, q, k, cos, sin, position_ids):
        """Apply rotary position embeddings"""
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
        
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        """Rotate half the hidden dims"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


# Create generator
generator = EAGLETreeGenerator(eagle_components, tree)


### **Step 4: Usage Example**

# Example: Generate tree tokens in ONE pass

# Assume you have:
# - target_model: Your LLaMA 3.1 8B model
# - input_ids: [1, 20] input tokens
# - fused_features: [1, 20, 4096] multi-level features from target

# Simulate fused features (in practice, extract from target model)
batch_size = 1
prefix_len = 20
hidden_dim = 4096

fused_features = torch.randn(
    batch_size, prefix_len, hidden_dim,
    dtype=torch.float16,
    device='cuda'
)

input_ids = torch.randint(
    0, 32000, (batch_size, prefix_len),
    device='cuda'
)

# Generate entire tree in ONE forward pass!
tree_tokens, tree_logits = generator.generate_tree_single_pass(
    fused_features=fused_features,
    input_ids=input_ids
)

print(f"Generated {tree_tokens.shape[1]} draft tokens in ONE pass!")
print(f"Tree tokens shape: {tree_tokens.shape}")  # [1, 121]
print(f"Tree logits shape: {tree_logits.shape}")  # [1, 121, 32000]
