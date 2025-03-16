# models.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import attr
from torchutils.param import Param

@attr.s
class FashionRecommenderParam(Param):
    name = attr.ib(default="FashionRecommender")
    num_users = attr.ib(default=32)
    backbone = attr.ib(default="resnet34")
    embd_dim = attr.ib(default=128)
    num_transformer_layers = attr.ib(default=2)
    num_heads = attr.ib(default=4)
    num_proto = attr.ib(default=16)
    dropout = attr.ib(default=0.3)
    num_points = attr.ib(default=0)
    num_seeds = attr.ib(default=1)
    loss_weight = attr.ib(factory=dict)
    use_visual = attr.ib(default=True)
    use_semantic = attr.ib(default=False)
    use_nn_feature = attr.ib(default=True)
    max_items = attr.ib(default=3)
    drop_path_rate = attr.ib(default=0.1)
    margin = attr.ib(default=0.5)
    temperature = attr.ib(default=0.1)
    l2_scale = attr.ib(default=0.01)
    l1_scale = attr.ib(default=0.001)
    # New parameters
    proto_reg_weight = attr.ib(default=0.1)
    momentum = attr.ib(default=0.9)
    curriculum_epochs = attr.ib(default=50)

class UserEmbedding(nn.Module):
    def __init__(self, num_users, dim):
        super().__init__()
        self.num_users = num_users
        self.encoder = nn.Embedding(num_users, dim)
        nn.init.normal_(self.encoder.weight, std=0.01)

    def forward(self, x):
        return self.encoder(x)

class AdaptiveUserAwareAttention(nn.Module):
    def __init__(self, dim, num_heads, user_dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scaling = self.head_dim ** -0.5

        # Query, Key, Value projections for user context
        self.user_q = nn.Linear(user_dim, dim)
        self.user_k = nn.Linear(user_dim, dim)
        self.user_v = nn.Linear(user_dim, dim)
        
        # Query, Key, Value projections for items
        self.item_q = nn.Linear(dim, dim)
        self.item_k = nn.Linear(dim, dim)
        self.item_v = nn.Linear(dim, dim)
        
        # Output projections
        self.out_proj = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        
        # Adaptive attention components
        self.attention_gate = nn.Sequential(
            nn.Linear(dim + user_dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, num_heads),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def generate_attention_mask(self, q_len, k_len, batch_size, attention_weights):
        # Create relative position encoding
        position_ids = torch.arange(k_len, dtype=torch.float, device=attention_weights.device)
        position_ids = position_ids.unsqueeze(0) - position_ids.unsqueeze(1)
        position_ids = position_ids.unsqueeze(0).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, self.num_heads, -1, -1)
        
        # Apply adaptive attention weights
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)
        position_bias = attention_weights * torch.sign(position_ids) * torch.log1p(torch.abs(position_ids))
        
        return position_bias[:, :, :q_len, :k_len]

    def forward(self, x, user_emb, mask=None):
        batch_size = x.size(0)
        
        # Project user embedding
        user_q = self.user_q(user_emb)
        user_k = self.user_k(user_emb)
        user_v = self.user_v(user_emb)
        
        # Project item features
        item_q = self.item_q(x)
        item_k = self.item_k(x)
        item_v = self.item_v(x)
        
        # Calculate adaptive attention weights
        combined_features = torch.cat([x.mean(1), user_emb], dim=-1)
        attention_weights = self.attention_gate(combined_features)
        
        # Reshape for multi-head attention
        def reshape_for_attention(x):
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Process queries, keys, and values
        q = reshape_for_attention(item_q)
        k = reshape_for_attention(item_k)
        v = reshape_for_attention(item_v)
        
        user_q = reshape_for_attention(user_q.unsqueeze(1).expand(-1, x.size(1), -1))
        user_k = reshape_for_attention(user_k.unsqueeze(1).expand(-1, x.size(1), -1))
        user_v = reshape_for_attention(user_v.unsqueeze(1).expand(-1, x.size(1), -1))
        
        # Cross attention between items and user context
        item_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        user_scores = torch.matmul(user_q, user_k.transpose(-2, -1)) * self.scaling
        
        # Generate and apply attention mask
        attention_mask = self.generate_attention_mask(
            q.size(-2), k.size(-2), batch_size, attention_weights
        )
        
        item_scores = item_scores + attention_mask
        user_scores = user_scores + attention_mask
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            item_scores = item_scores.masked_fill(~mask, -1e9)
            user_scores = user_scores.masked_fill(~mask, -1e9)
        
        # Apply softmax and dropout
        item_attn = torch.softmax(item_scores, dim=-1)
        user_attn = torch.softmax(user_scores, dim=-1)
        
        item_attn = self.dropout(item_attn)
        user_attn = self.dropout(user_attn)
        
        # Compute attention outputs
        item_out = torch.matmul(item_attn, v)
        user_out = torch.matmul(user_attn, user_v)
        
        # Combine and reshape outputs
        item_out = item_out.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        user_out = user_out.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        
        # Concatenate and project
        combined_out = torch.cat([item_out, user_out], dim=-1)
        out = self.out_proj(combined_out)
        
        return out

class AdaptiveResidualGate(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.dim = dim
        self.reduction = reduction
        
        # Pooling layers for different dimensions
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature reduction network
        reduced_dim = max(dim // reduction, 32)  # Ensure minimum size
        self.fc = nn.Sequential(
            nn.Linear(dim * 2, reduced_dim),  # Combine both inputs
            nn.LayerNorm(reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, dim),  # Output separate gates for each dimension
            nn.Sigmoid()
        )
        
    def forward(self, x, residual):
        # Get input shapes
        b, n, c = x.shape  # [batch, seq_len, channels]
        
        # Pool across sequence dimension
        x_pool = self.avg_pool(x.transpose(1, 2)).squeeze(-1)  # [b, c]
        r_pool = self.avg_pool(residual.transpose(1, 2)).squeeze(-1)  # [b, c]
        
        # Concatenate pooled features
        combined = torch.cat([x_pool, r_pool], dim=1)  # [b, 2*c]
        
        # Generate gates
        gates = self.fc(combined)  # [b, c]
        
        # Expand gates to match input dimension
        gates = gates.unsqueeze(1).expand(-1, n, -1)  # [b, n, c]
        
        return gates

class FocalResidualAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        self.scale = self.head_dim ** -0.5
        
        # Single projection for Q, K, V
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Focal attention parameters
        self.focal_alpha = nn.Parameter(torch.ones(1))
        self.focal_gamma = nn.Parameter(torch.ones(1) * 2.0)
        
    def forward(self, x, mask=None):
        B, N, C = x.shape  # batch, sequence length, channels
        
        # Project and reshape qkv
        qkv = self.qkv(x)  # [B, N, 3*C]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)  # [B, N, 3, H, D]
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, N, D]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        
        # Apply mask if provided
        if mask is not None:
            # Convert mask to boolean if needed
            if mask.dtype != torch.bool:
                mask = mask > 0.5
            
            # Reshape mask for attention
            mask = mask.view(B, 1, 1, N)  # [B, 1, 1, N]
            mask = mask.expand(-1, self.num_heads, N, -1)  # [B, H, N, N]
            attn = attn.masked_fill(~mask, float('-inf'))
        
        # Apply focal attention
        attn = F.softmax(attn, dim=-1)  # [B, H, N, N]
        attn = torch.pow(attn, self.focal_gamma) * self.focal_alpha
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Apply attention to values
        out = (attn @ v)  # [B, H, N, D]
        out = out.transpose(1, 2)  # [B, N, H, D]
        out = out.reshape(B, N, C)  # [B, N, C]
        
        # Final projection
        out = self.proj(out)
        return out

class ImbalanceAwareResidualBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, drop_path=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        # Attention
        self.attn = FocalResidualAttention(dim, num_heads)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * mlp_ratio, dim)
        )
        
        # Residual gates with explicit dimensions
        self.res_gate1 = AdaptiveResidualGate(dim)
        self.res_gate2 = AdaptiveResidualGate(dim)
        
        # Cross-gating with matched dimensions
        self.cross_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1)
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x, residual=None, mask=None):
        # Input shape verification
        b, n, c = x.shape
        assert c == self.dim, f"Input dimension {c} doesn't match block dimension {self.dim}"
        
        # Store input for residual
        if residual is None:
            residual = x
        # Skip connection and residual gates
        norm_x = self.norm1(x)
        attn_out = self.attn(norm_x, mask)
        
        # Residual gate application
        res_weight1 = self.res_gate1(attn_out, residual)
        x = x + self.drop_path(attn_out * res_weight1)

        # MLP and another residual gate
        norm_x = self.norm2(x)
        mlp_out = self.mlp(norm_x)
        
        res_weight2 = self.res_gate2(mlp_out, residual)
        x = x + self.drop_path(mlp_out * res_weight2)
        
        # Cross-gating for final residual
        if residual is not None:
            combined = torch.cat([self.norm3(x), self.norm3(residual)], dim=-1)
            cross_weights = self.cross_gate(combined)  # [b, n, 2]
            x = x * cross_weights[..., 0:1] + residual * cross_weights[..., 1:2]
        
        return x


    # Add shape verification helper
    def verify_tensor_shape(tensor, expected_shape, name="tensor"):
        """Helper function to verify tensor shapes during forward pass"""
        if tensor.shape != expected_shape:
            raise ValueError(f"Expected {name} shape {expected_shape}, got {tensor.shape}")
            
        return tensor

class EnhancedUShapeEncoder(nn.Module):
    def __init__(self, dim, depth=4, num_heads=8, mlp_ratio=4, drop_path=0.1):
        super().__init__()
        self.dim = dim
        self.depth = depth
        
        # Create encoder blocks
        self.encoders = nn.ModuleList([
            ImbalanceAwareResidualBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path * (i / depth)
            ) for i in range(depth)
        ])
        
        # Create decoder blocks
        self.decoders = nn.ModuleList([
            ImbalanceAwareResidualBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path * (i / depth)
            ) for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, mask=None):
        # Input shape verification
        b, n, c = x.shape
        assert c == self.dim, f"Input dimension {c} doesn't match encoder dimension {self.dim}"
        
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Encoder path
        for encoder in self.encoders:
            x = encoder(x, mask=mask)
            encoder_outputs.append(x)
            
        # Decoder path with skip connections
        for i, decoder in enumerate(self.decoders):
            skip_connection = encoder_outputs[-(i + 1)]
            x = decoder(x, residual=skip_connection, mask=mask)
            
        return self.norm(x)

    def create_attention_mask(mask, num_heads, device):
        """Helper function to create properly formatted attention masks"""
        if mask is None:
            return None
            
        # Ensure mask is on correct device
        mask = mask.to(device)
        
        # Convert to boolean if not already
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        
        # Add head dimension if needed
        if mask.dim() == 2:
            mask = mask.unsqueeze(1)  # [B, 1, N]
        
        # Expand for num_heads if needed
        if mask.dim() == 3:
            mask = mask.unsqueeze(1).expand(-1, num_heads, -1, -1)  # [B, num_heads, N, N]
            
        return mask

# Utility DropPath module (unchanged from original code)
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class DynamicTemperatureScale(nn.Module):
    def __init__(self, init_temp=1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))
        self.temp_scale = nn.Sequential(
            nn.Linear(2, 8),
            nn.LayerNorm(8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Softplus()  # Ensures positive temperature
        )
    
    def forward(self, logits, pos_ratio=None, score_variance=None):
        if pos_ratio is not None and score_variance is not None:
            # Create statistics tensor (make sure it's the right shape)
            stats = torch.tensor([pos_ratio, score_variance], device=logits.device).float()
            
            # Get dynamic temperature
            dynamic_temp = self.temp_scale(stats)
            
            # Combine with learned base temperature
            temp = torch.exp(self.log_temp) * dynamic_temp
        else:
            temp = torch.exp(self.log_temp)
        
        # Make sure temp has the right shape for broadcasting
        temp_value = temp.clamp(min=0.1, max=10.0).item()
            
        # Apply temperature scaling
        return logits / temp_value

class EnhancedFashionRecommenderUShape(nn.Module):
    def __init__(self, in_features, user_emb_dim, num_users, num_proto, num_heads, num_transformer_layers, dropout, max_items, drop_path_rate=0.1):
        super().__init__()
        self.embd_dim = user_emb_dim
        self.max_items = max_items
        self.user_embedding = UserEmbedding(num_users, user_emb_dim)
        
        # Enhanced input embedding with normalization
        self.embed = nn.Sequential(
            nn.Linear(in_features, user_emb_dim * 2),
            nn.LayerNorm(user_emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(user_emb_dim * 2, user_emb_dim),
            nn.LayerNorm(user_emb_dim)
        )
        
        # Linear decay for drop path rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_transformer_layers)]
        
        # Enhanced transformer blocks with cross-attention
        self.encoder = nn.ModuleList([
            EnhancedUShapeBlock(user_emb_dim, user_emb_dim, num_heads, dropout, drop_path=dpr[i])
            for i in range(num_transformer_layers)
        ])
        
        # Enhanced output transformation with gating
        self.output_transform = nn.Sequential(
            nn.Linear(user_emb_dim * 2, user_emb_dim),
            nn.LayerNorm(user_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(user_emb_dim, user_emb_dim // 2),
            nn.LayerNorm(user_emb_dim // 2),
            nn.ReLU(),
            nn.Linear(user_emb_dim // 2, 1)
        )
        
        # Global context aggregation
        self.global_context = nn.Parameter(torch.zeros(1, 1, user_emb_dim))
        nn.init.normal_(self.global_context, std=0.02)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, user_idx, mask=None):
        batch_size = x.size(0)
        
        # Get user embeddings
        user_emb = self.user_embedding(user_idx)
        
        # Embed input features
        h = self.embed(x)
        
        # Add global context
        global_context = self.global_context.expand(batch_size, -1, -1)
        h = torch.cat([global_context, h], dim=1)
        
        if mask is not None:
            # Extend mask for global context token
            context_mask = torch.ones(batch_size, 1, 1, device=mask.device)
            mask = torch.cat([context_mask, mask], dim=1)
        
        # Enhanced skip connections with previous layer inputs
        prev_skip = None
        for encoder in self.encoder:
            current_input = h
            h = encoder(h, user_emb, skip=prev_skip)
            prev_skip = current_input
        
        # Global pooling with mask handling
        if mask is not None:
            mask = mask.squeeze(-1)
            h = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
        else:
            h = h.mean(1)
        
        # Combine with user embedding and get final scores
        h = torch.cat([h, user_emb], dim=1)
        scores = self.output_transform(h).squeeze(-1)
        
        return scores

class MemoryManager:
    def __init__(self, num_users, embd_dim, num_proto, dropout=0.1, device='cpu'):
        self.num_users = num_users
        self.embd_dim = embd_dim
        self.num_proto = num_proto
        self.dropout = dropout
        self.device = device

        # Buffers for user embeddings, prototypes, and metrics
        self.user_embeddings = nn.Parameter(torch.zeros(num_users, embd_dim, device=device))
        self.user_prototypes = nn.Parameter(torch.zeros(num_users, num_proto // 2, embd_dim, device=device))
        self.shared_prototypes = nn.Parameter(torch.zeros(1, num_proto // 2, embd_dim, device=device))

        self.interaction_count = torch.zeros(num_users, device=device)
        self.success_rate = torch.zeros(num_users, device=device)
        self.feedback_history = torch.zeros(num_users, 100, device=device)  # Last 100 interactions

        # Initialization
        nn.init.normal_(self.user_embeddings, std=0.01)
        nn.init.normal_(self.user_prototypes, std=1.0 / math.sqrt(embd_dim))
        nn.init.normal_(self.shared_prototypes, std=1.0 / math.sqrt(embd_dim))

    def update_interaction(self, user_idx, success_mask):
        """Update interaction counts and success rates."""
        with torch.no_grad():
            user_idx = user_idx.to(self.device)  # Ensure user_idx is on the correct device
            success_mask = success_mask.to(self.device)

            # Update interaction counts
            self.interaction_count[user_idx] += 1

            # Update success rates and feedback history
            self.success_rate[user_idx] = (
                self.success_rate[user_idx] * self.dropout +
                success_mask * (1 - self.dropout)
            )
            self.feedback_history[user_idx] = torch.roll(self.feedback_history[user_idx], -1, dims=-1)
            self.feedback_history[user_idx, -1] = success_mask

    def update_prototypes(self, user_idx, user_emb, success_mask):
        """Update prototypes based on user embeddings and success."""
        with torch.no_grad():
            # Ensure tensors are on the correct device
            user_idx = user_idx.to(self.device)
            user_emb = user_emb.to(self.device)
            success_mask = success_mask.to(self.device)

            # Get user-specific prototypes
            user_protos = self.user_prototypes[user_idx]  # Shape: (batch_size, num_user_prototypes, embd_dim)

            # Expand shared prototypes to match batch size
            shared_protos = self.shared_prototypes.expand(user_emb.size(0), -1, -1)  # Shape: (batch_size, num_shared_prototypes, embd_dim)

            # Concatenate user-specific and shared prototypes
            all_protos = torch.cat([user_protos, shared_protos], dim=1)  # Shape: (batch_size, num_user_prototypes + num_shared_prototypes, embd_dim)

            # Compute prototype attention
            attn = torch.matmul(user_emb.unsqueeze(1), all_protos.transpose(1, 2))  # Shape: (batch_size, 1, num_user_prototypes + num_shared_prototypes)
            attn = F.softmax(attn / 0.1, dim=-1)  # Temperature = 0.1

            # Update user-specific prototypes
            proto_update = torch.matmul(attn.transpose(1, 2), user_emb.unsqueeze(1))  # Shape: (batch_size, num_user_prototypes + num_shared_prototypes, embd_dim)
            proto_update = proto_update[:, :user_protos.size(1), :]  # Keep only user-specific prototypes

            self.user_prototypes[user_idx] = (
                user_protos * self.dropout +
                proto_update * (1 - self.dropout)
            )

            # Update shared prototypes
            shared_attn = torch.matmul(user_emb.unsqueeze(1), self.shared_prototypes.transpose(1, 2))  # Shape: (batch_size, 1, num_shared_prototypes)
            shared_attn = F.softmax(shared_attn / 0.1, dim=-1)

            shared_update = torch.matmul(shared_attn.transpose(1, 2), user_emb.unsqueeze(1)).mean(0, keepdim=True)  # Shape: (1, num_shared_prototypes, embd_dim)

            self.shared_prototypes = (
                self.shared_prototypes * self.dropout +
                shared_update * (1 - self.dropout)
            )


    def get_cold_user_embedding(self, user_idx, group_embeddings, mean_embedding):
        """Generate embeddings for cold users based on group and shared prototypes."""
        with torch.no_grad():
            user_idx = user_idx.to(self.device)
            group_embeddings = group_embeddings.to(self.device)
            mean_embedding = mean_embedding.to(self.device)

            cold_mask = self.interaction_count[user_idx] < 5
            if cold_mask.any():
                group_ids = group_embeddings[user_idx]
                group_emb = group_embeddings[group_ids]
                shared_emb = self.shared_prototypes.expand(group_emb.size(0), -1, -1).mean(dim=1)

                # Combine embeddings
                cold_user_emb = 0.6 * group_emb + 0.4 * shared_emb
                self.user_embeddings[user_idx] = cold_user_emb

        return self.user_embeddings[user_idx]

class EnhancedMemoryManager(nn.Module):
    def __init__(self, num_users, embd_dim, num_proto, momentum=0.9, device='cuda'):
        super().__init__()
        self.momentum = momentum
        self.num_users = num_users
        self.embd_dim = embd_dim
        self.num_proto = num_proto
        self.device = device
        
        # Initialize prototype memories
        self.register_buffer('user_prototypes', 
            torch.randn(num_users, num_proto, embd_dim, device=device))
        self.register_buffer('shared_prototypes', 
            torch.randn(1, num_proto, embd_dim, device=device))
        self.register_buffer('interaction_count', 
            torch.zeros(num_users, device=device))
        
        nn.init.kaiming_uniform_(self.user_prototypes, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.shared_prototypes, a=math.sqrt(5))

    def ensure_long_index(self, idx):
        """Ensure index tensor is long type"""
        if not isinstance(idx, torch.Tensor):
            idx = torch.tensor(idx, device=self.device)
        return idx.long().to(self.device)

    def get_prototypes(self, user_idx):
        """Get combined prototypes for users based on curriculum learning"""
        user_idx = self.ensure_long_index(user_idx)
        
        # Calculate curriculum weights
        interaction_counts = self.interaction_count[user_idx]
        curriculum_weights = torch.sigmoid((interaction_counts/100) - 3)
        curriculum_weights = curriculum_weights.view(-1, 1, 1)
        
        # Get prototypes
        user_protos = self.user_prototypes[user_idx]
        shared_protos = self.shared_prototypes.expand(len(user_idx), -1, -1)
        
        return curriculum_weights * user_protos + (1 - curriculum_weights) * shared_protos

    def update_prototypes(self, user_idx, features, success_mask):
        """Momentum-based prototype update with curriculum learning with robust handling of different input shapes"""
        user_idx = self.ensure_long_index(user_idx)
        features = features.to(self.device)
        success_mask = success_mask.to(self.device)
        
        # Get current prototypes for these users
        current_protos = self.user_prototypes[user_idx]
        
        # Calculate adaptive momentum
        curr_momentum = torch.clamp(
            self.momentum + self.interaction_count[user_idx]/1000, 
            min=self.momentum, 
            max=0.99
        )
        curr_momentum = curr_momentum.view(-1, 1, 1)
        
        # Make sure dimensions match
        batch_size = features.size(0)
        
        # Process features and mask based on dimensionality
        if features.dim() == 3:  # [batch, seq_len, embd_dim]
            # Adjust success_mask dimensions if needed
            if success_mask.dim() == 3 and success_mask.size(2) == 1:  # [batch, seq_len, 1]
                pass  # Already in the right format
            elif success_mask.dim() == 2:  # [batch, seq_len]
                success_mask = success_mask.unsqueeze(-1)  # [batch, seq_len, 1]
            elif success_mask.dim() == 1:  # [batch]
                success_mask = success_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1]
                success_mask = success_mask.expand(-1, features.size(1), 1)  # [batch, seq_len, 1]
            
            # Mask and average features
            masked_features = features * success_mask
            sum_mask = success_mask.sum(dim=1).clamp(min=1e-6)  # Avoid division by zero
            feature_updates = (masked_features.sum(1) / sum_mask)  # Weighted average
        else:  # [batch, embd_dim]
            if success_mask.dim() == 3:  # [batch, seq_len, 1]
                # Average the mask across seq_len dimension
                success_mask = success_mask.mean(dim=1)  # [batch, 1]
            elif success_mask.dim() == 2 and success_mask.size(1) > 1:  # [batch, seq_len]
                success_mask = success_mask.mean(dim=1, keepdim=True)  # [batch, 1]
            elif success_mask.dim() == 1:  # [batch]
                success_mask = success_mask.unsqueeze(1)  # [batch, 1]
            
            # Mask features
            feature_updates = features * success_mask
        
        # Normalize and expand feature updates
        feature_updates = F.normalize(feature_updates, dim=-1)
        feature_updates = feature_updates.unsqueeze(1)
        feature_updates = feature_updates.expand(-1, self.num_proto, -1)
        
        # Update user prototypes
        self.user_prototypes[user_idx] = (
            curr_momentum * current_protos +
            (1 - curr_momentum) * feature_updates
        )
        
        # Update shared prototypes
        shared_updates = F.normalize(feature_updates.mean(0), dim=-1)
        self.shared_prototypes = (
            self.momentum * self.shared_prototypes +
            (1 - self.momentum) * shared_updates.unsqueeze(0)
        )
        
        # Update interaction counts
        self.interaction_count[user_idx] += 1

    def proto_diversity_reg(self, user_idx):
        """Log-determinant based diversity regularization"""
        user_idx = self.ensure_long_index(user_idx)
        user_protos = F.normalize(self.user_prototypes[user_idx], dim=-1)
        cov = torch.bmm(user_protos, user_protos.transpose(1, 2))
        eye = torch.eye(self.num_proto, device=self.device).unsqueeze(0)
        eye = eye.expand(user_protos.size(0), -1, -1)
        return torch.logdet(cov + 0.1 * eye).mean()


        
class EnhancedSetTransformer(nn.Module):
    """Enhanced Set Transformer with Position-aware Attention"""
    def __init__(self, in_dim, out_dim, num_heads, num_layers):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim)
        )
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=out_dim,
                nhead=num_heads,
                dim_feedforward=out_dim * 4,
                dropout=0.1,
                activation='gelu'
            ) for _ in range(num_layers)
        ])
        
        # Position-aware encoding
        self.position_embed = nn.Parameter(torch.randn(1, 1024, out_dim))
        
    def forward(self, x, mask=None):
        B, N, D = x.shape
        
        # First embed the input to the correct dimension
        x = self.embed(x)
        
        # Add learned positional embeddings
        positions = self.position_embed[:, :N]
        x = x + positions.expand(B, -1, -1)
        
        # Process mask for transformer blocks
        if mask is not None:
            # Ensure mask is float tensor
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.float32, device=x.device)
            elif mask.dtype != torch.float32:
                mask = mask.float()
            
            # Convert to boolean mask for transformer
            # TransformerEncoder expects True for tokens to mask out
            padding_mask = (mask < 0.5)
            
            # Ensure mask has correct shape [batch_size, sequence_length]
            if padding_mask.dim() == 3 and padding_mask.size(-1) == 1:
                padding_mask = padding_mask.squeeze(-1)
        else:
            padding_mask = None
        
        # Apply transformer blocks with masking if provided
        for block in self.blocks:
            x = block(x.transpose(0, 1), src_key_padding_mask=padding_mask).transpose(0, 1)
            
        return x


    
class HybridScorer(nn.Module):
    """Combines Transformer and Prototype-based Scoring"""
    def __init__(self, dim):
        super().__init__()
        self.transformer_score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )
        
        self.prototype_score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        
        # Learnable weighting parameter
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, h, prototypes):
        trans_score = self.transformer_score(h)
        
        # Compute prototype similarity scores
        h_norm = F.normalize(self.prototype_score(h), dim=-1)
        proto_norm = F.normalize(prototypes, dim=-1)
        proto_sim = torch.bmm(h_norm.unsqueeze(1), proto_norm.transpose(1, 2))
        proto_score = proto_sim.mean(dim=2, keepdim=True)
        
        # Combine scores with learned alpha
        alpha = torch.sigmoid(self.alpha)
        return alpha * trans_score + (1 - alpha) * proto_score
    
class EnhancedFashionRecommender(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.user_embed = UserEmbedding(param.num_users, param.embd_dim).to(self.device)
        
        # Replace feature encoder with new UShape encoder
        self.feature_encoder = EnhancedUShapeEncoder(
            dim=param.embd_dim,
            depth=param.num_transformer_layers,
            num_heads=param.num_heads,
            mlp_ratio=4,
            drop_path=param.drop_path_rate
        ).to(self.device)
        
        self.memory = EnhancedMemoryManager(
            num_users=param.num_users,
            embd_dim=param.embd_dim,
            num_proto=param.num_proto,
            momentum=param.momentum,
            device=self.device
        ).to(self.device)
        
        # Enhanced scorer with imbalance handling
        self.scorer = ImbalanceAwareScorer(param.embd_dim).to(self.device)
        
    def forward(self, x, user_idx, mask=None):
        # Get user embeddings
        user_emb = self.user_embed(user_idx)
        
        # Encode features
        features = self.feature_encoder(x, mask)  # [batch, num, dim]
        
        # Get prototypes from memory
        prototypes = self.memory.get_prototypes(user_idx)  # [batch, num_proto, dim]
        
        # Calculate positive ratio if mask is provided
        pos_ratio = None
        if mask is not None:
            pos_ratio = mask.float().mean().item()
        
        # Average features across sequence dimension
        avg_features = features.mean(dim=1)  # [batch, dim]
        
        # Generate final scores using hybrid scorer
        scores = self.scorer(avg_features, prototypes, pos_ratio)  # [batch]
        return scores

    def verify_scorer_shapes(h, prototypes):
        """Helper function to verify shapes in scorer"""
        print(f"Input feature shape: {h.shape}")
        print(f"Prototypes shape: {prototypes.shape}")
        print(f"Expected weight_net input shape: [batch, 2]")

class ImbalanceAwareScorer(nn.Module):
    """Enhanced scorer with better handling of imbalanced data using dynamic temperature scaling"""
    def __init__(self, dim):
        super().__init__()
        self.transformer_score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )
        
        self.prototype_score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Adaptive weighting mechanism
        self.weight_net = nn.Sequential(
            nn.Linear(2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # Add dynamic temperature scaling
        self.temp_scaler = DynamicTemperatureScale(init_temp=1.0)
        
    def forward(self, h, prototypes, pos_ratio=None):
        # Transform features
        trans_score = self.transformer_score(h)  # [batch, 1]
        h_proto = self.prototype_score(h)  # [batch, dim]
        
        # Compute prototype similarity
        h_norm = F.normalize(h_proto, dim=-1)
        proto_norm = F.normalize(prototypes, dim=-1)
        proto_sim = torch.bmm(h_norm.unsqueeze(1), proto_norm.transpose(1, 2))  # [batch, 1, num_proto]
        
        # Calculate variance of scores for temperature scaling
        if pos_ratio is not None:
            with torch.no_grad():
                # Fix dimension mismatch
                proto_mean = proto_sim.mean(dim=2)  # [batch, 1]
                # Now both trans_score and proto_mean are [batch, 1]
                all_scores = torch.cat([trans_score, proto_mean], dim=1)  # [batch, 2]
                score_variance = all_scores.var().item()
                
            # Apply dynamic temperature scaling
            proto_sim = self.temp_scaler(proto_sim, pos_ratio, score_variance)
        
        # Average prototype similarities
        proto_score = proto_sim.mean(dim=2, keepdim=True)  # [batch, 1, 1]
        proto_score = proto_score.squeeze(1)  # [batch, 1]
        
        # Combine scores directly
        combined = torch.cat([trans_score, proto_score], dim=1)  # [batch, 2]
        
        # Get adaptive weights
        weights = self.weight_net(combined)  # [batch, 2]
        
        # Apply weights and sum
        final_score = trans_score * weights[:, 0:1] + proto_score * weights[:, 1:2]
        
        return final_score.squeeze(-1)  # [batch]
    
class FashionRecommenderAdapter(nn.Module):
    def __init__(self, param: FashionRecommenderParam):
        super().__init__()
        self.param = param
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if param.use_visual:
            self.visual_feature = nn.Identity() if param.use_nn_feature else None
            
        self.in_features = 512
        self.feature_projection = nn.Sequential(
            nn.Linear(self.in_features, param.embd_dim),
            nn.LayerNorm(param.embd_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        ).to(self.device)
        
        self.model = EnhancedFashionRecommender(param).to(self.device)

    def train_batch(self, data: torch.Tensor, uidx: torch.Tensor, cate: torch.Tensor):
        # Move inputs to device
        uidx = uidx.long().to(self.device)
        data = data.to(self.device)
        cate = cate.to(self.device)
        
        # Process input data and get dimensions
        batch, pairs, num, *shape = data.shape
        data = data.view(-1, *shape)
        feat = data[..., -self.in_features:] if self.visual_feature is None else self.visual_feature(data)
        feat = feat.view(batch, pairs, num, -1)
        
        # Project features
        pos_feat = self.feature_projection(feat[:, 0])  # [batch, num, embd_dim]
        neg_feat = self.feature_projection(feat[:, 1])  # [batch, num, embd_dim]
        
        # Create masks
        pos_mask = (cate[:, 0] != -1).float()  # [batch, num]
        neg_mask = (cate[:, 1] != -1).float()  # [batch, num]
        
        # Get positive ratio for dynamic temperature scaling
        pos_ratio = pos_mask.mean().item()
        
        # Get model predictions
        pos_scores = self.model(pos_feat, uidx, pos_mask)  # [batch]
        neg_scores = self.model(neg_feat, uidx, neg_mask)  # [batch]
        
        # Calculate rank loss with improved focal weighting and dynamic temperature
        diff = pos_scores - neg_scores  # [batch]
        
        # CHANGE 2: Improved adaptive temperature scaling
        base_temp = 1.0 / max(pos_ratio, 0.1)
        if batch > 1:
            # Calculate the score gap between positives and negatives
            score_gap = (pos_scores.mean() - neg_scores.mean()).abs().item()
            # Adjust temperature based on score separation
            temp = base_temp * (1.0 - min(0.3, max(0.0, 0.3 - score_gap)))
        else:
            temp = base_temp
        
        # Create adaptive focal weight with enhanced parameters
        with torch.no_grad():
            # Safely calculate variance-based weight adjustment
            if batch > 1:  # Only compute variance if we have at least 2 samples
                scores_combined = torch.stack([pos_scores, neg_scores], dim=1)  # [batch, 2]
                score_variance = scores_combined.var(dim=1).detach()  # [batch]
                var_weight = torch.exp(-score_variance * 0.5).clamp(0.5, 2.0)  # [batch]
            else:
                # For single sample batches, use default weight
                var_weight = torch.ones_like(diff)
            
            # CHANGE 1: Enhanced focal weight with better parameters
            alpha = max(0.25, min(0.75, 1.0 - pos_ratio))  # Adaptive alpha based on positive ratio
            focal_gamma = 3.0  # Increase from 2.0 to focus more on hard examples
            focal_weight = alpha * torch.pow(torch.sigmoid(-diff * temp).detach(), focal_gamma) * var_weight
        
        # Apply weighted loss
        rank_loss = F.soft_margin_loss(diff, torch.ones_like(diff), reduction='none')
        rank_loss = (rank_loss * focal_weight).mean()
        
        # Calculate prototype loss with enhanced adaptive margin
        temp = 1.0 / max(pos_ratio, 0.1)
        
        pos_proto_sim = F.cosine_similarity(
            self.model.memory.user_prototypes[uidx].detach().mean(1),
            pos_feat.mean(1)
        ) / temp
        
        neg_proto_sim = F.cosine_similarity(
            self.model.memory.user_prototypes[uidx].detach().mean(1),
            neg_feat.mean(1)
        ) / temp
        
        # CHANGE 5: Enhanced adaptive margin
        base_margin = self.param.margin
        pos_weight = 1.0 + max(0.0, min(1.0, (0.5 - pos_ratio) * 3.0))
        margin = base_margin * pos_weight
        
        # Additional adjustment based on score distribution
        if batch > 1:
            pos_var = pos_proto_sim.var().item()
            neg_var = neg_proto_sim.var().item()
            
            # If variances are small, increase margin to create more separation
            if pos_var < 0.05 and neg_var < 0.05:
                margin = margin * 1.2
        
        # Calculate prototype loss
        proto_loss = F.margin_ranking_loss(
            pos_proto_sim, neg_proto_sim,
            torch.ones_like(pos_proto_sim),
            margin=margin
        )
        
        # Calculate regularization terms
        div_reg = self.model.memory.proto_diversity_reg(uidx)
        
        # CHANGE 3: Improved regularization mix
        l2reg = (torch.norm(pos_feat.view(batch, -1), p=2, dim=1) + 
                torch.norm(neg_feat.view(batch, -1), p=2, dim=1)).mean()
                
        # Add adaptive L1 regularization
        l1reg = (torch.norm(pos_feat.view(batch, -1), p=1, dim=1) + 
                torch.norm(neg_feat.view(batch, -1), p=1, dim=1)).mean()
        
        # Adjust regularization weights based on positive ratio
        l2_weight = self.param.l2_scale * (0.7 + min(pos_ratio * 1.5, 0.3))
        l1_weight = self.param.l1_scale * (1.5 - min(pos_ratio * 2, 1.0))
        
        # Combine losses with adaptive weights based on positive ratio
        proto_weight = 0.3 * (1.0 + (1.0 - min(pos_ratio * 2, 0.5)))  # Increase weight when fewer positives
        
        total_loss = (
            rank_loss + 
            proto_weight * proto_loss +
            self.param.proto_reg_weight * div_reg +
            l2_weight * l2reg +
            l1_weight * l1reg
        )
        
        # CHANGE 4: Enhanced memory update with better success indication
        with torch.no_grad():
            try:
                # Calculate confidence-based success mask with adaptive thresholding
                confidence = torch.sigmoid(diff)
                
                # Adaptive success threshold based on positive ratio
                success_threshold = max(0.45, min(0.65, 0.5 + (0.5 - pos_ratio)))
                
                # Binary success mask with adaptive threshold
                binary_success = (confidence > success_threshold).float()
                
                # Weighted success mask combining binary and continuous feedback
                success_mask = 0.7 * binary_success + 0.3 * confidence
                
                # Reshape for memory update
                success_mask = success_mask.view(batch, 1, 1)
                
                # Update prototypes with improved memory
                self.model.memory.update_prototypes(uidx, pos_feat.detach(), success_mask)
            except Exception as e:
                print(f"Warning: Memory update failed: {e}")
                # Fall back to simple binary mask update
                binary_mask = (diff > 0).float().view(batch, 1, 1)
                self.model.memory.update_prototypes(uidx, pos_feat.detach(), binary_mask)
        
        return {
            'rank_loss': rank_loss,
            'proto_loss': proto_loss,
            'div_reg': div_reg,
            'l2_reg': l2reg,
            'l1_reg': l1reg,
        }, {'accuracy': torch.gt(diff, 0.0)}
        
    @torch.no_grad()
    def test_batch(self, data: torch.Tensor, uidx: torch.Tensor, cate: torch.Tensor):
        batch, num, *shape = data.shape
        data = data.view(-1, *shape)
        feat = data[..., -self.in_features:] if self.visual_feature is None else self.visual_feature(data)
        feat = feat.view(batch, num, -1)
        
        feat = self.feature_projection(feat)
        mask = (cate != -1).float()  # [batch_size]
        scores = self.model(feat, uidx, mask)
        return scores

    def forward(self, *inputs):
        if self.training:
            return self.train_batch(*inputs)
        return self.test_batch(*inputs)
    
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, mask_a=None, mask_b=None):
        Q, K, V = self.fc_q(Q), self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        if mask_a is not None and mask_b is not None:
            mask_a = mask_a.repeat(self.num_heads, 1, 1)
            mask_b = mask_b.repeat(self.num_heads, 1, 1)
            
            dots = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_split)
            mask = torch.bmm(mask_a, mask_b.transpose(1, 2)) == 1.0
            dots.masked_fill_(~mask, -1e9)
            A = torch.softmax(dots, dim=2)
        else:
            A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_split), dim=2)

        H = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        H = H if getattr(self, "ln0", None) is None else self.ln0(H)
        H = H + F.relu(self.fc_o(H))
        H = H if getattr(self, "ln1", None) is None else self.ln1(H)
        return H

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, mask=None):
        return self.mab(X, X, mask, mask) if mask is not None else self.mab(X, X)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, mask=None):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, None, mask) if mask is not None else self.mab(self.S.repeat(X.size(0), 1, 1), X)

def stack_sab(dim, num_heads, num_sab, num_points=0):
    def _sab():
        return SAB(dim, dim, num_heads, ln=True)
    
    sabs = [_sab() for _ in range(num_sab)]
    return nn.ModuleList(sabs)

class SetTransformer(nn.Module):
    def __init__(self, in_features, embd_dim=128, num_sab=2, num_points=16, num_heads=4, num_seeds=4):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_features, embd_dim),
            nn.LayerNorm(embd_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.encoder = stack_sab(embd_dim, num_heads, num_sab)
        self.pma = PMA(embd_dim, num_heads, num_seeds, ln=True)
        
    def forward(self, x, mask=None):
        h = self.embed(x)
        for encoder in self.encoder:
            h = encoder(h, mask)
        out = self.pma(h, mask)
        return out.squeeze(1) if out.size(1) == 1 else out