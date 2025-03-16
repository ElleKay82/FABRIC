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
    use_nn_feature = attr.ib(default=False)
    cold_start = attr.ib(default=True)
    max_items = attr.ib(default=3)
    drop_path_rate = attr.ib(default=0.1)  # Add drop_path_rate parameter
    margin = attr.ib(default=0.5)
    temperature = attr.ib(default=0.1)
    l2_scale = attr.ib(default=0.01)
    l1_scale = attr.ib(default=0.001)

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

class EnhancedUShapeBlock(nn.Module):
    def __init__(self, dim, user_dim, num_heads, dropout=0.3, drop_path=0.1):
        super().__init__()
        self.attention = AdaptiveUserAwareAttention(dim, num_heads, user_dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Enhanced FFN with gating mechanism
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.LayerNorm(dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        
        # Feature gate for adaptive feature fusion
        self.feature_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, user_emb, skip=None):
        # Pre-norm for attention
        norm_x = self.norm1(x)
        
        # Apply adaptive attention
        attn_out = self.attention(norm_x, user_emb)
        
        # Residual connection with drop path
        x = x + self.drop_path(attn_out)
        
        # Pre-norm for FFN
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        
        # Adaptive feature fusion with skip connection
        if skip is not None:
            # Calculate feature importance gates
            combined_features = torch.cat([ffn_out, skip], dim=-1)
            gates = self.feature_gate(combined_features)
            
            # Apply gated skip connection
            ffn_out = ffn_out * gates + skip * (1 - gates)
        
        # Final residual connection with drop path
        x = x + self.drop_path(ffn_out)
        
        return x

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



class FashionRecommenderAdapter(nn.Module):
    def __init__(self, param: FashionRecommenderParam):
        super().__init__()
        self.param = param
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if param.use_visual:
            self.visual_feature = nn.Identity() if param.use_nn_feature else None

        self.in_features = 512
        # Using the enhanced model instead of the original
        self.model = EnhancedFashionRecommenderUShape(
            in_features=self.in_features,
            user_emb_dim=param.embd_dim,
            num_users=param.num_users,
            num_proto=param.num_proto,
            num_heads=param.num_heads,
            num_transformer_layers=param.num_transformer_layers,
            dropout=param.dropout,
            max_items=param.max_items,
            drop_path_rate=param.drop_path_rate
        ).to(self.device)

        # Initialize memory manager
        self.memory_manager = MemoryManager(
            num_users=param.num_users,
            embd_dim=param.embd_dim,
            num_proto=param.num_proto,
            dropout=param.dropout,
            device=self.device
        )

    def train_batch(self, data: torch.Tensor, uidx: torch.Tensor, cate: torch.Tensor):
        uidx = uidx.long().to(self.device)
        data = data.to(self.device)
        cate = cate.to(self.device)

        # Preprocess data
        batch, pairs, num, *shape = data.shape
        data = data.view(-1, *shape)
        feat = data[..., -self.in_features:] if self.visual_feature is None else self.visual_feature(data)
        feat = feat.view(batch, pairs, num, -1)

        pos_feat = feat[:, 0]
        neg_feat = feat[:, 1]

        pos_mask = (cate[:, 0] != -1).float().unsqueeze(-1).to(self.device)
        neg_mask = (cate[:, 1] != -1).float().unsqueeze(-1).to(self.device)

        # Compute scores
        pos_scores = self.model(pos_feat, uidx, pos_mask)
        neg_scores = self.model(neg_feat, uidx, neg_mask)

        diff = pos_scores - neg_scores

        # Calculate ranking loss
        rank_loss = F.soft_margin_loss(diff, torch.ones_like(diff), reduction="none")

        # Get user embeddings for regularization
        user_emb = self.memory_manager.user_embeddings[uidx]
        
        # Calculate L2 regularization
        l2reg = (torch.norm(user_emb.view(batch, -1), p=2, dim=1) + 
                 torch.norm(pos_feat.view(batch, -1), p=2, dim=1) + 
                 torch.norm(neg_feat.view(batch, -1), p=2, dim=1))
        l2reg = l2reg.mean()
        
        # Calculate L1 regularization
        l1reg = (torch.norm(user_emb.view(batch, -1), p=1, dim=1) + 
                 torch.norm(pos_feat.view(batch, -1), p=1, dim=1) + 
                 torch.norm(neg_feat.view(batch, -1), p=1, dim=1))
        l1reg = l1reg.mean()

        # Update embeddings and prototypes
        success_mask = (diff > 0).float().to(self.device)
        self.memory_manager.update_interaction(uidx, success_mask)
        self.memory_manager.update_prototypes(uidx, user_emb, success_mask)

        return {
            'rank_loss': rank_loss,
            'l1reg': l1reg,
            'l2reg': l2reg
        }, {'accuracy': torch.gt(diff, 0.0)}

    @torch.no_grad()
    def test_batch(self, data: torch.Tensor, uidx: torch.Tensor, cate: torch.Tensor):
        batch, num, *shape = data.shape
        data = data.view(-1, *shape)
        feat = data[..., -self.in_features:] if self.visual_feature is None else self.visual_feature(data)
        feat = feat.view(batch, num, -1)
        mask = (cate != -1).float().unsqueeze(-1)
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