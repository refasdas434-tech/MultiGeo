# models/layers/pooling.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class AttentionPooling(nn.Module):

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int = 4,
            dropout_rate: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0

        # 可学习的全局query (类似CLS token的作用)
        self.global_query = nn.Parameter(torch.randn(1, num_heads, self.head_dim))

        # K, V投影
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(
            self,
            node_features: torch.Tensor,
            batch_num_nodes: torch.Tensor
    ) -> torch.Tensor:
        batch_size = len(batch_num_nodes)
        device = node_features.device

        graph_features_list = []
        node_start = 0

        for i in range(batch_size):
            num_nodes = batch_num_nodes[i].item()
            node_end = node_start + num_nodes

            # 当前图的节点特征
            h = node_features[node_start:node_end]  # [num_nodes, hidden_dim]

            # K, V投影
            K = self.k_proj(h).view(num_nodes, self.num_heads, self.head_dim)
            V = self.v_proj(h).view(num_nodes, self.num_heads, self.head_dim)

            # Query: 使用全局可学习query
            Q = self.global_query  # [1, num_heads, head_dim]

            # 注意力分数: [1, num_heads, num_nodes]
            attn_scores = torch.einsum('qhd,nhd->qhn', Q, K) / self.scale

            # Softmax
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # 聚合: [1, num_heads, head_dim]
            context = torch.einsum('qhn,nhd->qhd', attn_weights, V)

            # Reshape: [1, hidden_dim]
            context = context.view(1, self.hidden_dim)

            # 输出投影
            graph_feat = self.out_proj(context)
            graph_feat = self.layer_norm(graph_feat)

            graph_features_list.append(graph_feat)
            node_start = node_end

        # 拼接所有图的特征: [batch_size, hidden_dim]
        graph_features = torch.cat(graph_features_list, dim=0)

        return graph_features


class GatedPooling(nn.Module):

    def __init__(
            self,
            hidden_dim: int,
            dropout_rate: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # 最终投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(
            self,
            node_features: torch.Tensor,
            batch_num_nodes: torch.Tensor
    ) -> torch.Tensor:
        batch_size = len(batch_num_nodes)
        device = node_features.device

        # 创建batch索引
        batch_indices = torch.repeat_interleave(
            torch.arange(batch_size, device=device),
            batch_num_nodes
        )

        # Mean pooling
        mean_features = torch.zeros(batch_size, self.hidden_dim, device=device)
        mean_features.index_add_(0, batch_indices, node_features)
        mean_features = mean_features / batch_num_nodes.unsqueeze(-1).float()

        # Max pooling (需要逐图处理)
        max_features_list = []
        node_start = 0
        for i in range(batch_size):
            num_nodes = batch_num_nodes[i].item()
            h = node_features[node_start:node_start + num_nodes]
            max_feat, _ = h.max(dim=0)
            max_features_list.append(max_feat.unsqueeze(0))
            node_start += num_nodes
        max_features = torch.cat(max_features_list, dim=0)

        # 门控融合
        combined = torch.cat([mean_features, max_features], dim=-1)
        gate_weights = self.gate(combined)

        # 加权组合
        graph_features = gate_weights * mean_features + (1 - gate_weights) * max_features

        # 输出投影
        graph_features = self.output_proj(graph_features)

        return graph_features


class ConformerAttentionPooling(nn.Module):

    def __init__(
            self,
            input_dim: int,
            hidden_dim: Optional[int] = None,
            dropout_rate: float = 0.1
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        self.score_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(
            self,
            conformer_reprs: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            score_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        scores = self.score_mlp(conformer_reprs).squeeze(-1)  # [B, K]

        if score_bias is not None:
            scores = scores + score_bias

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        weights = F.softmax(scores, dim=1)
        pooled = torch.sum(conformer_reprs * weights.unsqueeze(-1), dim=1)
        pooled = self.layer_norm(pooled)
        return pooled


class SetTransformerPooling(nn.Module):

    def __init__(
            self,
            hidden_dim: int,
            num_seeds: int = 4,
            num_heads: int = 4,
            dropout_rate: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_seeds = num_seeds
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0

        # 可学习的seed vectors
        self.seeds = nn.Parameter(torch.randn(num_seeds, hidden_dim))

        # Seed attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Seed间的自注意力
        self.seed_self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout_rate, batch_first=True
        )

        # 输出投影 (将num_seeds个向量聚合为一个)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * num_seeds, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = math.sqrt(self.head_dim)

    def forward(
            self,
            node_features: torch.Tensor,
            batch_num_nodes: torch.Tensor
    ) -> torch.Tensor:
        batch_size = len(batch_num_nodes)
        device = node_features.device

        graph_features_list = []
        node_start = 0

        for i in range(batch_size):
            num_nodes = batch_num_nodes[i].item()
            node_end = node_start + num_nodes
            h = node_features[node_start:node_end]  # [num_nodes, hidden_dim]

            # Seed -> Node attention (Pooling by Multihead Attention)
            seeds = self.seeds.unsqueeze(0)  # [1, num_seeds, hidden_dim]

            Q = self.q_proj(seeds).view(1, self.num_seeds, self.num_heads, self.head_dim)
            K = self.k_proj(h).view(1, num_nodes, self.num_heads, self.head_dim)
            V = self.v_proj(h).view(1, num_nodes, self.num_heads, self.head_dim)

            # [1, num_seeds, num_heads, num_nodes]
            attn_scores = torch.einsum('bshd,bnhd->bshn', Q, K) / self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # [1, num_seeds, num_heads, head_dim]
            context = torch.einsum('bshn,bnhd->bshd', attn_weights, V)
            context = context.view(1, self.num_seeds, self.hidden_dim)

            # 残差连接
            seed_repr = self.layer_norm1(seeds + context)

            # Seed间的自注意力
            seed_repr_attn, _ = self.seed_self_attn(seed_repr, seed_repr, seed_repr)
            seed_repr = self.layer_norm2(seed_repr + seed_repr_attn)

            # 展平并投影
            seed_repr = seed_repr.view(1, -1)  # [1, num_seeds * hidden_dim]
            graph_feat = self.output_proj(seed_repr)

            graph_features_list.append(graph_feat)
            node_start = node_end

        graph_features = torch.cat(graph_features_list, dim=0)
        return graph_features


class HierarchicalPooling(nn.Module):

    def __init__(
            self,
            hidden_dim: int,
            pool_ratio: float = 0.5,
            dropout_rate: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.pool_ratio = pool_ratio

        # 节点重要性评分
        self.score_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 最终聚合
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(
            self,
            node_features: torch.Tensor,
            batch_num_nodes: torch.Tensor
    ) -> torch.Tensor:
        batch_size = len(batch_num_nodes)
        device = node_features.device

        # 计算节点重要性分数
        scores = self.score_layer(node_features).squeeze(-1)  # [N_total]

        graph_features_list = []
        node_start = 0

        for i in range(batch_size):
            num_nodes = batch_num_nodes[i].item()
            node_end = node_start + num_nodes

            h = node_features[node_start:node_end]  # [num_nodes, hidden_dim]
            s = scores[node_start:node_end]  # [num_nodes]

            # 选择top-k重要节点
            k = max(1, int(num_nodes * self.pool_ratio))
            topk_scores, topk_indices = torch.topk(s, k)

            # 软选择: 用softmax权重而非硬选择
            topk_weights = F.softmax(topk_scores, dim=0)  # [k]

            # 选择对应节点
            topk_features = h[topk_indices]  # [k, hidden_dim]

            # 加权聚合
            graph_feat = (topk_weights.unsqueeze(-1) * topk_features).sum(dim=0, keepdim=True)

            graph_features_list.append(graph_feat)
            node_start = node_end

        graph_features = torch.cat(graph_features_list, dim=0)
        graph_features = self.output_proj(graph_features)

        return graph_features


class SAGPooling(nn.Module):

    def __init__(
            self,
            hidden_dim: int,
            pool_ratio: float = 0.5,
            dropout_rate: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.pool_ratio = pool_ratio

        # GCN-based 节点重要性评分 (类似 SAGPool 原论文)
        self.score_gcn = nn.Linear(hidden_dim, 1)

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(
            self,
            node_features: torch.Tensor,
            batch_num_nodes: torch.Tensor
    ) -> torch.Tensor:
        batch_size = len(batch_num_nodes)
        device = node_features.device

        # 计算节点重要性分数
        scores = self.score_gcn(node_features).squeeze(-1)  # [N_total]
        scores = torch.tanh(scores)  # 归一化到 [-1, 1]

        graph_features_list = []
        node_start = 0

        for i in range(batch_size):
            num_nodes = batch_num_nodes[i].item()
            node_end = node_start + num_nodes

            h = node_features[node_start:node_end]  # [num_nodes, hidden_dim]
            s = scores[node_start:node_end]  # [num_nodes]

            # 选择 top-k 重要节点
            k = max(1, int(num_nodes * self.pool_ratio))
            topk_scores, topk_indices = torch.topk(s, k)

            # 门控: 用分数对特征进行缩放
            topk_features = h[topk_indices]  # [k, hidden_dim]
            gated_features = topk_features * topk_scores.unsqueeze(-1)  # [k, hidden_dim]

            # Global Add Pool (sum pooling) - DTA-GTOmega 使用的方式
            graph_feat = gated_features.sum(dim=0, keepdim=True)  # [1, hidden_dim]

            graph_features_list.append(graph_feat)
            node_start = node_end

        graph_features = torch.cat(graph_features_list, dim=0)
        graph_features = self.output_proj(graph_features)

        return graph_features


class GlobalAddPool(nn.Module):

    def __init__(
            self,
            hidden_dim: int,
            dropout_rate: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(
            self,
            node_features: torch.Tensor,
            batch_num_nodes: torch.Tensor
    ) -> torch.Tensor:
        batch_size = len(batch_num_nodes)
        device = node_features.device

        # 创建 batch 索引
        batch_indices = torch.repeat_interleave(
            torch.arange(batch_size, device=device),
            batch_num_nodes
        )

        # Sum pooling (Global Add Pool)
        graph_features = torch.zeros(batch_size, self.hidden_dim, device=device)
        graph_features.index_add_(0, batch_indices, node_features)

        # 输出投影
        graph_features = self.output_proj(graph_features)

        return graph_features


class GraphLevelCoAttention(nn.Module):

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int = 8,
            dropout_rate: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0

        # 药物 -> 蛋白质 注意力
        self.drug_q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.protein_k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.protein_v_proj = nn.Linear(hidden_dim, hidden_dim)

        # 蛋白质 -> 药物 注意力
        self.protein_q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.drug_k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.drug_v_proj = nn.Linear(hidden_dim, hidden_dim)

        # 输出投影
        self.drug_output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.protein_output_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm_drug = nn.LayerNorm(hidden_dim)
        self.layer_norm_protein = nn.LayerNorm(hidden_dim)

        self.scale = math.sqrt(self.head_dim)

    def forward(
            self,
            drug_graph_repr: torch.Tensor,
            protein_graph_repr: torch.Tensor
    ) -> tuple:
        batch_size = drug_graph_repr.size(0)

        # === 药物通过关注蛋白质更新 ===
        q_drug = self.drug_q_proj(drug_graph_repr).view(
            batch_size, self.num_heads, self.head_dim
        )  # [B, H, D]
        k_protein = self.protein_k_proj(protein_graph_repr).view(
            batch_size, self.num_heads, self.head_dim
        )
        v_protein = self.protein_v_proj(protein_graph_repr).view(
            batch_size, self.num_heads, self.head_dim
        )

        # 单样本注意力 (每个样本的药物只关注对应的蛋白质)
        # [B, H, 1] -> 对角注意力
        attn_scores_drug = (q_drug * k_protein).sum(dim=-1, keepdim=True) / self.scale
        attn_weights_drug = F.softmax(attn_scores_drug, dim=-1)
        attn_weights_drug = self.dropout(attn_weights_drug)

        drug_context = attn_weights_drug * v_protein  # [B, H, D]
        drug_context = drug_context.view(batch_size, self.hidden_dim)
        drug_updated = self.drug_output_proj(drug_context)
        drug_updated = self.layer_norm_drug(drug_graph_repr + drug_updated)

        # === 蛋白质通过关注药物更新 ===
        q_protein = self.protein_q_proj(protein_graph_repr).view(
            batch_size, self.num_heads, self.head_dim
        )
        k_drug = self.drug_k_proj(drug_graph_repr).view(
            batch_size, self.num_heads, self.head_dim
        )
        v_drug = self.drug_v_proj(drug_graph_repr).view(
            batch_size, self.num_heads, self.head_dim
        )

        attn_scores_protein = (q_protein * k_drug).sum(dim=-1, keepdim=True) / self.scale
        attn_weights_protein = F.softmax(attn_scores_protein, dim=-1)
        attn_weights_protein = self.dropout(attn_weights_protein)

        protein_context = attn_weights_protein * v_drug
        protein_context = protein_context.view(batch_size, self.hidden_dim)
        protein_updated = self.protein_output_proj(protein_context)
        protein_updated = self.layer_norm_protein(protein_graph_repr + protein_updated)

        return drug_updated, protein_updated
