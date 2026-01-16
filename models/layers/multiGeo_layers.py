# models/layers/multiGeo_layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from typing import Optional, Tuple
import math


class DGLTransformerConv(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        edge_dim: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.edge_dim = edge_dim
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value投影
        self.q_linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.k_linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.v_linear = nn.Linear(in_dim, out_dim, bias=bias)

        # 边特征投影（如果有）
        if edge_dim is not None:
            self.edge_linear = nn.Linear(edge_dim, out_dim, bias=False)

        # 自身节点投影
        self.skip_linear = nn.Linear(in_dim, out_dim, bias=bias)

        # 输出投影
        self.out_linear = nn.Linear(out_dim, out_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.xavier_uniform_(self.skip_linear.weight)
        nn.init.xavier_uniform_(self.out_linear.weight)

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        num_nodes = node_feats.size(0)

        # 计算Q, K, V
        q = self.q_linear(node_feats).view(num_nodes, self.num_heads, self.head_dim)
        k = self.k_linear(node_feats).view(num_nodes, self.num_heads, self.head_dim)
        v = self.v_linear(node_feats).view(num_nodes, self.num_heads, self.head_dim)

        # 边特征处理
        if self.edge_dim is not None and edge_feats is not None:
            edge_v = self.edge_linear(edge_feats).view(-1, self.num_heads, self.head_dim)
        else:
            edge_v = None

        with graph.local_scope():
            graph.ndata['q'] = q
            graph.ndata['k'] = k
            graph.ndata['v'] = v

            if edge_v is not None:
                graph.edata['e'] = edge_v

            # 消息传递：计算注意力分数
            graph.apply_edges(self._compute_attention)

            # 归一化注意力权重
            graph.edata['a'] = dgl.ops.edge_softmax(graph, graph.edata['a'])
            graph.edata['a'] = self.dropout(graph.edata['a'])

            # 聚合消息
            if edge_v is not None:
                graph.apply_edges(lambda edges: {
                    'm': edges.data['a'].unsqueeze(-1) * (edges.src['v'] + edges.data['e'])
                })
            else:
                graph.apply_edges(lambda edges: {
                    'm': edges.data['a'].unsqueeze(-1) * edges.src['v']
                })

            graph.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'h'))

            h = graph.ndata['h'].view(num_nodes, -1)

        # 残差连接 + skip
        skip = self.skip_linear(node_feats)
        h = self.out_linear(h) + skip
        h = self.layer_norm(h)

        return h

    def _compute_attention(self, edges):
        """计算边的注意力分数"""
        q = edges.dst['q']  # [num_edges, num_heads, head_dim]
        k = edges.src['k']  # [num_edges, num_heads, head_dim]

        # 点积注意力
        a = (q * k).sum(dim=-1) * self.scale  # [num_edges, num_heads]

        return {'a': a}


class IntraGraphAttention(nn.Module):

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        num_heads: int = 4,
        edge_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()

        self.conv = DGLTransformerConv(
            in_dim=input_dim,
            out_dim=out_dim * num_heads,
            num_heads=num_heads,
            edge_dim=edge_dim,
            dropout=dropout
        )

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = F.elu(node_feats)
        return self.conv(graph, h, edge_feats)


class DGLSAGPooling(nn.Module):

    def __init__(
        self,
        in_dim: int,
        ratio: float = 0.5,
        min_score: float = -1.0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.in_dim = in_dim
        self.ratio = ratio
        self.dropout = nn.Dropout(dropout)

        # 使用PyG的SAGPooling（和MultiGeo一致）
        from torch_geometric.nn import SAGPooling, global_add_pool
        self.sag_pool = SAGPooling(in_dim, ratio=ratio, min_score=min_score)
        self.global_pool = global_add_pool

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 从DGL图提取PyG需要的格式
        src, dst = graph.edges()
        edge_index = torch.stack([src, dst], dim=0)

        # 创建batch索引
        batch_num_nodes = graph.batch_num_nodes()
        batch_idx = torch.repeat_interleave(
            torch.arange(len(batch_num_nodes), device=node_feats.device),
            batch_num_nodes
        )

        # PyG SAGPooling
        x_pool, edge_index_pool, _, batch_pool, perm, scores = self.sag_pool(
            node_feats, edge_index, batch=batch_idx
        )

        # global_add_pool（和MultiGeo一致）
        pooled = self.global_pool(x_pool, batch_pool)
        pooled = self.dropout(pooled)

        return pooled, scores


class CoAttentionLayer(nn.Module):

    def __init__(self, n_features: int):
        super().__init__()

        self.n_features = n_features

        # Query和Key投影
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features // 2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features // 2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features // 2))

        # 初始化
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))

    def forward(
        self,
        receiver: torch.Tensor,
        attendant: torch.Tensor
    ) -> torch.Tensor:
        keys = receiver @ self.w_k      # [batch, num_blocks, n_features//2]
        queries = attendant @ self.w_q  # [batch, num_blocks, n_features//2]

        # 计算注意力分数
        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a

        return e_scores


class RESCAL(nn.Module):

    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features

    def forward(
        self,
        heads: torch.Tensor,
        tails: torch.Tensor,
        alpha_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # L2归一化
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)

        # 计算双向交互分数
        scores = heads @ tails.transpose(-2, -1)   # [batch, num_blocks, num_blocks]
        scores2 = tails @ heads.transpose(-2, -1)  # [batch, num_blocks, num_blocks]

        # 应用注意力权重
        if alpha_scores is not None:
            scores = alpha_scores * scores
            scores2 = alpha_scores * scores2

        return scores, scores2


class MultiGeoBlock(nn.Module):

    def __init__(
        self,
        in_dim_drug: int,
        in_dim_protein: int,
        hidden_dim: int,
        num_heads: int = 4,
        drug_edge_dim: Optional[int] = None,
        protein_edge_dim: Optional[int] = None,
        pool_ratio: float = 0.5,
        dropout: float = 0.1
    ):
        super().__init__()

        out_dim = hidden_dim * num_heads

        # 药物TransformerConv
        self.drug_conv = DGLTransformerConv(
            in_dim=in_dim_drug,
            out_dim=out_dim,
            num_heads=num_heads,
            edge_dim=drug_edge_dim,
            dropout=dropout
        )

        # 蛋白质TransformerConv
        self.protein_conv = DGLTransformerConv(
            in_dim=in_dim_protein,
            out_dim=out_dim,
            num_heads=num_heads,
            edge_dim=protein_edge_dim,
            dropout=dropout
        )

        # 图内自注意力
        self.drug_intra_att = IntraGraphAttention(
            input_dim=out_dim,
            out_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.protein_intra_att = IntraGraphAttention(
            input_dim=out_dim,
            out_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # SAGPooling
        self.drug_pool = DGLSAGPooling(out_dim, ratio=pool_ratio, dropout=dropout)
        self.protein_pool = DGLSAGPooling(out_dim, ratio=pool_ratio, dropout=dropout)

        # GraphNorm
        self.drug_norm = nn.LayerNorm(out_dim)
        self.protein_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        drug_graph: dgl.DGLGraph,
        protein_graph: dgl.DGLGraph,
        drug_feats: torch.Tensor,
        protein_feats: torch.Tensor,
        drug_edge_feats: Optional[torch.Tensor] = None,
        protein_edge_feats: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # TransformerConv
        h_drug = self.drug_conv(drug_graph, drug_feats, drug_edge_feats)
        h_protein = self.protein_conv(protein_graph, protein_feats, protein_edge_feats)

        # IntraGraphAttention
        h_drug = self.drug_intra_att(drug_graph, h_drug)
        h_protein = self.protein_intra_att(protein_graph, h_protein)

        # SAGPooling得到图级表示
        drug_graph_repr, _ = self.drug_pool(drug_graph, h_drug)
        protein_graph_repr, _ = self.protein_pool(protein_graph, h_protein)

        # 归一化
        h_drug = F.elu(self.drug_norm(h_drug))
        h_protein = F.elu(self.protein_norm(h_protein))

        return h_drug, h_protein, drug_graph_repr, protein_graph_repr
