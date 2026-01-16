# models/dta_model_multiGeo.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from typing import Dict, List, Optional, Tuple
from torch_geometric.nn import GraphNorm
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.rnn import pack_padded_sequence

from models.layers.multiGeo_layers import (
    DGLSAGPooling,
    DGLTransformerConv,
    IntraGraphAttention
)
from models.layers.pooling import ConformerAttentionPooling


class MultiGeoDTAModel(nn.Module):

    def __init__(self, config: Dict):
        super().__init__()

        # 基础配置
        hidden_dim = config['model']['hidden_dim']
        num_heads = config['model']['num_attention_heads']
        num_blocks = config['model'].get('num_gnn_layers', 4)  # 默认4层，和论文一致
        dropout_rate = config['model']['dropout_rate']
        pool_ratio = config['model'].get('pool_ratio', 0.5)

        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # 特征维度（从config读取，支持ColabFold 256维或手工31维）
        protein_node_dim = config['features']['protein']['node_dim']
        drug_node_dim = config['features']['drug']['node_dim']
        drug_edge_dim = config['features']['drug']['edge_dim']

        # 输入LayerNorm（和论文一致）
        self.drug_input_norm = nn.LayerNorm(drug_node_dim)
        self.protein_input_norm = nn.LayerNorm(protein_node_dim)

        # 输入投影到hidden_dim
        self.drug_input_proj = nn.Linear(drug_node_dim, hidden_dim)
        self.protein_input_proj = nn.Linear(protein_node_dim, hidden_dim)

        # 多层Block - 药物
        out_dim = hidden_dim * num_heads
        self.out_dim = out_dim
        self.drug_blocks = nn.ModuleList()
        self.drug_norms = nn.ModuleList()
        for i in range(num_blocks):
            in_dim = hidden_dim if i == 0 else out_dim
            self.drug_blocks.append(
                MultiGeoSimpleBlock(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    edge_dim=drug_edge_dim if i == 0 else None,
                    pool_ratio=pool_ratio,
                    dropout=dropout_rate
                )
            )
            # 使用GraphNorm（和原版MultiGeo一致）
            self.drug_norms.append(GraphNorm(out_dim))

        # 多层Block - 蛋白质
        self.protein_blocks = nn.ModuleList()
        self.protein_norms = nn.ModuleList()
        for i in range(num_blocks):
            in_dim = hidden_dim if i == 0 else out_dim
            self.protein_blocks.append(
                MultiGeoSimpleBlock(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    edge_dim=None,
                    pool_ratio=pool_ratio,
                    dropout=dropout_rate
                )
            )
            # 使用GraphNorm（和原版MultiGeo一致）
            self.protein_norms.append(GraphNorm(out_dim))

        # Block Cross-Attention交互（替代CoAttention/RESCAL）
        self.cross_attention_fusion = BlockCrossAttentionFusion(
            hidden_dim=out_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )

        # 多构象融合（默认可只用第一个构象）
        self.conformer_fusion = config['model'].get('conformer_fusion', 'first')
        if self.conformer_fusion not in {'first', 'attention', 'mean'}:
            raise ValueError(f"Unsupported conformer_fusion: {self.conformer_fusion}")
        self.conformer_pool = None
        if self.conformer_fusion == 'attention':
            self.conformer_pool = ConformerAttentionPooling(
                input_dim=out_dim,
                hidden_dim=out_dim,
                dropout_rate=dropout_rate
            )
        self.conformer_score_weight = float(config['model'].get('conformer_score_weight', 0.0))
        self.use_checkpoint = bool(config['model'].get('use_checkpoint', False))

        # 动态构象建模（GRU）
        dynamic_cfg = config.get('model', {}).get('dynamic_conformer', {}) or {}
        self.use_dynamic_conformer = bool(dynamic_cfg.get('enabled', True)) if dynamic_cfg else False
        if self.use_dynamic_conformer:
            dynamic_hidden = int(dynamic_cfg.get('hidden_dim', out_dim))
            dynamic_layers = int(dynamic_cfg.get('num_layers', 1))
            dynamic_dropout = float(dynamic_cfg.get('dropout_rate', dropout_rate))
            dynamic_bidirectional = bool(dynamic_cfg.get('bidirectional', False))
            self.dynamic_fusion_mode = str(dynamic_cfg.get('fusion_mode', 'gate')).lower()
            if self.dynamic_fusion_mode not in {'gate', 'disagreement', 'orthogonal'}:
                raise ValueError(f"Unsupported dynamic fusion_mode: {self.dynamic_fusion_mode}")
            self.dynamic_modeler = ProteinDynamicModeler(
                input_dim=out_dim,
                hidden_dim=dynamic_hidden,
                num_layers=dynamic_layers,
                dropout=dynamic_dropout,
                bidirectional=dynamic_bidirectional
            )
            self.dynamic_dim = dynamic_hidden

            gate_dropout = float(dynamic_cfg.get('fusion_dropout', dropout_rate))
            self.dynamic_norm = nn.LayerNorm(self.dynamic_dim)
            self.dynamic_gate = nn.Sequential(
                nn.Linear(out_dim * 4, self.dynamic_dim),
                nn.ReLU(),
                nn.Linear(self.dynamic_dim, self.dynamic_dim)
            )
            self.dynamic_disagree_gate = None
            self.dynamic_ref_proj = None
            if self.dynamic_fusion_mode in {'disagreement', 'orthogonal'}:
                self.dynamic_ref_proj = (
                    nn.Identity()
                    if self.dynamic_dim == out_dim
                    else nn.Linear(out_dim, self.dynamic_dim)
                )
            if self.dynamic_fusion_mode == 'disagreement':
                self.dynamic_disagree_gate = nn.Sequential(
                    nn.Linear(self.dynamic_dim * 4, self.dynamic_dim),
                    nn.ReLU(),
                    nn.Linear(self.dynamic_dim, self.dynamic_dim)
                )
            self.dynamic_gamma = nn.Parameter(torch.tensor(0.0))
            self.dynamic_dropout = nn.Dropout(gate_dropout)
        else:
            self.dynamic_modeler = None
            self.dynamic_dim = 0
            self.dynamic_norm = None
            self.dynamic_gate = None
            self.dynamic_disagree_gate = None
            self.dynamic_ref_proj = None
            self.dynamic_gamma = None
            self.dynamic_dropout = None

        # 亲和力预测头（和论文一致）
        # 输入: rescal_output (out_dim * 2)
        fusion_dim = out_dim * 4
        if self.use_dynamic_conformer:
            fusion_dim += self.dynamic_dim

        self.affinity_predictor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(fusion_dim, 1)
        )

    def forward(
        self,
        protein_conformers: List[dgl.DGLGraph],
        drug_graph: dgl.DGLGraph,
        esm_features=None,  # 保持接口兼容，不使用
        protein_conformer_mask: Optional[torch.Tensor] = None,
        protein_conformer_scores: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # === 药物编码 ===
        drug_feats = self.drug_input_norm(drug_graph.ndata['feat'])
        h_drug = self.drug_input_proj(drug_feats)
        drug_edge_feats = drug_graph.edata.get('feat', None)

        # 获取batch索引（GraphNorm需要）
        drug_batch = self._get_batch_index(drug_graph)

        # 多层Block，每层输出图级表示
        drug_block_reprs = []
        for i, (block, norm) in enumerate(zip(self.drug_blocks, self.drug_norms)):
            edge_feats = drug_edge_feats if i == 0 else None
            h_drug, drug_repr = self._run_block(
                block,
                norm,
                drug_graph,
                h_drug,
                edge_feats,
                drug_batch
            )
            drug_block_reprs.append(drug_repr)

        # 堆叠 [batch_size, num_blocks, out_dim]
        drug_block_stack = torch.stack(drug_block_reprs, dim=1)

        # === 蛋白质编码 ===
        protein_conformer_reprs = []
        use_all_conformers = self.conformer_fusion != 'first' or self.use_dynamic_conformer
        protein_graphs = protein_conformers if use_all_conformers else protein_conformers[:1]
        for protein_graph in protein_graphs:
            protein_feats = self.protein_input_norm(protein_graph.ndata['feat'])
            h_protein = self.protein_input_proj(protein_feats)

            # 获取batch索引（GraphNorm需要）
            protein_batch = self._get_batch_index(protein_graph)

            protein_block_reprs = []
            for i, (block, norm) in enumerate(zip(self.protein_blocks, self.protein_norms)):
                h_protein, protein_repr = self._run_block(
                    block,
                    norm,
                    protein_graph,
                    h_protein,
                    None,
                    protein_batch
                )
                protein_block_reprs.append(protein_repr)
            protein_conformer_reprs.append(protein_block_reprs)

        # 堆叠 [batch_size, num_blocks, out_dim]
        if self.conformer_fusion == 'first' or len(protein_conformer_reprs) == 1:
            protein_block_stack = torch.stack(protein_conformer_reprs[0], dim=1)
        else:
            if protein_conformer_mask is not None and protein_conformer_mask.shape[1] != len(protein_conformer_reprs):
                protein_conformer_mask = protein_conformer_mask[:, :len(protein_conformer_reprs)]
            protein_block_stack = self._pool_conformer_blocks(
                protein_conformer_reprs,
                protein_conformer_mask,
                protein_conformer_scores
            )

        # 动态构象表示（GRU）
        protein_dynamic_repr = None
        if self.use_dynamic_conformer:
            conformer_graph_reprs = [conf_reprs[-1] for conf_reprs in protein_conformer_reprs]
            protein_dynamic_repr = self.dynamic_modeler(
                conformer_graph_reprs,
                mask=protein_conformer_mask
            )

        # === Block Cross-Attention交互 ===
        rescal_output = self.cross_attention_fusion(drug_block_stack, protein_block_stack)

        drug_repr = drug_block_stack[:, -1, :]
        protein_repr = protein_block_stack[:, -1, :]

        # === 预测 ===
        fusion_parts = [drug_repr, protein_repr, rescal_output]
        protein_dynamic_gated = None
        if protein_dynamic_repr is not None:
            dyn = self.dynamic_norm(protein_dynamic_repr)
            if self.dynamic_fusion_mode == 'disagreement':
                ref = self.dynamic_ref_proj(protein_repr) if self.dynamic_ref_proj is not None else protein_repr
                delta = dyn - ref
                gate_input = torch.cat([dyn, ref, delta, delta.abs()], dim=-1)
                gate = torch.sigmoid(self.dynamic_disagree_gate(gate_input))
                dynamic_feature = dyn
            elif self.dynamic_fusion_mode == 'orthogonal':
                ref = self.dynamic_ref_proj(protein_repr) if self.dynamic_ref_proj is not None else protein_repr
                ref_norm = ref / (ref.norm(dim=-1, keepdim=True) + 1e-8)
                proj = (dyn * ref_norm).sum(dim=-1, keepdim=True) * ref_norm
                dynamic_feature = dyn - proj
                ctx = torch.cat([drug_repr, protein_repr, rescal_output], dim=-1)
                gate = torch.sigmoid(self.dynamic_gate(ctx))
            else:
                ctx = torch.cat([drug_repr, protein_repr, rescal_output], dim=-1)
                gate = torch.sigmoid(self.dynamic_gate(ctx))
                dynamic_feature = dyn
            protein_dynamic_gated = self.dynamic_dropout(self.dynamic_gamma * gate * dynamic_feature)
            fusion_parts.append(protein_dynamic_gated)
        fusion_features = torch.cat(fusion_parts, dim=-1)
        predicted_affinity = self.affinity_predictor(fusion_features).squeeze(-1)

        return {
            'affinity': predicted_affinity,
            'drug_repr': drug_repr,
            'protein_repr': protein_repr,
            'protein_dynamic_repr': protein_dynamic_repr,
            'protein_dynamic_gated': protein_dynamic_gated
        }

    def _get_batch_index(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """从DGL batched graph获取batch索引（用于GraphNorm）"""
        batch_num_nodes = graph.batch_num_nodes()
        batch_idx = torch.repeat_interleave(
            torch.arange(len(batch_num_nodes), device=graph.device),
            batch_num_nodes
        )
        return batch_idx

    def _run_block(
        self,
        block: nn.Module,
        norm: nn.Module,
        graph: dgl.DGLGraph,
        h: torch.Tensor,
        edge_feats: Optional[torch.Tensor],
        batch_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def _forward(inner_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            out_h, graph_repr = block(graph, inner_h, edge_feats)
            out_h = F.elu(norm(out_h, batch_idx))
            return out_h, graph_repr

        if self.use_checkpoint and self.training:
            return checkpoint(_forward, h)
        return _forward(h)

    def _pool_conformer_blocks(
        self,
        protein_conformer_reprs: List[List[torch.Tensor]],
        mask: Optional[torch.Tensor],
        scores: Optional[torch.Tensor]
    ) -> torch.Tensor:
        pooled_blocks = []
        score_bias = self._build_score_bias(scores, mask) if scores is not None else None
        for block_idx in range(self.num_blocks):
            block_reprs = torch.stack(
                [conf_reprs[block_idx] for conf_reprs in protein_conformer_reprs],
                dim=1
            )
            if self.conformer_fusion == 'attention':
                pooled = self.conformer_pool(block_reprs, mask=mask, score_bias=score_bias)
            elif self.conformer_fusion == 'mean':
                pooled = self._masked_mean(block_reprs, mask)
            else:
                pooled = block_reprs[:, 0, :]
            pooled_blocks.append(pooled)
        return torch.stack(pooled_blocks, dim=1)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return values.mean(dim=1)
        mask = mask.unsqueeze(-1).to(values.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (values * mask).sum(dim=1) / denom

    def _build_score_bias(
        self,
        scores: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        score_bias = scores.float()
        if score_bias.numel() > 0 and score_bias.max() > 1.0:
            score_bias = score_bias / 100.0
        if mask is not None:
            score_bias = score_bias.masked_fill(~mask, 0.0)
        score_bias = score_bias - score_bias.mean(dim=1, keepdim=True)
        return score_bias * self.conformer_score_weight


class MultiGeoSimpleBlock(nn.Module):

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        edge_dim: Optional[int] = None,
        pool_ratio: float = 0.5,
        dropout: float = 0.1
    ):
        super().__init__()

        out_dim = hidden_dim * num_heads

        # 第一层TransformerConv
        self.conv = DGLTransformerConv(
            in_dim=in_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            edge_dim=edge_dim,
            dropout=dropout
        )

        # 第二层：IntraGraphAttention（另一个TransformerConv）
        self.intra_att = IntraGraphAttention(
            input_dim=out_dim,
            out_dim=hidden_dim,
            num_heads=num_heads,
            edge_dim=None,  # 第二层不使用边特征
            dropout=dropout
        )

        # 和原版MultiGeo一致：用min_score=-1而不是ratio，不做节点筛选
        self.pool = DGLSAGPooling(out_dim, ratio=1.0, min_score=-1.0, dropout=dropout)

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: Optional[torch.Tensor] = None
    ):
        # TransformerConv
        h = self.conv(graph, node_feats, edge_feats)
        # IntraGraphAttention（第二层TransformerConv）
        h = self.intra_att(graph, h)
        # SAGPooling
        graph_repr, _ = self.pool(graph, h)
        return h, graph_repr


class BlockCrossAttentionFusion(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for cross-attention")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_drug = nn.Linear(hidden_dim, hidden_dim)
        self.k_protein = nn.Linear(hidden_dim, hidden_dim)
        self.v_protein = nn.Linear(hidden_dim, hidden_dim)

        self.q_protein = nn.Linear(hidden_dim, hidden_dim)
        self.k_drug = nn.Linear(hidden_dim, hidden_dim)
        self.v_drug = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)
        fusion_dim = hidden_dim * 4
        self.proj = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_blocks, dim = x.shape
        x = x.view(bsz, num_blocks, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [B, H, L, Dh]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, heads, num_blocks, head_dim = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(bsz, num_blocks, heads * head_dim)

    def forward(self, drug_blocks: torch.Tensor, protein_blocks: torch.Tensor) -> torch.Tensor:
        # Drug attends to protein blocks
        qd = self._reshape_heads(self.q_drug(drug_blocks))
        kp = self._reshape_heads(self.k_protein(protein_blocks))
        vp = self._reshape_heads(self.v_protein(protein_blocks))

        attn_dp = torch.matmul(qd, kp.transpose(-2, -1)) * self.scale
        attn_dp = F.softmax(attn_dp, dim=-1)
        attn_dp = self.dropout(attn_dp)
        ctx_d = torch.matmul(attn_dp, vp)
        ctx_d = self._merge_heads(ctx_d)

        # Protein attends to drug blocks
        qp = self._reshape_heads(self.q_protein(protein_blocks))
        kd = self._reshape_heads(self.k_drug(drug_blocks))
        vd = self._reshape_heads(self.v_drug(drug_blocks))

        attn_pd = torch.matmul(qp, kd.transpose(-2, -1)) * self.scale
        attn_pd = F.softmax(attn_pd, dim=-1)
        attn_pd = self.dropout(attn_pd)
        ctx_p = torch.matmul(attn_pd, vd)
        ctx_p = self._merge_heads(ctx_p)

        drug_ctx = ctx_d.mean(dim=1) + drug_blocks.mean(dim=1)
        protein_ctx = ctx_p.mean(dim=1) + protein_blocks.mean(dim=1)

        fused = torch.cat(
            [
                drug_ctx,
                protein_ctx,
                drug_ctx * protein_ctx,
                torch.abs(drug_ctx - protein_ctx)
            ],
            dim=-1
        )
        return self.proj(fused)


class ProteinDynamicModeler(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        if bidirectional:
            self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.output_proj = nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        conformer_representations: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        sequence = torch.stack(conformer_representations, dim=1)  # [B, K, D]

        if mask is not None:
            lengths = mask.sum(dim=1).clamp(min=1).cpu()
            packed = pack_padded_sequence(
                sequence,
                lengths,
                batch_first=True,
                enforce_sorted=False
            )
            _, h_n = self.gru(packed)
        else:
            _, h_n = self.gru(sequence)

        if self.bidirectional:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            last_hidden = self.output_proj(last_hidden)
        else:
            last_hidden = h_n[-1]

        return self.layer_norm(last_hidden)
