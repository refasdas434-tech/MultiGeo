# data/preprocessing/protein_graph.py

import numpy as np
import torch
import dgl
from Bio.PDB import PDBParser, is_aa
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ProteinGraphBuilder:

    # 标准氨基酸列表
    AMINO_ACIDS = [
        'ALA', 'CYS', 'ASP', 'GLU', 'PHE',
        'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
        'MET', 'ASN', 'PRO', 'GLN', 'ARG',
        'SER', 'THR', 'VAL', 'TRP', 'TYR'
    ]

    # 氨基酸物理化学性质
    AA_PROPERTIES = {
        'hydrophobicity': {
            'ILE': 1.0, 'VAL': 0.9, 'LEU': 0.8, 'PHE': 0.8, 'CYS': 0.7,
            'MET': 0.6, 'ALA': 0.5, 'GLY': 0.4, 'THR': 0.3, 'SER': 0.2,
            'TRP': 0.2, 'TYR': 0.1, 'PRO': 0.0, 'HIS': -0.2, 'GLU': -0.5,
            'GLN': -0.6, 'ASP': -0.7, 'ASN': -0.7, 'LYS': -0.8, 'ARG': -1.0
        },
        'charge': {
            'ARG': 1.0, 'LYS': 1.0, 'HIS': 0.5,
            'ASP': -1.0, 'GLU': -1.0,
        },
        'polarity': {
            'SER': 1.0, 'THR': 1.0, 'ASN': 1.0, 'GLN': 1.0,
            'CYS': 0.5, 'TYR': 0.5,
        },
        'aromaticity': {
            'PHE': 1.0, 'TRP': 1.0, 'TYR': 1.0, 'HIS': 0.5,
        },
        'volume': {  # 归一化体积
            'GLY': 0.0, 'ALA': 0.2, 'SER': 0.3, 'CYS': 0.3, 'ASP': 0.4,
            'PRO': 0.4, 'ASN': 0.4, 'THR': 0.4, 'GLU': 0.5, 'VAL': 0.5,
            'GLN': 0.5, 'HIS': 0.6, 'MET': 0.6, 'ILE': 0.6, 'LEU': 0.6,
            'LYS': 0.7, 'ARG': 0.7, 'PHE': 0.8, 'TYR': 0.9, 'TRP': 1.0
        }
    }

    def __init__(
            self,
            contact_threshold: float = 8.0,
            secondary_structure_window: int = 5,
            use_plddt: bool = True,
            top_k: int = None,
            use_contact_map: bool = False,
            contact_map_dir: Optional[str] = None,
            pair_repr_dir: Optional[str] = None,
            max_protein_len: int = 800,
            colabfold_features_dir: Optional[str] = None
    ):
        """
        Args:
            contact_threshold: 空间接触距离阈值 (Å)
            secondary_structure_window: 二级结构窗口大小
            use_plddt: 是否使用pLDDT置信度（默认填充1.0）
            top_k: 每个残基保留的最近邻数量（接触边剪枝），None 表示不限制
            use_contact_map: 是否使用预计算的contact map来构建边（而非距离阈值）
            contact_map_dir: contact map文件目录路径
            pair_repr_dir: pair representation文件目录路径（用于边特征）
            max_protein_len: 蛋白质最大长度，超过则截断（与GTOmega一致，默认800）
            colabfold_features_dir: ColabFold single representation目录（256维节点特征）
        """
        self.contact_threshold = contact_threshold
        self.ss_window = secondary_structure_window
        self.use_plddt = use_plddt
        self.top_k = top_k
        self.use_contact_map = use_contact_map
        self.contact_map_dir = Path(contact_map_dir) if contact_map_dir else None
        self.pair_repr_dir = Path(pair_repr_dir) if pair_repr_dir else None
        self.max_protein_len = max_protein_len
        self.colabfold_features_dir = Path(colabfold_features_dir) if colabfold_features_dir else None
        self.pdb_parser = PDBParser(QUIET=True)

    def build_graph(
        self,
        pdb_file: str,
        target_id: Optional[str] = None,
        rank_idx: Optional[int] = None
    ):
        """
        从PDB文件构建蛋白质图

        Args:
            pdb_file: PDB文件路径
            target_id: 蛋白质ID，用于加载对应的contact map和pair_repr
            rank_idx: 构象rank索引（用于加载对应rank的特征/接触图）

        Returns:
            graph: DGLGraph对象，包含:
                - ndata['feat']: [num_nodes, node_dim] 节点特征
                - ndata['coord']: [num_nodes, 3] 坐标
                - edata['feat']: [num_edges, 128] 边特征（如果有pair_repr）
        """
        # 解析PDB
        structure = self.pdb_parser.get_structure('protein', pdb_file)
        model = structure[0]

        # 提取残基信息（包括pLDDT）
        residues, coords, plddt_scores = self._extract_residue_info(model)
        num_residues = len(residues)

        if num_residues == 0:
            raise ValueError(f"No valid residues found in {pdb_file}")

        # 截断到max_protein_len（与GTOmega一致）
        if self.max_protein_len and num_residues > self.max_protein_len:
            logger.debug(f"Truncating {target_id} from {num_residues} to {self.max_protein_len} residues")
            residues = residues[:self.max_protein_len]
            coords = coords[:self.max_protein_len]
            plddt_scores = plddt_scores[:self.max_protein_len]
            num_residues = self.max_protein_len

        # 简化的二级结构预测（基于局部几何）
        ss_info = self._predict_secondary_structure_simple(coords)

        # 构建节点特征：优先使用ColabFold 256维特征，否则用31维手工特征
        if self.colabfold_features_dir is not None and target_id is not None:
            node_features = self._load_colabfold_node_features(target_id, num_residues, rank_idx)
            if node_features is None:
                # 加载失败，抛出异常（不回退，因为会导致batch时维度不匹配）
                raise ValueError(f"ColabFold features not found for {target_id}, cannot mix 256d and 31d features")
        else:
            node_features = self._build_node_features(residues, ss_info, plddt_scores)

        # 构建边：使用contact map或距离阈值
        if self.use_contact_map and target_id is not None:
            edges = self._build_edges_from_contact_map(
                target_id, coords, num_residues, rank_idx
            )
        else:
            edges = self._build_edges_from_distance(coords, num_residues)

        # 创建DGLGraph
        src_nodes, dst_nodes = edges

        graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_residues)

        # 添加节点特征和坐标
        graph.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)
        graph.ndata['coord'] = torch.tensor(coords, dtype=torch.float32)

        # 加载pair_repr并提取边特征（在构图时完成，不在forward时）
        if self.pair_repr_dir is not None and target_id is not None:
            edge_feats = self._extract_edge_features_from_pair_repr(
                graph, target_id, num_residues, rank_idx
            )
            if edge_feats is not None:
                graph.edata['feat'] = edge_feats

        # 添加自环
        graph = dgl.add_self_loop(graph)

        # 如果有边特征，自环的边特征用零填充
        if 'feat' in graph.edata:
            num_edges_before = graph.num_edges() - num_residues  # 自环数量等于节点数
            num_self_loops = num_residues
            edge_feat_dim = graph.edata['feat'].shape[1]
            # 自环边的特征已经被add_self_loop自动处理为零了
            # 但DGL默认不会自动填充，需要手动处理
            # 获取当前边特征（不包括自环）
            current_edge_feats = graph.edata['feat'][:num_edges_before]
            # 创建自环的零特征
            self_loop_feats = torch.zeros(num_self_loops, edge_feat_dim, dtype=torch.float32)
            # 拼接
            graph.edata['feat'] = torch.cat([current_edge_feats, self_loop_feats], dim=0)

        return graph

    def _extract_edge_features_from_pair_repr(
            self,
            graph: dgl.DGLGraph,
            target_id: str,
            num_residues: int,
            rank_idx: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """
        从pair representation中提取边特征（在构图时完成）

        类似DTA-GTOmega的做法:
        protein_omega_edge_features = struct_edge[edges.T[:,0], edges.T[:,1]]

        Args:
            graph: 已构建的DGL图（有边但还没有边特征）
            target_id: 蛋白质ID
            num_residues: 残基数量

        Returns:
            edge_feats: [num_edges, 128] 边特征，如果加载失败返回None
        """
        if self.pair_repr_dir is None:
            return None

        # 搜索所有gpu目录
        gpu_dirs = []
        for subdir in self.pair_repr_dir.iterdir():
            if subdir.is_dir() and subdir.name.startswith('gpu'):
                gpu_dirs.append(subdir)

        if not gpu_dirs:
            gpu_dirs = [self.pair_repr_dir]

        # 兼容按target子目录存放
        search_dirs = []
        for base_dir in gpu_dirs:
            target_dir = base_dir / target_id
            if target_dir.is_dir():
                search_dirs.append(target_dir)
            if base_dir not in search_dirs:
                search_dirs.append(base_dir)

        patterns = []
        if rank_idx is not None:
            patterns.append(f"{target_id}_pair_repr_rank_{rank_idx:03d}_*.npy")
        patterns.append(f"{target_id}_pair_repr_rank_001_*.npy")
        patterns.append(f"{target_id}.npy")

        pair_repr_file = None
        for pattern in patterns:
            for search_dir in search_dirs:
                matching = list(search_dir.glob(pattern))
                if matching:
                    pair_repr_file = matching[0]
                    break
            if pair_repr_file is not None:
                break

        if pair_repr_file is None:
            logger.debug(f"Pair repr not found for {target_id}")
            return None

        # 加载pair_repr: [L, L, 128]
        pair_repr = np.load(pair_repr_file)
        L = pair_repr.shape[0]

        # 获取图的边
        src, dst = graph.edges()
        num_edges = graph.num_edges()

        # 对齐长度
        aligned_len = min(num_residues, L)

        # 向量化提取边特征
        edge_feats = np.zeros((num_edges, 128), dtype=np.float32)

        # 找出哪些边的两个端点都在pair_repr范围内
        src_np = src.numpy()
        dst_np = dst.numpy()
        valid_mask = (src_np < aligned_len) & (dst_np < aligned_len)

        if valid_mask.sum() > 0:
            valid_src = src_np[valid_mask]
            valid_dst = dst_np[valid_mask]
            # 向量化索引：pair_repr[src, dst, :] -> [num_valid_edges, 128]
            edge_feats[valid_mask] = pair_repr[valid_src, valid_dst, :]

        logger.debug(f"Extracted edge features for {target_id}: "
                     f"{valid_mask.sum()}/{num_edges} edges have pair_repr features")

        return torch.tensor(edge_feats, dtype=torch.float32)

    def _extract_residue_info(self, model) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """提取残基信息，包括pLDDT"""
        residues = []
        coords = []
        plddt_scores = []

        # 提取残基、Cα坐标和pLDDT
        for chain in model:
            for residue in chain:
                if is_aa(residue) and 'CA' in residue:
                    res_name = residue.get_resname()
                    if res_name in self.AMINO_ACIDS:
                        residues.append(res_name)
                        ca_atom = residue['CA']
                        ca_coord = ca_atom.get_coord()
                        coords.append(ca_coord)
                        # 从b-factor中提取pLDDT (ColabFold输出的PDB文件)
                        plddt_scores.append(ca_atom.bfactor)

        coords = np.array(coords)
        plddt_scores = np.array(plddt_scores)

        return residues, coords, plddt_scores

    def _predict_secondary_structure_simple(
            self,
            coords: np.ndarray
    ) -> np.ndarray:
        """
        简化的二级结构预测（基于局部几何）

        使用滑动窗口计算局部Cα距离和角度特征
        来粗略估计二级结构

        Args:
            coords: [num_residues, 3] Cα坐标

        Returns:
            ss_array: [num_residues, 3] (helix, sheet, coil)
        """
        num_residues = len(coords)
        ss_array = np.zeros((num_residues, 3), dtype=np.float32)

        if num_residues < 4:
            # 太短，全部标记为coil
            ss_array[:, 2] = 1.0
            return ss_array

        # 计算局部几何特征
        for i in range(num_residues):
            # 获取局部窗口
            start = max(0, i - 2)
            end = min(num_residues, i + 3)

            if end - start < 3:
                # 窗口太小，标记为coil
                ss_array[i, 2] = 1.0
                continue

            local_coords = coords[start:end]

            # 计算连续Cα距离
            distances = []
            for j in range(len(local_coords) - 1):
                dist = np.linalg.norm(local_coords[j + 1] - local_coords[j])
                distances.append(dist)

            avg_dist = np.mean(distances)
            std_dist = np.std(distances)

            # 简单规则判断：
            # α-helix: 连续Cα距离约3.8Å，规律性强
            # β-sheet: 连续Cα距离约3.3Å
            # coil: 其他

            if 3.6 <= avg_dist <= 4.0 and std_dist < 0.3:
                # 可能是helix
                ss_array[i, 0] = 0.7
                ss_array[i, 2] = 0.3
            elif 3.0 <= avg_dist <= 3.5 and std_dist < 0.4:
                # 可能是sheet
                ss_array[i, 1] = 0.6
                ss_array[i, 2] = 0.4
            else:
                # coil
                ss_array[i, 2] = 1.0

        # 平滑处理（避免单个残基的突变）
        ss_array_smoothed = np.copy(ss_array)
        window = 3
        for i in range(window, num_residues - window):
            ss_array_smoothed[i] = np.mean(
                ss_array[i - window:i + window + 1], axis=0
            )

        # 归一化
        row_sums = ss_array_smoothed.sum(axis=1, keepdims=True)
        ss_array_smoothed = ss_array_smoothed / (row_sums + 1e-8)

        return ss_array_smoothed

    def _load_colabfold_node_features(
            self,
            target_id: str,
            num_residues: int,
            rank_idx: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        加载ColabFold single representation作为节点特征

        文件格式: {target_id}_single_repr_rank_{rank:03d}_*.npy
        数据格式: (L, 256) float16

        Args:
            target_id: 蛋白质ID
            num_residues: 期望的残基数量（用于对齐）

        Returns:
            node_features: [num_residues, 256] 或 None（加载失败）
        """
        if self.colabfold_features_dir is None:
            return None

        # 搜索所有gpu目录
        gpu_dirs = []
        for subdir in self.colabfold_features_dir.iterdir():
            if subdir.is_dir() and subdir.name.startswith('gpu'):
                gpu_dirs.append(subdir)

        if not gpu_dirs:
            gpu_dirs = [self.colabfold_features_dir]

        # 兼容ColabFold特征按target子目录存放的结构（例如 Davis）
        search_dirs = []
        for base_dir in gpu_dirs:
            target_dir = base_dir / target_id
            if target_dir.is_dir():
                search_dirs.append(target_dir)
            if base_dir not in search_dirs:
                search_dirs.append(base_dir)

        # 搜索指定rank的single_repr文件
        patterns = []
        if rank_idx is not None:
            patterns.append(f"{target_id}_single_repr_rank_{rank_idx:03d}_*.npy")
        patterns.append(f"{target_id}_single_repr_rank_001_*.npy")

        found_file = None
        for pattern in patterns:
            for search_dir in search_dirs:
                matching = list(search_dir.glob(pattern))
                if matching:
                    found_file = matching[0]
                    break
            if found_file is not None:
                break

        if found_file is None:
            logger.debug(f"ColabFold single repr not found for {target_id}")
            return None

        try:
            # 加载single repr: (L, 256)，转换为float32
            single_repr = np.load(found_file).astype(np.float32)
            L = single_repr.shape[0]

            # 对齐长度
            if num_residues <= L:
                node_features = single_repr[:num_residues]
            else:
                # 蛋白质图节点数多于single_repr长度，用最后一个残基的特征填充
                node_features = np.zeros((num_residues, 256), dtype=np.float32)
                node_features[:L] = single_repr
                node_features[L:] = single_repr[-1]

            logger.debug(f"Loaded ColabFold features for {target_id}: {node_features.shape}")
            return node_features

        except Exception as e:
            logger.warning(f"Failed to load ColabFold features for {target_id}: {e}")
            return None

    def _build_node_features(
            self,
            residues: List[str],
            ss_info: np.ndarray,
            plddt_scores: np.ndarray
    ) -> np.ndarray:
        """
        构建节点特征

        Args:
            residues: 氨基酸名称列表
            ss_info: 二级结构信息 [num_residues, 3]
            plddt_scores: pLDDT置信度分数 [num_residues]，范围0-100

        Returns:
            features: [num_nodes, node_dim]
                - 21维: 氨基酸类型
                - 3维: 二级结构
                - 1维: pLDDT (归一化到0-1)
                - 1维: 位置编码
                - 5维: 物理化学性质
        """
        num_residues = len(residues)
        node_dim = 31
        features = np.zeros((num_residues, node_dim), dtype=np.float32)

        for i, res_name in enumerate(residues):
            offset = 0

            # 1. 氨基酸类型 (21维)
            if res_name in self.AMINO_ACIDS:
                aa_idx = self.AMINO_ACIDS.index(res_name)
                features[i, aa_idx] = 1.0
            else:
                features[i, 20] = 1.0  # UNK
            offset += 21

            # 2. 二级结构 (3维)
            features[i, offset:offset + 3] = ss_info[i]
            offset += 3

            # 3. pLDDT (1维) - 从b-factor提取，归一化到0-1
            if self.use_plddt:
                # pLDDT范围是0-100，归一化到0-1
                features[i, offset] = plddt_scores[i] / 100.0
            else:
                features[i, offset] = 0.0
            offset += 1

            # 4. 位置编码 (1维)
            features[i, offset] = i / num_residues
            offset += 1

            # 5. 物理化学性质 (5维)
            properties = [
                self.AA_PROPERTIES['hydrophobicity'].get(res_name, 0.0),
                self.AA_PROPERTIES['charge'].get(res_name, 0.0),
                self.AA_PROPERTIES['polarity'].get(res_name, 0.0),
                self.AA_PROPERTIES['aromaticity'].get(res_name, 0.0),
                self.AA_PROPERTIES['volume'].get(res_name, 0.5)
            ]
            features[i, offset:offset + 5] = properties

        return features

    def _build_edges_from_distance(
            self,
            coords: np.ndarray,
            num_residues: int
    ) -> Tuple[List[int], List[int]]:
        """
        基于距离阈值构建边（原有逻辑）

        Args:
            coords: [num_residues, 3] Cα坐标
            num_residues: 残基数量

        Returns:
            (src_nodes, dst_nodes): 边的源节点和目标节点列表
        """
        # 计算距离矩阵
        dist_matrix = np.linalg.norm(
            coords[:, np.newaxis, :] - coords[np.newaxis, :, :],
            axis=-1
        )

        src_nodes = []
        dst_nodes = []
        seen = set()

        for i in range(num_residues):
            dists = dist_matrix[i]
            # 候选邻居（排除自身）
            candidates = np.where((dists < self.contact_threshold) & (dists > 0))[0]
            if len(candidates) == 0:
                continue

            # 可选 top-k 裁剪
            if self.top_k is not None and self.top_k > 0:
                order = np.argsort(dists[candidates])
                candidates = candidates[order[:min(self.top_k, len(candidates))]]

            for j in candidates:
                a, b = sorted((int(i), int(j)))
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)
                # 添加双向边
                src_nodes.extend([a, b])
                dst_nodes.extend([b, a])

        return src_nodes, dst_nodes

    def _build_edges_from_contact_map(
            self,
            target_id: str,
            coords: np.ndarray,
            num_residues: int,
            rank_idx: Optional[int] = None
    ) -> Tuple[List[int], List[int]]:
        """
        基于预计算的contact map构建边

        Args:
            target_id: 蛋白质ID
            num_residues: PDB文件中解析出的残基数量

        Returns:
            (src_nodes, dst_nodes): 边的源节点和目标节点列表
        """
        if self.contact_map_dir is None:
            logger.warning(f"contact_map_dir is None, falling back to distance-based edges")
            return self._build_edges_from_distance(coords, num_residues)

        # 尝试加载contact map文件（优先rank特定）
        contact_map = None
        candidates = []
        if rank_idx is not None:
            candidates.extend([
                f"{target_id}_rank{rank_idx}.npy",
                f"{target_id}_rank_{rank_idx:03d}.npy",
                f"{target_id}_rank{rank_idx:03d}.npy"
            ])
        candidates.extend([f"{target_id}.npy", f"{target_id}_0.npy"])

        for name in candidates:
            contact_file = self.contact_map_dir / name
            if contact_file.exists():
                contact_map = np.load(contact_file)
                break

        if contact_map is None:
            logger.warning(f"Contact map not found for {target_id}, falling back to distance-based edges")
            return self._build_edges_from_distance(coords, num_residues)

        # contact map是bool类型，True表示有接触
        # 去掉对角线
        contact_map_no_diag = contact_map.copy().astype(bool)
        np.fill_diagonal(contact_map_no_diag, False)

        # 对齐长度：取contact map和PDB残基数的较小值
        min_len = min(num_residues, contact_map_no_diag.shape[0])
        contact_map_no_diag = contact_map_no_diag[:min_len, :min_len]

        # 提取边
        edges = np.argwhere(contact_map_no_diag)

        src_nodes = []
        dst_nodes = []
        seen = set()

        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            a, b = sorted((i, j))
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            # 添加双向边
            src_nodes.extend([a, b])
            dst_nodes.extend([b, a])

        logger.debug(f"Built {len(src_nodes) // 2} edges from contact map for {target_id} "
                     f"(contact_map: {contact_map.shape}, pdb_residues: {num_residues}, used: {min_len})")

        return src_nodes, dst_nodes
