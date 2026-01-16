# data/preprocessing/drug_graph.py

import numpy as np
import torch
import dgl
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class Drug3DGraphBuilder:
    """
    药物3D图构建器

    节点特征:
        - 150维: 丰富原子特征（默认）
        - 78维: DTA-GTOmega风格原子特征

    边特征 (10维):
        - 键类型 (4维: single, double, triple, aromatic)
        - 是否共轭 (1维)
        - 是否在环中 (1维)
        - 立体化学 (4维)
    """

    def __init__(
            self,
            edge_threshold: float = 5.0,
            max_conformers: int = 1,  # 只生成一个构象
            energy_minimize: bool = False,  # 禁用能量最小化
            node_feature_dim: int = 150,
            use_spatial_edges: bool = True,
            align_gtomega: bool = False
    ):
        self.edge_threshold = edge_threshold
        self.max_conformers = max_conformers
        self.energy_minimize = energy_minimize
        self.node_feature_dim = int(node_feature_dim)
        self.use_spatial_edges = bool(use_spatial_edges)
        self.align_gtomega = bool(align_gtomega)
        if self.align_gtomega:
            self._suppress_rdkit_warnings()
        """
        Args:
            edge_threshold: 空间边距离阈值 (Å)
            max_conformers: 最大构象数
            energy_minimize: 是否进行能量最小化
            node_feature_dim: 节点特征维度 (150 或 78)
            use_spatial_edges: 是否添加空间邻近边（默认True）
            align_gtomega: 对齐DTA-GTOmega药物图（不加H、仅化学键边）
        """

    @staticmethod
    def _suppress_rdkit_warnings():
        # DTA-GTOmega对齐时不加显式H，关闭RDKit提示以避免刷屏。
        try:
            RDLogger.DisableLog('rdApp.warning')
        except Exception:
            pass

    def build_graph(self, smiles: str) -> Optional[dgl.DGLGraph]:

        try:
            # 从SMILES创建分子
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error(f"Invalid SMILES: {smiles}")
                return None

            # 对齐DTA-GTOmega时不添加显式氢
            if not self.align_gtomega:
                mol = Chem.AddHs(mol)
            num_atoms = mol.GetNumAtoms()

            # 生成简单的3D坐标（不使用复杂的构象生成）
            try:
                # 使用ETKDG方法生成3D坐标
                params = AllChem.ETKDGv3()
                params.randomSeed = 42
                params.numThreads = 0
                AllChem.EmbedMolecule(mol, params)

                # 如果失败，使用更简单的方法
                if mol.GetNumConformers() == 0:
                    AllChem.Compute2DCoords(mol)
            except Exception as e:
                logger.warning(f"Coordinate generation failed for {smiles}: {e}")
                # 使用随机坐标作为备选
                num_atoms = mol.GetNumAtoms()
                coords = np.random.rand(num_atoms, 3).astype(np.float32)
                mol.RemoveAllConformers()
                conf = Chem.Conformer(num_atoms)
                for i in range(num_atoms):
                    conf.SetAtomPosition(i, Chem.Point3D(coords[i, 0], coords[i, 1], coords[i, 2]))
                mol.AddConformer(conf)

            # 获取坐标
            if mol.GetNumConformers() > 0:
                conformer = mol.GetConformer(0)
                coords = conformer.GetPositions()
            else:
                # 如果还是没有构象，使用随机坐标
                num_atoms = mol.GetNumAtoms()
                coords = np.random.rand(num_atoms, 3).astype(np.float32)

            # 构建节点特征
            node_features = self._build_atom_features(mol)

            # 构建边
            use_spatial_edges = self.use_spatial_edges and not self.align_gtomega
            if use_spatial_edges:
                src_nodes, dst_nodes, edge_features = self._build_edges(mol, coords)
            else:
                src_nodes, dst_nodes, edge_features = self._build_simple_edges(mol)

            # 创建图
            graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_atoms)

            # 添加特征
            graph.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)
            graph.ndata['coord'] = torch.tensor(coords, dtype=torch.float32)
            graph.edata['feat'] = torch.tensor(edge_features, dtype=torch.float32)

            # 对齐GTOmega药物图时不加自环
            if not self.align_gtomega:
                graph = dgl.add_self_loop(graph)

            return graph

        except Exception as e:
            logger.error(f"Graph building failed for {smiles}: {e}")
            return None

    def _build_simple_edges(self, mol: Chem.Mol) -> Tuple[list, list, np.ndarray]:
        """
        只构建化学键边，避免空间邻近边
        """
        src_nodes = []
        dst_nodes = []
        edge_features = []

        # 只考虑化学键边
        for bond in mol.GetBonds():
            i = int(bond.GetBeginAtomIdx())
            j = int(bond.GetEndAtomIdx())

            feat = self._get_bond_features(bond)

            # 无向图，添加双向边
            src_nodes.extend([i, j])
            dst_nodes.extend([j, i])
            edge_features.extend([feat, feat])

        # 如果没有边，按对齐策略处理
        if len(src_nodes) == 0:
            if self.align_gtomega:
                return [], [], np.zeros((0, 10), dtype=np.float32)
            num_atoms = mol.GetNumAtoms()
            src_nodes = list(range(num_atoms))
            dst_nodes = list(range(num_atoms))
            edge_features = [np.zeros(10, dtype=np.float32) for _ in range(num_atoms)]

        edge_features = np.array(edge_features, dtype=np.float32)

        return src_nodes, dst_nodes, edge_features


    def _build_atom_features(self, mol: Chem.Mol) -> np.ndarray:
        """
        构建原子特征
        """
        num_atoms = mol.GetNumAtoms()
        features = []

        for atom in mol.GetAtoms():
            if self.node_feature_dim == 78:
                feat = self._get_atom_features_gto(atom)
            else:
                feat = self._get_atom_features(atom)
            features.append(feat)

        return np.array(features, dtype=np.float32)

    @staticmethod
    def _one_of_k_encoding(value, allowable_set):
        return [value == s for s in allowable_set]

    @staticmethod
    def _one_of_k_encoding_unk(value, allowable_set):
        if value not in allowable_set:
            value = allowable_set[-1]
        return [value == s for s in allowable_set]

    @staticmethod
    def _get_implicit_valence(atom: Chem.Atom) -> int:
        # DTA-GTOmega uses GetImplicitValence() directly.
        return atom.GetImplicitValence()

    def _get_atom_features_gto(self, atom: Chem.Atom) -> np.ndarray:
        """
        DTA-GTOmega风格原子特征 (78维)
        """
        symbol_set = [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
            'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag',
            'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd',
            'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
        ]

        features = (
            self._one_of_k_encoding_unk(atom.GetSymbol(), symbol_set) +
            self._one_of_k_encoding(atom.GetDegree(), list(range(0, 11))) +
            self._one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(0, 11))) +
            self._one_of_k_encoding_unk(self._get_implicit_valence(atom), list(range(0, 11))) +
            [atom.GetIsAromatic()]
        )
        return np.array(features, dtype=np.float32)

    def _get_atom_features(self, atom: Chem.Atom) -> np.ndarray:
        """
        单个原子特征 (150维)

        包括:
            - 原子类型 (100维 one-hot)
            - 度数 (10维)
            - 形式电荷 (1维)
            - 杂化类型 (7维)
            - 芳香性 (1维)
            - 氢原子数 (10维)
            - 隐式氢数 (10维)
            - 是否在环中 (1维)
            - 其他 (10维)
        """
        # 原子类型 (100维 one-hot)
        atom_type = [0] * 100
        atomic_num = atom.GetAtomicNum()
        if atomic_num < 100:
            atom_type[atomic_num] = 1

        # 度数 (10维 one-hot, 最大度数=10)
        degree = [0] * 10
        d = min(atom.GetDegree(), 9)
        degree[d] = 1

        # 形式电荷
        formal_charge = [atom.GetFormalCharge()]

        # 杂化类型 (7维)
        hybridization = [0] * 7
        hyb_type = atom.GetHybridization()
        hyb_map = {
            Chem.HybridizationType.S: 0,
            Chem.HybridizationType.SP: 1,
            Chem.HybridizationType.SP2: 2,
            Chem.HybridizationType.SP3: 3,
            Chem.HybridizationType.SP3D: 4,
            Chem.HybridizationType.SP3D2: 5,
        }
        hyb_idx = hyb_map.get(hyb_type, 6)
        hybridization[hyb_idx] = 1

        # 芳香性
        aromatic = [int(atom.GetIsAromatic())]

        # 氢原子数 (10维)
        num_hs = [0] * 10
        hs = min(atom.GetTotalNumHs(), 9)
        num_hs[hs] = 1

        # 隐式氢数 (10维)
        implicit_hs = [0] * 10
        impl_hs = min(atom.GetNumImplicitHs(), 9)
        implicit_hs[impl_hs] = 1

        # 是否在环中
        in_ring = [int(atom.IsInRing())]

        # 其他特征 (10维)
        other = [
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            atom.GetMass() / 100.0,  # 归一化质量
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            int(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED),
            0, 0, 0, 0  # 预留
        ]

        # 拼接所有特征
        feat = (
                atom_type +
                degree +
                formal_charge +
                hybridization +
                aromatic +
                num_hs +
                implicit_hs +
                in_ring +
                other
        )

        return np.array(feat, dtype=np.float32)

    def _build_edges(
            self,
            mol: Chem.Mol,
            coords: np.ndarray
    ) -> Tuple[list, list, np.ndarray]:
        """
        构建边和边特征
        """
        src_nodes = []
        dst_nodes = []
        edge_features = []

        # 1. 化学键边 - 修复索引类型
        for bond in mol.GetBonds():
            i = int(bond.GetBeginAtomIdx())  # 确保转换为int
            j = int(bond.GetEndAtomIdx())  # 确保转换为int

            feat = self._get_bond_features(bond)

            # 无向图，添加双向边
            src_nodes.extend([i, j])
            dst_nodes.extend([j, i])
            edge_features.extend([feat, feat])

        # 2. 空间邻近边
        num_atoms = mol.GetNumAtoms()
        dist_matrix = np.linalg.norm(
            coords[:, np.newaxis, :] - coords[np.newaxis, :, :],
            axis=-1
        )

        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                if dist_matrix[i, j] < self.edge_threshold:
                    # 检查是否已有化学键
                    bond = mol.GetBondBetweenAtoms(i, j)
                    if bond is None:
                        # 空间边特征 (全0表示非化学键)
                        feat = np.zeros(10, dtype=np.float32)

                        src_nodes.extend([int(i), int(j)])  # 确保int类型
                        dst_nodes.extend([int(j), int(i)])  # 确保int类型
                        edge_features.extend([feat, feat])

        edge_features = np.array(edge_features, dtype=np.float32)

        return src_nodes, dst_nodes, edge_features

    def _get_bond_features(self, bond: Chem.Bond) -> np.ndarray:
        """
        键特征 (10维) - 和DTA-GTOmega一致

        包括:
            - 键类型 (4维: single, double, triple, aromatic)
            - 是否共轭 (1维)
            - 是否在环中 (1维)
            - 立体化学 (4维 one-hot: STEREONONE, STEREOANY, STEREOZ, STEREOE)
        """
        bond_type = [
            bond.GetBondType() == Chem.rdchem.BondType.SINGLE,
            bond.GetBondType() == Chem.rdchem.BondType.DOUBLE,
            bond.GetBondType() == Chem.rdchem.BondType.TRIPLE,
            bond.GetBondType() == Chem.rdchem.BondType.AROMATIC,
        ]
        bond_feats = bond_type + [bond.GetIsConjugated(), bond.IsInRing()]
        bond_feats += self._one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
        )
        return np.array(bond_feats, dtype=np.float32)
