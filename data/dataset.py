# data/dataset.py

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import dgl
from typing import Dict, List, Optional, Tuple
import logging
import pickle
from pathlib import Path
import hashlib

from data.preprocessing.protein_graph import ProteinGraphBuilder
from data.preprocessing.drug_graph import Drug3DGraphBuilder
from data.plddt_selector import get_plddt_from_pdb, select_conformers_by_plddt

logger = logging.getLogger(__name__)

# 全局缓存，用于跨 Dataset 实例共享（同一进程内）
_GLOBAL_PROTEIN_CACHE: Dict[str, Dict] = {}
_GLOBAL_DRUG_CACHE: Dict[str, Dict] = {}
_GLOBAL_PAIR_REPR_CACHE: Dict[str, Dict] = {}  # ColabFold pair representation缓存
_GLOBAL_COLABFOLD_CACHE: Dict[str, Dict] = {}  # ColabFold single representation缓存
_GLOBAL_CACHE_LOADED: bool = False


def _load_merged_cache(cache_dir: Path) -> Tuple[Dict, Dict]:
    """
    加载合并后的缓存文件（全局单例）

    Returns:
        (protein_cache, drug_cache)
        protein_cache: {target_id: {rank_idx: graph}}
        drug_cache: {drug_id: graph}
    """
    global _GLOBAL_PROTEIN_CACHE, _GLOBAL_DRUG_CACHE, _GLOBAL_CACHE_LOADED

    if _GLOBAL_CACHE_LOADED:
        return _GLOBAL_PROTEIN_CACHE, _GLOBAL_DRUG_CACHE

    merged_protein = cache_dir / "merged_protein_cache.pkl"
    merged_drug = cache_dir / "merged_drug_cache.pkl"

    if merged_protein.exists():
        logger.info(f"Loading merged protein cache from {merged_protein}...")
        with open(merged_protein, 'rb') as f:
            _GLOBAL_PROTEIN_CACHE = pickle.load(f)
        logger.info(f"Loaded {len(_GLOBAL_PROTEIN_CACHE)} proteins")

    if merged_drug.exists():
        logger.info(f"Loading merged drug cache from {merged_drug}...")
        with open(merged_drug, 'rb') as f:
            _GLOBAL_DRUG_CACHE = pickle.load(f)
        logger.info(f"Loaded {len(_GLOBAL_DRUG_CACHE)} drugs")

    _GLOBAL_CACHE_LOADED = True
    return _GLOBAL_PROTEIN_CACHE, _GLOBAL_DRUG_CACHE


def _atomic_pickle_dump(obj, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)


class MultiConformerDTADataset(Dataset):
    """
    支持多构象的药物-蛋白质亲和力数据集

    CSV格式: Drug_ID,Drug,Target_ID,Target,Y

    每个蛋白质需要5个构象（按能量排序）:
        - rank1.pdb (最稳定)
        - rank2.pdb
        - rank3.pdb
        - rank4.pdb
        - rank5.pdb (次稳定)
    """

    def __init__(
            self,
            csv_file: str,
            protein_conformer_dir: str,
            config: Dict,
            cache_dir: Optional[str] = None,
            use_cache: bool = True,
            label_mean: Optional[float] = None,
            label_std: Optional[float] = None,
            pair_repr_dir: Optional[str] = None,
            colabfold_features_dir: Optional[str] = None,
            **kwargs  # 忽略其他参数（向后兼容）
    ):
        """
        Args:
            csv_file: CSV文件路径
            protein_conformer_dir: 蛋白质构象目录
            config: 配置字典
            cache_dir: 缓存目录
            use_cache: 是否使用缓存
            pair_repr_dir: ColabFold pair representation目录（包含128维边特征）
            colabfold_features_dir: ColabFold single representation目录（256维节点特征）
        """
        self.data_df = pd.read_csv(csv_file)
        self.protein_conformer_dir = Path(protein_conformer_dir)
        self.config = config
        self.use_cache = use_cache
        self.label_mean = label_mean
        self.label_std = label_std

        # ColabFold pair representation目录（128维边特征）
        self.pair_repr_dir = Path(pair_repr_dir) if pair_repr_dir else None
        # ColabFold single representation目录（256维节点特征）
        self.colabfold_features_dir = Path(colabfold_features_dir) if colabfold_features_dir else None

        # 设置缓存目录
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        # 图构建器
        protein_cfg = config['graph_construction']['protein']
        self.use_contact_map = protein_cfg.get('use_contact_map', False)
        self.contact_map_dir = protein_cfg.get('contact_map_dir', None)
        if self.contact_map_dir:
            self.contact_map_dir = Path(self.contact_map_dir)
        self.max_protein_len = protein_cfg.get('max_protein_len', 800)
        self.max_protein_conformers = int(protein_cfg.get('max_conformers', 1))
        self.use_multi_conformer = bool(protein_cfg.get('use_multi_conformer', False))
        selection_cfg = protein_cfg.get('conformer_selection', {}) or {}
        self.use_plddt_selection = bool(selection_cfg.get('enabled', False))
        self.plddt_std_factor = float(selection_cfg.get('std_factor', protein_cfg.get('std_factor', 1.0)))
        self.plddt_fallback_to_all = bool(selection_cfg.get('fallback_to_all', True))
        self.plddt_use_cache = bool(selection_cfg.get('use_cache', True))
        self.plddt_selection_mode = str(selection_cfg.get('mode', 'threshold')).lower()
        self.diversity_max_similarity = float(selection_cfg.get('diversity_max_similarity', 0.9))
        self.plddt_scores_by_target: Dict[str, List[Tuple[int, float]]] = {}

        self.protein_builder = ProteinGraphBuilder(
            contact_threshold=protein_cfg['contact_threshold'],
            secondary_structure_window=protein_cfg['secondary_structure_window'],
            use_plddt=protein_cfg['use_plddt'],
            top_k=protein_cfg.get('top_k', None),
            use_contact_map=self.use_contact_map,
            contact_map_dir=self.contact_map_dir,
            pair_repr_dir=pair_repr_dir,  # 在构图时直接提取边特征
            max_protein_len=self.max_protein_len,
            colabfold_features_dir=colabfold_features_dir
        )

        drug_node_dim = config['features']['drug']['node_dim']
        align_gtomega = bool(config.get('model', {}).get('align_gtomega', False))
        self.drug_builder = Drug3DGraphBuilder(
            edge_threshold=config['graph_construction']['drug']['edge_threshold'],
            max_conformers=config['graph_construction']['drug']['max_conformers'],
            energy_minimize=config['graph_construction']['drug']['energy_minimize'],
            node_feature_dim=drug_node_dim,
            use_spatial_edges=config['graph_construction']['drug'].get('use_spatial_edges', True),
            align_gtomega=align_gtomega
        )

        # 预处理所有蛋白质构象
        logger.info("Preprocessing protein conformers...")
        self.protein_conformers_dict = {}
        self.global_plddt_std = None  # 仅在需要时计算
        if self.use_multi_conformer and self.use_plddt_selection:
            self._prepare_plddt_scores(self.data_df['Target_ID'].unique())
        self._preprocess_protein_conformers()

        # 预加载所有药物图到内存（大幅加速训练）
        logger.info("Preprocessing drug graphs...")
        self.drug_graphs_dict = {}
        self._preprocess_drug_graphs()

        # pair_repr边特征已在构图时直接提取到graph.edata['feat']，无需单独加载
        # ProteinGraphBuilder会根据pair_repr_dir自动提取边特征

        logger.info(f"Dataset loaded: {len(self.data_df)} samples")

    def _preprocess_protein_conformers(self):
        """预处理蛋白质构象（支持多构象）"""
        unique_target_ids = self.data_df['Target_ID'].unique()
        logger.info(f"Preprocessing {len(unique_target_ids)} proteins for this dataset...")

        if self.use_multi_conformer:
            default_ranks = list(range(1, self.max_protein_conformers + 1))
        else:
            default_ranks = [1]

        # 尝试从合并缓存加载（仅当不使用contact_map时）
        if self.cache_dir and self.use_cache and not self.use_contact_map:
            merged_cache, _ = _load_merged_cache(self.cache_dir)
            if merged_cache:
                loaded_count = 0
                for target_id in unique_target_ids:
                    if target_id in merged_cache:
                        rank_data = merged_cache[target_id]
                        conformer_graphs = []
                        selected_ranks = self._get_selected_ranks(target_id, default_ranks)
                        for rank_idx in selected_ranks:
                            if rank_idx not in rank_data:
                                continue
                            cached = rank_data[rank_idx]
                            graph = cached[0] if isinstance(cached, tuple) else cached
                            score = self._get_plddt_score(target_id, rank_idx)
                            conformer_graphs.append((graph, score))
                        if len(conformer_graphs) >= 1:
                            self.protein_conformers_dict[target_id] = conformer_graphs
                            loaded_count += 1
                logger.info(f"Loaded {loaded_count}/{len(unique_target_ids)} proteins from merged cache")
                # 如果全部加载成功，直接返回
                if loaded_count == len(unique_target_ids):
                    return

        for target_id in unique_target_ids:
            target_conformer_dir = self.protein_conformer_dir / str(target_id)

            selected_ranks = self._get_selected_ranks(target_id, default_ranks)
            if not selected_ranks:
                logger.warning(f"No conformers selected for {target_id}, skipping")
                continue

            conformer_graphs = []
            loaded_ranks = set()

            # 先尝试从缓存加载（即使没有PDB目录也可以）
            if self.cache_dir and self.use_cache:
                for rank_idx in selected_ranks:
                    cache_file = self.cache_dir / f"protein_{target_id}_rank{rank_idx}.pkl"
                    if cache_file.exists():
                        try:
                            with open(cache_file, 'rb') as f:
                                cached = pickle.load(f)

                            graph = cached[0] if isinstance(cached, tuple) else cached
                            score = self._get_plddt_score(target_id, rank_idx)
                            conformer_graphs.append((graph, score))
                            loaded_ranks.add(rank_idx)
                        except Exception as e:
                            logger.warning(f"Failed to load cache {cache_file}: {e}")

            if not target_conformer_dir.exists():
                if len(conformer_graphs) >= 1:
                    self.protein_conformers_dict[target_id] = conformer_graphs
                else:
                    logger.warning(f"Target directory not found: {target_conformer_dir}")
                continue

            for rank_idx in selected_ranks:  # 只加载选中的conformer
                if rank_idx in loaded_ranks:
                    continue

                cache_file = None
                if self.cache_dir:
                    cache_file = self.cache_dir / f"protein_{target_id}_rank{rank_idx}.pkl"

                # 缓存不存在或加载失败，重新构建图
                pdb_file = self._find_pdb_file(target_conformer_dir, target_id, rank_idx)
                if pdb_file is None:
                    continue

                try:
                    graph = self.protein_builder.build_graph(
                        str(pdb_file),
                        target_id=target_id,
                        rank_idx=rank_idx
                    )
                    score = self._get_plddt_score(target_id, rank_idx)
                    conformer_graphs.append((graph, score))

                    # 保存缓存
                    if cache_file is not None:
                        with open(cache_file, 'wb') as f:
                            pickle.dump(graph, f)
                        logger.info(f"Saved cache: {cache_file.name}")

                except Exception as e:
                    logger.error(f"Failed to build graph for {pdb_file}: {e}")
                    continue

            # 保存成功加载的conformer（至少1个）
            if len(conformer_graphs) >= 1:
                self.protein_conformers_dict[target_id] = conformer_graphs
            else:
                logger.warning(f"Failed to load conformer for {target_id}, skipping")

    def _prepare_plddt_scores(self, target_ids: List[str]):
        """
        预计算每个蛋白质的pLDDT分数（用于动态选择构象）
        """
        cache_file = None
        if self.cache_dir and self.use_cache and self.plddt_use_cache:
            cache_file = self.cache_dir / "plddt_scores.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cached = pickle.load(f)
                    if isinstance(cached, dict) and 'scores_by_target' in cached:
                        scores_by_target = cached.get('scores_by_target', {})
                        cached_std = cached.get('global_std', None)
                        cached_max = cached.get('max_conformers', None)
                        if cached_max is None or cached_max == self.max_protein_conformers:
                            self.plddt_scores_by_target = scores_by_target
                            if cached_std is None:
                                cached_std = self._compute_global_std(scores_by_target)
                            self.global_plddt_std = cached_std
                            logger.info(
                                f"Loaded cached pLDDT scores: {len(scores_by_target)} proteins, "
                                f"std={self.global_plddt_std:.2f}"
                            )
                            return
                    elif isinstance(cached, dict):
                        # 兼容旧格式：直接是scores_by_target
                        self.plddt_scores_by_target = cached
                        self.global_plddt_std = self._compute_global_std(cached)
                        logger.info(
                            f"Loaded cached pLDDT scores: {len(cached)} proteins, "
                            f"std={self.global_plddt_std:.2f}"
                        )
                        return
                except Exception as e:
                    logger.warning(f"Failed to load pLDDT cache {cache_file}: {e}")

        scores_by_target: Dict[str, List[Tuple[int, float]]] = {}
        all_scores: List[float] = []

        for target_id in target_ids:
            target_conformer_dir = self.protein_conformer_dir / str(target_id)
            if not target_conformer_dir.exists():
                continue

            target_scores: List[Tuple[int, float]] = []
            for rank_idx in range(1, self.max_protein_conformers + 1):
                pdb_file = self._find_pdb_file(
                    target_conformer_dir, target_id, rank_idx, warn_on_missing=False
                )
                if pdb_file is None:
                    continue
                score = get_plddt_from_pdb(pdb_file)
                if score > 0:
                    target_scores.append((rank_idx, score))
                    all_scores.append(score)

            if target_scores:
                scores_by_target[target_id] = target_scores

        self.plddt_scores_by_target = scores_by_target
        if all_scores:
            global_std = float(np.std(all_scores))
            global_mean = float(np.mean(all_scores))
            logger.info(f"pLDDT统计: mean={global_mean:.2f}, std={global_std:.2f}, n={len(all_scores)}")
        else:
            logger.warning("无法读取pLDDT分数，使用默认全局标准差 5.0")
            global_std = 5.0

        self.global_plddt_std = global_std

        if cache_file:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(
                        {
                            'scores_by_target': scores_by_target,
                            'global_std': global_std,
                            'max_conformers': self.max_protein_conformers
                        },
                        f
                    )
                logger.info(f"Saved pLDDT cache: {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save pLDDT cache {cache_file}: {e}")

    @staticmethod
    def _compute_global_std(scores_by_target: Dict[str, List[Tuple[int, float]]]) -> float:
        scores = [score for target_scores in scores_by_target.values() for _, score in target_scores]
        if not scores:
            return 5.0
        return float(np.std(scores))

    def _load_contact_map_for_rank(self, target_id: str, rank_idx: int) -> Optional[np.ndarray]:
        if self.contact_map_dir is None:
            return None
        candidates = [
            f"{target_id}_rank{rank_idx}.npy",
            f"{target_id}_rank_{rank_idx:03d}.npy",
            f"{target_id}_rank{rank_idx:03d}.npy",
        ]
        for name in candidates:
            contact_file = self.contact_map_dir / name
            if contact_file.exists():
                return np.load(contact_file).astype(bool)
        return None

    def _contact_map_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        min_len = min(a.shape[0], b.shape[0])
        if min_len == 0:
            return 0.0
        a_sub = a[:min_len, :min_len]
        b_sub = b[:min_len, :min_len]
        np.fill_diagonal(a_sub, False)
        np.fill_diagonal(b_sub, False)
        inter = np.logical_and(a_sub, b_sub).sum()
        union = np.logical_or(a_sub, b_sub).sum()
        if union == 0:
            return 0.0
        return float(inter / union)

    def _select_diverse_ranks(self, target_id: str, default_ranks: List[int]) -> List[int]:
        scores = self.plddt_scores_by_target.get(target_id, [])
        if not scores:
            return default_ranks if self.plddt_fallback_to_all else []

        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        contact_maps: Dict[int, np.ndarray] = {}
        for rank_idx, _ in sorted_scores:
            cm = self._load_contact_map_for_rank(target_id, rank_idx)
            if cm is not None:
                contact_maps[rank_idx] = cm

        if not contact_maps:
            return default_ranks if self.plddt_fallback_to_all else []

        selected: List[int] = []
        for rank_idx, _ in sorted_scores:
            if len(selected) >= self.max_protein_conformers:
                break
            if rank_idx not in contact_maps:
                continue
            if not selected:
                selected.append(rank_idx)
                continue
            max_sim = 0.0
            for sel in selected:
                if sel not in contact_maps:
                    continue
                sim = self._contact_map_similarity(contact_maps[rank_idx], contact_maps[sel])
                if sim > max_sim:
                    max_sim = sim
            if max_sim <= self.diversity_max_similarity:
                selected.append(rank_idx)

        if not selected and self.plddt_fallback_to_all:
            return default_ranks
        if len(selected) < 1 and sorted_scores:
            selected = [sorted_scores[0][0]]
        return sorted(selected)

    def _get_selected_ranks(self, target_id: str, default_ranks: List[int]) -> List[int]:
        if not self.use_multi_conformer:
            return default_ranks
        if not self.use_plddt_selection:
            return default_ranks
        if self.plddt_selection_mode == 'soft':
            return default_ranks
        if self.plddt_selection_mode == 'diverse':
            return self._select_diverse_ranks(target_id, default_ranks)

        scores = self.plddt_scores_by_target.get(target_id, [])
        if not scores:
            return default_ranks if self.plddt_fallback_to_all else []

        global_std = self.global_plddt_std if self.global_plddt_std is not None else 5.0
        selected = select_conformers_by_plddt(
            scores,
            global_std=global_std,
            std_factor=self.plddt_std_factor,
            max_conformers=self.max_protein_conformers
        )
        if not selected and self.plddt_fallback_to_all:
            return default_ranks
        return selected

    def _get_plddt_score(self, target_id: str, rank_idx: int) -> float:
        scores = self.plddt_scores_by_target.get(target_id, [])
        for rank, score in scores:
            if rank == rank_idx:
                return float(score)
        return 0.0

    @staticmethod
    def _find_pdb_file(
        target_conformer_dir: Path,
        target_id: str,
        rank_idx: int,
        warn_on_missing: bool = True
    ):
        """查找指定rank的PDB文件"""
        rank_str = f"{rank_idx:03d}"  # 转换为三位数，如 001, 002
        pattern = f"{target_id}_unrelaxed_rank_{rank_str}_alphafold2_ptm_model_*_seed_000.pdb"
        matching_files = list(target_conformer_dir.glob(pattern))

        if not matching_files:
            if warn_on_missing:
                logger.warning(f"Missing conformer for pattern: {pattern}")
            return None

        if len(matching_files) > 1:
            logger.warning(f"Multiple files match pattern {pattern}, using first one: {matching_files[0]}")

        return matching_files[0]

    def _preprocess_drug_graphs(self):
        """预加载所有唯一药物的图到内存，避免每次getitem都重新构建"""
        unique_drugs = self.data_df[['Drug_ID', 'Drug']].drop_duplicates()
        built = 0
        failed = 0

        # 尝试从合并缓存加载
        if self.cache_dir and self.use_cache:
            _, merged_drug_cache = _load_merged_cache(self.cache_dir)
            if merged_drug_cache:
                loaded_count = 0
                for _, row in unique_drugs.iterrows():
                    drug_id = str(row['Drug_ID'])
                    if drug_id in merged_drug_cache:
                        self.drug_graphs_dict[drug_id] = merged_drug_cache[drug_id]
                        loaded_count += 1
                logger.info(f"Loaded {loaded_count}/{len(unique_drugs)} drugs from merged cache")
                if loaded_count == len(unique_drugs):
                    return

        for _, row in unique_drugs.iterrows():
            drug_id = row['Drug_ID']
            smiles = row['Drug']

            # 已从合并缓存加载，跳过
            if drug_id in self.drug_graphs_dict:
                built += 1
                continue

            # 先检查缓存
            cache_file = None
            if self.cache_dir:
                cache_file = self.cache_dir / f"drug_{drug_id}.pkl"

            if cache_file and cache_file.exists() and self.use_cache:
                try:
                    if cache_file.stat().st_size == 0:
                        cache_file.unlink()
                except OSError:
                    pass
                try:
                    with open(cache_file, 'rb') as f:
                        graph = pickle.load(f)
                    self.drug_graphs_dict[drug_id] = graph
                    built += 1
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load drug cache: {e}")
                    try:
                        cache_file.unlink()
                    except OSError:
                        pass

            # 构建图
            try:
                graph = self.drug_builder.build_graph(smiles)
                if graph is not None and graph.num_nodes() > 0:
                    self.drug_graphs_dict[drug_id] = graph
                    built += 1

                    # 保存缓存
                    if cache_file:
                        _atomic_pickle_dump(graph, cache_file)
                else:
                    failed += 1
            except Exception as e:
                logger.warning(f"Failed to build drug graph for {drug_id}: {e}")
                failed += 1

        logger.info(f"Drug graphs loaded: {built}, failed: {failed}")

    def _build_drug_graph(self, smiles: str, drug_id: str) -> Optional[dgl.DGLGraph]:
        """构建药物图（带缓存）"""
        cache_file = None
        if self.cache_dir:
            cache_file = self.cache_dir / f"drug_{drug_id}.pkl"

        # 尝试从缓存加载
        if cache_file and cache_file.exists() and self.use_cache:
            try:
                if cache_file.stat().st_size == 0:
                    cache_file.unlink()
            except OSError:
                pass
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load drug cache: {e}")
                try:
                    cache_file.unlink()
                except OSError:
                    pass

        # 构建图
        try:
            graph = self.drug_builder.build_graph(smiles)

            # 保存缓存
            if cache_file and graph is not None:
                _atomic_pickle_dump(graph, cache_file)

            return graph

        except Exception as e:
            logger.error(f"Failed to build drug graph for {smiles}: {e}")
            return None

    def __len__(self) -> int:
        return len(self.data_df)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        try:
            row = self.data_df.iloc[idx]
            target_id = row['Target_ID']
            drug_id = row['Drug_ID']
            smiles = row['Drug']
            raw_affinity = row['Y']

            # 验证亲和力值范围 (Davis: 0-15, KIBA: 0-18)
            if not (0 <= raw_affinity <= 18):
                logger.warning(f"Invalid affinity value {raw_affinity} for {drug_id}-{target_id}")
                return None

            affinity = raw_affinity
            if self.label_mean is not None and self.label_std is not None:
                affinity = (affinity - self.label_mean) / self.label_std

            # 获取蛋白质构象
            if target_id not in self.protein_conformers_dict:
                return None

            protein_conformers = self.protein_conformers_dict[target_id]

            # 至少需要1个conformer
            if len(protein_conformers) < 1:
                return None
            protein_graphs = []
            protein_scores = []
            for item in protein_conformers:
                if isinstance(item, tuple):
                    graph = item[0]
                    score = 0.0
                    if len(item) >= 2:
                        candidate = item[-1] if len(item) > 2 else item[1]
                        if torch.is_tensor(candidate):
                            score = float(candidate.item())
                        elif isinstance(candidate, (int, float, np.floating)):
                            score = float(candidate)
                    protein_graphs.append(graph)
                    protein_scores.append(score)
                else:
                    protein_graphs.append(item)
                    protein_scores.append(0.0)

            # 获取预加载的药物图（如果有），否则构建
            if drug_id in self.drug_graphs_dict:
                drug_graph = self.drug_graphs_dict[drug_id]
            else:
                drug_graph = self._build_drug_graph(smiles, drug_id)
                if drug_graph is None:
                    return None
                # 验证图节点数量
                if drug_graph.num_nodes() == 0:
                    return None

            return {
                'protein_conformers': protein_graphs,
                'protein_conformer_scores': torch.tensor(protein_scores, dtype=torch.float32),
                'drug_graph': drug_graph,
                'affinity': torch.tensor(affinity, dtype=torch.float32),
                'target_id': target_id,
                'drug_id': drug_id
            }

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            return None


def collate_batch(samples: List[Optional[Dict]]) -> Optional[Dict]:
    """
    批处理函数

    Returns:
        {
            'protein_conformers': List[dgl.DGLGraph],  # 长度=K (K>=1)
            'protein_conformer_mask': torch.BoolTensor [batch_size, K],
            'protein_conformer_scores': torch.FloatTensor [batch_size, K],
            'drug_graphs': dgl.DGLGraph,
            'affinities': torch.Tensor [batch_size],
            'target_ids': List[str],
            'drug_ids': List[str]
        }
    """
    # 过滤None样本
    samples = [s for s in samples if s is not None]

    if len(samples) == 0:
        return None

    # 收集数据
    max_conformers = max(len(s['protein_conformers']) for s in samples)
    if max_conformers < 1:
        return None

    def _make_dummy_graph(template_graph: dgl.DGLGraph) -> dgl.DGLGraph:
        num_feat = template_graph.ndata['feat'].shape[1]
        dummy = dgl.graph(([0], [0]), num_nodes=1)
        dummy.ndata['feat'] = torch.zeros(1, num_feat)
        if 'coord' in template_graph.ndata:
            dummy.ndata['coord'] = torch.zeros(1, 3)
        if 'feat' in template_graph.edata:
            edge_dim = template_graph.edata['feat'].shape[1]
            dummy.edata['feat'] = torch.zeros(1, edge_dim)
        return dummy

    protein_graphs_by_rank = [[] for _ in range(max_conformers)]
    conformer_mask = torch.zeros(len(samples), max_conformers, dtype=torch.bool)
    conformer_scores = torch.zeros(len(samples), max_conformers, dtype=torch.float32)

    template_graph = samples[0]['protein_conformers'][0]
    dummy_graph = _make_dummy_graph(template_graph)

    drug_graphs = []
    affinities = []
    target_ids = []
    drug_ids = []

    for sample_idx, sample in enumerate(samples):
        conformers = sample['protein_conformers']
        scores = sample.get('protein_conformer_scores', None)
        for rank_idx in range(max_conformers):
            if rank_idx < len(conformers):
                protein_graphs_by_rank[rank_idx].append(conformers[rank_idx])
                conformer_mask[sample_idx, rank_idx] = True
                if scores is not None and rank_idx < len(scores):
                    conformer_scores[sample_idx, rank_idx] = scores[rank_idx]
            else:
                protein_graphs_by_rank[rank_idx].append(dummy_graph)
        drug_graphs.append(sample['drug_graph'])
        affinities.append(sample['affinity'])
        target_ids.append(sample['target_id'])
        drug_ids.append(sample['drug_id'])

    # Batch蛋白质图（每个rank一个batch）
    batched_protein_conformers = [dgl.batch(graphs) for graphs in protein_graphs_by_rank]

    # Batch药物图
    batched_drug_graphs = dgl.batch(drug_graphs)

    # Stack亲和力
    affinities_tensor = torch.stack(affinities)

    return {
        'protein_conformers': batched_protein_conformers,
        'protein_conformer_mask': conformer_mask,
        'protein_conformer_scores': conformer_scores,
        'drug_graphs': batched_drug_graphs,
        'affinities': affinities_tensor,
        'target_ids': target_ids,
        'drug_ids': drug_ids
    }


def create_dataloaders(
        data_root: str,
        split_type: str,
        protein_conformer_dir: str,
        config: Dict,
        pair_repr_dir: Optional[str] = None,
        colabfold_features_dir: Optional[str] = None,
        dataset: str = "KIBA",
        **kwargs  # 忽略其他参数（向后兼容）
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_root = Path(data_root)

    # 检查数据集子目录 - 支持两种路径格式:
    # 1. data_root=data, dataset=KIBA -> data/KIBA/split_data
    # 2. data_root=data/KIBA, dataset=KIBA -> data/KIBA/split_data (避免重复)
    if (data_root / "split_data").exists():
        # data_root已经是数据集目录
        split_dir = data_root / "split_data" / split_type
    elif (data_root / dataset / "split_data").exists():
        # data_root是父目录，需要加上dataset
        split_dir = data_root / dataset / "split_data" / split_type
    else:
        # 回退：直接在data_root下查找
        split_dir = data_root / split_type

    # CSV文件路径
    train_csv = split_dir / 'train.csv'
    val_csv = split_dir / 'valid.csv'
    test_csv = split_dir / 'test.csv'

    # 检查文件存在
    for csv_file in [train_csv, val_csv, test_csv]:
        if not csv_file.exists():
            raise FileNotFoundError(f"Data file not found: {csv_file}")

    # 创建缓存目录
    cache_dir = None
    if config['data'].get('cache_dir'):
        cache_dir = Path(config['data']['cache_dir']) / split_type

    # 读取标签标准化配置
    label_norm = config['data'].get('label_normalization', False)
    label_mean = None
    label_std = None
    if label_norm:
        stats_file = split_dir / 'train.csv'
        df_stats = pd.read_csv(stats_file)
        label_mean = df_stats['Y'].mean()
        label_std = df_stats['Y'].std() + 1e-8

    # 创建数据集
    train_dataset = MultiConformerDTADataset(
        csv_file=str(train_csv),
        protein_conformer_dir=protein_conformer_dir,
        config=config,
        cache_dir=cache_dir,
        use_cache=config['data']['use_cache'],
        label_mean=label_mean,
        label_std=label_std,
        pair_repr_dir=pair_repr_dir,
        colabfold_features_dir=colabfold_features_dir
    )

    val_dataset = MultiConformerDTADataset(
        csv_file=str(val_csv),
        protein_conformer_dir=protein_conformer_dir,
        config=config,
        cache_dir=cache_dir,
        use_cache=config['data']['use_cache'],
        label_mean=label_mean,
        label_std=label_std,
        pair_repr_dir=pair_repr_dir,
        colabfold_features_dir=colabfold_features_dir
    )

    test_dataset = MultiConformerDTADataset(
        csv_file=str(test_csv),
        protein_conformer_dir=protein_conformer_dir,
        config=config,
        cache_dir=cache_dir,
        use_cache=config['data']['use_cache'],
        label_mean=label_mean,
        label_std=label_std,
        pair_repr_dir=pair_repr_dir,
        colabfold_features_dir=colabfold_features_dir
    )

    num_workers = config['data']['num_workers']
    pin_memory = config['data']['pin_memory']
    # 仅在多进程加载时开启预取和持久化 worker
    persistent_workers = config['data'].get('persistent_workers', num_workers > 0 and num_workers is not None)
    prefetch_factor = config['data'].get('prefetch_factor', 2)

    common_loader_kwargs = {
        'batch_size': config['training']['batch_size'],
        'collate_fn': collate_batch,
        'num_workers': num_workers,
        'pin_memory': pin_memory
    }
    if num_workers and num_workers > 0:
        common_loader_kwargs['prefetch_factor'] = prefetch_factor
        common_loader_kwargs['persistent_workers'] = persistent_workers

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **common_loader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_loader_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_loader_kwargs
    )

    return train_loader, val_loader, test_loader
