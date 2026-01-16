import numpy as np
from pathlib import Path
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def _get_plddt_from_pdb_fast(pdb_file: Path) -> float:
    """
    使用快速文本解析提取平均pLDDT分数
    """
    scores = []
    try:
        with pdb_file.open('r') as f:
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    try:
                        scores.append(float(line[60:66]))
                    except ValueError:
                        continue
    except Exception as e:
        logger.warning(f"Error reading {pdb_file}: {e}")
        return 0.0

    if not scores:
        return 0.0

    return float(np.mean(scores))


def get_plddt_from_pdb(pdb_file: Path) -> float:
    """
    从PDB文件的b-factor列中提取平均pLDDT分数

    ColabFold输出的PDB文件中，b-factor列存储的就是pLDDT分数(0-100)

    Args:
        pdb_file: PDB文件路径

    Returns:
        平均pLDDT分数 (0-100)
    """
    return _get_plddt_from_pdb_fast(pdb_file)


def calculate_global_std(protein_dir: Path, target_ids: List[str]) -> float:
    """
    计算指定蛋白质集合的pLDDT全局标准差

    Args:
        protein_dir: 蛋白质conformer根目录
        target_ids: 该数据集包含的蛋白质ID列表

    Returns:
        global_std: 全局标准差
    """
    scores = []

    for target_id in target_ids:
        target_conformer_dir = protein_dir / str(target_id)
        if not target_conformer_dir.exists():
            continue

        # 读取该蛋白质所有conformer的pLDDT
        for rank_idx in range(1, 6):
            rank_str = f"{rank_idx:03d}"
            pattern = f"{target_id}_unrelaxed_rank_{rank_str}_alphafold2_ptm_model_*_seed_000.pdb"
            matching_files = list(target_conformer_dir.glob(pattern))

            if matching_files:
                score = get_plddt_from_pdb(matching_files[0])
                if score > 0:
                    scores.append(score)

    if not scores:
        logger.warning("无法读取pLDDT分数，使用默认全局标准差 5.0")
        return 5.0

    global_std = float(np.std(scores))
    global_mean = float(np.mean(scores))

    logger.info(f"pLDDT统计: mean={global_mean:.2f}, std={global_std:.2f}, n={len(scores)}")

    return global_std


def select_conformers_by_plddt(
    plddt_scores: List[Tuple[int, float]],
    global_std: float,
    std_factor: float = 1.0,
    max_conformers: int = 5
) -> List[int]:
    """
    基于全局标准差的conformer选择策略

    策略:
    1. 阈值 = 当前蛋白质最高分 - std_factor * 全局标准差
    2. 选择所有 >= 阈值的conformer
    3. 保底：至少选1个（最高分）

    Args:
        plddt_scores: List of (rank_idx, plddt_score)
        global_std: 全局标准差（整个数据集的）
        std_factor: 容忍几倍全局标准差（默认1.0）
        max_conformers: 最多选择几个（默认5）

    Returns:
        选中的rank索引列表，例如 [1, 2, 3]
    """
    if not plddt_scores:
        return []

    # 按pLDDT分数降序排序
    sorted_scores = sorted(plddt_scores, key=lambda x: x[1], reverse=True)

    # 当前蛋白质的最高分
    max_plddt = sorted_scores[0][1]

    # 阈值 = 最高分 - factor * 全局标准差
    threshold = max_plddt - (std_factor * global_std)

    selected_ranks = []
    for rank_idx, plddt in sorted_scores:
        if plddt >= threshold and len(selected_ranks) < max_conformers:
            selected_ranks.append(rank_idx)

    # 保底：至少选1个（最高分）
    if not selected_ranks:
        selected_ranks = [sorted_scores[0][0]]

    # 按rank索引排序（保持原始顺序）
    selected_ranks.sort()

    return selected_ranks


def get_selected_conformers(
    protein_dir: Path,
    target_id: str,
    global_std: float,
    std_factor: float = 1.0,
    max_conformers: int = 5,
    fallback_to_all: bool = True
) -> List[int]:
    """
    为指定蛋白质获取应该加载的conformer索引

    Args:
        protein_dir: 蛋白质目录
        target_id: 目标ID
        global_std: 全局标准差（需要预先计算）
        std_factor: 容忍几倍全局标准差（默认1.0）
        max_conformers: 最多选择几个
        fallback_to_all: 如果找不到PDB文件，是否回退到加载所有conformer

    Returns:
        应该加载的rank索引列表，例如 [1, 2, 3]
    """
    target_conformer_dir = protein_dir / str(target_id)

    if not target_conformer_dir.exists():
        if fallback_to_all:
            return [1, 2, 3, 4, 5]
        else:
            return []

    # 读取所有conformer的pLDDT分数
    plddt_scores = []

    for rank_idx in range(1, 6):  # rank 1-5
        rank_str = f"{rank_idx:03d}"
        pattern = f"{target_id}_unrelaxed_rank_{rank_str}_alphafold2_ptm_model_*_seed_000.pdb"
        matching_files = list(target_conformer_dir.glob(pattern))

        if not matching_files:
            continue

        pdb_file = matching_files[0]
        plddt = get_plddt_from_pdb(pdb_file)
        plddt_scores.append((rank_idx, plddt))

    if not plddt_scores:
        if fallback_to_all:
            return [1, 2, 3, 4, 5]
        else:
            return []

    # 动态选择conformer（基于全局标准差）
    selected = select_conformers_by_plddt(
        plddt_scores,
        global_std=global_std,
        std_factor=std_factor,
        max_conformers=max_conformers
    )

    return selected
