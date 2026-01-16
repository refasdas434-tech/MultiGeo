#!/usr/bin/env python3

import os
import json
import numpy as np
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def extract_contact_map_from_pae(scores_json_file, pae_threshold=8.0):
    """
    从ColabFold的scores JSON文件提取contact map（基于PAE矩阵）

    Args:
        scores_json_file: ColabFold输出的scores JSON文件路径
        pae_threshold: PAE阈值(Å)，小于此值认为有接触，默认8.0Å

    Returns:
        contact_map: 接触图矩阵 (bool类型)
    """
    with open(scores_json_file, 'r') as f:
        data = json.load(f)

    pae = np.array(data['pae'])
    contact_map = pae < pae_threshold

    return contact_map


def _parse_ranks(ranks_arg):
    if not ranks_arg:
        return None
    ranks = []
    for part in ranks_arg.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = part.split('-', 1)
            ranks.extend(range(int(start), int(end) + 1))
        else:
            ranks.append(int(part))
    ranks = sorted(set(ranks))
    return ranks


def generate_all_contact_maps(colabfold_dir, output_dir, pae_threshold=8.0, ranks=None):
    """
    从ColabFold输出目录提取所有蛋白质的contact map

    Args:
        colabfold_dir: ColabFold结果目录（包含各蛋白质子目录）
        output_dir: 输出目录
        pae_threshold: PAE阈值(Å)
    """
    colabfold_dir = Path(colabfold_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    # 遍历所有蛋白质目录
    ranks_to_use = [1] if ranks is None else ranks
    use_legacy_name = ranks is None

    for protein_dir in colabfold_dir.iterdir():
        if protein_dir.is_dir():
            protein_name = protein_dir.name

            for rank_idx in ranks_to_use:
                scores_files = list(
                    protein_dir.glob(f"*_scores_rank_{rank_idx:03d}_*.json")
                )
                if not scores_files and use_legacy_name and rank_idx == 1:
                    scores_files = list(protein_dir.glob("*_scores_*.json"))

                if not scores_files:
                    skipped += 1
                    continue

                scores_file = sorted(scores_files)[0]

                try:
                    contact_map = extract_contact_map_from_pae(scores_file, pae_threshold)

                    # 保存为numpy数组
                    if use_legacy_name and rank_idx == 1:
                        output_file = output_dir / f"{protein_name}.npy"
                    else:
                        output_file = output_dir / f"{protein_name}_rank_{rank_idx:03d}.npy"
                    np.save(output_file, contact_map)

                    processed += 1

                except Exception as e:
                    print(f"Error processing {protein_name} rank{rank_idx}: {e}")
                    skipped += 1

    print(f"\nDone! Processed: {processed}, Skipped: {skipped}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='从ColabFold输出提取contact map (基于PAE)')
    parser.add_argument('--colabfold_dir', type=str, default='data/Davis/colabfold_results',
                        help='ColabFold结果目录')
    parser.add_argument('--output_dir', type=str, default='data/pdb_contact_map',
                        help='输出目录')
    parser.add_argument('--pae_threshold', type=float, default=8.0,
                        help='PAE阈值(Å)，小于此值认为有接触')
    parser.add_argument('--ranks', type=str, default=None,
                        help='指定rank，例如 "1,2,3" 或 "1-5"，默认只生成rank1且使用旧命名')

    args = parser.parse_args()

    print(f"从ColabFold提取contact map (基于PAE)")
    print(f"输入目录: {args.colabfold_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"PAE阈值: {args.pae_threshold} Å")
    print()

    ranks = _parse_ranks(args.ranks)
    generate_all_contact_maps(args.colabfold_dir, args.output_dir, args.pae_threshold, ranks=ranks)


if __name__ == "__main__":
    main()
