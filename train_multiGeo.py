# train_multiGeo.py
# MultiGeo风格模型训练脚本

import os
import argparse
import yaml
import torch
import wandb
import logging
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm

from models.dta_model_multiGeo import MultiGeoDTAModel
from data.dataset import create_dataloaders
from training.trainer import DTATrainer
from utils.metrics import print_metrics, log_metrics_to_wandb

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    # 加载配置
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    # 设置随机种子
    set_seed(config['general']['seed'])

    # 启用TF32以提升矩阵计算吞吐（Ampere+ GPU）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # 初始化WandB
    if config['general']['use_wandb']:
        wandb.init(
            project=config['general']['project_name'],
            name=f"{args.dataset}_{args.split_type}",
            config=config,
            tags=[args.dataset, args.split_type],
            settings=wandb.Settings(_disable_stats=True)
        )

    # 创建数据加载器
    logger.info(f"Loading data: {args.dataset} - {args.split_type}")

    # ColabFold节点特征目录
    colabfold_features_dir = args.colabfold_features_dir
    if colabfold_features_dir:
        logger.info(f"Using ColabFold single representations as node features from: {colabfold_features_dir}")

    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=args.data_root,
        split_type=args.split_type,
        protein_conformer_dir=args.protein_conformer_dir,
        config=config,
        dataset=args.dataset,
        pair_repr_dir=args.pair_repr_dir,
        colabfold_features_dir=colabfold_features_dir
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    # 创建模型
    model = MultiGeoDTAModel(config)
    logger.info("Using MultiGeo-style DTA Model")
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Protein node feature dim: {config['features']['protein']['node_dim']}")

    # 创建保存目录
    save_dir = Path(args.save_dir) / args.dataset / args.split_type
    save_dir.mkdir(parents=True, exist_ok=True)

    # 创建训练器
    trainer = DTATrainer(model, config, device)
    label_mean = getattr(train_loader.dataset, 'label_mean', None)
    label_std = getattr(train_loader.dataset, 'label_std', None)
    trainer.label_mean = label_mean
    trainer.label_std = label_std

    # 训练模型
    logger.info("\nStarting training...")
    trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=config['training']['num_epochs'],
        save_dir=str(save_dir)
    )

    # 测试模型
    logger.info("\n" + "=" * 60)
    logger.info("Testing Best Model")
    logger.info("=" * 60)

    # 加载最佳模型
    checkpoint = torch.load(trainer.best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 收集测试集预测
    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            if batch is None:
                continue

            protein_conformers = [g.to(device) for g in batch['protein_conformers']]
            drug_graph = batch['drug_graphs'].to(device)
            affinity = batch['affinities'].to(device)

            protein_mask = batch.get('protein_conformer_mask', None)
            protein_scores = batch.get('protein_conformer_scores', None)
            if protein_mask is not None:
                protein_mask = protein_mask[:, :len(protein_conformers)].to(device)
            if protein_scores is not None:
                protein_scores = protein_scores[:, :len(protein_conformers)].to(device)
            if protein_mask is not None or protein_scores is not None:
                outputs = model(
                    protein_conformers,
                    drug_graph,
                    protein_conformer_mask=protein_mask,
                    protein_conformer_scores=protein_scores
                )
            else:
                outputs = model(protein_conformers, drug_graph)
            pred_affinity = outputs['affinity']

            all_preds.extend(pred_affinity.cpu().numpy())
            all_targets.extend(affinity.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    # 反标准化
    if label_mean is not None and label_std is not None:
        all_preds = all_preds * label_std + label_mean
        all_targets = all_targets * label_std + label_mean

    # 计算并记录指标
    test_metrics = log_metrics_to_wandb(
        all_targets, all_preds,
        prefix="test/",
        dataset_name=args.dataset,
        split_type=args.split_type
    )

    # 打印结果
    print_metrics(all_targets, all_preds, args.dataset, args.split_type)

    # 保存测试结果
    save_test_results(all_targets, all_preds, test_metrics, checkpoint, args, save_dir)

    # 打印最终摘要
    print_final_summary(test_metrics, checkpoint, args)

    if config['general']['use_wandb'] and wandb.run is not None:
        wandb.finish()


def save_test_results(all_targets, all_preds, test_metrics, checkpoint, args, save_dir):
    """保存测试结果"""
    results_file = save_dir / 'test_results.json'
    with open(results_file, 'w') as f:
        import json
        json.dump({
            'dataset': args.dataset,
            'split_type': args.split_type,
            'best_epoch': checkpoint.get('epoch', 0),
            'best_val_loss': checkpoint.get('val_metrics', {}).get('val_loss', 0),
            'test_metrics': test_metrics,
            'predictions': all_preds.tolist(),
            'targets': all_targets.tolist()
        }, f, indent=2)

    logger.info(f"Test results saved to {results_file}")


def print_final_summary(test_metrics, checkpoint, args):
    """打印最终摘要"""
    logger.info(f"\n{'=' * 60}")
    logger.info("Final Results Summary")
    logger.info(f"{'=' * 60}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Split Type: {args.split_type}")
    logger.info(f"Best Epoch: {checkpoint.get('epoch', 0)}")
    logger.info(f"Best Validation Loss: {checkpoint.get('val_metrics', {}).get('val_loss', 0):.4f}")
    logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}")
    logger.info(f"Test MAE: {test_metrics['mae']:.4f}")
    logger.info(f"Test Pearson: {test_metrics['pearson']:.4f}")
    logger.info(f"Test Spearman: {test_metrics['spearman']:.4f}")
    logger.info(f"Test Rm2: {test_metrics['rm2']:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MultiGeo-style DTA Model')

    # 配置文件
    parser.add_argument(
        '--config_file',
        type=str,
        default='configs/config.yaml',
        help='配置文件路径'
    )

    # 数据路径
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='数据根目录'
    )
    parser.add_argument(
        '--protein_conformer_dir',
        type=str,
        required=True,
        help='蛋白质构象目录'
    )
    parser.add_argument(
        '--pair_repr_dir',
        type=str,
        default=None,
        help='ColabFold pair representation目录'
    )

    # 数据集和划分
    parser.add_argument(
        '--dataset',
        type=str,
        default='Davis',
        choices=['Davis', 'KIBA'],
        help='数据集名称'
    )
    parser.add_argument(
        '--split_type',
        type=str,
        required=True,
        choices=['cold_drug', 'cold_target', 'cold_drug_target'],
        help='划分类型'
    )

    # 保存目录
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoints',
        help='模型保存目录'
    )

    # ColabFold节点特征
    parser.add_argument(
        '--colabfold_features_dir',
        type=str,
        default=None,
        help='ColabFold single representation目录（256维节点特征）'
    )

    args = parser.parse_args()

    main(args)
