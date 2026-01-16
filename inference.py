import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm

from models.dta_model_multiGeo import MultiGeoDTAModel
from data.dataset import create_dataloaders
from utils.metrics import compute_metrics

def main(args):
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 先加载 checkpoint 获取配置
    checkpoint_path = args.model_path
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_epoch = checkpoint.get('epoch')
    if ckpt_epoch is not None:
        print(f"Checkpoint epoch: {ckpt_epoch}")

    # 优先使用 checkpoint 中保存的配置
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"Using config from checkpoint (hidden_dim={config['model']['hidden_dim']})")
    else:
        # 兼容旧版本 checkpoint，从文件加载配置
        print(f"Config not found in checkpoint, loading from: {args.config_file}")
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)

    # ColabFold特征目录
    colabfold_features_dir = args.colabfold_features_dir
    if colabfold_features_dir:
        print(f"Using ColabFold single representations as node features from: {colabfold_features_dir}")

    # 创建数据加载器（只需要测试集）
    print("Loading test data...")
    _, _, test_loader = create_dataloaders(
        data_root=args.data_root,
        split_type=args.split_type,
        protein_conformer_dir=args.protein_conformer_dir,
        config=config,
        pair_repr_dir=args.pair_repr_dir,
        colabfold_features_dir=colabfold_features_dir
    )
    print(f"Test batches: {len(test_loader)}")

    # 创建模型
    model = MultiGeoDTAModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # 运行推理
    print("\n" + "=" * 60)
    print("Running Inference on Test Set")
    print("=" * 60)

    all_preds = []
    all_targets = []
    batch_count = 0

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

            outputs = model(
                protein_conformers,
                drug_graph,
                protein_conformer_mask=protein_mask,
                protein_conformer_scores=protein_scores
            )
            pred_affinity = outputs['affinity']

            all_preds.extend(pred_affinity.cpu().numpy())
            all_targets.extend(affinity.cpu().numpy())
            batch_count += 1

    print(f"Processed {batch_count} batches")

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 反标准化（如果训练时做了label_normalization）
    label_mean = getattr(test_loader.dataset, 'label_mean', None)
    label_std = getattr(test_loader.dataset, 'label_std', None)
    if label_mean is not None and label_std is not None:
        all_preds = all_preds * label_std + label_mean
        all_targets = all_targets * label_std + label_mean

    # 计算指标
    test_metrics = compute_metrics(all_targets, all_preds)

    # 打印结果
    print(f"\n{'=' * 60}")
    print("Test Results Summary")
    print(f"{'=' * 60}")
    print(f"Dataset: {args.dataset}")
    print(f"Split Type: {args.split_type}")
    print(f"Number of test samples: {len(all_preds)}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"MAE: {test_metrics['mae']:.4f}")
    print(f"MSE: {test_metrics['mse']:.4f}")
    print(f"RM²: {test_metrics['rm2']:.4f}")
    print(f"Pearson correlation: {test_metrics['pearson']:.4f}")
    print(f"Spearman correlation: {test_metrics['spearman']:.4f}")
    print(f"CI (Concordance Index): {test_metrics['ci']:.4f}")

    # 保存结果
    save_results = {
        'predictions': all_preds.tolist(),
        'targets': all_targets.tolist(),
        'metrics': test_metrics,
        'dataset': args.dataset,
        'split_type': args.split_type,
        'model_path': args.model_path
    }

    import json
    results_file = f"inference_results_{args.dataset}_{args.split_type}.json"
    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference with trained model')

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

    # 模型路径
    parser.add_argument(
        '--model_path',
        type=str,
        default='checkpoints/Davis/cold_drug/best_model.pt',
        help='训练好的模型路径'
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

    parser.add_argument(
        '--colabfold_features_dir',
        type=str,
        default=None,
        help='ColabFold single representation目录'
    )
    parser.add_argument(
        '--pair_repr_dir',
        type=str,
        default=None,
        help='ColabFold pair representation目录（可选）'
    )

    args = parser.parse_args()

    main(args)
