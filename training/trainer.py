# training/trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau, LambdaLR
from typing import Dict
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path
import logging
import time

from utils.metrics import compute_metrics, log_metrics_to_wandb

logger = logging.getLogger(__name__)


class DTATrainer:
    """药物-靶标亲和力预测训练器"""

    def __init__(
            self,
            model: nn.Module,
            config: Dict,
            device: str = 'cuda',
            num_protein_conformers: int | None = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.training_history = []
        if num_protein_conformers is None:
            num_protein_conformers = int(config.get('training', {}).get('num_protein_conformers', 5))
        self.num_protein_conformers = int(num_protein_conformers)
        if self.num_protein_conformers <= 0:
            raise ValueError(f"num_protein_conformers must be >= 1, got {self.num_protein_conformers}")

        # 优化器
        optimizer_name = str(config['training'].get('optimizer', 'adamw')).lower()
        optimizer_kwargs = dict(
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        if optimizer_name == 'adam':
            self.optimizer = Adam(model.parameters(), **optimizer_kwargs)
        else:
            self.optimizer = AdamW(model.parameters(), **optimizer_kwargs)

        # 损失函数
        self.criterion_mse = nn.MSELoss()
        self.criterion_mae = nn.L1Loss()

        # 训练状态跟踪 - 支持多种监控指标
        # monitor支持: loss/mse/rmse/val_loss/pearson (loss默认等价于mse)
        self.monitor = config['training']['early_stopping'].get('monitor', 'pearson')
        self.monitor_metric, self.monitor_mode = self._resolve_monitor(self.monitor)
        if self.monitor_mode == 'min':
            self.best_val_metric = float('inf')  # 越小越好
        else:
            self.best_val_metric = float('-inf')  # 越大越好

        self.best_val_metrics = {}
        self.best_model_path = None
        self.patience_counter = 0
        self.epoch = 0

        # 标签标准化（可选）
        self.label_mean = None
        self.label_std = None

        if self.monitor_metric == self.monitor:
            logger.info(f"Model selection monitor: {self.monitor_metric} (mode: {self.monitor_mode})")
        else:
            logger.info(
                f"Model selection monitor: {self.monitor} -> {self.monitor_metric} (mode: {self.monitor_mode})"
            )

        # 梯度累积
        self.accumulation_steps = config['training'].get('accumulation_steps', 4)

        # AMP混合精度
        self.use_amp = bool(config['training'].get('use_amp', False)) and str(self.device).startswith('cuda')
        amp_dtype = str(config['training'].get('amp_dtype', 'fp16')).lower()
        self.amp_dtype = torch.bfloat16 if amp_dtype == 'bf16' else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp and self.amp_dtype == torch.float16
        )

        # 学习率调度器（与monitor对齐）
        scheduler_config = config['training']['scheduler']
        scheduler_type = scheduler_config.get('type', 'cosine_annealing_warm_restarts')
        eta_min = float(scheduler_config.get('eta_min', 1e-6))
        self.warmup_epochs = int(scheduler_config.get('warmup_epochs', 0))
        self.total_warmup_steps = max(1, self.warmup_epochs)

        def lr_lambda(current_step, trainer=self):
            if trainer.warmup_epochs > 0 and current_step < trainer.total_warmup_steps:
                return float(current_step) / float(max(1, trainer.total_warmup_steps))
            return 1.0

        self.warmup_scheduler = None
        if self.warmup_epochs > 0:
            self.warmup_scheduler = LambdaLR(self.optimizer, lr_lambda)

        # 根据类型创建调度器
        self.scheduler_type = scheduler_type
        if scheduler_type == 'cosine_annealing':
            # 不重启的cosine调度器
            T_max = scheduler_config.get('T_max', config['training']['num_epochs'])
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=eta_min
            )
        elif scheduler_type == 'reduce_on_plateau':
            # 基于与early-stopping一致的验证指标
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=self.monitor_mode,  # min: loss/rmse/mse, max: pearson
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5),
                min_lr=eta_min
            )
        else:
            # 默认: cosine_annealing_warm_restarts
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=scheduler_config.get('T_0', 10),
                T_mult=scheduler_config.get('T_mult', 2),
                eta_min=eta_min
            )

    @staticmethod
    def _resolve_monitor(monitor: str) -> tuple[str, str]:
        """
        将配置中的monitor映射到val_metrics里的实际键名和优化方向。
        """
        mapping = {
            # loss相关（越小越好）
            'loss': ('mse', 'min'),      # 向后兼容：loss= MSE（和DTA-GTOmega一致）
            'mse': ('mse', 'min'),
            'rmse': ('rmse', 'min'),
            'val_loss': ('val_loss', 'min'),  # validate()里 val_loss = rmse
            # 相关性（越大越好）
            'pearson': ('pearson', 'max'),
        }
        if monitor not in mapping:
            raise ValueError(
                f"Unsupported early_stopping.monitor: {monitor}. "
                f"Supported: {sorted(mapping.keys())}"
            )
        return mapping[monitor]

    def _pearson_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Pearson相关性损失 (1 - pearson)
        数值稳定版本
        """
        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()

        pred_std = torch.sqrt(pred_centered.pow(2).mean() + 1e-8)
        target_std = torch.sqrt(target_centered.pow(2).mean() + 1e-8)

        corr = (pred_centered * target_centered).mean() / (pred_std * target_std + 1e-8)
        corr = corr.clamp(-1.0, 1.0)

        return 1.0 - corr

    def _ranking_loss(self, pred: torch.Tensor, target: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
        """
        排序损失 - 提升CI
        对于target_i > target_j的样本对，希望pred_i > pred_j
        使用margin ranking loss，数值稳定
        """
        # 这里在训练循环中提供按target/drug分组的子批次，避免跨分布比较
        if pred.numel() < 2:
            return torch.tensor(0.0, device=pred.device)

        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)  # [B, B]
        target_diff = target.unsqueeze(1) - target.unsqueeze(0)  # [B, B]
        valid_mask = target_diff.abs() > 0.1

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        target_sign = torch.sign(target_diff)
        loss_matrix = torch.relu(-target_sign * pred_diff + margin)
        return loss_matrix[valid_mask].mean()

    def _listmle_loss(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """
        ListMLE排序损失 - 基于排列似然的listwise排序损失
        比pairwise的margin ranking更适合排序任务

        公式: -sum_i log(exp(s_pi(i)) / sum_j>=i exp(s_pi(j)))
        其中pi是按target降序排列的排列
        """
        if pred.numel() < 2:
            return torch.tensor(0.0, device=pred.device)

        # 按target降序排序，获取排列索引
        _, indices = torch.sort(target, descending=True)
        pred_sorted = pred[indices]

        # 计算ListMLE loss
        # 对于位置i，计算log softmax over remaining items
        n = pred_sorted.size(0)

        # 使用cumsum从后往前计算 log(sum_j>=i exp(s_j))
        # 为数值稳定性，减去最大值
        max_pred = pred_sorted.max()
        pred_shifted = pred_sorted - max_pred

        # 从后往前累积求和
        exp_pred = torch.exp(pred_shifted)
        cum_sum = torch.flip(torch.cumsum(torch.flip(exp_pred, [0]), dim=0), [0])

        # ListMLE loss = -sum(pred_sorted - log(cum_sum))
        log_cum_sum = torch.log(cum_sum + eps)
        loss = -torch.mean(pred_shifted - log_cum_sum)

        return loss

    def _contrastive_loss(self, pred: torch.Tensor, target: torch.Tensor,
                          embeddings: torch.Tensor = None, temperature: float = 0.1) -> torch.Tensor:
        """
        对比学习损失 - 基于亲和力的对比学习
        高亲和力的样本对应该在embedding空间中更接近

        使用soft contrastive: 根据亲和力差异计算soft label
        """
        if pred.numel() < 2:
            return torch.tensor(0.0, device=pred.device)

        # 如果没有提供embeddings，使用prediction作为1D embedding
        if embeddings is None:
            embeddings = pred.unsqueeze(-1)

        # 归一化embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        # 计算相似度矩阵
        sim_matrix = torch.mm(embeddings, embeddings.t()) / temperature

        # 计算亲和力差异作为soft label
        # 亲和力接近的样本应该有更高的相似度
        target_diff = torch.abs(target.unsqueeze(1) - target.unsqueeze(0))
        # 将差异转换为soft label (差异小 -> 标签接近1)
        soft_labels = torch.exp(-target_diff / 2.0)  # sigma=2.0

        # 对角线设为0（不和自己比较）
        mask = torch.eye(pred.size(0), device=pred.device).bool()
        soft_labels = soft_labels.masked_fill(mask, 0)

        # 归一化soft labels
        soft_labels = soft_labels / (soft_labels.sum(dim=1, keepdim=True) + 1e-8)

        # 计算log_softmax
        log_prob = F.log_softmax(sim_matrix, dim=1)

        # KL散度损失
        loss = -torch.sum(soft_labels * log_prob) / pred.size(0)

        return loss

    def compute_loss(self, predictions, targets, current_epoch):
        """
        计算多任务损失函数

        组成：
        1. MSE/MAE: 基础回归损失
        2. Pearson Loss: 提升相关性指标
        3. Ranking Loss: 提升CI（排序一致性）
        4. Variance Loss: 防止预测值收缩
        """
        # predictions可以是tensor或{'value': tensor, 'grouped': [(pred_g, target_g), ...]}
        if isinstance(predictions, dict):
            pred_tensor = predictions['value']
            grouped = predictions.get('grouped', [])
        else:
            pred_tensor = predictions
            grouped = []
        batch_size = pred_tensor.size(0)

        # 1. 基础回归损失
        mse_loss = self.criterion_mse(pred_tensor, targets)
        mae_loss = self.criterion_mae(pred_tensor, targets)

        # 2. Pearson相关性损失 - 提升Pearson/Spearman
        pearson_loss = self._pearson_loss(pred_tensor, targets)

        # 3. 排序损失 - 提升CI
        # 默认全局ranking
        ranking_loss = self._ranking_loss(pred_tensor, targets, margin=0.5)
        # 如果提供了分组ranking，覆盖默认
        if grouped:
            rank_loss_total = 0.0
            for pred_g, target_g in grouped:
                rank_loss_total = rank_loss_total + self._ranking_loss(pred_g, target_g, margin=0.5)
            ranking_loss = rank_loss_total / max(len(grouped), 1)

        # 4. 方差匹配损失 - 防止预测值方差过小
        pred_std = pred_tensor.std()
        target_std = targets.std()
        variance_loss = F.l1_loss(pred_std, target_std)

        # 5. 高亲和力样本额外关注
        high_affinity_mask = targets > 7.0
        if high_affinity_mask.sum() > 0:
            high_affinity_loss = F.mse_loss(
                pred_tensor[high_affinity_mask],
                targets[high_affinity_mask]
            )
        else:
            high_affinity_loss = torch.tensor(0.0, device=pred_tensor.device)

        # 检查配置文件中的loss_weights
        config_loss_weights = self.config['training'].get('loss_weights', {})
        ranking_boost = config_loss_weights.get('ranking_boost', False)

        # 如果配置了显式的loss权重，使用配置的权重
        # 支持: mse, ranking, pearson, listmle, contrastive
        if 'mse' in config_loss_weights:
            mse_weight = config_loss_weights.get('mse', 1.0)
            ranking_weight = config_loss_weights.get('ranking', 0.0)
            pearson_weight = config_loss_weights.get('pearson', 0.0)
            listmle_weight = config_loss_weights.get('listmle', 0.0)
            contrastive_weight = config_loss_weights.get('contrastive', 0.0)

            # 支持分组ranking：训练循环可传入 {'grouped': [(pred_g, target_g), ...]}
            if isinstance(predictions, dict) and 'grouped' in predictions:
                rank_loss_total = 0.0
                groups = predictions['grouped']
                for pred_g, target_g in groups:
                    rank_loss_total = rank_loss_total + self._ranking_loss(pred_g, target_g, margin=0.5)
                ranking_loss = rank_loss_total / max(len(groups), 1)

            # 计算ListMLE损失（如果权重>0）
            listmle_loss = torch.tensor(0.0, device=pred_tensor.device)
            if listmle_weight > 0:
                listmle_loss = self._listmle_loss(pred_tensor, targets)

            # 计算Contrastive损失（如果权重>0）
            contrastive_loss = torch.tensor(0.0, device=pred_tensor.device)
            if contrastive_weight > 0:
                contrastive_loss = self._contrastive_loss(pred_tensor, targets)

            total_loss = (
                mse_weight * mse_loss +
                ranking_weight * ranking_loss +
                pearson_weight * pearson_loss +
                listmle_weight * listmle_loss +
                contrastive_weight * contrastive_loss
            )

            # 数值安全检查
            if not torch.isfinite(total_loss):
                logger.warning(f"Non-finite loss detected, fallback to MSE only")
                total_loss = mse_loss

            return total_loss

        # 自适应权重调度
        # 前期：专注基础回归
        # 中期：引入排序和相关性
        # 后期：平衡所有目标
        if ranking_boost:
            # ranking_boost模式：大幅加强ranking权重优化CI
            if current_epoch < 10:
                weights = {
                    'mse': 0.5,
                    'mae': 0.1,
                    'pearson': 0.1,
                    'ranking': 0.2,
                    'variance': 0.05,
                    'high_affinity': 0.05
                }
            elif current_epoch < 30:
                weights = {
                    'mse': 0.3,
                    'mae': 0.05,
                    'pearson': 0.15,
                    'ranking': 0.35,
                    'variance': 0.1,
                    'high_affinity': 0.05
                }
            else:
                weights = {
                    'mse': 0.2,
                    'mae': 0.05,
                    'pearson': 0.15,
                    'ranking': 0.45,
                    'variance': 0.1,
                    'high_affinity': 0.05
                }
        elif current_epoch < 15:
            # 热身阶段：主要学习基础回归
            weights = {
                'mse': 0.6,
                'mae': 0.2,
                'pearson': 0.1,
                'ranking': 0.05,
                'variance': 0.05,
                'high_affinity': 0.0
            }
        elif current_epoch < 40:
            # 中期：逐步引入排序和相关性
            weights = {
                'mse': 0.4,
                'mae': 0.1,
                'pearson': 0.2,
                'ranking': 0.15,
                'variance': 0.1,
                'high_affinity': 0.05
            }
        else:
            # 后期：加大排序和相关性权重
            weights = {
                'mse': 0.3,
                'mae': 0.1,
                'pearson': 0.25,
                'ranking': 0.2,
                'variance': 0.1,
                'high_affinity': 0.05
            }

        total_loss = (
            weights['mse'] * mse_loss +
            weights['mae'] * mae_loss +
            weights['pearson'] * pearson_loss +
            weights['ranking'] * ranking_loss +
            weights['variance'] * variance_loss +
            weights['high_affinity'] * high_affinity_loss
        )

        # 数值安全检查
        if not torch.isfinite(total_loss):
            logger.warning(f"Non-finite loss detected, fallback to MSE only")
            total_loss = mse_loss

        return total_loss

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        num_batches = 0
        total_data_time = 0.0
        total_compute_time = 0.0

        pbar = tqdm(train_loader, desc=f'Training Epoch {self.epoch + 1}')
        total_loader_time = 0.0
        t_loader_start = time.time()
        for batch_idx, batch in enumerate(pbar):
            total_loader_time += time.time() - t_loader_start
            if batch is None:
                t_loader_start = time.time()
                continue

            t0 = time.time()
            # 移动到设备
            protein_graphs = batch['protein_conformers'][:self.num_protein_conformers]
            protein_conformers = [g.to(self.device) for g in protein_graphs]
            drug_graph = batch['drug_graphs'].to(self.device)
            affinity = batch['affinities'].to(self.device, non_blocking=True)

            total_data_time += time.time() - t0

            t1 = time.time()
            # 前向传播
            protein_mask = batch.get('protein_conformer_mask', None)
            protein_scores = batch.get('protein_conformer_scores', None)
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                if protein_mask is not None:
                    protein_mask = protein_mask[:, :len(protein_conformers)].to(self.device)
                if protein_scores is not None:
                    protein_scores = protein_scores[:, :len(protein_conformers)].to(self.device)
                if protein_mask is not None or protein_scores is not None:
                    outputs = self.model(
                        protein_conformers,
                        drug_graph,
                        protein_conformer_mask=protein_mask,
                        protein_conformer_scores=protein_scores
                    )
                else:
                    outputs = self.model(protein_conformers, drug_graph)
                pred_affinity = outputs['affinity'].float()

            # 计算损失
            # 构建按target分组的ranking输入（可选）
            grouping = []
            if 'target_ids' in batch:
                ids = batch['target_ids']
                id_to_idx = {}
                for idx, tid in enumerate(ids):
                    id_to_idx.setdefault(tid, []).append(idx)
                for idx_list in id_to_idx.values():
                    if len(idx_list) > 1:
                        grouping.append((pred_affinity[idx_list], affinity[idx_list]))
            loss_input = pred_affinity
            if grouping:
                loss_input = {'value': pred_affinity, 'grouped': grouping}
            loss = self.compute_loss(loss_input, affinity, self.epoch)
            loss = loss / self.accumulation_steps

            # 反向传播
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度累积
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.use_amp and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm']
                )
                if self.use_amp and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                if self.warmup_scheduler:
                    self.warmup_scheduler.step()

            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            total_compute_time += time.time() - t1

            # 更新进度条
            current_lr = self.optimizer.param_groups[0]['lr']
            avg_d = total_data_time / num_batches
            avg_c = total_compute_time / num_batches
            avg_l = total_loader_time / num_batches
            pbar.set_postfix({
                'loss': f'{loss.item() * self.accumulation_steps:.4f}',
                'lr': f'{current_lr:.2e}',
                'load': f'{avg_l:.3f}',
                'd_t': f'{avg_d:.3f}',
                'c_t': f'{avg_c:.3f}'
            })
            t_loader_start = time.time()

        # 处理剩余的梯度
        if len(train_loader) % self.accumulation_steps != 0:
            if self.use_amp and self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['max_grad_norm']
            )
            if self.use_amp and self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_data_time = total_data_time / max(num_batches, 1)
        avg_compute_time = total_compute_time / max(num_batches, 1)
        logger.info(f"Avg data time/batch: {avg_data_time:.3f}s, compute time/batch: {avg_compute_time:.3f}s")
        return {'train_loss': avg_loss, 'data_time': avg_data_time, 'compute_time': avg_compute_time}

    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """验证"""
        self.model.eval()

        all_preds = []
        all_targets = []
        total_data_time = 0.0
        total_compute_time = 0.0
        num_batches = 0

        for batch in tqdm(val_loader, desc='Validating'):
            if batch is None:
                continue

            t0 = time.time()
            protein_graphs = batch['protein_conformers'][:self.num_protein_conformers]
            protein_conformers = [g.to(self.device) for g in protein_graphs]
            drug_graph = batch['drug_graphs'].to(self.device)
            affinity = batch['affinities'].to(self.device, non_blocking=True)

            total_data_time += time.time() - t0

            t1 = time.time()
            protein_mask = batch.get('protein_conformer_mask', None)
            protein_scores = batch.get('protein_conformer_scores', None)
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                if protein_mask is not None:
                    protein_mask = protein_mask[:, :len(protein_conformers)].to(self.device)
                if protein_scores is not None:
                    protein_scores = protein_scores[:, :len(protein_conformers)].to(self.device)
                if protein_mask is not None or protein_scores is not None:
                    outputs = self.model(
                        protein_conformers,
                        drug_graph,
                        protein_conformer_mask=protein_mask,
                        protein_conformer_scores=protein_scores
                    )
                else:
                    outputs = self.model(protein_conformers, drug_graph)
            pred_affinity = outputs['affinity'].float()

            all_preds.extend(pred_affinity.cpu().numpy())
            all_targets.extend(affinity.cpu().numpy())
            total_compute_time += time.time() - t1
            num_batches += 1

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        # 反标准化
        if getattr(self, 'label_mean', None) is not None and getattr(self, 'label_std', None) is not None:
            all_preds = all_preds * self.label_std + self.label_mean
            all_targets = all_targets * self.label_std + self.label_mean

        # 计算指标
        metrics = compute_metrics(all_targets, all_preds)
        metrics['val_loss'] = metrics['mse']  # 和原版GTOmega一致，用MSE作为val_loss
        metrics['data_time'] = total_data_time / max(num_batches, 1)
        metrics['compute_time'] = total_compute_time / max(num_batches, 1)

        return metrics

    def train_model(self, train_loader, val_loader, num_epochs: int, save_dir: str = 'checkpoints'):
        """训练模型"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            self.epoch = epoch

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'=' * 60}")

            # 设置warmup步数（基于当前训练集大小）
            if self.warmup_epochs > 0 and self.total_warmup_steps <= self.warmup_epochs:
                self.total_warmup_steps = self.warmup_epochs * len(train_loader)

            # 训练
            train_metrics = self.train_epoch(train_loader)

            # 验证
            val_metrics = self.validate(val_loader)

            # 学习率调度
            if self.warmup_scheduler:
                self.warmup_scheduler.step()
            # ReduceLROnPlateau需要传入与monitor一致的验证指标
            if self.scheduler_type == 'reduce_on_plateau':
                self.scheduler.step(val_metrics[self.monitor_metric])
            else:
                self.scheduler.step()

            # 保存训练历史
            epoch_history = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['train_loss'],
                'val_loss': val_metrics['val_loss'],
                **val_metrics
            }
            self.training_history.append(epoch_history)

            # 打印指标
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Learning Rate: {current_lr:.2e}")
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"Val MSE: {val_metrics['mse']:.4f}")
            logger.info(f"Val RMSE: {val_metrics['rmse']:.4f}")
            logger.info(f"Val MAE: {val_metrics['mae']:.4f}")
            logger.info(f"Val Rm²: {val_metrics['rm2']:.4f}")
            logger.info(f"Val Pearson: {val_metrics['pearson']:.4f}")
            logger.info(f"Val Spearman: {val_metrics['spearman']:.4f}")
            logger.info(f"Val CI: {val_metrics['ci']:.4f}")
            if 'data_time' in train_metrics:
                logger.info(f"Train avg data/compute time: {train_metrics['data_time']:.3f}s/{train_metrics['compute_time']:.3f}s")
            if 'data_time' in val_metrics:
                logger.info(f"Val avg data/compute time: {val_metrics['data_time']:.3f}s/{val_metrics['compute_time']:.3f}s")

            # WandB记录
            if self.config['general']['use_wandb'] and wandb.run is not None:
                wandb.log({
                    'epoch': epoch + 1,
                    'learning_rate': current_lr,
                    'train_loss': train_metrics['train_loss'],
                    'val_loss': val_metrics['val_loss'],
                    'val_mse': val_metrics['mse'],
                    'val_rmse': val_metrics['rmse'],
                    'val_mae': val_metrics['mae'],
                    'val_rm2': val_metrics['rm2'],
                    'val_pearson': val_metrics['pearson'],
                    'val_spearman': val_metrics['spearman'],
                    'val_ci': val_metrics['ci']
                })

            # 保存最佳模型 - 支持多种监控指标
            min_delta = self.config['training']['early_stopping'].get('min_delta', 0.0)

            # 根据monitor获取当前指标值
            current_metric = val_metrics[self.monitor_metric]
            if self.monitor_mode == 'min':
                improvement = self.best_val_metric - current_metric  # 越小越好
            else:
                improvement = current_metric - self.best_val_metric  # 越大越好

            if improvement > min_delta:
                self.best_val_metric = current_metric
                self.best_val_metrics = val_metrics
                self.patience_counter = 0

                self.best_model_path = save_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_metrics': val_metrics,
                    'config': self.config
                }, self.best_model_path)

                if self.monitor_metric == self.monitor:
                    monitor_desc = self.monitor_metric
                else:
                    monitor_desc = f"{self.monitor}({self.monitor_metric})"
                logger.info(
                    f"✓ Saved best model (val_{monitor_desc}: {current_metric:.4f}, improvement: {improvement:.4f})"
                )
            else:
                self.patience_counter += 1
                logger.info(f"No improvement (improvement: {improvement:.4f} <= min_delta: {min_delta:.4f}), patience: {self.patience_counter}/{self.config['training']['early_stopping']['patience']}")

            # 早停检查
            if self.patience_counter >= self.config['training']['early_stopping']['patience']:
                logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

            # 定期保存检查点
            if (epoch + 1) % self.config['general']['save_interval'] == 0:
                checkpoint_path = save_dir / f'checkpoint_epoch{epoch + 1}.pt'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'config': self.config
                }, checkpoint_path)
                logger.info(f"✓ Saved checkpoint")

        if self.monitor_metric == self.monitor:
            monitor_desc = self.monitor_metric
        else:
            monitor_desc = f"{self.monitor}({self.monitor_metric})"
        logger.info(f"\nTraining completed! Best val_{monitor_desc}: {self.best_val_metric:.4f}")
