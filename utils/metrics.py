# utils/metrics.py

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import wandb


def compute_ci(y_true, y_pred):
    """计算一致性指数 (Concordance Index)"""
    n = len(y_true)
    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] > y_true[j]:
                if y_pred[i] > y_pred[j]:
                    concordant += 1
                elif y_pred[i] < y_pred[j]:
                    discordant += 1
            elif y_true[i] < y_true[j]:
                if y_pred[i] < y_pred[j]:
                    concordant += 1
                elif y_pred[i] > y_pred[j]:
                    discordant += 1

    if concordant + discordant == 0:
        return 0.5
    return concordant / (concordant + discordant)


def r_squared_error(y_obs, y_pred):
    """使用你提供的原始R²计算方法"""
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    """计算通过原点的斜率"""
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    """计算通过原点模型的R²"""
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def compute_rm2(y_obs, y_pred):
    """使用你提供的原始RM²计算方法"""
    r2 = r_squared_error(y_obs, y_pred)
    r02 = squared_error_zero(y_obs, y_pred)
    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))

def compute_metrics(y_true, y_pred):
    """计算评估指标（包含MSE、Spearman和一致性指标CI）"""
    # 添加数据验证
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            'mse': 0.0,
            'rmse': 0.0,
            'mae': 0.0,
            'pearson': 0.0,
            'spearman': 0.0,
            'ci': 0.0,
            'rm2': 0.0
        }

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    rm2 = compute_rm2(y_true, y_pred)

    # 相关性指标
    pearson = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else 0.0
    spearman = spearmanr(y_true, y_pred)[0] if len(y_true) > 1 else 0.0

    # 一致性指标
    ci_score = compute_ci(y_true, y_pred)

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'pearson': float(pearson),
        'spearman': float(spearman),
        'ci': float(ci_score),
        'rm2': float(rm2)
    }


def print_metrics(y_true, y_pred, dataset_name="", split_strategy=""):
    """打印评估指标（包含MSE、Spearman和一致性指标CI）"""
    metrics = compute_metrics(y_true, y_pred)

    print(f'{dataset_name} {split_strategy}')
    print('MSE:', metrics['mse'])
    print('RMSE:', metrics['rmse'])
    print('MAE:', metrics['mae'])
    print('Pearson:', metrics['pearson'])
    print('Spearman:', metrics['spearman'])
    print('CI:', metrics['ci'])
    print('RM2:', metrics['rm2'])

    return metrics


def log_metrics_to_wandb(y_true, y_pred, prefix="", dataset_name="", split_type=""):
    """记录指标到WandB（包含MSE、Spearman和一致性指标CI）"""
    metrics = compute_metrics(y_true, y_pred)

    # 只在wandb可用时记录
    try:
        if wandb.run is not None:
            # 记录所有核心指标
            wandb_metrics = {
                f"{prefix}mse": metrics['mse'],
                f"{prefix}rmse": metrics['rmse'],
                f"{prefix}mae": metrics['mae'],
                f"{prefix}pearson": metrics['pearson'],
                f"{prefix}spearman": metrics['spearman'],
                f"{prefix}ci": metrics['ci'],
                f"{prefix}rm2": metrics['rm2']
            }

            # 只在测试时记录散点图
            if prefix == "test/":
                create_scatter_plot(y_true, y_pred, prefix, dataset_name, split_type)

            wandb.log(wandb_metrics)
    except Exception as e:
        pass  # wandb不可用时跳过

    return metrics


def create_scatter_plot(y_true, y_pred, prefix, dataset_name, split_type):
    """创建散点图"""
    data = [[true, pred] for true, pred in zip(y_true, y_pred)]
    table = wandb.Table(data=data, columns=["True", "Predicted"])

    wandb.log({
        f"{prefix}prediction_scatter": wandb.plot.scatter(
            table, "True", "Predicted",
            title=f"True vs Predicted - {dataset_name} {split_type}"
        )
    })