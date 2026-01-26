# 5-Fold Cross-Validation Results

## Davis Dataset

### Performance Comparison of Different Models

| Model | MSE | PCC | Rm2 |
|-------|-----|-----|-----|
| GraphDTA(GAT) | 0.232 | - | 0.663 |
| GraphDTA(GIN) | 0.229 | - | 0.662 |
| WGNN-DTA(GAT) | 0.214 | 0.848 | - |
| WGNN-DTA(GCN) | 0.215 | 0.848 | - |
| Deepdta | 0.261 | - | 0.630 |
| DeepDtaGen | 0.214 | - | 0.705 |
| DTA-GTOmega | 0.264 | 0.789 | 0.578 |
| **MultiGeo** | **0.186** | **0.851** | **0.722** |

## KIBA Dataset

### Performance Comparison of Different Models

| Model | MSE | PCC | Rm2 |
|-------|-----|-----|-----|
| GraphDTA(GAT) | 0.179 | - | 0.687 |
| GraphDTA(GIN) | 0.147 | - | 0.665 |
| WGNN-DTA(GAT) | 0.149 | 0.884 | - |
| WGNN-DTA(GCN) | 0.155 | 0.889 | - |
| Deepdta | 0.194 | - | 0.630 |
| DeepDtaGen | 0.146 | - | 0.748 |
| DTA-GTOmega | 0.174 | 0.868 | 0.707 |
| **MultiGeo** | **0.143** | **0.892** | **0.756** |

## Notes
- "-" indicates the metric was not reported for that model
- **Bold** values indicate the best performance across all models
- Lower MSE values indicate better performance
- Higher PCC and Rm2 values indicate better performance
- All models use the same data split to ensure fair comparison
