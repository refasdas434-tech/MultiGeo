# KIBA Dataset 5-Fold Cross-Validation Results

## Performance Comparison of Different Models

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
