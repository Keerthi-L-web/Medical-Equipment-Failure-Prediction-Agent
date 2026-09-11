# XGBoost Predictive Maintenance Model Evaluation

## 1. Class Distribution

- Total Training Samples: `7,500`
- Total Testing Samples: `2,500`
- Positive Class Count (Test): `3,733`
- Negative Class Count (Test): `6,267`
- Imbalance Weight (`scale_pos_weight`): `1.68`

## 2. Evaluation Metrics

| Metric | Value |
| --- | --- |
| **Accuracy** | 0.7572 |
| **Precision** | 0.6281 |
| **Recall** (Priority) | **0.8564** |
| **F1-Score** | 0.7247 |
| **ROC-AUC** | 0.8361 |
| **PR-AUC** | 0.7002 |

## 3. Confusion Matrix

```
[[ True Negatives: 1094   False Positives: 473   ]
 [ False Negatives: 134   True Positives:  799   ]]
```

## 4. Top Feature Importances

- `events_last_5_years`: 0.2765
- `previous_event_count`: 0.2147
- `events_last_3_years`: 0.0770
- `manufacturer_event_rate`: 0.0731
- `days_since_last_event`: 0.0730
- `manufacturer_device_count`: 0.0490
- `manufacturer_event_count`: 0.0314
- `manufacturer_recall_count`: 0.0171
- `devtype_Diagnostic`: 0.0110
- `risk_class_encoded`: 0.0095
