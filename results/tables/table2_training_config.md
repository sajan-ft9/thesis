**Training Configuration**

| Hyperparameter | Value |
| --- | --- |
| Backbone | efficientnet_b0 |
| Pretrained | ImageNet |
| Image size | 224x224 |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Weight decay | 0.0001 |
| LR schedule | 3-epoch warmup + cosine |
| Epochs | 20 |
| Unfreeze epoch | 8 |
| Label smoothing | 0.05 |
| Early stopping patience | 7 |
| Gradient clipping | 1.0 |
| Mixed precision | True |
| Seed | 42 |
