**Quantization Results (backend: qnnpack)**

| Variant | Size (MB) | Size ↓ (%) | Accuracy | Acc drop (%) | AUC | AUC drop (%) | Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP32 | 17.667 | 0.0 | 0.9183 | 0.0 | 0.9678 | 0.0 | 133.407 |
| INT8 dynamic | 16.688 | 5.5 | 0.9199 | -0.16 | 0.9682 | -0.04 | 135.504 |
| INT8 static (PTQ) | 5.219 | 70.5 | 0.8109 | 10.74 | 0.9444 | 2.34 | 16.294 |
