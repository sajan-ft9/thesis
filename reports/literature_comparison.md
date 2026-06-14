# Literature Comparison (template for the literature review)

Complete this table from your own reading to position this work and surface the
research gap. **Cite every literature value from its source**, and never mix
literature numbers with our measured results (our row is filled from
`results/` via the rendered thesis, not hand-typed here).

A machine-generated version (CSV / Markdown / LaTeX) is produced at
`results/tables/literature_comparison_template.*` when you run `make report`.

| Author | Year | Dataset | Model | Accuracy | AUC | Explainability | Quantization | Edge Deployment Focus |
|---|---|---|---|---|---|---|---|---|
| This work | 2026 | Kermany (pediatric CXR) | EfficientNet-B0 + Grad-CAM++ + INT8 | see Table 4 | see Table 4 | Integrated Grad-CAM++ | Dynamic + static PTQ | Yes |
| Rajpurkar et al. | 2017 | ChestX-ray14 | CheXNet (DenseNet-121) | — | 0.768* | No | No | No |
| Kermany et al. | 2018 | Kermany (pediatric CXR) | Inception-v3 (transfer) | ~0.92* | — | Occlusion maps | No | No |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |

\* Example placeholders — replace with exact values and citations from the papers.

## How to use this for the research gap
1. Fill rows for the most relevant prior CXR pneumonia / lightweight-CNN / XAI /
   quantization studies.
2. Note which columns are typically **empty** in prior work (often: integrated
   explainability *and* quantization *and* edge focus *and* reproducibility).
3. State the gap explicitly: few works jointly optimise and **correctly measure**
   performance, efficiency, and explainability with a fully reproducible pipeline —
   which is the contribution of this thesis.
