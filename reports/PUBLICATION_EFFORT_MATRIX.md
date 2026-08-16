# Publication Effort vs Doability

Assumptions: no purchases, no new hardware, no radiologist or participant interaction, and the existing local datasets/checkpoints remain available.

| Action | Doability | Effort | Value | Recommendation |
|---|---|---:|---:|---|
| Lock threshold from validation, not test | High | 0.5–1 day | Very high | Do first |
| Correct file-size and deployment-artifact measurement | High | 0.5 day | Very high | Do first |
| Fresh-process RSS benchmark | High | 1 day | Very high | Do first |
| Record actual quantization qconfig/fallback | High | 0.5 day | High | Do first |
| Three-seed EfficientNet reruns | High if compute is available | 1–3 days | Very high | Do |
| Three-seed runs for all three models | Medium | 3–7 days | Medium | Optional |
| Add calibration-set sensitivity analysis | High | 0.5–1 day | High | Do |
| Add Grad-CAM deletion/occlusion faithfulness metric | High | 1–2 days | Medium-high | Do if explainability remains in title |
| Near-duplicate/perceptual-hash check | High | 0.5–1 day | Medium | Do if easy; disclose scope |
| Patient-level split verification | Low with current Kermany files | 1–3 days | High | State as unavailable; do not invent it |
| Additional public external dataset | Medium | 1–4 days plus download risk | Medium | Keep RSNA only for minimal paper |
| Real device benchmark | Low without hardware | Not feasible under constraint | High | Future work only |
| Radiologist reader study | Low without human interaction/ethics | Not feasible under constraint | Very high | Future work only |
| Quantization-aware training | High | 1–3 days | Medium | Optional; not required for minimal paper |
| Formal regulatory/clinical validation | Low | Weeks/months | Very high | Out of scope |
| CLAIM/TRIPOD+AI checklist and reporting cleanup | High | 0.5–1 day | Very high | Do |

## Minimal submission path

The best cost-benefit path is: fix threshold leakage, fix size/RSS semantics, run three primary seeds, add calibration/qconfig sensitivity, tighten claims, complete reporting checklists, and keep the RSNA result as secondary evidence. Do not add another dataset, hardware, or human study unless a journal explicitly requires it.
