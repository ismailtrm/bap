# Architecture (High‑Level)
**Updated:** 2025-09-16 11:26

```
Camera → Detector (YOLOv8n) ─┐
                             ├─> Assigner/Tracker (SORT‑like) → Predictive aim (feed‑forward)
CV Fallback (HSV + contour) ─┘

Tracker IDs → Decision Layer
  ├─ Stage‑1: small/big via area → fire gate
  ├─ Stage‑2: friend/foe via color gate → fire gate
  └─ Stage‑3: board cue (ROI) + shape+color double‑check → fire gate

Fire Gate (conservative):
  fire = is_enemy & is_correct_shape & conf_det>τ & conf_color>τc & id_stable≥N & outside_no_fire_mask
```

**Scoring Harness:** reads event logs (hits, wrong hits, timestamps) → computes A1/A2/A3 caps and BSP.
