# Hava Savunma Demo System — Project Summary
**Version:** 0.1 • **Updated:** 2025-09-16 11:26

## Goal
Build a compact, safe, and fast demo air‑defense system that can (1) detect and track moving balloon targets, (2) perform friend/foe discrimination by color, and (3) follow a “given engagement” cue to autonomously select the correct platform and destroy the correct target by **shape+color**, maximizing score and **Bonus Time Points (BSP)**.

## Competition Constraints (for quick reference)
- Max dimension of the system: **< 100 cm** (extra points if ≤ 60 cm on the longest edge).
- Target distance: **~5–10 m** (moving platform with balloons).
- Each stage time: **5 minutes** (300 s).  
  **BSP formula:** `BSP = 20 × (remaining_seconds / 300)`.
- Stages & base scoring caps:
  - **Stage‑1:** Big=5, Small=15, **cap=60**.
  - **Stage‑2:** Big=10, Small=20, **cap=100**. Wrong friend hit: **−30**, 2+ wrong friend hits ⇒ stage **0**.
  - **Stage‑3:** Target by given engagement (shape+color), **cap=140**. Wrong target: **−50**, 3+ wrong ⇒ stage **0**.
- Safety: emergency stop (hardware), “no‑fire zone” (software mask + mechanical stopper), smooth edges.

## Our Core Idea
A vision‑first, low‑latency pipeline on an **edge device** (RPi5 / Jetson) with:
- **Detection:** lightweight YOLO fine‑tune (single class “balloon”) + classic CV (HSV + contour‑shape) as a robust fallback.
- **Tracking:** fast “SORT‑like” association (Kalman optional) for stable IDs and predictive aiming.
- **Decision:** friend/foe (color), Stage‑3 board cue, shape+color double‑check before firing (conservative gate).
- **Control:** PID/MPC ready; for the demo pack we produce a software fire‑gate and a mock gimbal feed‑forward.
- **Scoring:** real‑time score/BSP computation and logs → auto reports.

## Deliverables (Evidence of Work)
1. **Docs** (this folder): purpose, tasks, running notes (chat summary) for agents.
2. **Data & Training Proof:** synthetic dataset generator + YOLO training script and metrics (PR, mAP, confusion).
3. **Algorithms:** CV balloon/shape detection, tracking module, friend/foe color classifier, Stage‑3 board cue detector.
4. **Integration:** Streamlit UI with “Engagement Accepted” button, no‑fire mask, live score overlay.
5. **Scoring Tools:** log‑to‑score harness (A1/A2/A3 + BSP).
6. **Short Demo Clips:** A1/A2/A3 capabilities (to be recorded by team).

## Success Criteria
- Stable real‑time detection (>30 FPS with small models / lower res).
- Stage‑1/2/3 **base caps reached** in scripted tests; **no wrong hits** in A2/A3.
- Demonstrable safety features: no‑fire mask + hardware E‑Stop.
- Clear, reproducible metrics & logs (code + PDFs + CSV).
