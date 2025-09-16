# Tasks & Backlog
**Version:** 0.1 • **Updated:** 2025-09-16 11:26

## P0 — Critical Path (this week)
- [ ] Prepare synthetic dataset (200–500 imgs) with color balloons + shapes; export COCO labels.
- [ ] Train YOLOv8‑nano for **single class "balloon"** (fine‑tune); export ONNX.
- [ ] Implement CV fallback (HSV + contour) for balloon + shape classification.
- [ ] Implement light tracker (SORT‑like) with stable ID; produce a short tracking demo.
- [ ] Implement friend/foe color gate with conservative thresholds and “no‑fire on doubt” policy.
- [ ] Implement Stage‑3 cue parsing (board ROI + shape+color double‑check).
- [ ] Implement **score harness** (A1=60/A2=100/A3=140 caps + BSP) over logs.
- [ ] Implement Streamlit UI: live view, “Engagement Accepted” button, **no‑fire mask**, live score panel.
- [ ] Record short capability clips for A1/A2/A3.

## P1 — Optimizations
- [ ] Quantize / TensorRT export; measure FPS on RPi5/Jetson (512p input baseline).
- [ ] ByteTrack option; ID‑switch minimization.
- [ ] Robust color normalization: WB, gamma, CLAHE; illumination tests.

## P2 — Safety & Mech Integration
- [ ] Hardware E‑Stop wiring diagram; E‑Stop functional test video.
- [ ] Mechanical hard‑stops matched with software mask; combined test.

## Artifacts to Produce
- `runs/train/results.png`, `PR_curve.png`, `confusion_matrix.png`, `train_settings.json`
- `videos/` A1/A2/A3 raw clips
- `reports/score_report.csv`, `score_report.pdf`
