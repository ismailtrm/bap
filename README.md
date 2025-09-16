# Hava Savunma Demo — Starter Pack
Updated: 2025-09-16 11:35

## What you have
- **Synthetic data generator** (balloons with color & shape) → COCO/YOLO labels
- **Classic CV detection** (HSV + contour + shape)
- **YOLO train/infer** scripts (ultralytics)
- **Tracker** (lightweight SORT-like, ID stability)
- **Friend/Foe** classifier (color gate, conservative policy)
- **Stage-3** cue helper (board ROI + shape+color double-check)
- **Score harness** (A1/A2/A3 + BSP from logs)
- **Streamlit UI** (engagement button, no-fire mask, score overlay)

## Quick start (suggested order)
1. Create a venv and install `pip install -r requirements.txt`.
2. Generate synthetic data: `python scripts/synthetic_generator.py --out data/synth_samples --train 200 --val 60`.
3. Train YOLO: `python scripts/yolo_train.py --data data/yolo/data.yaml --epochs 60`.
4. Test classic CV: `python scripts/cv_color_shape_detect.py --source 0` (or a video path).
5. Run tracker demo: `python scripts/track_demo.py --source 0 --det cv` (or `--det yolo` after training).
6. Streamlit UI: `streamlit run scripts/ui_app.py`.
7. After recording a session, compute scores: `python scripts/score_harness.py --stage 1 --log logs/events.csv --end 300`.

## Notes
- All scripts include **English comments** and safe defaults.
- Stage-2/3 penalties are handled by the score harness.
- Real hardware integration (gimbal, E-Stop) is outside this pack; we provide software no-fire mask and clean interfaces.
