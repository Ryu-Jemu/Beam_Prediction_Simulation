# Beam Prediction with PaP-GPT2 — Refactored (U=20, H=5)

**Decisions applied**
- Values follow PPTX/PDF defaults where available; otherwise reasonable placeholders.
- Language: English.
- Speed mode: **fixed** (speed in [5, 20] m/s set once per trajectory).
- Angle error **normalized** by π for reporting (`nMSE_theta` in [0,1]).
- Extra information metrics: best-prediction stats, top-k-like summaries, gain-regret approximations.

**Pipeline**
1. **Mobility** (Markov with fixed speed) → trajectory \( (x_t,y_t), v, \text{heading} \).
2. **Channel (SV)** → h_t, AoD φ_t, optimal beam index q_t using a DFT codebook.
3. **Dataset** builds sequences with **U=20** past steps and **H=5** future steps:
   - Features: normalized past beam index, position, heading (sin/cos), speed, AoD (optional).
   - Labels: future continuous AoD (sin, cos) over horizon H.
4. **Model (Req. #6)**: input preprocessing → **patch embedding** → **class-variable attention** → **patch reprogramming** → **PaP** → **GPT-2** (or offline fallback) → projection to (sin,cos)^H.
5. **Training**: circular MSE on sin/cos; metrics include normalized angle MSE, best prediction, hit-rate within tolerance, simple gain-regret proxy.
6. **Visualization**: plots MSE curves; optionally normalised position error using a simple kinematic integration from predicted headings.

> Note: `transformers` GPT-2 requires internet to download the weights. For offline demo, set `use_gpt2 = False` in `setting.py` and the model falls back to a light TransformerEncoder.

## Quick start

```bash
# Optional: create venv and install deps
pip install torch numpy matplotlib transformers==4.44.*

# Train and evaluate
python main.py
```

## Files
- `setting.py` — global config.
- `utils.py` — math helpers, metrics, seeding.
- `mobility.py` — Markov mobility with fixed speed option.
- `channel.py` — SV channel, DFT codebook, optimal beam.
- `dataset.py` — U/H slicing, features, labels.
- `pap_gpt2.py` — PaP prompt embeddings, prototype token set, reprogramming attention utilities.
- `model.py` — PatchEmbedding, ClassVariableAttention, PatchReprogrammer, PaP-GPT2 regressor.
- `train.py` — train/valid loop, metrics logging.
- `visualize.py` — plots for MSE and optional trajectory errors.
- `main.py` — orchestration.

## Metrics
- `nMSE_theta` — normalized angle MSE, wrapping error (Δθ/π)^2.
- `MAE_deg` — mean absolute error in degrees.
- `hit@10deg` — fraction with absolute error ≤ 10°.
- `best_pred_deg` — minimum absolute error across H per sample.
- `gain_regret_db` — proxy from nearest codebook beam mismatch.

## Notes
- All random seeds are set via `setting.seed`.
- For PaP prompts, we use domain, instruction, and statistics prefixes derived from the paper's figures and text. In absence of exact figure captions as raw text, we follow a faithful template that encodes the same semantics.
