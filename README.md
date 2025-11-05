# Beam Prediction with PaP-LLM (SV Channel, DFT Codebook)

This repository contains a Jupyter notebook (`Sim_ver_01.ipynb`) that simulates a mmWave downlink with user mobility, builds a sparse-vector (SV) channel, finds the optimal DFT beam per time step, and trains a GPT-2 model to predict future beams using Prefix-as-Prompt (PaP). The evaluation uses the Normalized Beamforming Gain (NBG).

> **Scope**: 1 trajectory, slides-faithful Markov mobility + specular reflections, SV channel (LOS + NLOS), GPT-2 PaP, NBG metric and basic visualization.

---

## Key Features

- **Mobility (slides-faithful)**
  - Markov model with probabilistic heading changes: `p_turn = 0.1`, `Δθ = ±10°`.
  - **Specular reflections** at area boundaries (vertical, horizontal, and corner cases).

- **Channel**
  - **Sparse-vector model (SV)**: 1 LOS path + `(L-1)` NLOS paths (Rayleigh).
  - **ULA**: `M=32` with half-wavelength spacing at `f_c = 30 GHz`.
  - **DFT codebook**: `Q=32` beams; optimal beam `q* = argmax |h^H f_q|^2`.

- **LLM (PaP)**
  - GPT-2 Causal LM with **beam tokens** `<B0> .. <B31>`.
  - **Prefix-as-Prompt**: first `U` beams are context; predict `H` future beams.
  - Trainer uses CE loss; **pad_token** is safely added.

- **Metrics**
  - **NBG** (normalized beamforming gain) averaged over horizon.
  - Optional per-step NBG@k (extendable).

- **Engineering**
  - **MPS/CUDA/CPU** device auto-selection.
  - Complex math kept **MPS-safe** (no `torch.vdot`, no `1j` literals).
  - **Reproducible** seeds (PyTorch & NumPy).

---

## File

- `Sim_ver_01.ipynb` — end-to-end notebook (simulation → channel → optimal beams → PaP training → NBG evaluation → plots)

---

## Requirements

- Python 3.10+
- PyTorch 2.x (CUDA or MPS optional)
- `transformers`, `tqdm`, `matplotlib`
- (optional) `seaborn`, `pandas` for enhanced plots

Example:
```bash
pip install torch torchvision torchaudio
pip install transformers tqdm matplotlib seaborn pandas