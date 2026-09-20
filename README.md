# MARSE

Minimal inference release for **Multi-Array Region Speech Extraction (MARSE)**.
This directory intentionally includes only the final model path and WAV
inference code. Training, dataset generation, losses, and experiment-specific
branches are not included. The complete codebase will be released in a future
update.

## Setup

```bash
pip install -r requirements.txt
```

Place the trained Lightning checkpoint anywhere on disk. The loader accepts
both the original checkpoint (`state_dict` entries prefixed with `model.`) and
a plain PyTorch state dictionary.

## Input

Inference expects three synchronized 16-kHz WAV files, one per array. Each WAV
must contain six channels in the same microphone order used during training.
The ROI is supplied as lower and upper local azimuth bounds, in degrees, for
the three arrays. An interval with a lower bound greater than its upper bound
crosses the `-180/180` degree boundary.

## Inference

Run from this directory:

```bash
python infer.py \
  --checkpoint /path/to/model.ckpt \
  --input array_1.wav array_2.wav array_3.wav \
  --theta-l -30 -45 -20 \
  --theta-h  20  10  35 \
  --output estimate.wav
```

Use `--device cpu`, `--device cuda`, or leave the default `auto`. The command
writes a mono WAV and reports the selected array, array scores, and predicted
silence probability.
