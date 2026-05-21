# LOSO 10-Model Counting Experiment Plan

## Position

This experiment is a standalone cough-counting model comparison. It is designed to evaluate end-to-end count-oriented models under subject-independent validation.

## Evidence Blocks

| Block | Role |
|---|---|
| Data preparation | Build the EdgeAI manifest, subject splits, STFT features, and density targets |
| 10-model LOSO comparison | Compare model families under a matched 0.7M-parameter budget |
| Report package | Preserve ranked results, fold-level metrics, runtime environment, and exact configs |

## Models

```text
CNN1D
DSCNN
ResCNN
CRNN
BiCRNN
BiGRU
TCN
TCN-Attn
TCN+UniGRU
TCN+BiGRU
```

All models use the same dataset, windowing, density target, LOSO split logic, training epochs, optimizer settings, and primary metric.

## Interpretation

The experiment should be described as a capacity-controlled model-family comparison:

> Under a similar parameter budget, recurrent and attention-based temporal models outperform purely convolutional baselines. BiCRNN obtains the lowest Count MAE, while TCN-Attn and TCN+GRU variants form the next tier.

Avoid describing this as strict component ablation. Strict ablation should start from one fixed model and remove one component at a time.
