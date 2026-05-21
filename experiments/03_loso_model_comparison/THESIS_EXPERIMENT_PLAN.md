# Chapter 5 Experiment Plan

## Position

This experiment supports the cough-counting section of Chapter 5. The final thesis line is:

```text
Original WST window-level cough detector can support cough response prompting
but event counting needs a dedicated count-oriented model.
```

The current model experiment is no longer framed as TCN/GRU ablation. It is a unified 10-model structure comparison under a matched 0.7M-parameter budget.

## Evidence Blocks

| Block | Status | Thesis role |
|---|---|---|
| WST window detector + merge counting | Completed separately | Shows the original WST system is sensitive to cough events but is not an ideal event counter |
| 10-model LOSO comparison | Completed / reproducible by this folder | Selects and explains count-oriented model structures |
| Release report package | Generated from final summaries | Provides tables, fold details, configs, and environment evidence |

## Main Models

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

Avoid writing it as a strict component ablation.
