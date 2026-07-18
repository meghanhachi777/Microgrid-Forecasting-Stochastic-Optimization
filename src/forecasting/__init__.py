"""Forecasting models.

Submodules (import the one you need; heavy dependencies load lazily):
- persistence : constant and daily-seasonal baselines (numpy only)
- cnn_lstm    : CNN-LSTM v1/v2 (TensorFlow)
- patchtst    : PatchTST transformer (PyTorch + transformers)
- chronos     : Chronos-2 zero-shot pipeline (chronos-forecasting)
"""

from . import persistence  # lightweight; safe to import eagerly
