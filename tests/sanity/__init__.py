"""Sanity tests for catching common ML bugs.

These tests verify that the model and training pipeline work correctly
by checking for common issues like:
- Training on random labels (should fail)
- Overfitting small batches (should succeed)
- Constant predictions (should be avoided)
- Gradient flow (should work)
- Determinism (should be reproducible)

All tests use synthetic data and are designed to run quickly on CPU.
"""
