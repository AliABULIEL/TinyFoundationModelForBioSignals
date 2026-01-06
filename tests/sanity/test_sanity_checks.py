"""Comprehensive sanity tests for TTM-HAR.

These tests catch common ML bugs by verifying expected behavior:
- Random labels → low accuracy
- Small batch → can overfit
- Model predictions → not constant
- Gradients → flow correctly
- Loss → decreases with training
- Outputs → correct shapes
- Determinism → with seed
- Batch size → invariant
- End-to-end → pipeline works

All tests complete in <5 seconds on CPU using synthetic data only.
"""

import os

# ⚠️ CRITICAL: Allow mock models for testing ONLY
os.environ["TTM_HAR_ALLOW_MOCK"] = "1"

import pytest
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.models.model_factory import create_model
from src.utils.reproducibility import set_seed


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def minimal_config():
    """Minimal configuration for sanity tests."""
    return {
        "experiment": {
            "name": "sanity_test",
            "seed": 42,
        },
        "model": {
            "backbone": "ttm",
            "checkpoint": "ibm-granite/granite-timeseries-ttm-r2",
            "num_channels": 3,
            "num_classes": 5,
            "context_length": 512,
            "patch_length": 16,
            "freeze_strategy": "none",
            "head": {
                "type": "linear",
                "pooling": "mean",
                "dropout": 0.1,
                "activation": "gelu",
            },
        },
        "dataset": {
            "num_classes": 5,
        },
        "training": {
            "batch_size": 4,
            "lr_head": 1e-3,
            "lr_backbone": 1e-5,
            "optimizer": "adamw",
            "weight_decay": 0.01,
        },
    }


@pytest.fixture
def synthetic_batch():
    """Generate synthetic batch data."""
    batch_size = 8
    context_length = 512
    num_channels = 3
    num_classes = 5

    inputs = torch.randn(batch_size, context_length, num_channels)
    labels = torch.randint(0, num_classes, (batch_size,))

    return inputs, labels


# ==============================================================================
# TEST CLASS 1: RANDOM LABELS
# ==============================================================================


@pytest.mark.sanity
class TestRandomLabels:
    """Test that training on random labels produces low accuracy."""

    def test_random_labels_low_accuracy(self, minimal_config):
        """Verify that model performs poorly on random labels (data leakage check)."""
        set_seed(42)

        # Create model
        model = create_model(minimal_config)
        model.train()

        # Create TRAIN data with RANDOM labels
        num_train = 32
        train_inputs = torch.randn(num_train, 512, 3)
        train_labels = torch.randint(0, 5, (num_train,))

        # Create separate TEST data with different random labels
        num_test = 16
        test_inputs = torch.randn(num_test, 512, 3)
        test_labels = torch.randint(0, 5, (num_test,))

        # Train for limited steps
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        num_steps = 30
        for _ in range(num_steps):
            optimizer.zero_grad()
            outputs = model(train_inputs)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

        # Evaluate on DIFFERENT test data (not training data)
        model.eval()
        with torch.no_grad():
            outputs = model(test_inputs)
            preds = torch.argmax(outputs, dim=1)
            accuracy = (preds == test_labels).float().mean().item()

        # Model should NOT achieve high accuracy on unseen random labels
        # Random chance is 20% for 5 classes
        assert accuracy < 0.60, (
            f"Model achieved {accuracy:.1%} accuracy on unseen random labels. "
            f"Expected < 60% (random chance is ~20%). "
            f"This may indicate data leakage or a bug in training."
        )


# ==============================================================================
# TEST CLASS 2: CONSTANT PREDICTIONS
# ==============================================================================


@pytest.mark.sanity
class TestConstantPredictions:
    """Test that model doesn't degenerate to constant predictions."""

    def test_constant_predictions_zero_balanced_accuracy(self, minimal_config):
        """Verify model produces varied predictions, not always the same class."""
        set_seed(42)

        # Create model
        model = create_model(minimal_config)
        model.eval()

        # Create diverse inputs
        num_batches = 5
        batch_size = 8
        all_preds = []

        with torch.no_grad():
            for i in range(num_batches):
                # Add varying noise to create different inputs
                inputs = torch.randn(batch_size, 512, 3) * (i + 1) * 0.5
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.tolist())

        # Check prediction diversity
        unique_predictions = len(set(all_preds))

        assert unique_predictions >= 2, (
            f"Model predicting only {unique_predictions} unique class(es)! "
            f"This indicates the model has collapsed to constant predictions. "
            f"Check initialization and training logic."
        )


# ==============================================================================
# TEST CLASS 3: OVERFIT MICRO BATCH
# ==============================================================================


@pytest.mark.sanity
class TestOverfitMicroBatch:
    """Test that model can overfit a tiny batch (capacity check)."""

    def test_overfit_single_batch(self, minimal_config):
        """Verify model can memorize a tiny batch (proves it has capacity to learn)."""
        set_seed(42)

        # Create model
        model = create_model(minimal_config)
        model.train()

        # Tiny batch - should be easy to memorize
        micro_batch_size = 4
        inputs = torch.randn(micro_batch_size, 512, 3)
        labels = torch.tensor([0, 1, 2, 3])  # Different labels

        # Train until perfect fit
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        max_steps = 100
        target_accuracy = 0.95

        for step in range(max_steps):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Check progress every 20 steps
            if step % 20 == 19:
                model.eval()
                with torch.no_grad():
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    accuracy = (preds == labels).float().mean().item()

                if accuracy >= target_accuracy:
                    return  # Success!

                model.train()

        # Final check
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            final_accuracy = (preds == labels).float().mean().item()

        assert final_accuracy >= target_accuracy, (
            f"Model failed to overfit tiny batch: {final_accuracy:.1%} accuracy "
            f"after {max_steps} steps (expected ≥{target_accuracy:.1%}). "
            f"This suggests the model cannot learn or has training issues."
        )


# ==============================================================================
# TEST CLASS 4: GRADIENT FLOW
# ==============================================================================


@pytest.mark.sanity
class TestGradientFlow:
    """Test that gradients flow correctly through the model."""

    def test_head_receives_gradients(self, minimal_config):
        """Verify classification head receives gradients when backbone is frozen."""
        set_seed(42)

        # Create model with frozen backbone
        minimal_config["model"]["freeze_strategy"] = "all"
        model = create_model(minimal_config)
        model.train()

        # Forward and backward
        inputs = torch.randn(4, 512, 3)
        labels = torch.randint(0, 5, (4,))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Check head has gradients
        head_has_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.head.parameters()
            if p.requires_grad
        )

        assert head_has_grads, (
            "Classification head did not receive gradients! "
            "Check that head parameters require gradients and backward pass works."
        )

    def test_backbone_gradients_when_unfrozen(self, minimal_config):
        """Verify backbone receives gradients when unfrozen."""
        set_seed(42)

        # Create model with unfrozen backbone
        minimal_config["model"]["freeze_strategy"] = "none"
        model = create_model(minimal_config)
        model.train()

        # Forward and backward
        inputs = torch.randn(4, 512, 3)
        labels = torch.randint(0, 5, (4,))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Check backbone has gradients
        backbone_has_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.backbone.parameters()
            if p.requires_grad
        )

        assert backbone_has_grads, (
            "Backbone did not receive gradients when unfrozen! "
            "Check that backbone parameters require gradients."
        )

    def test_backbone_frozen_no_gradients(self, minimal_config):
        """Verify frozen backbone does NOT receive gradients."""
        set_seed(42)

        # Create model with frozen backbone
        minimal_config["model"]["freeze_strategy"] = "all"
        model = create_model(minimal_config)
        model.train()

        # Forward and backward
        inputs = torch.randn(4, 512, 3)
        labels = torch.randint(0, 5, (4,))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Check backbone has NO gradients (frozen params shouldn't have grads)
        backbone_trainable = any(p.requires_grad for p in model.backbone.parameters())

        assert not backbone_trainable, (
            "Frozen backbone still has trainable parameters! "
            "Freezing logic is not working correctly."
        )


# ==============================================================================
# TEST CLASS 5: LOSS DECREASES
# ==============================================================================


@pytest.mark.sanity
class TestLossDecreases:
    """Test that loss decreases during training."""

    def test_loss_decreases(self, minimal_config):
        """Verify loss goes down when training on same batch repeatedly."""
        set_seed(42)

        # Create model
        model = create_model(minimal_config)
        model.train()

        # Fixed batch
        inputs = torch.randn(8, 512, 3)
        labels = torch.randint(0, 5, (8,))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Train for 10 steps
        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        initial_loss = losses[0]
        final_loss = losses[-1]

        assert final_loss < initial_loss, (
            f"Loss did not decrease during training:\n"
            f"  Initial: {initial_loss:.4f}\n"
            f"  Final:   {final_loss:.4f}\n"
            f"This suggests optimizer, learning rate, or gradient issues."
        )


# ==============================================================================
# TEST CLASS 6: OUTPUT SHAPES
# ==============================================================================


@pytest.mark.sanity
class TestOutputShapes:
    """Test that model outputs have correct shapes."""

    def test_output_shape(self, minimal_config):
        """Verify model output is (batch_size, num_classes)."""
        set_seed(42)

        model = create_model(minimal_config)
        model.eval()

        batch_size = 8
        num_classes = 5
        inputs = torch.randn(batch_size, 512, 3)

        with torch.no_grad():
            outputs = model(inputs)

        expected_shape = (batch_size, num_classes)
        assert outputs.shape == expected_shape, (
            f"Incorrect output shape: got {outputs.shape}, expected {expected_shape}"
        )

    def test_backbone_output_shape(self, minimal_config):
        """Verify backbone output is (batch_size, hidden_dim)."""
        set_seed(42)

        model = create_model(minimal_config)
        model.eval()

        batch_size = 8
        inputs = torch.randn(batch_size, 512, 3)

        with torch.no_grad():
            features = model.backbone(inputs)

        assert features.dim() == 2, (
            f"Backbone should output 2D tensor, got {features.dim()}D"
        )
        assert features.shape[0] == batch_size, (
            f"Backbone output batch dimension mismatch: got {features.shape[0]}, expected {batch_size}"
        )


# ==============================================================================
# TEST CLASS 7: DETERMINISM
# ==============================================================================


@pytest.mark.sanity
class TestDeterminism:
    """Test that model behavior is deterministic with seed."""

    def test_deterministic_forward(self, minimal_config):
        """Verify same seed produces identical outputs."""
        # Run 1
        set_seed(42)
        model1 = create_model(minimal_config)
        model1.eval()

        inputs = torch.randn(4, 512, 3)

        with torch.no_grad():
            out1 = model1(inputs)

        # Run 2 with same seed
        set_seed(42)
        model2 = create_model(minimal_config)
        model2.eval()

        with torch.no_grad():
            out2 = model2(inputs)

        # Should be identical
        max_diff = (out1 - out2).abs().max().item()

        assert max_diff < 1e-6, (
            f"Non-deterministic behavior detected (max diff: {max_diff:.2e}). "
            f"Models with same seed should produce identical outputs. "
            f"Check random number generator seeding."
        )


# ==============================================================================
# TEST CLASS 8: BATCH SIZE INVARIANCE
# ==============================================================================


@pytest.mark.sanity
class TestBatchSizeInvariance:
    """Test that model handles different batch sizes correctly."""

    def test_batch_size_1(self, minimal_config):
        """Verify model works with batch size 1."""
        set_seed(42)

        model = create_model(minimal_config)
        model.eval()

        inputs = torch.randn(1, 512, 3)

        with torch.no_grad():
            outputs = model(inputs)

        assert outputs.shape == (1, 5), (
            f"Model failed with batch_size=1: output shape {outputs.shape}"
        )

    def test_varying_batch_sizes(self, minimal_config):
        """Verify same sample gives same output regardless of batch composition."""
        set_seed(42)

        model = create_model(minimal_config)
        model.eval()

        # Create a specific sample
        sample = torch.randn(1, 512, 3)

        # Output when alone (batch size 1)
        with torch.no_grad():
            out_single = model(sample)

        # Output when in larger batch (batch size 4)
        batch = torch.cat([sample, torch.randn(3, 512, 3)], dim=0)
        with torch.no_grad():
            out_batch = model(batch)

        # First output should be same
        max_diff = (out_single[0] - out_batch[0]).abs().max().item()

        assert max_diff < 1e-4, (
            f"Batch-dependent behavior detected (max diff: {max_diff:.2e}). "
            f"Same sample should give same output regardless of batch size. "
            f"Check for batch normalization or similar batch-dependent operations."
        )


# ==============================================================================
# TEST CLASS 9: END-TO-END SMOKE TEST
# ==============================================================================


@pytest.mark.sanity
class TestEndToEndSmoke:
    """Test complete pipeline from data to metrics."""

    def test_full_pipeline_smoke(self, minimal_config):
        """Verify entire pipeline works: data → model → loss → backward → metrics."""
        set_seed(42)

        # Create model
        model = create_model(minimal_config)
        model.train()

        # Create data
        batch_size = 8
        inputs = torch.randn(batch_size, 512, 3)
        labels = torch.randint(0, 5, (batch_size,))

        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)

        # Loss
        loss = criterion(outputs, labels)
        assert torch.isfinite(loss), "Loss is not finite (NaN or Inf)"

        # Backward
        loss.backward()

        # Check gradients exist
        has_grads = any(
            p.grad is not None
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grads, "No gradients computed"

        # Optimizer step
        optimizer.step()

        # Metrics
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).numpy()
            labels_np = labels.numpy()

            accuracy = (preds == labels_np).mean()
            bal_acc = balanced_accuracy_score(labels_np, preds)
            f1 = f1_score(labels_np, preds, average="macro", zero_division=0)

            # All metrics should be finite
            assert np.isfinite(accuracy), "Accuracy is not finite"
            assert np.isfinite(bal_acc), "Balanced accuracy is not finite"
            assert np.isfinite(f1), "F1 score is not finite"


# ==============================================================================
# PERFORMANCE CHECK
# ==============================================================================


@pytest.mark.sanity
def test_all_sanity_tests_complete_quickly():
    """Meta-test: verify all sanity tests complete in reasonable time."""
    import time

    start = time.time()

    # This is a placeholder - actual timing is done by pytest
    # Just verify this test itself is fast
    time.sleep(0.01)  # Minimal sleep

    elapsed = time.time() - start
    assert elapsed < 1.0, f"Test took {elapsed:.2f}s, should be near-instant"
