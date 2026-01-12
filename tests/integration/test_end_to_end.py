"""Integration tests for end-to-end workflows.

REQUIREMENTS:
- Real IBM TTM model must be installed
- Real CAPTURE-24 data must be available for full integration tests

Tests will be skipped if requirements are not met.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import os


# ==============================================================================
# ENVIRONMENT CHECKS
# ==============================================================================

def _check_ttm_available():
    """Check if real TTM is available."""
    try:
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
        return True
    except ImportError:
        try:
            from granite_tsfm.models import TinyTimeMixerForPrediction
            return True
        except ImportError:
            return False


def _check_data_available(data_path="data/capture24"):
    """Check if real CAPTURE-24 data is available."""
    path = Path(data_path)
    if not path.exists():
        return False
    # Check for at least one participant folder
    participant_dirs = [d for d in path.iterdir() if d.is_dir() and d.name.startswith("P")]
    return len(participant_dirs) > 0


TTM_AVAILABLE = _check_ttm_available()
DATA_PATH = os.environ.get("CAPTURE24_DATA_PATH", "data/capture24")
DATA_AVAILABLE = _check_data_available(DATA_PATH)


# Skip markers
requires_ttm = pytest.mark.skipif(
    not TTM_AVAILABLE,
    reason="Requires real IBM TTM model"
)

requires_data = pytest.mark.skipif(
    not DATA_AVAILABLE,
    reason=f"Requires real CAPTURE-24 data at {DATA_PATH}"
)

requires_full_setup = pytest.mark.skipif(
    not (TTM_AVAILABLE and DATA_AVAILABLE),
    reason="Requires both TTM model and CAPTURE-24 data"
)


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

@pytest.mark.integration
@requires_full_setup
class TestEndToEndTraining:
    """Integration tests for complete training pipeline."""

    @pytest.fixture
    def integration_config(self):
        """Config for integration tests with real data."""
        return {
            "experiment": {
                "name": "integration_test",
                "seed": 42,
                "output_dir": "outputs",
            },
            "dataset": {
                "name": "capture24",
                "data_path": DATA_PATH,
                "num_classes": 5,
                "train_split": 0.7,
                "val_split": 0.15,
                "test_split": 0.15,
            },
            "preprocessing": {
                "sampling_rate_original": 100,
                "sampling_rate_target": 30,
                "context_length": 512,
                "patch_length": 16,
                "window_stride_train": 256,
                "window_stride_eval": 512,
                "resampling_method": "polyphase",
                "normalization": {"method": "zscore", "epsilon": 1e-8},
                "gravity_removal": {"enabled": False},
            },
            "model": {
                "backbone": "ttm",
                "checkpoint": "ibm-granite/granite-timeseries-ttm-r2",
                "num_channels": 3,
                "num_classes": 5,
                "context_length": 512,
                "patch_length": 16,
                "freeze_strategy": "all",
                "head": {
                    "type": "linear",
                    "pooling": "mean",
                    "dropout": 0.1,
                    "activation": "gelu",
                },
            },
            "training": {
                "strategy": "linear_probe",
                "epochs": 2,  # Quick test
                "batch_size": 8,
                "lr_head": 1e-3,
                "lr_backbone": 1e-5,
                "weight_decay": 0.01,
                "optimizer": "adamw",
                "scheduler": "cosine",
                "warmup_ratio": 0.1,
                "gradient_clip_norm": 1.0,
                "loss": {"type": "weighted_ce"},
            },
            "hardware": {
                "device": None,
                "num_workers": 0,
                "pin_memory": False,
                "mixed_precision": False,
            },
        }

    def test_complete_training_pipeline(self, integration_config, temp_dir, device):
        """Test complete training pipeline from data to trained model."""
        from src.data.datamodule import HARDataModule
        from src.models.model_factory import create_model
        from src.training.trainer import Trainer

        # Create data module with real data
        data_module = HARDataModule(config=integration_config)
        data_module.setup()

        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        # Should have data
        assert len(train_loader) > 0
        assert len(val_loader) > 0

        # Create model
        model = create_model(integration_config)
        assert model is not None

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=integration_config,
            device=device,
            callbacks=[],
        )

        # Train for 2 epochs
        history = trainer.train()

        # Check training completed
        assert history["num_epochs"] == 2
        assert len(history["train_losses"]) == 2
        assert len(history["val_metrics"]) == 2

        # Check metrics are reasonable
        final_metrics = history["val_metrics"][-1]
        assert "accuracy" in final_metrics
        assert 0 <= final_metrics["accuracy"] <= 1

    def test_checkpoint_save_and_load(self, integration_config, temp_dir, device):
        """Test saving and loading checkpoints."""
        from src.data.datamodule import HARDataModule
        from src.models.model_factory import create_model
        from src.training.trainer import Trainer

        integration_config["training"]["epochs"] = 1

        # Create data module
        data_module = HARDataModule(config=integration_config)
        data_module.setup()

        # Create and train model
        model = create_model(integration_config)
        trainer = Trainer(
            model=model,
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader(),
            config=integration_config,
            device=device,
            callbacks=[],
        )

        # Train
        trainer.train()

        # Save checkpoint
        checkpoint_path = temp_dir / "test_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))

        assert checkpoint_path.exists()

        # Create new trainer and load checkpoint
        new_model = create_model(integration_config)
        new_trainer = Trainer(
            model=new_model,
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader(),
            config=integration_config,
            device=device,
            callbacks=[],
        )

        new_trainer.load_checkpoint(str(checkpoint_path))

        # Check epoch was restored
        assert new_trainer.current_epoch == trainer.current_epoch

    def test_evaluation_pipeline(self, integration_config, temp_dir, device):
        """Test complete evaluation pipeline."""
        from src.data.datamodule import HARDataModule
        from src.models.model_factory import create_model
        from src.training.trainer import Trainer
        from src.evaluation import Evaluator

        integration_config["training"]["epochs"] = 1

        # Create data
        data_module = HARDataModule(config=integration_config)
        data_module.setup()

        # Create and train model
        model = create_model(integration_config)
        trainer = Trainer(
            model=model,
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader(),
            config=integration_config,
            device=device,
            callbacks=[],
        )
        trainer.train()

        # Evaluate
        evaluator = Evaluator(
            model=model,
            device=device,
            label_names=list(data_module.get_label_map().values()),
        )

        results = evaluator.evaluate(
            data_module.test_dataloader(),
            include_per_class=True,
            include_confusion_matrix=True,
        )

        # Check results
        assert "accuracy" in results
        assert "confusion_matrix" in results
        assert "per_class" in results

        # Check per-class metrics
        per_class = results["per_class"]
        assert len(per_class["precision"]) == 5  # 5 classes
        assert len(per_class["recall"]) == 5
        assert len(per_class["f1"]) == 5


@pytest.mark.integration
@requires_full_setup
class TestDataPipeline:
    """Integration tests for data pipeline."""

    @pytest.fixture
    def data_config(self):
        """Config for data pipeline tests."""
        return {
            "experiment": {"seed": 42},
            "dataset": {
                "name": "capture24",
                "data_path": DATA_PATH,
                "num_classes": 5,
                "train_split": 0.7,
                "val_split": 0.15,
                "test_split": 0.15,
            },
            "preprocessing": {
                "sampling_rate_original": 100,
                "sampling_rate_target": 30,
                "context_length": 512,
                "patch_length": 16,
                "window_stride_train": 256,
                "window_stride_eval": 512,
                "resampling_method": "polyphase",
                "normalization": {"method": "zscore"},
                "gravity_removal": {"enabled": False},
            },
            "training": {"batch_size": 4},
            "hardware": {"num_workers": 0, "pin_memory": False},
        }

    def test_data_module_setup(self, data_config):
        """Test data module setup and data loading."""
        from src.data.datamodule import HARDataModule

        data_module = HARDataModule(config=data_config)
        data_module.setup()

        # Check splits created
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()

        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0

    def test_data_module_batch_format(self, data_config):
        """Test that batches have correct format."""
        from src.data.datamodule import HARDataModule

        data_module = HARDataModule(config=data_config)
        data_module.setup()

        train_loader = data_module.train_dataloader()

        # Get one batch
        batch = next(iter(train_loader))

        # Check format
        assert "signal" in batch
        assert "label" in batch

        # Check shapes
        assert batch["signal"].shape[0] == 4  # Batch size
        assert batch["signal"].shape[1] == 512  # Context length
        assert batch["signal"].shape[2] == 3  # 3 channels
        assert batch["label"].shape[0] == 4

    def test_no_subject_leakage(self, data_config):
        """Test that there's no subject leakage across splits."""
        from src.data.datamodule import HARDataModule

        data_module = HARDataModule(config=data_config)
        data_module.setup()

        # Get datasets
        train_dataset = data_module.train_dataset
        val_dataset = data_module.val_dataset
        test_dataset = data_module.test_dataset

        # Get participant IDs from each dataset
        train_subjects = set(train_dataset.participant_ids)
        val_subjects = set(val_dataset.participant_ids)
        test_subjects = set(test_dataset.participant_ids)

        # Check no overlap
        assert len(train_subjects & val_subjects) == 0, "Subject leakage between train and val"
        assert len(train_subjects & test_subjects) == 0, "Subject leakage between train and test"
        assert len(val_subjects & test_subjects) == 0, "Subject leakage between val and test"


@pytest.mark.integration
@requires_ttm
class TestModelPipeline:
    """Integration tests for model pipeline (only needs TTM, not data)."""

    @pytest.fixture
    def model_config(self):
        """Config for model tests."""
        return {
            "experiment": {"seed": 42},
            "dataset": {"num_classes": 5},
            "model": {
                "backbone": "ttm",
                "checkpoint": "ibm-granite/granite-timeseries-ttm-r2",
                "num_channels": 3,
                "num_classes": 5,
                "context_length": 512,
                "patch_length": 16,
                "freeze_strategy": "all",
                "head": {
                    "type": "linear",
                    "pooling": "mean",
                    "dropout": 0.1,
                    "activation": "gelu",
                },
            },
            "training": {"lr_head": 1e-3, "lr_backbone": 1e-5},
        }

    def test_model_forward_pass(self, model_config, sample_batch, device):
        """Test model forward pass."""
        from src.models.model_factory import create_model

        model = create_model(model_config)
        model = model.to(device)
        model.eval()

        # Move batch to device
        inputs = sample_batch["signal"].to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)

        # Check output shape
        assert outputs.shape[0] == inputs.shape[0]  # Batch size
        assert outputs.shape[1] == 5  # Num classes

    def test_model_backward_pass(self, model_config, sample_batch, device):
        """Test model backward pass."""
        from src.models.model_factory import create_model

        model = create_model(model_config)
        model = model.to(device)
        model.train()

        # Unfreeze for gradient test
        model.unfreeze_backbone()

        # Forward pass
        inputs = sample_batch["signal"].to(device)
        labels = sample_batch["label"].to(device)

        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)

        # Backward pass
        loss.backward()

        # Check gradients exist for trainable parameters
        has_grads = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grads = True
                break

        assert has_grads, "No gradients computed"

    def test_freeze_unfreeze(self, model_config, device):
        """Test freezing and unfreezing model."""
        from src.models.model_factory import create_model

        model = create_model(model_config)
        model = model.to(device)

        # Freeze backbone
        model.freeze_backbone(strategy="all")

        # Check backbone frozen
        for param in model.backbone.parameters():
            assert not param.requires_grad

        # Head should still be trainable
        for param in model.head.parameters():
            assert param.requires_grad

        # Unfreeze backbone
        model.unfreeze_backbone()

        # Check backbone unfrozen
        for param in model.backbone.parameters():
            assert param.requires_grad


@pytest.mark.integration
@pytest.mark.slow
@requires_full_setup
class TestFullWorkflow:
    """Test complete workflow from preprocessing to evaluation."""

    @pytest.fixture
    def full_config(self):
        """Full config for workflow test."""
        return {
            "experiment": {"name": "full_workflow_test", "seed": 42},
            "dataset": {
                "name": "capture24",
                "data_path": DATA_PATH,
                "num_classes": 5,
                "train_split": 0.7,
                "val_split": 0.15,
                "test_split": 0.15,
            },
            "preprocessing": {
                "sampling_rate_original": 100,
                "sampling_rate_target": 30,
                "context_length": 512,
                "patch_length": 16,
                "window_stride_train": 256,
                "window_stride_eval": 512,
                "resampling_method": "polyphase",
                "normalization": {"method": "zscore"},
                "gravity_removal": {"enabled": False},
            },
            "model": {
                "backbone": "ttm",
                "checkpoint": "ibm-granite/granite-timeseries-ttm-r2",
                "num_channels": 3,
                "num_classes": 5,
                "context_length": 512,
                "patch_length": 16,
                "freeze_strategy": "all",
                "head": {"type": "linear", "pooling": "mean", "dropout": 0.1},
            },
            "training": {
                "strategy": "linear_probe",
                "epochs": 2,
                "batch_size": 8,
                "lr_head": 1e-3,
                "lr_backbone": 1e-5,
                "weight_decay": 0.01,
                "optimizer": "adamw",
                "scheduler": "cosine",
                "warmup_ratio": 0.1,
                "gradient_clip_norm": 1.0,
                "loss": {"type": "weighted_ce"},
            },
            "hardware": {"num_workers": 0, "pin_memory": False},
        }

    def test_full_workflow(self, full_config, temp_dir, device):
        """Test complete workflow: data -> train -> evaluate."""
        from src.data.datamodule import HARDataModule
        from src.models.model_factory import create_model
        from src.training.trainer import Trainer
        from src.evaluation import Evaluator

        # 1. Setup data
        data_module = HARDataModule(config=full_config)
        data_module.setup()

        # 2. Create model
        model = create_model(full_config)

        # 3. Train
        trainer = Trainer(
            model=model,
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader(),
            config=full_config,
            device=device,
            callbacks=[],
        )

        history = trainer.train()

        # 4. Save checkpoint
        checkpoint_path = temp_dir / "model.pt"
        trainer.save_checkpoint(str(checkpoint_path))

        # 5. Evaluate
        evaluator = Evaluator(
            model=model,
            device=device,
            label_names=list(data_module.get_label_map().values()),
        )

        results = evaluator.evaluate(data_module.test_dataloader())

        # 6. Verify results
        assert history["num_epochs"] == 2
        assert "accuracy" in results
        assert 0 <= results["accuracy"] <= 1
        assert checkpoint_path.exists()
