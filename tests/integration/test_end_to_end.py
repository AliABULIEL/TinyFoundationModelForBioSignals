"""Integration tests for end-to-end workflows."""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.data.datamodule import HARDataModule
from src.models.model_factory import create_model
from src.training.trainer import Trainer
from src.evaluation import Evaluator


@pytest.mark.integration
class TestEndToEndTraining:
    """Integration tests for complete training pipeline."""

    def test_complete_training_pipeline(self, sample_config, temp_dir, device):
        """Test complete training pipeline from data to trained model."""
        # Modify config for quick test
        sample_config["training"]["epochs"] = 2
        sample_config["training"]["batch_size"] = 8
        sample_config["dataset"]["data_path"] = str(temp_dir)

        # Create data module (will use synthetic data)
        data_module = HARDataModule(config=sample_config)
        data_module.setup()

        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        # Should have data
        assert len(train_loader) > 0
        assert len(val_loader) > 0

        # Create model
        model = create_model(sample_config)
        assert model is not None

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=sample_config,
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


    def test_checkpoint_save_and_load(self, sample_config, temp_dir, device):
        """Test saving and loading checkpoints."""
        # Setup quick training
        sample_config["training"]["epochs"] = 1
        sample_config["training"]["batch_size"] = 8
        sample_config["dataset"]["data_path"] = str(temp_dir)

        # Create data module
        data_module = HARDataModule(config=sample_config)
        data_module.setup()

        # Create and train model
        model = create_model(sample_config)
        trainer = Trainer(
            model=model,
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader(),
            config=sample_config,
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
        new_model = create_model(sample_config)
        new_trainer = Trainer(
            model=new_model,
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader(),
            config=sample_config,
            device=device,
            callbacks=[],
        )

        new_trainer.load_checkpoint(str(checkpoint_path))

        # Check epoch was restored
        assert new_trainer.current_epoch == trainer.current_epoch


    def test_evaluation_pipeline(self, sample_config, temp_dir, device):
        """Test complete evaluation pipeline."""
        # Setup
        sample_config["training"]["epochs"] = 1
        sample_config["training"]["batch_size"] = 8
        sample_config["dataset"]["data_path"] = str(temp_dir)

        # Create data
        data_module = HARDataModule(config=sample_config)
        data_module.setup()

        # Create and train model
        model = create_model(sample_config)
        trainer = Trainer(
            model=model,
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader(),
            config=sample_config,
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
class TestDataPipeline:
    """Integration tests for data pipeline."""

    def test_data_module_setup(self, sample_config, temp_dir):
        """Test data module setup and data loading."""
        sample_config["dataset"]["data_path"] = str(temp_dir)

        data_module = HARDataModule(config=sample_config)
        data_module.setup()

        # Check splits created
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()

        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0

    def test_data_module_batch_format(self, sample_config, temp_dir):
        """Test that batches have correct format."""
        sample_config["dataset"]["data_path"] = str(temp_dir)
        sample_config["training"]["batch_size"] = 4

        data_module = HARDataModule(config=sample_config)
        data_module.setup()

        train_loader = data_module.train_dataloader()

        # Get one batch
        batch = next(iter(train_loader))

        # Check format
        assert "signal" in batch
        assert "label" in batch

        # Check shapes
        assert batch["signal"].shape[0] == 4  # Batch size
        assert batch["signal"].shape[2] == 3  # 3 channels
        assert batch["label"].shape[0] == 4

    def test_no_subject_leakage(self, sample_config, temp_dir):
        """Test that there's no subject leakage across splits."""
        sample_config["dataset"]["data_path"] = str(temp_dir)

        data_module = HARDataModule(config=sample_config)
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
        assert len(train_subjects & val_subjects) == 0
        assert len(train_subjects & test_subjects) == 0
        assert len(val_subjects & test_subjects) == 0


@pytest.mark.integration
class TestModelPipeline:
    """Integration tests for model pipeline."""

    def test_model_forward_pass(self, sample_config, sample_batch, device):
        """Test model forward pass."""
        model = create_model(sample_config)
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

    def test_model_backward_pass(self, sample_config, sample_batch, device):
        """Test model backward pass."""
        model = create_model(sample_config)
        model = model.to(device)
        model.train()

        # Forward pass
        inputs = sample_batch["signal"].to(device)
        labels = sample_batch["label"].to(device)

        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)

        # Backward pass
        loss.backward()

        # Check gradients exist for trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_freeze_unfreeze(self, sample_config, device):
        """Test freezing and unfreezing model."""
        model = create_model(sample_config)
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
class TestFullWorkflow:
    """Test complete workflow from preprocessing to evaluation."""

    def test_full_workflow(self, sample_config, temp_dir, device):
        """Test complete workflow: data -> train -> evaluate."""
        # Configure for quick test
        sample_config["training"]["epochs"] = 2
        sample_config["training"]["batch_size"] = 8
        sample_config["dataset"]["data_path"] = str(temp_dir)

        # 1. Setup data
        data_module = HARDataModule(config=sample_config)
        data_module.setup()

        # 2. Create model
        model = create_model(sample_config)

        # 3. Train
        trainer = Trainer(
            model=model,
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader(),
            config=sample_config,
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
