"""Tests for training utilities and trainers."""

import json
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.trainers import (
    EarlyStopping,
    FocalLoss,
    TrainerBase,
    TrainerClf,
    TrainerReg,
    create_optimizer,
    create_scheduler,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_dim: int = 10, output_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestFocalLoss:
    """Test focal loss implementation."""
    
    def test_focal_loss_shape(self):
        """Test focal loss output shape."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        inputs = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))
        
        loss = loss_fn(inputs, targets)
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
    
    def test_focal_loss_reduction(self):
        """Test different reduction modes."""
        inputs = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))
        
        # Mean reduction
        loss_mean = FocalLoss(reduction='mean')
        loss_val_mean = loss_mean(inputs, targets)
        assert loss_val_mean.shape == torch.Size([])
        
        # Sum reduction
        loss_sum = FocalLoss(reduction='sum')
        loss_val_sum = loss_sum(inputs, targets)
        assert loss_val_sum.shape == torch.Size([])
        
        # Sum should be larger than mean for batch size > 1
        assert loss_val_sum > loss_val_mean
        
        # None reduction
        loss_none = FocalLoss(reduction='none')
        loss_val_none = loss_none(inputs, targets)
        assert loss_val_none.shape == torch.Size([32])


class TestEarlyStopping:
    """Test early stopping callback."""
    
    def test_early_stopping_patience(self):
        """Test early stopping with patience."""
        early_stop = EarlyStopping(patience=3, mode='min', verbose=False)
        
        # Decreasing scores (good for min mode)
        scores = [1.0, 0.9, 0.8, 0.7]
        for score in scores:
            should_stop = early_stop(score)
            assert not should_stop
        
        # Increasing scores (bad for min mode)
        bad_scores = [0.8, 0.9, 1.0, 1.1]
        stop_triggered = False
        for i, score in enumerate(bad_scores):
            should_stop = early_stop(score)
            if should_stop:
                stop_triggered = True
                assert i == 3  # Should stop after patience=3
                break
        
        assert stop_triggered
    
    def test_early_stopping_max_mode(self):
        """Test early stopping in max mode."""
        early_stop = EarlyStopping(patience=2, mode='max', verbose=False)
        
        # Increasing scores (good for max mode)
        scores = [0.5, 0.6, 0.7, 0.8]
        for score in scores:
            should_stop = early_stop(score)
            assert not should_stop
        
        # Decreasing scores (bad for max mode)
        bad_scores = [0.7, 0.6, 0.5]
        stop_triggered = False
        for i, score in enumerate(bad_scores):
            should_stop = early_stop(score)
            if should_stop:
                stop_triggered = True
                assert i == 2  # Should stop after patience=2
                break
        
        assert stop_triggered


class TestTrainerBase:
    """Test base trainer functionality."""
    
    def create_dummy_data(self, n_samples=100, input_dim=10, output_dim=2):
        """Create dummy dataset."""
        X = torch.randn(n_samples, input_dim)
        y = torch.randint(0, output_dim, (n_samples,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=16, shuffle=True)
    
    def test_trainer_init(self):
        """Test trainer initialization."""
        model = SimpleModel()
        train_loader = self.create_dummy_data()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = TrainerBase(
                model=model,
                train_loader=train_loader,
                criterion=nn.CrossEntropyLoss(),
                checkpoint_dir=tmpdir
            )
            
            assert trainer.model is not None
            assert trainer.train_loader is not None
            assert trainer.criterion is not None
            assert trainer.optimizer is not None
    
    def test_train_step(self):
        """Test single training step."""
        model = SimpleModel()
        train_loader = self.create_dummy_data()
        
        trainer = TrainerBase(
            model=model,
            train_loader=train_loader,
            criterion=nn.CrossEntropyLoss(),
            device='cpu'
        )
        
        # Get one batch
        batch = next(iter(train_loader))
        
        # Perform train step
        metrics = trainer.train_step(batch)
        
        assert 'loss' in metrics
        assert metrics['loss'] >= 0
    
    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        model = SimpleModel()
        train_loader = self.create_dummy_data()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = TrainerBase(
                model=model,
                train_loader=train_loader,
                criterion=nn.CrossEntropyLoss(),
                checkpoint_dir=tmpdir
            )
            
            # Save checkpoint
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
            trainer.save_checkpoint(str(checkpoint_path))
            
            assert checkpoint_path.exists()
            
            # Load checkpoint
            loaded = trainer.load_checkpoint(str(checkpoint_path))
            
            assert 'model_state_dict' in loaded
            assert 'optimizer_state_dict' in loaded
            assert 'epoch' in loaded


class TestTrainerClf:
    """Test classification trainer."""
    
    def create_classification_data(self, n_samples=200, n_features=10, n_classes=3):
        """Create classification dataset."""
        X = torch.randn(n_samples, n_features)
        y = torch.randint(0, n_classes, (n_samples,))
        
        train_dataset = TensorDataset(X[:150], y[:150])
        val_dataset = TensorDataset(X[150:], y[150:])
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        return train_loader, val_loader
    
    def test_classification_trainer_init(self):
        """Test classification trainer initialization."""
        model = SimpleModel(output_dim=3)
        train_loader, val_loader = self.create_classification_data()
        
        trainer = TrainerClf(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=3,
            device='cpu'
        )
        
        assert trainer.num_classes == 3
        assert isinstance(trainer.criterion, nn.CrossEntropyLoss)
    
    def test_classification_metrics(self):
        """Test classification metrics computation."""
        model = SimpleModel(output_dim=3)
        train_loader, _ = self.create_classification_data()
        
        trainer = TrainerClf(
            model=model,
            train_loader=train_loader,
            num_classes=3,
            device='cpu'
        )
        
        # Create dummy outputs and targets
        outputs = torch.randn(50, 3)
        targets = torch.randint(0, 3, (50,))
        
        metrics = trainer.compute_metrics(outputs, targets)
        
        assert 'accuracy' in metrics
        assert 'mean_per_class_accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['mean_per_class_accuracy'] <= 1
    
    def test_overfit_tiny_dataset(self):
        """Test that model can overfit a tiny dataset."""
        # Create tiny dataset that should be easy to overfit
        X = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        y = torch.tensor([0, 1, 0, 1])
        
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        model = SimpleModel(input_dim=2, output_dim=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = TrainerClf(
                model=model,
                train_loader=train_loader,
                val_loader=None,
                num_classes=2,
                device='cpu',
                use_amp=False,
                checkpoint_dir=tmpdir
            )
            
            # Train for many epochs to ensure overfitting
            initial_loss = None
            for epoch in range(1, 51):
                metrics = trainer.train_epoch(epoch)
                if initial_loss is None:
                    initial_loss = metrics['loss']
            
            # Loss should decrease significantly
            final_loss = metrics['loss']
            assert final_loss < initial_loss * 0.5
    
    def test_focal_loss_integration(self):
        """Test trainer with focal loss."""
        model = SimpleModel(output_dim=3)
        train_loader, _ = self.create_classification_data()
        
        trainer = TrainerClf(
            model=model,
            train_loader=train_loader,
            num_classes=3,
            use_focal_loss=True,
            focal_alpha=0.25,
            focal_gamma=2.0,
            device='cpu'
        )
        
        assert isinstance(trainer.criterion, FocalLoss)
        
        # Test one epoch runs without error
        metrics = trainer.train_epoch(1)
        assert 'loss' in metrics


class TestTrainerReg:
    """Test regression trainer."""
    
    def create_regression_data(self, n_samples=200, n_features=10):
        """Create regression dataset."""
        X = torch.randn(n_samples, n_features)
        # Create targets with some structure
        y = X[:, 0] * 2 + X[:, 1] * 0.5 + torch.randn(n_samples) * 0.1
        y = y.unsqueeze(1)
        
        train_dataset = TensorDataset(X[:150], y[:150])
        val_dataset = TensorDataset(X[150:], y[150:])
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        return train_loader, val_loader
    
    def test_regression_trainer_init(self):
        """Test regression trainer initialization."""
        model = SimpleModel(output_dim=1)
        train_loader, val_loader = self.create_regression_data()
        
        trainer = TrainerReg(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_type='mse',
            device='cpu'
        )
        
        assert isinstance(trainer.criterion, nn.MSELoss)
    
    def test_regression_metrics(self):
        """Test regression metrics computation."""
        model = SimpleModel(output_dim=1)
        train_loader, _ = self.create_regression_data()
        
        trainer = TrainerReg(
            model=model,
            train_loader=train_loader,
            device='cpu'
        )
        
        # Create dummy outputs and targets
        outputs = torch.randn(50, 1)
        targets = torch.randn(50, 1)
        
        metrics = trainer.compute_metrics(outputs, targets)
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert metrics['mse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['rmse'] >= 0
    
    def test_different_loss_types(self):
        """Test different loss types for regression."""
        model = SimpleModel(output_dim=1)
        train_loader, _ = self.create_regression_data()
        
        # Test MSE
        trainer_mse = TrainerReg(
            model=model,
            train_loader=train_loader,
            loss_type='mse',
            device='cpu'
        )
        assert isinstance(trainer_mse.criterion, nn.MSELoss)
        
        # Test MAE
        trainer_mae = TrainerReg(
            model=model,
            train_loader=train_loader,
            loss_type='mae',
            device='cpu'
        )
        assert isinstance(trainer_mae.criterion, nn.L1Loss)
        
        # Test Huber
        trainer_huber = TrainerReg(
            model=model,
            train_loader=train_loader,
            loss_type='huber',
            device='cpu'
        )
        assert isinstance(trainer_huber.criterion, nn.SmoothL1Loss)


class TestOptimizersSchedulers:
    """Test optimizer and scheduler creation."""
    
    def test_create_optimizer(self):
        """Test optimizer creation."""
        model = SimpleModel()
        
        # Test AdamW
        opt = create_optimizer(model, optimizer_type='adamw', lr=0.001)
        assert isinstance(opt, torch.optim.AdamW)
        
        # Test Adam
        opt = create_optimizer(model, optimizer_type='adam', lr=0.001)
        assert isinstance(opt, torch.optim.Adam)
        
        # Test SGD
        opt = create_optimizer(model, optimizer_type='sgd', lr=0.01, momentum=0.9)
        assert isinstance(opt, torch.optim.SGD)
    
    def test_create_scheduler(self):
        """Test scheduler creation."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Test cosine scheduler
        scheduler = create_scheduler(
            optimizer,
            scheduler_type='cosine',
            num_epochs=10,
            warmup_epochs=2
        )
        assert scheduler is not None
        
        # Test linear scheduler
        scheduler = create_scheduler(
            optimizer,
            scheduler_type='linear',
            num_epochs=10,
            warmup_epochs=1
        )
        assert scheduler is not None
    
    def test_scheduler_warmup(self):
        """Test that scheduler includes warmup."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        scheduler = create_scheduler(
            optimizer,
            scheduler_type='cosine',
            num_epochs=10,
            warmup_epochs=2
        )
        
        # Get initial LR
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Step through warmup
        scheduler.step()
        lr_after_step = optimizer.param_groups[0]['lr']
        
        # LR should change during warmup
        assert lr_after_step != initial_lr


class TestCheckpointingMetrics:
    """Test checkpointing and metrics saving."""
    
    def test_save_load_full_training_state(self):
        """Test saving and loading full training state."""
        model = SimpleModel()
        X = torch.randn(50, 10)
        y = torch.randint(0, 2, (50,))
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=16)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and train for a few epochs
            trainer = TrainerClf(
                model=model,
                train_loader=train_loader,
                num_classes=2,
                device='cpu',
                checkpoint_dir=tmpdir
            )
            
            # Train for 2 epochs
            for epoch in range(1, 3):
                trainer.train_epoch(epoch)
            
            # Save checkpoint
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            trainer.save_checkpoint(str(checkpoint_path))
            
            # Create new trainer and load checkpoint
            new_model = SimpleModel()
            new_trainer = TrainerClf(
                model=new_model,
                train_loader=train_loader,
                num_classes=2,
                device='cpu',
                checkpoint_dir=tmpdir
            )
            
            loaded = new_trainer.load_checkpoint(str(checkpoint_path))
            
            # Check state is restored
            assert new_trainer.epoch == 2
            assert len(new_trainer.train_history) == 2
    
    def test_metrics_json_saving(self):
        """Test saving metrics to JSON."""
        model = SimpleModel()
        X = torch.randn(50, 10)
        y = torch.randint(0, 2, (50,))
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=16)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = TrainerClf(
                model=model,
                train_loader=train_loader,
                num_classes=2,
                device='cpu',
                checkpoint_dir=tmpdir
            )
            
            # Add some dummy metrics
            trainer.train_history = [{'loss': 0.5}, {'loss': 0.4}]
            trainer.val_history = [{'loss': 0.6, 'accuracy': 0.8}]
            trainer.best_val_metric = 0.8
            trainer.epoch = 2
            
            # Save metrics
            metrics_path = Path(tmpdir) / "metrics.json"
            trainer.save_metrics(str(metrics_path))
            
            # Load and verify
            with open(metrics_path, 'r') as f:
                loaded_metrics = json.load(f)
            
            assert loaded_metrics['final_epoch'] == 2
            assert loaded_metrics['best_val_metric'] == 0.8
            assert len(loaded_metrics['train_history']) == 2
            assert len(loaded_metrics['val_history']) == 1
