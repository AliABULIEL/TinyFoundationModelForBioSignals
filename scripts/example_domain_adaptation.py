#!/usr/bin/env python3
"""
Example: Domain Adaptation for BUT-PPG Fine-Tuning
==================================================

This script shows how to use domain adaptation to fine-tune
your SSL foundation model (trained on VitalDB) for BUT-PPG signal quality.

Steps:
1. Load SSL foundation model with domain adapter
2. Progressive fine-tuning (3 phases)
3. Evaluate on BUT-PPG test set

Author: Claude Code
Date: October 2025
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.domain_adaptation import (
    create_domain_adapted_model,
    ProgressiveFineTuner
)
from src.data.butppg_loader import get_butppg_dataloaders


def train_one_phase(model, train_loader, val_loader, optimizer, num_epochs, device='cuda'):
    """Train for one phase of progressive fine-tuning."""
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}, Acc={100.*correct/total:.2f}%")

        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

    return val_loss, val_acc


def main():
    print("=" * 80)
    print("DOMAIN ADAPTATION EXAMPLE - BUT-PPG FINE-TUNING")
    print("=" * 80)

    # Configuration
    ssl_checkpoint = 'artifacts/foundation_model/best_model.pt'
    output_dir = Path('artifacts/butppg_domain_adapted')
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Step 1: Create domain-adapted model
    print("\n[Step 1/4] Creating domain-adapted model...")
    print(f"  SSL checkpoint: {ssl_checkpoint}")
    print(f"  Adaptation type: projection")

    model = create_domain_adapted_model(
        ssl_checkpoint_path=ssl_checkpoint,
        adaptation_type='projection',  # Simple projection adapter
        num_classes=2,  # Binary: good/bad signal quality
        device=device
    )

    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} total parameters")

    # Step 2: Load BUT-PPG data
    print("\n[Step 2/4] Loading BUT-PPG data...")

    train_loader, val_loader, test_loader = get_butppg_dataloaders(
        data_dir='data/processed/butppg/windows',
        batch_size=64,
        num_workers=4,
        window_size=1024,
        overlap=0
    )

    print(f"✓ Data loaded:")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val:   {len(val_loader.dataset)} samples")
    print(f"  Test:  {len(test_loader.dataset)} samples")

    # Step 3: Progressive fine-tuning
    print("\n[Step 3/4] Progressive fine-tuning...")

    tuner = ProgressiveFineTuner(
        encoder=model.encoder,
        adapter=model.adapter,
        task_head=model.head
    )

    # Phase 1: Freeze encoder, train adapter + head (warm-up)
    print("\n--- PHASE 1: Warm-up (5 epochs) ---")
    print("Frozen: SSL encoder")
    print("Trainable: Domain adapter + Task head")

    optimizer_p1 = tuner.get_optimizer(phase=1, base_lr=1e-3)
    val_loss_p1, val_acc_p1 = train_one_phase(
        model, train_loader, val_loader, optimizer_p1,
        num_epochs=5, device=device
    )

    # Save phase 1 checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'phase': 1,
        'val_loss': val_loss_p1,
        'val_acc': val_acc_p1
    }, output_dir / 'checkpoint_phase1.pt')
    print(f"✓ Phase 1 checkpoint saved")

    # Phase 2: Partial unfreezing
    print("\n--- PHASE 2: Partial Unfreezing (10 epochs) ---")
    print("Encoder LR: 1e-5 (0.1x)")
    print("Adapter LR: 1e-4 (1.0x)")
    print("Head LR:    1e-4 (1.0x)")

    optimizer_p2 = tuner.get_optimizer(phase=2, base_lr=1e-4)
    val_loss_p2, val_acc_p2 = train_one_phase(
        model, train_loader, val_loader, optimizer_p2,
        num_epochs=10, device=device
    )

    # Save phase 2 checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'phase': 2,
        'val_loss': val_loss_p2,
        'val_acc': val_acc_p2
    }, output_dir / 'checkpoint_phase2.pt')
    print(f"✓ Phase 2 checkpoint saved")

    # Phase 3: Full fine-tuning
    print("\n--- PHASE 3: Full Fine-Tuning (15 epochs) ---")
    print("Encoder LR: 1e-6 (0.01x)")
    print("Adapter LR: 1e-5 (0.1x)")
    print("Head LR:    1e-4 (1.0x)")

    optimizer_p3 = tuner.get_optimizer(phase=3, base_lr=1e-4)
    val_loss_p3, val_acc_p3 = train_one_phase(
        model, train_loader, val_loader, optimizer_p3,
        num_epochs=15, device=device
    )

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'phase': 3,
        'val_loss': val_loss_p3,
        'val_acc': val_acc_p3
    }, output_dir / 'best_model.pt')
    print(f"✓ Final model saved")

    # Step 4: Evaluate on test set
    print("\n[Step 4/4] Evaluating on test set...")

    model.eval()
    test_correct = 0
    test_total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = 100. * test_correct / test_total

    # Compute AUROC
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    from sklearn.metrics import roc_auc_score, f1_score

    auroc = roc_auc_score(all_labels, all_probs[:, 1])
    f1 = f1_score(all_labels, all_probs[:, 1] > 0.5)

    # Results summary
    print("\n" + "=" * 80)
    print("DOMAIN ADAPTATION COMPLETE!")
    print("=" * 80)

    print(f"\n📊 Test Set Results:")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  AUROC:    {auroc:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    print(f"\n📈 Progressive Fine-Tuning Summary:")
    print(f"  Phase 1 (Warm-up):         Val Acc={val_acc_p1:.2f}%")
    print(f"  Phase 2 (Partial):         Val Acc={val_acc_p2:.2f}%")
    print(f"  Phase 3 (Full):            Val Acc={val_acc_p3:.2f}%")
    print(f"  Final Test:                Test Acc={test_acc:.2f}%, AUROC={auroc:.4f}")

    print(f"\n✓ Model saved to: {output_dir / 'best_model.pt'}")

    print("\n" + "=" * 80)

    # Expected vs Actual comparison
    print("\n🎯 Expected Performance Comparison:")
    print(f"{'Method':<30} {'AUROC':<10} {'Status'}")
    print("-" * 50)
    print(f"{'Random Init':<30} {'0.505':<10} {'✅ Baseline'}")
    print(f"{'IBM TTM (no SSL)':<30} {'0.622':<10} {'✅ Pretrained baseline'}")
    print(f"{'SSL (no adaptation)':<30} {'~0.60':<10} {'⚠️  Domain gap'}")
    print(f"{'SSL + Domain Adaptation':<30} {f'{auroc:.3f}':<10} {'🏆 YOUR RESULT'}")

    if auroc > 0.70:
        print("\n🎉 SUCCESS! Domain adaptation worked!")
        print("   AUROC >0.70 means SSL foundation model + domain adaptation")
        print("   successfully transferred VitalDB knowledge to BUT-PPG!")
    elif auroc > 0.622:
        print("\n✅ GOOD! Beat IBM baseline, domain adaptation helping")
    else:
        print("\n⚠️  Result not as expected. Possible issues:")
        print("   - SSL model may not have trained on VitalDB")
        print("   - Need more fine-tuning epochs")
        print("   - Try adversarial adaptation instead of projection")


if __name__ == '__main__':
    main()
