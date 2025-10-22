#!/bin/bash
# Quick-Start Examples for Enhanced Fine-Tuning
#
# These are ready-to-run commands for the enhanced fine-tuning script.
# Each example demonstrates different configurations.

set -e  # Exit on error

# ANSI color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Enhanced Fine-Tuning Examples${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ============================================================================
# EXAMPLE 1: Quality Classification (RECOMMENDED)
# ============================================================================
echo -e "${GREEN}[Example 1] Quality Classification with Projection Adapter${NC}"
echo -e "${YELLOW}This is the recommended starting point${NC}"
echo ""

read -p "Run Example 1? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python scripts/finetune_enhanced.py \
        --pretrained artifacts/foundation_model/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --task quality \
        --adaptation projection \
        --epochs 30 \
        --head-only-epochs 5 \
        --partial-epochs 15 \
        --lr 1e-4 \
        --batch-size 64 \
        --early-stopping \
        --patience 10 \
        --output-dir artifacts/butppg_quality_projection

    echo -e "${GREEN}✓ Example 1 complete!${NC}"
    echo -e "Results saved to: artifacts/butppg_quality_projection/"
    echo ""
fi

# ============================================================================
# EXAMPLE 2: Baseline (No Domain Adaptation)
# ============================================================================
echo -e "${GREEN}[Example 2] Baseline Quality Classification (No Adaptation)${NC}"
echo -e "${YELLOW}Use this to establish baseline performance${NC}"
echo ""

read -p "Run Example 2? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python scripts/finetune_enhanced.py \
        --pretrained artifacts/foundation_model/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --task quality \
        --no-adaptation \
        --epochs 30 \
        --head-only-epochs 5 \
        --partial-epochs 15 \
        --lr 1e-4 \
        --batch-size 64 \
        --early-stopping \
        --patience 10 \
        --output-dir artifacts/butppg_quality_baseline

    echo -e "${GREEN}✓ Example 2 complete!${NC}"
    echo -e "Results saved to: artifacts/butppg_quality_baseline/"
    echo ""
fi

# ============================================================================
# EXAMPLE 3: Heart Rate Regression
# ============================================================================
echo -e "${GREEN}[Example 3] Heart Rate Regression${NC}"
echo -e "${YELLOW}Demonstrates regression task${NC}"
echo ""

read -p "Run Example 3? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python scripts/finetune_enhanced.py \
        --pretrained artifacts/foundation_model/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --task hr \
        --adaptation projection \
        --epochs 30 \
        --head-only-epochs 5 \
        --partial-epochs 15 \
        --lr 5e-5 \
        --batch-size 64 \
        --early-stopping \
        --patience 10 \
        --output-dir artifacts/butppg_hr

    echo -e "${GREEN}✓ Example 3 complete!${NC}"
    echo -e "Results saved to: artifacts/butppg_hr/"
    echo ""
fi

# ============================================================================
# EXAMPLE 4: Blood Pressure Estimation
# ============================================================================
echo -e "${GREEN}[Example 4] Blood Pressure Estimation (Adversarial Adapter)${NC}"
echo -e "${YELLOW}Demonstrates adversarial domain adaptation${NC}"
echo ""

read -p "Run Example 4? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python scripts/finetune_enhanced.py \
        --pretrained artifacts/foundation_model/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --task blood_pressure \
        --adaptation adversarial \
        --epochs 40 \
        --head-only-epochs 8 \
        --partial-epochs 20 \
        --lr 1e-4 \
        --batch-size 64 \
        --early-stopping \
        --patience 15 \
        --output-dir artifacts/butppg_bp_adversarial

    echo -e "${GREEN}✓ Example 4 complete!${NC}"
    echo -e "Results saved to: artifacts/butppg_bp_adversarial/"
    echo ""
fi

# ============================================================================
# EXAMPLE 5: Quick Test (1 epoch, fast)
# ============================================================================
echo -e "${GREEN}[Example 5] Quick Test (1 epoch)${NC}"
echo -e "${YELLOW}Fast sanity check - verifies script works${NC}"
echo ""

read -p "Run Example 5? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python scripts/finetune_enhanced.py \
        --pretrained artifacts/foundation_model/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --task quality \
        --adaptation projection \
        --epochs 1 \
        --head-only-epochs 1 \
        --partial-epochs 0 \
        --lr 1e-4 \
        --batch-size 32 \
        --output-dir artifacts/butppg_quick_test

    echo -e "${GREEN}✓ Example 5 complete!${NC}"
    echo -e "Results saved to: artifacts/butppg_quick_test/"
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}All Examples Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "To compare results:"
echo "  ls -lh artifacts/butppg_*/test_metrics.json"
echo "  cat artifacts/butppg_quality_projection/test_metrics.json"
echo ""
echo "To visualize training history:"
echo "  python scripts/plot_training_history.py artifacts/butppg_quality_projection/history.json"
echo ""
