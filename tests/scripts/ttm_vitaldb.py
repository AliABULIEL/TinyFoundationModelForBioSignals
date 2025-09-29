#!/usr/bin/env python3
"""
Main CLI entry point for TTM × VitalDB pipeline.

Commands:
    preprocess: Download and preprocess VitalDB data
    pretrain: Pre-train TTM on VitalDB
    finetune: Fine-tune for specific task
    evaluate: Evaluate on test set
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TTM × VitalDB: Foundation Model for Biosignals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Download and preprocess VitalDB data")
    preprocess_parser.add_argument("--config", type=str, default="configs/run.yaml", help="Run config path")
    preprocess_parser.add_argument("--mode", type=str, choices=["fasttrack", "full"], default="full")
    
    # Pretrain command
    pretrain_parser = subparsers.add_parser("pretrain", help="Pre-train TTM on VitalDB")
    pretrain_parser.add_argument("--config", type=str, default="configs/run.yaml", help="Run config path")
    pretrain_parser.add_argument("--mode", type=str, choices=["fasttrack", "full"], default="full")
    
    # Finetune command
    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune for specific task")
    finetune_parser.add_argument("--config", type=str, default="configs/run.yaml", help="Run config path")
    finetune_parser.add_argument("--task", type=str, required=True, help="Task name (e.g., ppg_quality)")
    finetune_parser.add_argument("--unfreeze-last-n", type=int, default=0, help="Number of blocks to unfreeze")
    finetune_parser.add_argument("--lora-rank", type=int, default=0, help="LoRA rank (0 to disable)")
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate on test set")
    evaluate_parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    evaluate_parser.add_argument("--config", type=str, default="configs/run.yaml", help="Run config path")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # TODO: Implement commands in Prompt 8
    print(f"Command '{args.command}' will be implemented in Prompt 8")
    print(f"Args: {args}")


if __name__ == "__main__":
    main()
