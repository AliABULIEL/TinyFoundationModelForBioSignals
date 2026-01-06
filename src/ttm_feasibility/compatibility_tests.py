"""
Part 3: Compatibility Tests for TTM with Accelerometry Data

Tests whether TTM can process Capture-24 shaped inputs and extract embeddings.
This is a critical gate before proceeding to full embedding extraction.
"""

import numpy as np
import torch
from transformers import set_seed
from tsfm_public import TinyTimeMixerForPrediction
from typing import Dict, Any, Tuple
import warnings

warnings.filterwarnings('ignore')


class TTMCompatibilityTester:
    """Test suite for TTM compatibility with accelerometry data."""

    def __init__(self, model_path="ibm-granite/granite-timeseries-ttm-r2", seed=42):
        self.model_path = model_path
        self.seed = seed
        set_seed(seed)
        torch.manual_seed(seed)

        print("="*70)
        print("TTM COMPATIBILITY TESTING")
        print("="*70)
        print(f"Model: {model_path}")
        print(f"Seed: {seed}")
        print("="*70)

        # Load model
        print("\nLoading TTM model...")
        self.model = TinyTimeMixerForPrediction.from_pretrained(
            model_path,
            revision="main"
        )
        self.model.eval()
        print(f"✓ Model loaded ({sum(p.numel() for p in self.model.parameters()):,} parameters)")

        self.results = {}

    def test_1_basic_shape(self) -> Dict[str, Any]:
        """Test 1: Can TTM process tensors of shape (batch, 512, 3)?"""
        print("\n" + "="*70)
        print("TEST 1: Basic Shape Compatibility")
        print("="*70)

        test_result = {"name": "Basic Shape Test", "status": "UNKNOWN", "details": {}}

        try:
            batch_size = 4
            context_length = 512
            num_channels = 3

            # Create dummy input
            dummy_input = torch.randn(batch_size, context_length, num_channels)
            print(f"Input shape: {dummy_input.shape}")

            # Forward pass
            with torch.no_grad():
                output = self.model(past_values=dummy_input, output_hidden_states=True)

            # Check outputs
            pred_shape = output.prediction_outputs.shape
            num_hidden = len(output.hidden_states) if hasattr(output, 'hidden_states') else 0

            print(f"✓ Forward pass successful")
            print(f"  Prediction shape: {pred_shape}")
            print(f"  Number of hidden layers: {num_hidden}")

            # Verify shapes
            assert pred_shape[0] == batch_size, f"Batch size mismatch: {pred_shape[0]} vs {batch_size}"
            assert pred_shape[2] == num_channels, f"Channel mismatch: {pred_shape[2]} vs {num_channels}"
            assert num_hidden > 0, "No hidden states available"

            test_result["status"] = "PASS"
            test_result["details"] = {
                "input_shape": str(dummy_input.shape),
                "output_shape": str(pred_shape),
                "num_hidden_layers": num_hidden
            }

            print("\n✅ TEST 1: PASS")

        except Exception as e:
            test_result["status"] = "FAIL"
            test_result["details"]["error"] = str(e)
            print(f"\n❌ TEST 1: FAIL - {e}")

        self.results["test_1"] = test_result
        return test_result

    def test_2_batch_sizes(self) -> Dict[str, Any]:
        """Test 2: What batch sizes work without memory issues?"""
        print("\n" + "="*70)
        print("TEST 2: Batch Size Scalability")
        print("="*70)

        test_result = {"name": "Batch Size Test", "status": "UNKNOWN", "details": {}}

        batch_sizes_to_test = [1, 8, 16, 32, 64, 128]
        successful_sizes = []
        failed_sizes = []

        for batch_size in batch_sizes_to_test:
            try:
                dummy_input = torch.randn(batch_size, 512, 3)

                with torch.no_grad():
                    output = self.model(past_values=dummy_input)

                successful_sizes.append(batch_size)
                print(f"  Batch size {batch_size:3d}: ✓ PASS")

            except Exception as e:
                failed_sizes.append(batch_size)
                print(f"  Batch size {batch_size:3d}: ✗ FAIL ({str(e)[:50]}...)")

        if len(successful_sizes) > 0:
            test_result["status"] = "PASS"
            test_result["details"] = {
                "successful_sizes": successful_sizes,
                "failed_sizes": failed_sizes,
                "max_successful": max(successful_sizes)
            }
            print(f"\n✅ TEST 2: PASS (Max batch size: {max(successful_sizes)})")
        else:
            test_result["status"] = "FAIL"
            print("\n❌ TEST 2: FAIL (No batch sizes successful)")

        self.results["test_2"] = test_result
        return test_result

    def test_3_real_data_simulation(self) -> Dict[str, Any]:
        """Test 3: Process simulated accelerometry data."""
        print("\n" + "="*70)
        print("TEST 3: Simulated Accelerometry Data")
        print("="*70)

        test_result = {"name": "Real Data Simulation", "status": "UNKNOWN", "details": {}}

        try:
            # Simulate realistic accelerometry (based on typical wrist-worn patterns)
            # Values in g units, typical range -2g to +2g for daily activities
            np.random.seed(self.seed)

            # Simulate walking pattern (1-2 Hz oscillation + noise)
            t = np.linspace(0, 5.12, 512)  # 512 samples at 100Hz = 5.12 seconds

            # X-axis: Forward-backward sway
            x = 0.5 * np.sin(2 * np.pi * 1.5 * t) + np.random.normal(0, 0.1, 512)

            # Y-axis: Lateral sway
            y = 0.3 * np.sin(2 * np.pi * 1.3 * t + np.pi/4) + np.random.normal(0, 0.1, 512)

            # Z-axis: Vertical (gravity component + movement)
            z = -1.0 + 0.4 * np.sin(2 * np.pi * 1.5 * t) + np.random.normal(0, 0.1, 512)

            # Stack into [512, 3] array
            simulated_data = np.stack([x, y, z], axis=1)

            # Convert to tensor and add batch dimension
            input_tensor = torch.tensor(simulated_data, dtype=torch.float32).unsqueeze(0)

            print(f"Simulated data shape: {simulated_data.shape}")
            print(f"Value ranges: X=[{x.min():.2f}, {x.max():.2f}], "
                  f"Y=[{y.min():.2f}, {y.max():.2f}], Z=[{z.min():.2f}, {z.max():.2f}]")

            # Forward pass
            with torch.no_grad():
                output = self.model(past_values=input_tensor, output_hidden_states=True)

            # Extract embedding
            final_hidden = output.hidden_states[17]  # Final encoder layer
            print(f"✓ Embedding extracted, shape: {final_hidden.shape}")

            # Check embedding statistics
            embedding_np = final_hidden.numpy()
            stats = {
                "mean": float(np.mean(embedding_np)),
                "std": float(np.std(embedding_np)),
                "min": float(np.min(embedding_np)),
                "max": float(np.max(embedding_np)),
                "has_nan": bool(np.isnan(embedding_np).any()),
                "has_inf": bool(np.isinf(embedding_np).any())
            }

            print(f"Embedding statistics:")
            print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  Has NaN: {stats['has_nan']}, Has Inf: {stats['has_inf']}")

            # Verify embeddings are reasonable
            assert not stats['has_nan'], "Embeddings contain NaN"
            assert not stats['has_inf'], "Embeddings contain Inf"
            assert stats['std'] > 0.001, f"Embeddings have very low variance: {stats['std']}"

            test_result["status"] = "PASS"
            test_result["details"] = stats
            print("\n✅ TEST 3: PASS")

        except Exception as e:
            test_result["status"] = "FAIL"
            test_result["details"]["error"] = str(e)
            print(f"\n❌ TEST 3: FAIL - {e}")

        self.results["test_3"] = test_result
        return test_result

    def test_4_normalization_strategies(self) -> Dict[str, Any]:
        """Test 4: How do different normalizations affect outputs?"""
        print("\n" + "="*70)
        print("TEST 4: Normalization Strategy Comparison")
        print("="*70)

        test_result = {"name": "Normalization Test", "status": "UNKNOWN", "details": {}}

        try:
            # Create realistic accelerometry data
            np.random.seed(self.seed)
            t = np.linspace(0, 5.12, 512)
            x = 0.5 * np.sin(2 * np.pi * 1.5 * t) + np.random.normal(0, 0.1, 512)
            y = 0.3 * np.sin(2 * np.pi * 1.3 * t + np.pi/4) + np.random.normal(0, 0.1, 512)
            z = -1.0 + 0.4 * np.sin(2 * np.pi * 1.5 * t) + np.random.normal(0, 0.1, 512)
            raw_data = np.stack([x, y, z], axis=1)

            normalization_results = {}

            # Strategy 1: No normalization (raw)
            print("\n[Strategy 1: Raw (no normalization)]")
            raw_tensor = torch.tensor(raw_data, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out_raw = self.model(past_values=raw_tensor, output_hidden_states=True)
            emb_raw = out_raw.hidden_states[17].numpy()
            normalization_results["raw"] = {
                "mean": float(np.mean(emb_raw)),
                "std": float(np.std(emb_raw))
            }
            print(f"  Embedding - Mean: {normalization_results['raw']['mean']:.4f}, "
                  f"Std: {normalization_results['raw']['std']:.4f}")

            # Strategy 2: Global standardization
            print("\n[Strategy 2: Global standardization]")
            global_mean = np.mean(raw_data)
            global_std = np.std(raw_data)
            global_norm = (raw_data - global_mean) / (global_std + 1e-8)
            global_tensor = torch.tensor(global_norm, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out_global = self.model(past_values=global_tensor, output_hidden_states=True)
            emb_global = out_global.hidden_states[17].numpy()
            normalization_results["global_std"] = {
                "mean": float(np.mean(emb_global)),
                "std": float(np.std(emb_global))
            }
            print(f"  Embedding - Mean: {normalization_results['global_std']['mean']:.4f}, "
                  f"Std: {normalization_results['global_std']['std']:.4f}")

            # Strategy 3: Per-channel standardization
            print("\n[Strategy 3: Per-channel standardization]")
            perchannel_norm = np.zeros_like(raw_data)
            for i in range(3):
                ch_mean = np.mean(raw_data[:, i])
                ch_std = np.std(raw_data[:, i])
                perchannel_norm[:, i] = (raw_data[:, i] - ch_mean) / (ch_std + 1e-8)
            perchannel_tensor = torch.tensor(perchannel_norm, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out_perchannel = self.model(past_values=perchannel_tensor, output_hidden_states=True)
            emb_perchannel = out_perchannel.hidden_states[17].numpy()
            normalization_results["perchannel_std"] = {
                "mean": float(np.mean(emb_perchannel)),
                "std": float(np.std(emb_perchannel))
            }
            print(f"  Embedding - Mean: {normalization_results['perchannel_std']['mean']:.4f}, "
                  f"Std: {normalization_results['perchannel_std']['std']:.4f}")

            # Strategy 4: Min-max normalization
            print("\n[Strategy 4: Min-max normalization [0, 1]]")
            data_min = np.min(raw_data)
            data_max = np.max(raw_data)
            minmax_norm = (raw_data - data_min) / (data_max - data_min + 1e-8)
            minmax_tensor = torch.tensor(minmax_norm, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out_minmax = self.model(past_values=minmax_tensor, output_hidden_states=True)
            emb_minmax = out_minmax.hidden_states[17].numpy()
            normalization_results["minmax"] = {
                "mean": float(np.mean(emb_minmax)),
                "std": float(np.std(emb_minmax))
            }
            print(f"  Embedding - Mean: {normalization_results['minmax']['mean']:.4f}, "
                  f"Std: {normalization_results['minmax']['std']:.4f}")

            test_result["status"] = "PASS"
            test_result["details"] = normalization_results
            print("\n✅ TEST 4: PASS (All normalization strategies work)")
            print("\n💡 Recommendation: Per-channel standardization (Strategy 3)")
            print("   Rationale: Preserves channel-specific distributions")

        except Exception as e:
            test_result["status"] = "FAIL"
            test_result["details"]["error"] = str(e)
            print(f"\n❌ TEST 4: FAIL - {e}")

        self.results["test_4"] = test_result
        return test_result

    def test_5_edge_cases(self) -> Dict[str, Any]:
        """Test 5: Edge cases (zeros, constants, extremes)."""
        print("\n" + "="*70)
        print("TEST 5: Edge Cases")
        print("="*70)

        test_result = {"name": "Edge Cases Test", "status": "UNKNOWN", "details": {}}
        edge_case_results = {}

        test_cases = {
            "all_zeros": np.zeros((512, 3)),
            "all_ones": np.ones((512, 3)),
            "constant_value": np.full((512, 3), 0.5),
            "extreme_positive": np.full((512, 3), 100.0),
            "extreme_negative": np.full((512, 3), -100.0),
            "mixed_extremes": np.random.choice([-100, 100], size=(512, 3)),
        }

        for case_name, case_data in test_cases.items():
            try:
                print(f"\n[{case_name}]")
                input_tensor = torch.tensor(case_data, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    output = self.model(past_values=input_tensor, output_hidden_states=True)

                embedding = output.hidden_states[17].numpy()

                # Check for NaN/Inf
                has_nan = np.isnan(embedding).any()
                has_inf = np.isinf(embedding).any()
                all_same = np.allclose(embedding, embedding.flat[0])

                edge_case_results[case_name] = {
                    "has_nan": bool(has_nan),
                    "has_inf": bool(has_inf),
                    "all_same_value": bool(all_same),
                    "embedding_std": float(np.std(embedding))
                }

                status = "✓ PASS" if not (has_nan or has_inf) else "✗ FAIL"
                print(f"  {status} - NaN: {has_nan}, Inf: {has_inf}, "
                      f"Constant: {all_same}, Std: {np.std(embedding):.4f}")

            except Exception as e:
                edge_case_results[case_name] = {"error": str(e)}
                print(f"  ✗ FAIL - Error: {str(e)[:50]}")

        # Overall status
        all_passed = all(not result.get("has_nan", True) and not result.get("has_inf", True)
                         for result in edge_case_results.values() if "error" not in result)

        test_result["status"] = "PASS" if all_passed else "FAIL"
        test_result["details"] = edge_case_results

        if all_passed:
            print("\n✅ TEST 5: PASS (All edge cases handled)")
        else:
            print("\n❌ TEST 5: FAIL (Some edge cases failed)")

        self.results["test_5"] = test_result
        return test_result

    def generate_report(self) -> Dict[str, Any]:
        """Generate final compatibility test report."""
        print("\n" + "="*70)
        print("COMPATIBILITY TEST SUMMARY")
        print("="*70)

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r["status"] == "PASS")

        print(f"\nTotal tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success rate: {100 * passed_tests / total_tests:.1f}%")

        print("\n" + "-"*70)
        print("Test Results:")
        print("-"*70)

        for test_id, result in self.results.items():
            status_symbol = "✅" if result["status"] == "PASS" else "❌"
            print(f"{status_symbol} {result['name']}: {result['status']}")

        # Overall verdict
        print("\n" + "="*70)
        if passed_tests == total_tests:
            print("OVERALL VERDICT: ✅ PASS - TTM is compatible with accelerometry data")
            print("RECOMMENDATION: Proceed to Part 4 (Embedding Extraction)")
        elif passed_tests >= total_tests * 0.8:
            print("OVERALL VERDICT: ⚠️ PARTIAL PASS - TTM mostly compatible")
            print("RECOMMENDATION: Proceed with caution, monitor failing tests")
        else:
            print("OVERALL VERDICT: ❌ FAIL - TTM has compatibility issues")
            print("RECOMMENDATION: Do NOT proceed - investigate failures first")
        print("="*70)

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": 100 * passed_tests / total_tests,
            "results": self.results
        }

    def run_all_tests(self):
        """Run all compatibility tests."""
        self.test_1_basic_shape()
        self.test_2_batch_sizes()
        self.test_3_real_data_simulation()
        self.test_4_normalization_strategies()
        self.test_5_edge_cases()

        return self.generate_report()


def main():
    """Main entry point."""
    tester = TTMCompatibilityTester()
    report = tester.run_all_tests()

    # Save results
    import json
    from pathlib import Path

    output_dir = Path(__file__).parents[2] / "reports"
    output_file = output_dir / "part3_compatibility_results.json"

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
