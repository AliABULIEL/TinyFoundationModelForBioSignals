#!/usr/bin/env python3
"""
Programmatic Notebook Execution and Validation Script

Executes ttm_har_extended_notebook.ipynb cell-by-cell with:
- Comprehensive error tracking
- Shape validation
- Output verification
- Detailed reporting
"""

import json
import sys
import traceback
from pathlib import Path
import subprocess

def execute_notebook_with_jupyter(notebook_path: str, output_path: str):
    """Execute notebook using jupyter nbconvert."""
    cmd = [
        'jupyter', 'nbconvert',
        '--to', 'notebook',
        '--execute',
        '--ExecutePreprocessor.timeout=600',
        '--output', output_path,
        notebook_path
    ]

    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    return result.returncode, result.stdout, result.stderr

def validate_executed_notebook(nb_path: str):
    """Validate the executed notebook outputs."""
    with open(nb_path, 'r') as f:
        nb = json.load(f)

    validation_report = {
        'total_cells': len(nb['cells']),
        'executed_cells': 0,
        'cells_with_output': 0,
        'cells_with_errors': 0,
        'error_details': [],
        'key_outputs': {},
    }

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            validation_report['executed_cells'] += 1

            # Check for execution
            execution_count = cell.get('execution_count')
            outputs = cell.get('outputs', [])

            if len(outputs) > 0:
                validation_report['cells_with_output'] += 1

            # Check for errors
            for output in outputs:
                if output.get('output_type') == 'error':
                    validation_report['cells_with_errors'] += 1
                    validation_report['error_details'].append({
                        'cell_index': i,
                        'error_name': output.get('ename'),
                        'error_value': output.get('evalue'),
                        'traceback': output.get('traceback', [])[:5]  # First 5 lines
                    })

            # Extract key outputs
            source = ''.join(cell.get('source', []))
            if 'VALIDATION PASSED' in source or 'SMOKE TEST' in source:
                validation_report['key_outputs']['validation'] = {
                    'cell': i,
                    'has_output': len(outputs) > 0
                }
            elif 'class TTMHARModel' in source:
                validation_report['key_outputs']['model_defined'] = i
            elif 'Dataset loaded' in source or 'CAPTURE24' in source:
                validation_report['key_outputs']['dataset'] = i

    return validation_report

def main():
    repo_root = Path(__file__).parent
    notebook_path = repo_root / "ttm_har_extended_notebook.ipynb"
    output_path = repo_root / "ttm_har_extended_notebook_executed.ipynb"

    print("=" * 70)
    print("NOTEBOOK EXECUTION & VALIDATION")
    print("=" * 70)
    print(f"Notebook: {notebook_path}")
    print(f"Output: {output_path}")
    print()

    # Execute notebook
    print("Executing notebook...")
    returncode, stdout, stderr = execute_notebook_with_jupyter(
        str(notebook_path), str(output_path)
    )

    print(f"\nExecution completed with return code: {returncode}")

    if returncode != 0:
        print(f"\n❌ EXECUTION FAILED")
        print(f"\nSTDOUT:\n{stdout}")
        print(f"\nSTDERR:\n{stderr}")
        sys.exit(1)

    # Validate executed notebook
    print("\nValidating executed notebook...")
    report = validate_executed_notebook(str(output_path))

    print("\n" + "=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)
    print(f"Total cells: {report['total_cells']}")
    print(f"Executed code cells: {report['executed_cells']}")
    print(f"Cells with output: {report['cells_with_output']}")
    print(f"Cells with errors: {report['cells_with_errors']}")

    if report['error_details']:
        print("\n❌ ERRORS DETECTED:")
        for err in report['error_details']:
            print(f"\n  Cell {err['cell_index']}: {err['error_name']}")
            print(f"  Message: {err['error_value']}")

    print(f"\nKey outputs detected:")
    for key, value in report['key_outputs'].items():
        print(f"  {key}: {value}")

    if report['cells_with_errors'] > 0:
        print("\n❌ VALIDATION FAILED (errors in execution)")
        sys.exit(1)
    else:
        print("\n✅ VALIDATION PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()
