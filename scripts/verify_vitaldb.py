from src.data.vitaldb_loader import load_channel, list_cases, get_available_case_sets

# Load PPG data for pre-training
cases = list_cases(
    required_channels=['PLETH'],
    case_set='bis',  # High-quality BIS cases
    max_cases=100
)

for case in cases:
    try:
        signal, fs = load_channel(case['case_id'], 'PLETH')
        # Signal is now guaranteed to be numeric and cleaned
        # Process for TTM pre-training...
    except Exception as e:
        print(f"Skipping case {case['case_id']}: {e}")