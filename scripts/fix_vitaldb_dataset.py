# Fixed usage for VitalDBDataset class
# Add this to your biosignal/data.py or wherever VitalDBDataset is defined

"""
Fix for VitalDBDataset to use correct VitalDB API
Replace the problematic line in __init__ method:
"""

# OLD INCORRECT CODE:
# self.cases = self.vitaldb.find_cases(self.track_name)

# NEW CORRECT CODE:
# Import the fixed loader functions
from src.data.vitaldb_loader import find_cases_with_track, get_available_case_sets, load_case

# In the __init__ method of VitalDBDataset, replace the case loading section:

def _get_cases_for_dataset(self):
    """Get cases using the correct VitalDB API."""
    # Get case set from config
    vitaldb_config = self.config.config.get('vitaldb', {})
    case_set_name = vitaldb_config.get('case_set', 'bis')  # Default to high-quality BIS cases
    
    # Option 1: Use pre-filtered case sets directly (recommended)
    case_sets = get_available_case_sets()
    if case_set_name in case_sets:
        self.cases = list(case_sets[case_set_name])
    else:
        # Option 2: Find cases with specific track
        self.cases = find_cases_with_track(self.track_name, case_set=case_set_name)
    
    # Convert to integers
    self.cases = [int(c) for c in self.cases]
    
    # Apply case limit if specified
    cases_limit = vitaldb_config.get('cases_limit')
    if cases_limit:
        self.cases = self.cases[:cases_limit]
    
    return self.cases


# Also fix the load_case method to use the correct API:

def _load_signal_from_case(self, case_id: int):
    """Load signal from VitalDB case using correct API."""
    try:
        # Use VitalFile to load the data
        vf = self.vitaldb.VitalFile(case_id)
        
        # Get available tracks
        tracks = vf.get_track_names()
        
        # Find the track we need
        actual_track = None
        track_upper = self.track_name.upper()
        
        for track in tracks:
            if track_upper in track.upper():
                actual_track = track
                break
        
        if not actual_track:
            # Try common mappings
            track_map = {
                'PLETH': ['PLETH', 'PPG', 'SNUADC/PLETH'],
                'ECG_II': ['ECG_II', 'SNUADC/ECG_II', 'ECG'],
                'ABP': ['ABP', 'ART', 'SNUADC/ABP'],
                'BIS/BIS': ['BIS/BIS', 'BIS']
            }
            
            if self.track_name in track_map:
                for possible in track_map[self.track_name]:
                    if possible in tracks:
                        actual_track = possible
                        break
        
        if not actual_track:
            print(f"Track {self.track_name} not found in case {case_id}")
            return None
        
        # Load the data (e.g., first 60 seconds for testing)
        data = vf.to_numpy([actual_track], 0, 60)
        
        if data is not None and len(data) > 0:
            # Handle 2D data
            if data.ndim == 2:
                data = data[:, 0]
            return data
        else:
            return None
            
    except Exception as e:
        print(f"Error loading case {case_id}: {e}")
        return None


# Complete example of how to use in your code:

"""
from src.data.vitaldb_loader import get_available_case_sets, load_channel
import ssl

# Fix SSL certificate issue
ssl._create_default_https_context = ssl._create_unverified_context

# Get available case sets
case_sets = get_available_case_sets()
print(f"Available case sets: {list(case_sets.keys())}")

# Use BIS cases (high quality)
bis_cases = list(case_sets['bis'])[:50]  # First 50 for testing

# Load PPG data from a case
case_id = bis_cases[0]
ppg_signal, fs = load_channel(case_id, 'PLETH', duration_sec=10)
print(f"Loaded {len(ppg_signal)} samples at {fs} Hz")
"""
