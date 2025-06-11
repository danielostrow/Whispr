from typing import List


def assign_locations(speakers: List[dict]):
    """Assign dummy (x, y) positions along x-axis for each speaker."""
    spacing = 1.0
    for idx, spk in enumerate(speakers):
        spk["location"] = [idx * spacing, 0.0] 