import numpy as np
import sys
from collections import defaultdict

def load_structured_data(file_path):
    try:
        archive = np.load(file_path)
        records = defaultdict(lambda: {})
        
        for item_key in archive.files:
            segments = item_key.split('*', 2)
            
            if len(segments) < 3:
                print(f"Skipping malformed key: {item_key}")
                continue
                
            record_id = f"{segments[0]}*{segments[1]}"
            attribute = segments[2]
            records[record_id][attribute] = archive[item_key]
            
        return records
        
    except Exception as err:
        print(f"Error reconstructing data: {err}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python reconstruct_npz.py <file.npz>")
    else:
        load_structured_data(sys.argv[1])