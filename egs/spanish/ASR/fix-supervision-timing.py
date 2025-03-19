#!/usr/bin/env python3

import json
import gzip
import sys
from pathlib import Path

def fix_supervisions(input_file, output_file, tolerance=0.001):
    """
    Fix supervision timing issues in a cuts JSONL file.
    
    Args:
        input_file: Path to input JSONL file (gzipped)
        output_file: Path to output JSONL file (will be gzipped)
        tolerance: Tolerance for end time (seconds)
    """
    print(f"Processing {input_file} -> {output_file}")
    
    fixed_count = 0
    total_count = 0
    
    with gzip.open(input_file, 'rt') as fin, gzip.open(output_file, 'wt') as fout:
        for line in fin:
            total_count += 1
            cut = json.loads(line.strip())
            
            # Check if this cut has supervisions
            if 'supervisions' in cut and len(cut['supervisions']) > 0:
                fixed_this_cut = False
                
                for sup in cut['supervisions']:
                    # Calculate supervision end time
                    sup_end = sup['start'] + sup['duration']
                    
                    # If supervision ends after cut duration, fix it
                    if sup_end > cut['duration'] + tolerance:
                        # Option 1: Truncate the supervision
                        sup['duration'] = cut['duration'] - sup['start']
                        
                        # Option 2 (alternative): Extend the cut duration
                        # cut['duration'] = sup_end
                        
                        fixed_this_cut = True
                        fixed_count += 1
                
                if fixed_this_cut:
                    print(f"Fixed cut: {cut['id']}")
            
            # Write the fixed cut
            fout.write(json.dumps(cut) + '\n')
    
    print(f"Fixed {fixed_count} supervisions out of {total_count} cuts")
    return fixed_count

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} input.jsonl.gz output.jsonl.gz")
        sys.exit(1)
    
    fixed = fix_supervisions(sys.argv[1], sys.argv[2])
    sys.exit(0 if fixed >= 0 else 1)