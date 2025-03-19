#!/usr/bin/env python3

import json
import gzip
import os
import sys

def convert_json_to_jsonl(input_file, output_file):
    """Convert a JSON array file to JSONL format (one JSON object per line)"""
    print(f"Converting {input_file} to {output_file}")
    
    # Read the input file
    if input_file.endswith('.gz'):
        with gzip.open(input_file, 'rt') as f:
            data = json.load(f)
    else:
        with open(input_file, 'r') as f:
            data = json.load(f)
    
    # Ensure data is a list
    if not isinstance(data, list):
        print(f"Error: Expected JSON array, got {type(data)}")
        return False
    
    # Write each item as a separate line
    if output_file.endswith('.gz'):
        with gzip.open(output_file, 'wt') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    else:
        with open(output_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    
    print(f"Successfully converted {len(data)} items to {output_file}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} input.json.gz output.jsonl.gz")
        sys.exit(1)
    
    success = convert_json_to_jsonl(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)