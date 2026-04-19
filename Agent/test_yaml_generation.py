#!/usr/bin/env python3
"""Test YAML generation from data folder"""

import sys
from pathlib import Path

# Add task_templates to path
sys.path.insert(0, str(Path(__file__).parent))

from task_templates import generate_remapping_yaml_config
import yaml

# Test with a sample data folder
data_folder = "/pscratch/sd/q/qiuyf/dataset/climatology"

print(f"Testing YAML generation for: {data_folder}")
print("-" * 60)

# Check if folder exists
if not Path(data_folder).exists():
    print(f"ERROR: Folder does not exist: {data_folder}")
    sys.exit(1)

# List .nc files in the folder
nc_files = sorted(Path(data_folder).glob("*.nc"))
print(f"Found {len(nc_files)} .nc files:")
for f in nc_files:
    print(f"  - {f.name}")
print("-" * 60)

# Generate YAML config
config = generate_remapping_yaml_config(data_folder)

if config is None:
    print("ERROR: generate_remapping_yaml_config returned None")
    print("Possible reasons:")
    print("  1. yaml module not installed (pip install pyyaml)")
    print("  2. No .nc files found in folder")
    print("  3. Template parsing failed")
    sys.exit(1)

# Print generated YAML
print("Generated YAML configuration:")
print("=" * 60)
print(yaml.dump(config, default_flow_style=False, sort_keys=False))
print("=" * 60)

# Verify the three key replacements
stream = config.get("stream", {})
print("\nVerification:")
print(f"✓ path_prefix: {stream.get('path_prefix')}")

for substream in stream.get("substreams", []):
    name = substream.get("name")
    filenames = substream.get("filenames")
    if name in ["mesh", "data"]:
        print(f"✓ {name} filenames: {filenames}")

print("\nYAML generation test complete!")
