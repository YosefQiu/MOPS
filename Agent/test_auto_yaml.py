#!/usr/bin/env python3
"""Demo: Auto-generate YAML from data folder"""

import json
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from task_templates import generate_remapping_yaml_config
import yaml

print("=" * 70)
print("DEMO: Auto-YAML Generation for MOPS Agent")
print("=" * 70)

# Example: User provides data folder instead of YAML path
data_folder = "/pscratch/sd/q/qiuyf/dataset/climatology"
output_dir = Path(__file__).parent / "generated"

print(f"\n1. User specifies data folder: {data_folder}")
print(f"2. Checking if folder exists...")

if not Path(data_folder).is_dir():
    print(f"   ERROR: Folder not found")
    sys.exit(1)

print(f"   ✓ Folder exists")

# Count .nc files
nc_files = list(Path(data_folder).glob("*.nc"))
print(f"   ✓ Found {len(nc_files)} .nc files")

print(f"\n3. Generating YAML configuration...")
config = generate_remapping_yaml_config(data_folder)

if not config:
    print("   ERROR: Generation failed")
    sys.exit(1)

print(f"   ✓ YAML config generated")

# Save YAML
output_dir.mkdir(parents=True, exist_ok=True)
yaml_path = output_dir / "auto_generated.yaml"

with open(yaml_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"\n4. Saved to: {yaml_path}")

# Show key settings
stream = config['stream']
print(f"\n5. Verification:")
print(f"   path_prefix: {stream['path_prefix']}")
for sub in stream['substreams']:
    name = sub['name']
    files = sub['filenames']
    print(f"   {name:6s} files: {files}")

print(f"\n6. How to use:")
print(f"   - LLM agent detects 'data_folder' in user request")
print(f"   - Auto-generates YAML: {yaml_path}")
print(f"   - Uses this YAML for remapping job")

print("\n" + "=" * 70)
print("SUCCESS: Auto-YAML generation works!")
print("=" * 70)
