#!/usr/bin/env python3
"""
Flask backend server for MOPS frontend.
Handles remapping requests from the web UI and returns generated images.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "Agent"
GENERATED_DIR = AGENT_DIR / "generated"

# Add Agent directory to Python path for imports
sys.path.insert(0, str(AGENT_DIR))


@app.route('/api/remapping', methods=['POST'])
def run_remapping():
    """
    Execute remapping task from frontend request.

    Expected JSON payload:
    {
        "request": "user natural language request",
        "data_folder": "/path/to/data",  (optional)
        "yaml_path": "path/to/file.yaml"   (optional)
        "stream_output": true/false        (optional, default: false)
    }

    Returns JSON:
    {
        "success": true/false,
        "manifest": {...},
        "images": [...],
        "message": "...",
        "error": "..." (if failed)
    }
    """
    try:
        data = request.get_json()
        user_request = data.get('request', '')
        data_folder = data.get('data_folder', '')
        yaml_path = data.get('yaml_path', '')
        stream_output = data.get('stream_output', False)  # For real-time output streaming

        if not user_request:
            return jsonify({
                'success': False,
                'error': 'No request provided'
            }), 400

        # Build command for llm_task_agent.py
        cmd = [
            sys.executable,
            str(AGENT_DIR / 'llm_task_agent.py'),
            '--request', user_request,
        ]

        # Add optional parameters
        if data_folder:
            cmd.extend(['--data-folder', data_folder])
        if yaml_path:
            cmd.extend(['--yaml-path', yaml_path])

        # Execute the agent
        print(f"[Backend] Executing: {' '.join(cmd)}")
        print(f"[Backend] Working directory: {PROJECT_ROOT}")
        print("-" * 80)

        if stream_output:
            # Real-time streaming mode - output goes directly to console
            print("[Backend] Streaming mode: output will appear in real-time")
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                timeout=300
            )
            stdout_output = "(streamed to console)"
            stderr_output = ""
        else:
            # Capture mode - output is captured and printed after completion
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            stdout_output = result.stdout
            stderr_output = result.stderr

            # Print captured output for debugging
            if result.stdout:
                print("[Backend] Agent stdout:")
                print(result.stdout)
            if result.stderr:
                print("[Backend] Agent stderr:")
                print(result.stderr)

        print("-" * 80)

        if result.returncode != 0:
            print(f"[Backend] Agent failed with return code {result.returncode}")
            return jsonify({
                'success': False,
                'error': f'Agent failed with code {result.returncode}',
                'stdout': stdout_output,
                'stderr': stderr_output
            }), 500

        print(f"[Backend] Agent completed successfully (exit code 0)")

        # Find the most recent manifest.json
        manifest_path = find_latest_manifest()
        if not manifest_path:
            return jsonify({
                'success': False,
                'error': 'No manifest.json found after execution'
            }), 500

        # Load manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Get output directory from manifest
        output_subdir = manifest.get('output_subdir', 'Agent/outputs/remapping')
        output_dir = PROJECT_ROOT / output_subdir

        # Select the 5 most useful images
        # Want: output_0 ch0,ch1,ch2 (u,v,speed) + output_1 ch0,ch1 (salinity,temp)
        # Skip: output_0 ch3 (unused channel)
        channel_map = manifest.get('channel_map', [])

        # Filter for useful quantities in the correct order
        useful_quantities = ['velocity_u', 'velocity_v', 'velocity_speed', 'salinity', 'temperature']
        selected_images = []

        # First, try to select by quantity name (most reliable)
        for quantity in useful_quantities:
            for item in channel_map:
                if item.get('quantity') == quantity:
                    selected_images.append(item)
                    break

        # Fallback: if quantity matching didn't work, select by output_index and channel
        if len(selected_images) < 5:
            selected_images = []
            for item in channel_map:
                output_idx = item.get('output_index', 0)
                channel = item.get('channel', 0)
                # Take ch0,ch1,ch2 from output_0 and ch0,ch1 from output_1
                if (output_idx == 0 and channel in [0, 1, 2]) or \
                   (output_idx == 1 and channel in [0, 1]):
                    selected_images.append(item)

        # Ensure we have exactly 5 images
        selected_images = selected_images[:5]

        # Log selected images for debugging
        print(f"[Backend] Selected {len(selected_images)} images:")
        for i, item in enumerate(selected_images):
            print(f"  {i+1}. {item.get('file')} - {item.get('label')} ({item.get('quantity')})")

        # Build image URLs (relative paths for frontend)
        images = []
        for item in selected_images:
            images.append({
                'label': item['label'],
                'file': item['file'],
                'colorbar': item['colorbar'],
                'quantity': item.get('quantity', ''),
                'image_url': f'/api/output/{output_subdir}/{item["file"]}',
                'colorbar_url': f'/api/output/{output_subdir}/{item["colorbar"]}'
            })

        return jsonify({
            'success': True,
            'manifest': manifest,
            'images': images,
            'message': f'Remapping completed successfully. Generated {len(images)} images.',
            'output_dir': str(output_dir)
        })

    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'error': 'Agent execution timed out (>5 minutes)'
        }), 500
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/output/<path:filepath>')
def serve_output(filepath):
    """Serve generated output files (images, colorbars, etc.)"""
    file_path = PROJECT_ROOT / filepath
    if not file_path.exists():
        return jsonify({'error': 'File not found'}), 404
    return send_from_directory(file_path.parent, file_path.name)


@app.route('/')
def index():
    """Serve the frontend HTML"""
    return send_from_directory(Path(__file__).parent, 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static frontend files (CSS, JS, etc.)"""
    frontend_dir = Path(__file__).parent
    file_path = frontend_dir / filename
    if file_path.exists() and file_path.is_file():
        return send_from_directory(frontend_dir, filename)
    return jsonify({'error': 'File not found'}), 404


@app.route('/api/browse', methods=['GET'])
def browse_directory():
    """
    Browse server-side directories and files.
    Query params:
      - path: directory path to browse (default: project root)
    Returns: {directories: [...], files: [...], current_path: "..."}
    """
    try:
        # Get path from query param
        requested_path = request.args.get('path', '')

        # Security: only allow browsing within /pscratch/sd/q/qiuyf/ and project root
        allowed_roots = [
            '/pscratch/sd/q/qiuyf',
            str(PROJECT_ROOT),
        ]

        if not requested_path:
            # Return common NERSC paths as suggestions
            return jsonify({
                'suggestions': [
                    '/pscratch/sd/q/qiuyf/dataset/climatology',
                    '/pscratch/sd/q/qiuyf/MOPS_Tutorial/Agent/outputs/remapping',
                    str(PROJECT_ROOT / 'Agent' / 'outputs'),
                ],
                'current_path': '',
                'directories': [],
                'files': []
            })

        browse_path = Path(requested_path).resolve()

        # Security check
        allowed = any(str(browse_path).startswith(root) for root in allowed_roots)
        if not allowed:
            return jsonify({
                'error': 'Access denied: path outside allowed directories'
            }), 403

        if not browse_path.exists():
            return jsonify({
                'error': f'Path does not exist: {requested_path}'
            }), 404

        if not browse_path.is_dir():
            return jsonify({
                'error': f'Not a directory: {requested_path}'
            }), 400

        # List directory contents
        directories = []
        files = []

        for item in sorted(browse_path.iterdir()):
            item_info = {
                'name': item.name,
                'path': str(item)
            }

            if item.is_dir():
                directories.append(item_info)
            elif item.is_file():
                # Include file extension info
                item_info['extension'] = item.suffix
                files.append(item_info)

        return jsonify({
            'current_path': str(browse_path),
            'parent_path': str(browse_path.parent) if browse_path.parent != browse_path else None,
            'directories': directories[:50],  # Limit to 50 items
            'files': files[:50],
            'total_dirs': len(directories),
            'total_files': len(files)
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'project_root': str(PROJECT_ROOT),
        'agent_available': (AGENT_DIR / 'llm_task_agent.py').exists()
    })


def find_latest_manifest():
    """Find the most recently created manifest.json in output directories"""
    search_dirs = [
        PROJECT_ROOT / 'Agent' / 'outputs' / 'remapping',
        PROJECT_ROOT / 'Agent' / 'outputs' / 'streamline',
        PROJECT_ROOT / 'Agent' / 'outputs' / 'pathline',
    ]

    latest = None
    latest_time = 0

    for dir_path in search_dirs:
        if not dir_path.exists():
            continue
        manifest = dir_path / 'manifest.json'
        if manifest.exists():
            mtime = manifest.stat().st_mtime
            if mtime > latest_time:
                latest_time = mtime
                latest = manifest

    return latest


if __name__ == '__main__':
    print(f"[Backend] Starting MOPS Frontend Backend Server")
    print(f"[Backend] Project root: {PROJECT_ROOT}")
    print(f"[Backend] Agent directory: {AGENT_DIR}")

    # Start Flask server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
