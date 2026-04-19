#!/usr/bin/env python3
"""
Export MOPS pathline trajectory data to binary format for web visualization.

Output formats:
1. .bin - raw binary format (double precision lat/lon pairs)
2. .json - JSON format for JavaScript consumption
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any
import struct


def xyz_to_lat_lon_depth(x, y, z, R=6_371_000.0):
    """Convert ECEF XYZ to lat/lon/depth."""
    lon = np.degrees(np.arctan2(y, x))
    r = np.sqrt(x*x + y*y + z*z)
    lat = np.degrees(np.arcsin(z / r))
    depth = R - r
    return lat, lon, depth


def export_pathlines_to_binary(
    trajectory_lines: List[Dict[str, Any]],
    output_path: str,
    include_velocity: bool = False,
    include_scalars: bool = False
) -> Dict[str, Any]:
    """
    Export pathline trajectories to binary format.

    Binary format (per particle):
    - num_points: int32 (4 bytes)
    - For each point:
      - lat: float64 (8 bytes)
      - lon: float64 (8 bytes)
      - [optional] velocity_u: float64
      - [optional] velocity_v: float64
      - [optional] speed: float64
      - [optional] temperature: float64
      - [optional] salinity: float64

    Returns metadata dict describing the binary structure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "format_version": "1.0",
        "num_particles": len(trajectory_lines),
        "fields": ["lat", "lon"],
        "data_type": "float64",
        "byte_order": "little",
        "particle_offsets": []
    }

    if include_velocity:
        metadata["fields"].extend(["velocity_u", "velocity_v", "speed"])
    if include_scalars:
        metadata["fields"].extend(["temperature", "salinity"])

    with open(output_path, 'wb') as f:
        # Write header: number of particles
        f.write(struct.pack('<i', len(trajectory_lines)))

        for idx, line in enumerate(trajectory_lines):
            particle_start = f.tell()

            if not isinstance(line, dict):
                continue

            P = np.asarray(line.get("points", []))
            if P.shape[0] < 1:
                # Write empty particle
                f.write(struct.pack('<i', 0))
                metadata["particle_offsets"].append({"start": particle_start, "points": 0})
                continue

            # Convert XYZ to lat/lon
            lat, lon, depth = xyz_to_lat_lon_depth(P[:, 0], P[:, 1], P[:, 2])

            # Get optional data
            V = np.asarray(line.get("velocity", [])) if include_velocity else None
            T = np.asarray(line.get("temperature", [])) if include_scalars else None
            S = np.asarray(line.get("salinity", [])) if include_scalars else None

            num_points = len(lat)
            f.write(struct.pack('<i', num_points))

            for i in range(num_points):
                # Always write lat/lon
                f.write(struct.pack('<d', lat[i]))
                f.write(struct.pack('<d', lon[i]))

                # Optional velocity fields
                if include_velocity and V is not None and V.shape[0] == P.shape[0]:
                    vel_u = V[i, 0] if V.shape[1] > 0 else 0.0
                    vel_v = V[i, 1] if V.shape[1] > 1 else 0.0
                    speed = np.linalg.norm(V[i])
                    f.write(struct.pack('<d', vel_u))
                    f.write(struct.pack('<d', vel_v))
                    f.write(struct.pack('<d', speed))
                elif include_velocity:
                    # Write zeros if velocity requested but not available
                    f.write(struct.pack('<ddd', 0.0, 0.0, 0.0))

                # Optional scalar fields
                if include_scalars:
                    temp = T[i] if T is not None and T.shape[0] > i else 0.0
                    sali = S[i] if S is not None and S.shape[0] > i else 0.0
                    f.write(struct.pack('<d', temp))
                    f.write(struct.pack('<d', sali))

            metadata["particle_offsets"].append({
                "start": particle_start,
                "points": num_points
            })

    # Write metadata JSON
    meta_path = output_path.with_suffix('.meta.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Exported {len(trajectory_lines)} particles to {output_path}")
    print(f"Metadata written to {meta_path}")

    return metadata


def export_pathlines_to_json(
    trajectory_lines: List[Dict[str, Any]],
    output_path: str,
    decimation_factor: int = 1
) -> None:
    """
    Export pathlines to JSON format for direct JavaScript consumption.

    JSON structure:
    {
      "particles": [
        {
          "id": 0,
          "points": [[lat, lon], [lat, lon], ...],
          "velocity": [[u, v, w], ...],  // optional
          "temperature": [...],  // optional
          "salinity": [...]  // optional
        },
        ...
      ]
    }
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    particles = []

    for idx, line in enumerate(trajectory_lines):
        if not isinstance(line, dict):
            continue

        P = np.asarray(line.get("points", []))
        if P.shape[0] < 1:
            continue

        # Convert XYZ to lat/lon
        lat, lon, depth = xyz_to_lat_lon_depth(P[:, 0], P[:, 1], P[:, 2])

        # Decimate if requested
        if decimation_factor > 1:
            indices = np.arange(0, len(lat), decimation_factor)
            lat = lat[indices]
            lon = lon[indices]

        particle = {
            "id": idx,
            "points": [[float(lat[i]), float(lon[i])] for i in range(len(lat))]
        }

        # Optional fields
        V = line.get("velocity")
        if V is not None and len(V) > 0:
            V = np.asarray(V)
            if decimation_factor > 1:
                V = V[indices]
            particle["velocity"] = [[float(v) for v in V[i]] for i in range(min(len(V), len(lat)))]

        T = line.get("temperature")
        if T is not None and len(T) > 0:
            T = np.asarray(T)
            if decimation_factor > 1:
                T = T[indices]
            particle["temperature"] = [float(t) for t in T[:len(lat)]]

        S = line.get("salinity")
        if S is not None and len(S) > 0:
            S = np.asarray(S)
            if decimation_factor > 1:
                S = S[indices]
            particle["salinity"] = [float(s) for s in S[:len(lat)]]

        particles.append(particle)

    output_data = {
        "format": "mops-pathlines-v1",
        "num_particles": len(particles),
        "particles": particles
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Exported {len(particles)} particles to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("This module provides pathline export functions.")
    print("Import and use: export_pathlines_to_binary() or export_pathlines_to_json()")
