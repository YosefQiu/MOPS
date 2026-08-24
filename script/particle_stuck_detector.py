#!/usr/bin/env python3
"""
Particle Stuck Detector - Identify stuck particles in merged VTP trajectory files.

This script analyzes particle trajectories that have been split into multiple line
segments and detects particles that remain stationary for extended periods.
"""

import numpy as np
import sys
import os
import argparse

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


def extract_line_segments(mesh):
    """
    Extract individual line segments from VTP mesh.

    Args:
        mesh: PyVista mesh object

    Returns:
        List of line segment point arrays
    """
    segments = []
    lines = mesh.lines
    n_lines = mesh.n_lines

    idx = 0
    for line_id in range(n_lines):
        n_points = lines[idx]
        idx += 1
        point_ids = lines[idx:idx+n_points]
        idx += n_points

        points = mesh.points[point_ids]
        segments.append(points)

    return segments


def merge_particle_trajectories(segments, num_particles):
    """
    Merge line segments into complete particle trajectories.

    Assumes segments are ordered cyclically: particle_0_seg_0, particle_1_seg_0, ...,
    particle_N_seg_0, particle_0_seg_1, particle_1_seg_1, ...

    Args:
        segments: List of all line segment arrays from VTP
        num_particles: Number of actual particles

    Returns:
        List of complete trajectory arrays for each particle
    """
    if len(segments) % num_particles != 0:
        print(f"WARNING: {len(segments)} segments not evenly divisible by {num_particles} particles")
        print(f"         Results may be incorrect!")

    num_time_periods = len(segments) // num_particles
    print(f"\nMerging trajectories:")
    print(f"  Total segments: {len(segments)}")
    print(f"  Number of particles: {num_particles}")
    print(f"  Segments per particle: {num_time_periods}")

    trajectories = [[] for _ in range(num_particles)]

    for seg_id, seg in enumerate(segments):
        particle_id = seg_id % num_particles
        trajectories[particle_id].append(seg)

    # Concatenate all segments for each particle
    merged_trajectories = []
    for particle_id in range(num_particles):
        all_points = []
        for seg in trajectories[particle_id]:
            all_points.extend(seg)

        merged_traj = np.array(all_points)
        merged_trajectories.append(merged_traj)

        print(f"  Particle {particle_id}: {len(trajectories[particle_id])} segments -> {len(merged_traj)} points")

    return merged_trajectories


def check_stuck_particles(trajectories, distance_threshold=1e-6, consecutive_threshold=5):
    """
    Check for stuck particles in trajectories.

    Args:
        trajectories: List of particle trajectory arrays
        distance_threshold: Minimum movement distance (meters) to be considered moving
        consecutive_threshold: Minimum consecutive stationary steps to flag as stuck

    Returns:
        List of dictionaries containing stuck particle information
    """
    stuck_particles = []

    for particle_id, traj in enumerate(trajectories):
        n_points = len(traj)

        if n_points < 2:
            continue

        # Calculate distances between consecutive points
        displacements = np.diff(traj, axis=0)
        distances = np.linalg.norm(displacements, axis=1)

        # Find where particle is not moving
        stationary = distances < distance_threshold

        # Count consecutive stationary steps
        max_consecutive = 0
        current_consecutive = 0
        stuck_start = -1
        stuck_segments = []

        for i, is_stationary in enumerate(stationary):
            if is_stationary:
                if current_consecutive == 0:
                    stuck_start = i
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                if current_consecutive >= consecutive_threshold:
                    stuck_segments.append({
                        'start_step': stuck_start,
                        'end_step': i,
                        'duration': current_consecutive,
                        'position': traj[stuck_start]
                    })
                current_consecutive = 0

        # Check if stuck at the end
        if current_consecutive >= consecutive_threshold:
            stuck_segments.append({
                'start_step': stuck_start,
                'end_step': len(stationary),
                'duration': current_consecutive,
                'position': traj[stuck_start]
            })

        # Calculate statistics
        total_distance = np.sum(distances)
        mean_step_distance = np.mean(distances)
        stuck_fraction = np.sum(stationary) / len(stationary) if len(stationary) > 0 else 0

        if stuck_segments or max_consecutive >= consecutive_threshold:
            stuck_particles.append({
                'particle_id': particle_id,
                'total_steps': n_points,
                'max_consecutive_stuck': max_consecutive,
                'stuck_segments': stuck_segments,
                'total_distance': total_distance,
                'mean_step_distance': mean_step_distance,
                'stuck_fraction': stuck_fraction,
                'start_pos': traj[0],
                'end_pos': traj[-1]
            })

    return stuck_particles


def print_stuck_particle_report(stuck_particles, total_particles):
    """Print detailed report of stuck particles."""
    print("\n" + "="*80)
    print("STUCK PARTICLE ANALYSIS REPORT (MERGED TRAJECTORIES)")
    print("="*80)

    print(f"\nTotal particles: {total_particles}")
    print(f"Stuck particles: {len(stuck_particles)}")
    print(f"Stuck percentage: {100.0 * len(stuck_particles) / total_particles:.2f}%")

    if not stuck_particles:
        print("\nNo stuck particles detected!")
        return

    print("\n" + "-"*80)
    print("STUCK PARTICLE DETAILS")
    print("-"*80)

    for p in stuck_particles:
        print(f"\nParticle {p['particle_id']}:")
        print(f"  Total steps: {p['total_steps']}")
        print(f"  Max consecutive stuck steps: {p['max_consecutive_stuck']}")
        print(f"  Total distance traveled: {p['total_distance']:.6f} m")
        print(f"  Mean step distance: {p['mean_step_distance']:.6e} m")
        print(f"  Fraction of time stuck: {p['stuck_fraction']*100:.2f}%")
        print(f"  Start position: ({p['start_pos'][0]:.2f}, {p['start_pos'][1]:.2f}, {p['start_pos'][2]:.2f})")
        print(f"  End position: ({p['end_pos'][0]:.2f}, {p['end_pos'][1]:.2f}, {p['end_pos'][2]:.2f})")

        if p['stuck_segments']:
            print(f"  Number of stuck segments: {len(p['stuck_segments'])}")
            for i, seg in enumerate(p['stuck_segments'][:3]):
                print(f"    Segment {i+1}: steps {seg['start_step']}-{seg['end_step']} "
                      f"(duration: {seg['duration']})")
                print(f"      Position: ({seg['position'][0]:.2f}, {seg['position'][1]:.2f}, {seg['position'][2]:.2f})")
            if len(p['stuck_segments']) > 3:
                print(f"    ... and {len(p['stuck_segments'])-3} more segments")


def save_stuck_particle_summary(stuck_particles, total_particles, output_file):
    """Save summary to CSV file."""
    with open(output_file, 'w') as f:
        f.write("# Stuck Particle Analysis Summary (Merged Trajectories)\n")
        f.write(f"# Total particles: {total_particles}\n")
        f.write(f"# Stuck particles: {len(stuck_particles)}\n")
        f.write(f"# Stuck percentage: {100.0 * len(stuck_particles) / total_particles:.2f}%\n")
        f.write("\n")
        f.write("ParticleID,TotalSteps,MaxConsecutiveStuck,TotalDistance,MeanStepDistance,StuckFraction,")
        f.write("StartX,StartY,StartZ,EndX,EndY,EndZ,NumStuckSegments\n")

        for p in stuck_particles:
            f.write(f"{p['particle_id']},{p['total_steps']},{p['max_consecutive_stuck']},")
            f.write(f"{p['total_distance']},{p['mean_step_distance']},{p['stuck_fraction']},")
            f.write(f"{p['start_pos'][0]},{p['start_pos'][1]},{p['start_pos'][2]},")
            f.write(f"{p['end_pos'][0]},{p['end_pos'][1]},{p['end_pos'][2]},")
            f.write(f"{len(p['stuck_segments'])}\n")


def main():
    """Main function with argument parsing."""
    if not HAS_PYVISTA:
        print("ERROR: pyvista is not installed.")
        print("Install with: pip install pyvista")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description='Detect stuck particles in merged VTP trajectory files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s trajectory.vtp 100
  %(prog)s output.vtp 50 --threshold 1e-5 --consecutive 10
  %(prog)s data.vtp 200 -o results.csv

Notes:
  - VTP file should contain merged particle trajectories with line segments
  - Segments are assumed to be ordered cyclically by particle ID
        """
    )

    parser.add_argument(
        'vtp_file',
        help='Path to VTP trajectory file'
    )

    parser.add_argument(
        'num_particles',
        type=int,
        help='Number of actual particles (not line segments)'
    )

    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=1e-6,
        metavar='DIST',
        help='Distance threshold in meters for stuck detection (default: 1e-6)'
    )

    parser.add_argument(
        '-c', '--consecutive',
        type=int,
        default=5,
        metavar='N',
        help='Minimum consecutive stationary steps to flag as stuck (default: 5)'
    )

    parser.add_argument(
        '-o', '--output',
        metavar='FILE',
        help='Save CSV summary to file (default: <vtp_file>_stuck_merged.csv)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.vtp_file):
        print(f"ERROR: File not found: {args.vtp_file}")
        sys.exit(1)

    print(f"Analyzing: {args.vtp_file}")
    print(f"Expected number of particles: {args.num_particles}")

    # Read VTP file
    mesh = pv.read(args.vtp_file)
    print(f"\nLoaded VTP file:")
    print(f"  Total points: {mesh.n_points}")
    print(f"  Total line segments: {mesh.n_lines}")

    # Extract line segments
    segments = extract_line_segments(mesh)

    # Merge into complete trajectories
    trajectories = merge_particle_trajectories(segments, args.num_particles)

    # Analyze for stuck particles
    print(f"\nChecking for stuck particles...")
    print(f"  Distance threshold: {args.threshold} meters")
    print(f"  Consecutive threshold: {args.consecutive} steps")

    stuck_particles = check_stuck_particles(
        trajectories,
        distance_threshold=args.threshold,
        consecutive_threshold=args.consecutive
    )

    # Print report
    print_stuck_particle_report(stuck_particles, args.num_particles)

    # Save to file
    output_csv = args.output if args.output else args.vtp_file.replace('.vtp', '_stuck_merged.csv')
    save_stuck_particle_summary(stuck_particles, args.num_particles, output_csv)
    print(f"\nSummary saved to: {output_csv}")


if __name__ == "__main__":
    main()
