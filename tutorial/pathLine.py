from pyMOPSAPI import *

# =============================================================================
# Per-Particle Depth Support
# =============================================================================
# Now MOPS supports different depths for each particle!
# No more grouping by depth needed - just pass all particles at once.
# =============================================================================


def check_pathlines(lines: List[Dict], verbose: bool = True) -> bool:
    """
    Validate pathline results for consistency and data quality.
    
    Checks:
      1. No NaN values in positions
      2. No NaN values in velocities  
      3. All trajectories have the same length (same number of points)
    
    Parameters:
      lines: List of trajectory dicts from run_pathline_from_lat_lon_depth() or pl.run()
      verbose: If True, print detailed check results
      
    Returns:
      True if all checks pass, False otherwise
    """
    if not lines:
        if verbose:
            print("[CHECK] ❌ No trajectories to check (empty list)")
        return False
    
    all_passed = True
    n_lines = len(lines)
    
    # Collect sizes for uniformity check
    position_sizes = []
    velocity_sizes = []
    nan_position_count = 0
    nan_velocity_count = 0
    nan_position_lines = []
    nan_velocity_lines = []
    
    for i, line in enumerate(lines):
        points = np.asarray(line.get("points", []))
        velocity = np.asarray(line.get("velocity", []))
        
        # Track sizes
        position_sizes.append(len(points))
        velocity_sizes.append(len(velocity))
        
        # Check for NaN in positions
        if points.size > 0 and np.any(np.isnan(points)):
            nan_count = np.sum(np.isnan(points))
            nan_position_count += nan_count
            nan_position_lines.append(i)
        
        # Check for NaN in velocities
        if velocity.size > 0 and np.any(np.isnan(velocity)):
            nan_count = np.sum(np.isnan(velocity))
            nan_velocity_count += nan_count
            nan_velocity_lines.append(i)
    
    # Check 1: No NaN in positions
    if nan_position_count > 0:
        all_passed = False
        if verbose:
            print(f"[CHECK] ❌ Found {nan_position_count} NaN values in positions")
            print(f"        Affected lines: {nan_position_lines[:10]}{'...' if len(nan_position_lines) > 10 else ''}")
    else:
        if verbose:
            print(f"[CHECK] ✅ No NaN in positions ({n_lines} trajectories)")
    
    # Check 2: No NaN in velocities
    if nan_velocity_count > 0:
        all_passed = False
        if verbose:
            print(f"[CHECK] ❌ Found {nan_velocity_count} NaN values in velocities")
            print(f"        Affected lines: {nan_velocity_lines[:10]}{'...' if len(nan_velocity_lines) > 10 else ''}")
    else:
        if verbose:
            print(f"[CHECK] ✅ No NaN in velocities ({n_lines} trajectories)")
    
    # Check 3: All trajectories have uniform length
    unique_pos_sizes = set(position_sizes)
    unique_vel_sizes = set(velocity_sizes)
    
    if len(unique_pos_sizes) > 1:
        all_passed = False
        if verbose:
            min_size = min(position_sizes)
            max_size = max(position_sizes)
            print(f"[CHECK] ❌ Position sizes not uniform: min={min_size}, max={max_size}")
            # Show distribution
            size_counts = {}
            for s in position_sizes:
                size_counts[s] = size_counts.get(s, 0) + 1
            print(f"        Size distribution: {dict(sorted(size_counts.items()))}")
    else:
        if verbose:
            print(f"[CHECK] ✅ All trajectories have uniform position length: {position_sizes[0]} points")
    
    if len(unique_vel_sizes) > 1:
        all_passed = False
        if verbose:
            min_size = min(velocity_sizes)
            max_size = max(velocity_sizes)
            print(f"[CHECK] ❌ Velocity sizes not uniform: min={min_size}, max={max_size}")
    else:
        if verbose:
            print(f"[CHECK] ✅ All trajectories have uniform velocity length: {velocity_sizes[0]} points")
    
    # Summary
    if verbose:
        print(f"\n[CHECK] {'✅ ALL CHECKS PASSED' if all_passed else '❌ SOME CHECKS FAILED'}")
        print(f"        Total trajectories: {n_lines}")
        if position_sizes:
            print(f"        Points per trajectory: {position_sizes[0] if len(unique_pos_sizes) == 1 else f'{min(position_sizes)}-{max(position_sizes)}'}")
    
    return all_passed

def run_pathline_from_lat_lon_depth(
    pl: "MOPSPathline",
    sy: int, sm: int, ey: int, em: int, direction: str,
    lat_lon_depth: np.ndarray | str,
    is_file: bool = True,
    csv_delim: str = ",",
    method: str = "rk4",
    delta_minutes: int = 1,
    record_every_minutes: int = 6,
    earth_radius: float = EARTH_RADIUS_M,
) -> List[Dict]:
    """
    Run pathline tracing with per-particle depths.
    
    Now supports different depths for each particle - no grouping needed!
    
    Parameters:
      pl: MOPSPathline instance (already .init() done)
      sy, sm, ey, em: start/end year and month
      direction: "forward" or "backward"
      lat_lon_depth: (N,3) array or path to file with columns [lat, lon, depth]
      is_file: if True, lat_lon_depth is a file path
      method: "rk4" or "euler"
      delta_minutes: integration time step
      record_every_minutes: output sampling interval
      earth_radius: Earth radius in meters
      
    Returns:
      List of trajectory dicts, one per particle
    """
    # 0) Load data
    if is_file:
        if isinstance(lat_lon_depth, str) and lat_lon_depth.lower().endswith(".npy"):
            arr = np.load(lat_lon_depth)
        elif isinstance(lat_lon_depth, str):
            arr = np.loadtxt(lat_lon_depth, delimiter=csv_delim)
        else:
            raise ValueError("lat_lon_depth must be path when is_file=True")
    else:
        arr = np.asarray(lat_lon_depth)
    
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"lat_lon_depth must be (N,3) [lat, lon, depth], got {arr.shape}")

    lat, lon, depths = arr[:, 0], arr[:, 1], arr[:, 2]
    
    # Convert lat/lon to unit XYZ, then place at each particle's depth
    unit_xyz = lat_lon_to_unit_xyz(lat, lon)
    
    # Place each particle at its own depth
    seeds_xyz = np.zeros_like(unit_xyz)
    for i in range(len(depths)):
        r = earth_radius - depths[i]  # radius at this depth
        seeds_xyz[i] = unit_xyz[i] * r
    
    # 1) Configure time
    pl.set_time(sy, sm, ey, em, direction=direction)

    # 2) Set seeds with per-particle depths (no grouping needed!)
    pl.set_seed(
        depths=depths,          # per-particle depths array
        points=seeds_xyz,       # XYZ coordinates
        follow_last=True
    )

    # 3) Run pathline
    lines = pl.run(
        method=method,
        delta_minutes=delta_minutes,
        record_every_minutes=record_every_minutes
    )
    
    # Note: depth is now automatically included in each line dict from MOPS
    # (set by TrajectoryLine.depth in C++ and returned via bindings.cpp)
    
    return lines


def example1_uniform_depth(yaml_path: str, pts):
    """Example: uniform depth with custom seed points"""
    mops = MOPSPathline(yaml_path)
    mops.init("gpu")
    mops.set_time(1, 1, 1, 12, direction="forward")
    mops.set_seed(
        depth=10.0,
        lat_range=(35.0, 45.0),
        lon_range=(-90.0, -65.0),
        grid=(2, 2),
        points=pts
    )
    lines = mops.run(method="rk4", delta_minutes=1, record_every_minutes=6)
    Vis_PathLines(lines, save_path="pathlines_pts.png", color_by="velocity", title="Pathlines from custom seeds")


def example2_per_particle_depth(yaml_path: str, npy_path: str):
    """Example: per-particle depths from NPY file (no grouping needed!)"""
    pl = MOPSPathline(yaml_path).init("gpu")

    lines = run_pathline_from_lat_lon_depth(
        pl,
        sy=1, sm=1,
        ey=1, em=12,
        direction="forward",
        lat_lon_depth=npy_path,
        is_file=True,
        method="rk4",
        delta_minutes=1,
        record_every_minutes=6
    )

    # Validate results
    check_pathlines(lines)

    Vis_PathLines(lines, save_path="pathlines_npy.png", color_by="velocity", title="Pathlines from NPY seeds")
    
    # Print depth statistics
    depths = [d["depth"] for d in lines]
    print(f"Traced {len(lines)} particles with depths from {min(depths):.2f}m to {max(depths):.2f}m")


def example3_explicit_multi_depth(yaml_path: str):
    """Example: explicitly set different depths for each particle"""
    pl = MOPSPathline(yaml_path).init("gpu")
    pl.set_time(1, 1, 1, 12, direction="forward")
    
    # Create particles at different depths (same lat/lon, different depths)
    lats = np.array([40.0, 40.0, 40.0, 40.0])
    lons = np.array([-50.0, -50.0, -50.0, -50.0])
    depths = np.array([10.0, 50.0, 100.0, 200.0])  # Different depth for each particle!
    
    # Convert to XYZ
    unit_xyz = lat_lon_to_unit_xyz(lats, lons)
    seeds_xyz = np.zeros_like(unit_xyz)
    for i in range(len(depths)):
        r = EARTH_RADIUS_M - depths[i]
        seeds_xyz[i] = unit_xyz[i] * r
    
    # Set seeds with per-particle depths
    pl.set_seed(
        depths=depths,      # Each particle has its own depth!
        points=seeds_xyz,
        follow_last=True
    )
    
    lines = pl.run(method="rk4", delta_minutes=1, record_every_minutes=6)
    
    # Validate results
    check_pathlines(lines)
    
    print(f"Traced {len(lines)} particles at depths: {depths}")
    Vis_PathLines(lines, save_path="pathlines_multi_depth.png", color_by="velocity",
                  title="Pathlines at Multiple Depths")


if __name__ == "__main__":
    yaml_path = "/pscratch/sd/q/qiuyf/MOPS_Tutorial/test_ab_climatology.yaml"
    
    # Generate test data with varying depths
    anchor = np.array([1908930.101867, -5174124.236251, 3189701.032088], dtype=float)
    pts = generate_points_from_anchor(anchor, n=20, lon_step_deg=2.0)
    lat, lon, depth = xyz_to_lat_lon_depth(pts[:, 0], pts[:, 1], pts[:, 2])
    
    # Assign different depths to particles (10m to 200m)
    depths_varied = np.linspace(10.0, 200.0, len(lat))
    
    particles = np.column_stack((lat, lon, depths_varied)).astype(float)
    print("Particles with varied depths:")
    print(particles)
    np.save("seed_lat_lon_depth.npy", particles)
    
    # Run with per-particle depths (no grouping needed!)
    example2_per_particle_depth(yaml_path=yaml_path, npy_path="seed_lat_lon_depth.npy")
    
    # Or run the explicit multi-depth example
    # example3_explicit_multi_depth(yaml_path=yaml_path)
    
    