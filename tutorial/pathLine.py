from pyMOPSAPI import *

def run_pathline_from_lat_lon_depth(
    pl: "MOPSPathline",               # your existing object (already .init(...) done)
    sy: int, sm: int, ey: int, em: int, direction: str,
    lat_lon_depth: np.ndarray | str,
    is_file: bool = True,
    csv_delim: str = ",",
    depth_tol_m: float = 1e-6,
    method: str = "rk4",
    delta_minutes: int = 1,
    record_every_minutes: int = 6,
    earth_radius: float = EARTH_RADIUS_M,
) -> List[Dict]:
    """
    Loop depth-groups externally and call your existing single-depth run() for each group.
    """
    # 0) load data
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

    lat, lon, dep = arr[:,0], arr[:,1], arr[:,2]
    unit_xyz = lat_lon_to_unit_xyz(lat, lon)

    # 1) configure time (pathline)
    pl.set_time(sy, sm, ey, em, direction=direction)

    # 2) group by depth
    groups = group_depths(dep, tol_m=depth_tol_m)

    merged: List[Dict] = []
    for depth_val, idx in groups:
        seeds_xyz = place_on_depth(unit_xyz[idx], depth_val, earth_radius=earth_radius)

        # use a single scalar depth for cfg.depth + pass the points
        pl.set_seed(depth=depth_val, points=seeds_xyz, follow_last=True)

        seg = pl.run(method=method, delta_minutes=delta_minutes, record_every_minutes=record_every_minutes)

        for d in seg:
            d["depth"] = depth_val
        merged.extend(seg)

    return merged

def example1(yaml_path: str, pts):
    mops = MOPSPathline(yaml_path)
    mops.init("gpu")
    mops.set_time(1, 1, 1, 12, direction="forward")
    mops.set_seed(
        depth=10.0,
        lat_range=(35.0, 45.0),
        lon_range=(-90.0, -65.0),
        grid=(2, 2),
        points = pts
    )
    lines = mops.run(method="rk4", delta_minutes=1, record_every_minutes=6)
    Vis_PathLines(lines, save_path="pathlines_pts.png", color_by="velocity", title="Pathlines from custom seeds")


def example2(yaml_path: str, npy_path: str):
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

    Vis_PathLines(lines, save_path="pathlines_npy.png", color_by="velocity", title="Pathlines from NPY seeds")
    

if __name__ == "__main__":

    yaml_path   = "/pscratch/sd/q/qiuyf/MOPS_Tutorial/test_ab_climatology.yaml"
    
    anchor = np.array([1908930.101867, -5174124.236251, 3189701.032088], dtype=float)
    pts = generate_points_from_anchor(anchor, n=20, lon_step_deg=2.0)
    lat, lon, depth = xyz_to_lat_lon_depth(pts[:,0], pts[:,1], pts[:,2])
    depth = abs(depth)
    particles = np.column_stack((lat, lon, depth)).astype(float)
    print(particles)
    np.save("seed_lat_lon_depth.npy", particles)
    
    # example1(yaml_path=yaml_path, pts=pts)
    example2(yaml_path=yaml_path, npy_path="seed_lat_lon_depth.npy")
    
    