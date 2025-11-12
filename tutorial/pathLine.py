from pyMOPSAPI import *

def run_pathline_from_latlon_depth(
    pl: "MOPSPathline",               # your existing object (already .init(...) done)
    sy: int, sm: int, ey: int, em: int, direction: str,
    latlon_depth: np.ndarray | str,
    is_file: bool = True,
    csv_delim: str = ",",
    depth_tol_m: float = 1e-6,
    first_point: np.ndarray | None = None,
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
        if isinstance(latlon_depth, str) and latlon_depth.lower().endswith(".npy"):
            arr = np.load(latlon_depth)
        elif isinstance(latlon_depth, str):
            arr = np.loadtxt(latlon_depth, delimiter=csv_delim)
        else:
            raise ValueError("latlon_depth must be path when is_file=True")
    else:
        arr = np.asarray(latlon_depth)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"latlon_depth must be (N,3) [lat, lon, depth], got {arr.shape}")

    lat, lon, dep = arr[:,0], arr[:,1], arr[:,2]
    unit_xyz = latlon_to_unit_xyz(lat, lon)

    # 1) configure time (pathline)
    pl.set_time(sy, sm, ey, em, direction=direction)

    # 2) group by depth
    groups = group_depths(dep, tol_m=depth_tol_m)

    merged: List[Dict] = []
    for depth_val, idx in groups:
        seeds_xyz = place_on_depth(unit_xyz[idx], depth_val, earth_radius=earth_radius)

        # use a single scalar depth for cfg.depth + pass the points
        pl.set_seed(depth=depth_val, points=seeds_xyz, first_point=first_point, follow_last=True)

        seg = pl.run(method=method, delta_minutes=delta_minutes, record_every_minutes=record_every_minutes)

        for d in seg:
            d["depth"] = depth_val
        merged.extend(seg)

    return merged



if __name__ == "__main__":

    yaml_path   = "/pscratch/sd/q/qiuyf/MOPS_Tutorial/test.yaml"
    mops = MOPSPathline(yaml_path)

    mops.init("gpu")
    mops.set_time(18, 1, 18, 2, direction="forward")
    mops.set_seed(
        depth=10.0,
        lat_range=(35.0, 45.0),
        lon_range=(-90.0, -65.0),
        grid=(2, 2),
        first_point=[1908930.101867, -5174124.236251, 3189701.032088],  # optional
    )
    lines = mops.run(method="rk4", delta_minutes=1, record_every_minutes=6)

    