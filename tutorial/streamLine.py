from pyMOPSAPI import *


def run_streamline_from_lat_lon_depth(
    sl: "MOPSStreamline",            # your existing object (already .init(...) done)
    start_date: str,                 # e.g., "0018-01-01"
    duration_seconds: int | None = None,
    duration_ymd: Tuple[int,int,int] | None = None,
    latlon_depth: np.ndarray | str = None,  # (N,3) array or path to .npy/.csv
    is_file: bool = True,
    csv_delim: str = ",",
    depth_tol_m: float = 1e-6,
    first_point: np.ndarray | None = None,  # optional override for the first seed
    method: str = "rk4",
    delta_minutes: int = 1,
    record_every_minutes: int = 6,
    earth_radius: float = EARTH_RADIUS_M,
) -> List[Dict]:
    """
      1) Load (lat, lon, depth).
      2) Build unit directions on the sea surface.
            3) Place each particle on its own depth radius r = R - depth.
            4) Run once with per-particle depths.
            5) Inject depth into each line-dict for provenance.
    """
    # 0) load data
    if is_file:
        if isinstance(latlon_depth, str) and latlon_depth.lower().endswith(".npy"):
            arr = np.load(latlon_depth)
        elif isinstance(latlon_depth, str):
            arr = np.loadtxt(latlon_depth, delimiter=csv_delim)
        else:
            raise ValueError("latlon_depth must be a path to .npy or .csv when is_file=True")
    else:
        arr = np.asarray(latlon_depth)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"latlon_depth must be (N,3) [lat, lon, depth], got {arr.shape}")

    lat, lon, dep = arr[:,0], arr[:,1], arr[:,2]
    unit_xyz = lat_lon_to_unit_xyz(lat, lon)

    # 1) set time (single timestep streamline)
    sl.set_time(start=start_date, duration_seconds=duration_seconds, duration_ymd=duration_ymd)

    # 2) place each seed at its own depth and run once
    seeds_xyz = np.zeros_like(unit_xyz)
    for i in range(len(dep)):
        r = earth_radius - dep[i]
        seeds_xyz[i] = unit_xyz[i] * r

    sl.set_seed(depths=dep, points=seeds_xyz, first_point=first_point, follow_last=True)
    seg = sl.run(method=method, delta_minutes=delta_minutes, record_every_minutes=record_every_minutes)

    # 3) annotate depth for provenance (streamline binding does not include depth yet)
    for i, d in enumerate(seg):
        if i < len(dep):
            d["depth"] = float(dep[i])
    return seg


def run_streamline_from_latlon_depth(*args, **kwargs):
    """Backward-compatible alias."""
    return run_streamline_from_lat_lon_depth(*args, **kwargs)

def example1():
    yaml_path   = "/pscratch/sd/q/qiuyf/MOPS_Tutorial/test.yaml"
    mops = MOPSStreamline(yaml_path)

    mops.init("gpu")
    mops.set_time(start="0015-01-01", duration_ymd=(2, 0, 0))
    mops.set_seed(
        depth=10.0,
        lat_range=(35.0, 45.0),
        lon_range=(-90.0, -65.0),
        grid=(2, 2),
        first_point=[1908930.101867, -5174124.236251, 3189701.032088],  # optional
    )
    lines = mops.run(method="rk4", delta_minutes=1, record_every_minutes=6)
    
def example2():
    yaml_path   = "/pscratch/sd/q/qiuyf/MOPS_Tutorial/test.yaml"
    sl = MOPSStreamline(yaml_path).init("gpu")
    lines = run_streamline_from_lat_lon_depth(
        sl,
        start_date="0015-01-01",
        duration_ymd=(0,0,5),
        latlon_depth="seeds_at_55N_for_forward_trajectory.npy",  # (N,3)
        is_file=True,
        depth_tol_m=0.01,        # group seeds whose depths differ <1cm
        method="rk4",
    )
    # lines[i]["points"], ["velocity"], ... and extra ["depth"] per line
    Vis_PathLines(lines, save_path="sl_multi_depth.png", color_by="speed")

    
    

if __name__ == "__main__":
    example2()
    
    