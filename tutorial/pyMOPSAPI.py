import sys
from pathlib import Path
sys.path.append("../tools/pyMOPS/pyMOPS/")
import pyMOPS
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

import warnings
import functools

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from typing import List, Tuple, Dict, Iterable, Optional, Sequence


print("Available API:", dir(pyMOPS))
print("\n\n")
    
def deprecated(reason: str = ""):
    def decorator(cls_or_func):
        msg = f"{cls_or_func.__name__} is deprecated."
        if reason:
            msg += f" {reason}"

        if isinstance(cls_or_func, type):
            orig_init = cls_or_func.__init__
            @functools.wraps(orig_init)
            def new_init(self, *args, **kwargs):
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
                return orig_init(self, *args, **kwargs)
            cls_or_func.__init__ = new_init
            return cls_or_func

        @functools.wraps(cls_or_func)
        def wrapped(*args, **kwargs):
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return cls_or_func(*args, **kwargs)
        return wrapped
    return decorator


EARTH_RADIUS_M = 6_371_000.0

def lat_lon_to_unit_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Return unit-length Cartesian directions for points on the sphere (sea surface)."""
    lat = np.deg2rad(lat_deg.astype(np.float64))
    lon = np.deg2rad(lon_deg.astype(np.float64))
    cx, sx = np.cos(lat), np.sin(lat)
    cy, sy = np.cos(lon), np.sin(lon)
    x = cx * cy
    y = cx * sy
    z = sx
    return np.stack([x, y, z], axis=1)  # unit vectors

def place_on_depth(unit_xyz: np.ndarray, depth_m: float, earth_radius: float = EARTH_RADIUS_M) -> np.ndarray:
    """Project unit directions to ECEF coordinates at given depth: r = R - depth."""
    r = float(earth_radius - depth_m)
    return (unit_xyz * r).astype(np.float64, copy=False)

def group_depths(depths: np.ndarray, tol_m: float = 1e-6) -> List[Tuple[float, np.ndarray]]:
    """
    Group indices by (approximately) equal depth using a tolerance (meters).
    Returns a list of (depth_value, indices).
    """
    d = np.asarray(depths, dtype=np.float64)
    order = np.argsort(d)
    groups: List[Tuple[float, np.ndarray]] = []
    if d.size == 0:
        return groups
    start = 0
    for i in range(1, d.size + 1):
        if i == d.size or abs(d[order][i] - d[order][start]) > tol_m:
            group_idx = order[start:i]
            depth_val = float(np.mean(d[group_idx]))
            groups.append((depth_val, group_idx))
            start = i
    return groups


def xyz_to_lat_lon(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.where(r < 1e-8, np.nan, r)
    lat = np.arcsin(z / r) * 180 / np.pi
    lon = np.arctan2(y, x) * 180 / np.pi
    return lat, lon

def xyz_to_lat_lon_depth(x, y, z, R=EARTH_RADIUS_M):
    """
    Returns (lat_deg, lon_deg, depth_m)
    depth positive downward, consistent with your C++.
    """
    lon = np.degrees(np.arctan2(y, x))
    r = np.sqrt(x*x + y*y + z*z)
    lat = np.degrees(np.arcsin(z / r))
    depth = R - r
    return lat, lon, depth

def lat_lon_depth_to_xyz(lat_deg, lon_deg, depth, R=EARTH_RADIUS_M):
    r = R - depth
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.array([x, y, z], dtype=float)

def make_same_lat_depth_diff_lon(p_xyz, delta_lon_deg):
    lat, lon, depth = xyz_to_lat_lon_depth(p_xyz[0], p_xyz[1], p_xyz[2])
    lon = lon + delta_lon_deg
    # wrap to [-180, 180]
    if lon > 180.0:
        lon -= 360.0
    if lon < -180.0:
        lon += 360.0
    return lat_lon_depth_to_xyz(lat, lon, depth)

def generate_points_from_anchor(anchor_xyz, n=15, lon_step_deg=2.0):
    """
    anchor_xyz: (3,)
    returns pts: (n,3), where pts[0]=anchor and pts[i] shifts lon by lon_step_deg*i
    """
    pts = np.empty((n, 3), dtype=float)
    pts[0] = np.asarray(anchor_xyz, dtype=float)
    for i in range(1, n):
        pts[i] = make_same_lat_depth_diff_lon(pts[0], lon_step_deg * i)
    return pts

@deprecated("You should implement your own visualization function. pyMOPS does not include any visualization code.")
def Vis_PathLines(
    trajectory_lines,
    save_path="pathlines.png",
    region_extent=None,                   # [lon_min, lon_max, lat_min, lat_max]
    color_by=None,                        # None / 'temperature' / 'salinity' / 'speed'
    cmap="viridis",
    vmin=None, vmax=None,
    linewidth=1.0,
    show_colorbar=True,
    title="Pathlines (Lat/Lon)"
):
    """
    trajectory_lines: 
      - newFormat MOPS_RunPathLine -> list[dict]
    """
    
    def _build_segments(lons, lats, values=None):
        lons = np.asarray(lons); lats = np.asarray(lats)
        ok = ~np.isnan(lons) & ~np.isnan(lats)
        lons = lons[ok]; lats = lats[ok]
        if len(lons) < 2:
            return np.empty((0, 2, 2)), np.empty((0,)) if values is not None else None

        dlon = ((lons[1:] - lons[:-1] + 180.0) % 360.0) - 180.0
        good = np.abs(dlon) < 170.0  

        segs = np.stack([
            np.column_stack([lons[:-1], lats[:-1]])[good],
            np.column_stack([lons[1:],  lats[1:]])[good]
        ], axis=1)

        if values is None:
            return segs, None
        values = np.asarray(values)
        values = values[ok]
        if len(values) != len(lons):
            return segs, None
        vals_seg = 0.5 * (values[:-1] + values[1:])
        vals_seg = vals_seg[good]
        return segs, vals_seg

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    if region_extent:
        ax.set_extent(region_extent, crs=ccrs.PlateCarree())
    else:
        ax.set_global()

    ax.stock_img()
    ax.coastlines(linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.6, linestyle='--')
    try:
        gl.top_labels = gl.right_labels = False
    except Exception:
        pass

    all_lats, all_lons = [], []
    lcs = []  # Collects all LineCollection for uniform normalization.
    scal_min, scal_max = np.inf, -np.inf

    for line in trajectory_lines:
        if isinstance(line, dict):
            P = np.asarray(line["points"])            # (N,3)
            V = np.asarray(line.get("velocity", []))  # (N,3) 
            T = np.asarray(line.get("temperature", []))  # (N,) 
            S = np.asarray(line.get("salinity", []))     # (N,) 

        if P.shape[0] < 2:
            continue

        lat, lon = xyz_to_lat_lon(P[:, 0], P[:, 1], P[:, 2])
        all_lats.extend(lat[np.isfinite(lat)])
        all_lons.extend(lon[np.isfinite(lon)])

        values = None
        if color_by is None:
            values = None
        elif color_by.lower() in ("temperature", "temp"):
            values = T if T.size == P.shape[0] else None
        elif color_by.lower() in ("salinity", "sali", "salt"):
            values = S if S.size == P.shape[0] else None
        elif color_by.lower() in ("speed", "velocity", "vel"):
            if V.size == P.shape[0] * 3:
                values = np.linalg.norm(V, axis=1)
            else:
                values = None
        else:
            values = None

        segs, vals_seg = _build_segments(lon, lat, values)

        if segs.shape[0] == 0:
            continue

        if vals_seg is None or len(vals_seg) == 0:
            lc = LineCollection(
                segs,
                linewidths=linewidth,
                colors="white",
                transform=ccrs.PlateCarree()
            )
        else:
            lc = LineCollection(
                segs,
                linewidths=linewidth,
                cmap=cmap,
                array=vals_seg,
                transform=ccrs.PlateCarree()
            )


        ax.add_collection(lc)
        lcs.append(lc)

    # Automatic setting range
    if region_extent is None and len(all_lats) > 0:
        margin = 2.0
        ax.set_extent([
            float(np.nanmin(all_lons)) - margin, float(np.nanmax(all_lons)) + margin,
            float(np.nanmin(all_lats)) - margin, float(np.nanmax(all_lats)) + margin
        ], crs=ccrs.PlateCarree())

    # Uniform Normalization & colorbar
    if color_by is not None and len(lcs) > 0 and np.isfinite([scal_min, scal_max]).all():
        if vmin is None: vmin = scal_min
        if vmax is None: vmax = scal_max
        if vmin == vmax:  
            vmin -= 1e-12
            vmax += 1e-12

        norm = Normalize(vmin=vmin, vmax=vmax)
        for lc in lcs:
            if lc.get_array() is not None:
                lc.set_norm(norm)
                lc.set_cmap(cmap)

        if show_colorbar:
            h = next((lc for lc in lcs if lc.get_array() is not None), None)
            if h is not None:
                cb = plt.colorbar(h, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
                label_map = {
                    "temperature": "Temperature",
                    "temp": "Temperature",
                    "salinity": "Salinity",
                    "sali": "Salinity",
                    "salt": "Salinity",
                    "speed": "Speed (|v|)",
                    "velocity": "Speed (|v|)",
                    "vel": "Speed (|v|)"
                }
                cb.set_label(label_map.get(color_by.lower(), color_by))

    plt.title(title)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


class MOPSRemapping:
    """
    Python wrapper for remapping, mirroring tutorial/reMapping.cpp.

    Workflow:
      1) init(device, time_stamp, time_step)
      2) run(...)
      3) optionally save outputs to png / npy
    """

    def __init__(self, yaml_path: str):
        self.yaml_path = str(yaml_path)

        self.grid = pyMOPS.MPASOGrid()
        self.sol = pyMOPS.MPASOSolution()

        self._device: Optional[str] = None
        self._time_stamp: Optional[str] = None
        self._time_step: int = 0
        self._initialized = False
        self._active_solution_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Init / load
    # ------------------------------------------------------------------
    def init(
        self,
        device: str = "gpu",
        time_stamp: str = "0015-01-01",
        time_step: int = 0,
        add_temperature: bool = True,
        add_salinity: bool = True,
    ):
        """
        Initialize runtime, load grid + one solution snapshot, and register them.
        Mirrors tutorial/reMapping.cpp.

        Parameters
        ----------
        device : str
            "gpu" or "cpu"
        time_stamp : str
            e.g. "0015-01-01"
        time_step : int
            usually 0
        add_temperature : bool
            whether to expose temperature attribute
        add_salinity : bool
            whether to expose salinity attribute
        """
        self._device = device
        self._time_stamp = time_stamp
        self._time_step = int(time_step)

        pyMOPS.MOPS_Init(device)

        # Load grid
        self.grid.init_from_reader(
            pyMOPS.MPASOReader.readGridData(self.yaml_path)
        )

        # Load solution at given time stamp
        self.sol.init_from_reader(
            pyMOPS.MPASOReader.readSolData(
                self.yaml_path, time_stamp, self._time_step
            )
        )

        # Match tutorial/reMapping.cpp behavior
        if add_temperature:
            self.sol.add_attribute("temperature", pyMOPS.AttributeFormat.kFloat)
        if add_salinity:
            self.sol.add_attribute("salinity", pyMOPS.AttributeFormat.kFloat)

        pyMOPS.MOPS_Begin()
        pyMOPS.MOPS_AddGridMesh(self.grid)
        pyMOPS.MOPS_AddAttribute(self.sol.getID(), self.sol)
        pyMOPS.MOPS_End()
        pyMOPS.MOPS_ActiveAttribute(self.sol.getID(), None)

        self._active_solution_id = self.sol.getID()
        self._initialized = True
        return self

    # ------------------------------------------------------------------
    # Config builder
    # ------------------------------------------------------------------
    def build_config(
        self,
        width: int = 3601,
        height: int = 1801,
        lat_range: Tuple[float, float] = (-90.0, 90.0),
        lon_range: Tuple[float, float] = (-180.0, 180.0),
        fixed_depth: float = 10.0,
        time_step: int = 0,
        calc_type=None,
        vis_type=None,
        position_type=None,
        save_type=None,
    ):
        """
        Build VisualizationSettings for remapping.

        Notes
        -----
        The C++ tutorial relies on defaults for CalcType / VisType / PositionType.
        So these are optional here too. Only override them when you explicitly know
        which enum you want.
        """
        cfg = pyMOPS.VisualizationSettings()
        cfg.imageSize = (int(width), int(height))
        cfg.LatRange = (float(lat_range[0]), float(lat_range[1]))
        cfg.LonRange = (float(lon_range[0]), float(lon_range[1]))
        cfg.FixedDepth = float(fixed_depth)
        cfg.TimeStep = int(time_step)

        if calc_type is not None:
            cfg.CalcType = calc_type
        if vis_type is not None:
            cfg.VisType = vis_type
        if position_type is not None:
            cfg.PositionType = position_type
        if save_type is not None:
            cfg.SaveType = save_type

        return cfg

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(
        self,
        width: int = 3601,
        height: int = 1801,
        lat_range: Tuple[float, float] = (-90.0, 90.0),
        lon_range: Tuple[float, float] = (-180.0, 180.0),
        fixed_depth: float = 10.0,
        time_step: int = 0,
        calc_type=None,
        vis_type=None,
        position_type=None,
        save_type=None,
        return_numpy: bool = True,
    ) -> List[np.ndarray]:
        """
        Execute remapping and return a list of images.

        Each output image is expected to be shape (H, W, 4).
        """
        if not self._initialized:
            raise RuntimeError(
                "MOPSRemapping is not initialized. Call .init(...) first."
            )

        cfg = self.build_config(
            width=width,
            height=height,
            lat_range=lat_range,
            lon_range=lon_range,
            fixed_depth=fixed_depth,
            time_step=time_step,
            calc_type=calc_type,
            vis_type=vis_type,
            position_type=position_type,
            save_type=save_type,
        )

        imgs = pyMOPS.MOPS_RunRemapping(cfg)

        if return_numpy:
            return [np.asarray(img) for img in imgs]
        return imgs

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
        """
        Normalize arbitrary float array to uint8 in [0,255].
        NaNs/Infs are handled safely.
        """
        x = np.asarray(arr, dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        xmin = np.min(x)
        xmax = np.max(x)
        if xmax <= xmin:
            return np.zeros_like(x, dtype=np.uint8)

        y = (x - xmin) / (xmax - xmin)
        y = np.clip(y * 255.0, 0, 255)
        return y.astype(np.uint8)

    @staticmethod
    def save_npy(images: Sequence[np.ndarray], out_dir: str, prefix: str = "remap"):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(images):
            np.save(out / f"{prefix}_{i}.npy", np.asarray(img))

    @staticmethod
    def save_png_channels(
        images: Sequence[np.ndarray],
        out_dir: str,
        prefix: str = "output",
        channels: Iterable[int] = (0, 1, 2),
    ):
        """
        Save selected channels of each remapped image as grayscale PNGs,
        matching the spirit of tutorial/reMapping.cpp.
        """
        from PIL import Image

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(images):
            arr = np.asarray(img)
            if arr.ndim != 3 or arr.shape[2] < 4:
                raise ValueError(
                    f"Expected image shape (H,W,4), got {arr.shape} for image {i}"
                )

            for ch in channels:
                if ch < 0 or ch >= arr.shape[2]:
                    raise ValueError(f"Invalid channel {ch} for image {i}")

                u8 = MOPSRemapping._normalize_to_uint8(arr[:, :, ch])
                Image.fromarray(u8, mode="L").save(out / f"{prefix}_{i}_ch{ch}.png")

    @staticmethod
    def save_colormap_png(
        images: Sequence[np.ndarray],
        out_dir: str,
        prefix: str = "output",
        channel: int = 3,
        cmap_name: str = "viridis",
        save_colorbar: bool = True,
    ):
        """
        Save one selected channel from each remapped image as a pseudocolor PNG,
        matching the logic of the C++ SaveToPNG:
        - use one channel only
        - ignore NaNs when computing min/max
        - NaNs become transparent black
        - normal values mapped through colormap
        """
        from PIL import Image
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        import matplotlib
        cmap = matplotlib.colormaps.get_cmap(cmap_name)

        for i, img in enumerate(images):
            arr = np.asarray(img)
            if arr.ndim != 3:
                raise ValueError(f"Expected image shape (H,W,C), got {arr.shape} for image {i}")
            if channel < 0 or channel >= arr.shape[2]:
                raise ValueError(
                    f"Invalid channel={channel}, image {i} only has {arr.shape[2]} channels"
                )

            channel_data = np.asarray(arr[:, :, channel], dtype=np.float64)
            h, w = channel_data.shape

            valid_mask = np.isfinite(channel_data)
            if not np.any(valid_mask):
                # all invalid -> fully transparent black
                rgba_u8 = np.zeros((h, w, 4), dtype=np.uint8)
                min_val, max_val = 0.0, 1.0
            else:
                valid_vals = channel_data[valid_mask]
                min_val = float(np.min(valid_vals))
                max_val = float(np.max(valid_vals))
                if min_val >= max_val:
                    max_val = min_val + 1e-5

                norm = np.zeros_like(channel_data, dtype=np.float64)
                norm[valid_mask] = (channel_data[valid_mask] - min_val) / (max_val - min_val)
                norm = np.clip(norm, 0.0, 1.0)

                rgba = cmap(norm)  # float RGBA in [0,1], shape (H,W,4)
                rgba_u8 = (rgba * 255.0).astype(np.uint8)

                # NaNs -> transparent black
                rgba_u8[~valid_mask, 0] = 0
                rgba_u8[~valid_mask, 1] = 0
                rgba_u8[~valid_mask, 2] = 0
                rgba_u8[~valid_mask, 3] = 0

                # valid pixels fully opaque
                rgba_u8[valid_mask, 3] = 255

            Image.fromarray(rgba_u8, mode="RGBA").save(out / f"{prefix}_{i}_ch{channel}.png")

            if save_colorbar:
                # Create vertical colorbar (changed from horizontal for frontend display)
                fig, ax = plt.subplots(figsize=(1.5, 6))  # Narrow and tall for vertical
                fig.subplots_adjust(left=0.5)  # Adjust left margin for vertical orientation

                norm_obj = mcolors.Normalize(vmin=min_val, vmax=max_val)
                sm = cm.ScalarMappable(norm=norm_obj, cmap=cmap)
                sm.set_array([])

                cbar = fig.colorbar(sm, cax=ax, orientation="vertical")  # Changed to vertical
                # cbar.set_label(f"Channel {channel}")

                fig.savefig(
                    out / f"{prefix}_{i}_ch{channel}_colorbar.png",
                    dpi=200,
                    bbox_inches="tight",
                    transparent=False,
                )
                plt.close(fig)

    @staticmethod
    def save_colormap_pngs(
        images: Sequence[np.ndarray],
        out_dir: str,
        prefix: str = "output",
        channels: Iterable[int] = (0, 1, 2, 3),
        cmap_name: str = "coolwarm",
        save_colorbar: bool = True,
    ):
        """
        Save multiple channels, one pseudocolor PNG per channel per image.
        """
        for ch in channels:
            MOPSRemapping.save_colormap_png(
                images=images,
                out_dir=out_dir,
                prefix=prefix,
                channel=ch,
                cmap_name=cmap_name,
                save_colorbar=save_colorbar,
            )


class MOPSReGrid:
    """
    Python wrapper for regridding at fixed latitude.

    This class mirrors the functionality of tutorial/reGrid.cpp, which performs
    regridding along a fixed latitude cross-section through the ocean depth.

    Workflow:
      1) init(device, time_stamp, time_step)
      2) run(...)
      3) save outputs using save_to_binary() or save_to_png()
    """

    def __init__(self, yaml_path: str):
        self.yaml_path = str(yaml_path)

        self.grid = pyMOPS.MPASOGrid()
        self.sol = pyMOPS.MPASOSolution()

        self._device: Optional[str] = None
        self._time_stamp: Optional[str] = None
        self._time_step: int = 0
        self._initialized = False
        self._active_solution_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Init / load
    # ------------------------------------------------------------------
    def init(
        self,
        device: str = "gpu",
        time_stamp: str = "0015-01-01",
        time_step: int = 0,
        add_temperature: bool = True,
        add_salinity: bool = True,
    ):
        """
        Initialize runtime, load grid + one solution snapshot, and register them.

        Parameters
        ----------
        device : str
            "gpu" or "cpu"
        time_stamp : str
            e.g. "0015-01-01"
        time_step : int
            usually 0
        add_temperature : bool
            whether to expose temperature attribute
        add_salinity : bool
            whether to expose salinity attribute
        """
        self._device = device
        self._time_stamp = time_stamp
        self._time_step = int(time_step)

        pyMOPS.MOPS_Init(device)

        # Load grid
        self.grid.init_from_reader(
            pyMOPS.MPASOReader.readGridData(self.yaml_path)
        )

        # Load solution at given time stamp
        self.sol.init_from_reader(
            pyMOPS.MPASOReader.readSolData(
                self.yaml_path, time_stamp, self._time_step
            )
        )

        # Add attributes
        if add_temperature:
            self.sol.add_attribute("temperature", pyMOPS.AttributeFormat.kFloat)
        if add_salinity:
            self.sol.add_attribute("salinity", pyMOPS.AttributeFormat.kFloat)

        pyMOPS.MOPS_Begin()
        pyMOPS.MOPS_AddGridMesh(self.grid)
        pyMOPS.MOPS_AddAttribute(self.sol.getID(), self.sol)
        pyMOPS.MOPS_End()
        pyMOPS.MOPS_ActiveAttribute(self.sol.getID(), None)

        self._active_solution_id = self.sol.getID()
        self._initialized = True
        return self

    # ------------------------------------------------------------------
    # Config builder
    # ------------------------------------------------------------------
    def build_config(
        self,
        width: int = 720,
        height: int = 100,
        lon_range: Tuple[float, float] = (-180.0, 180.0),
        depth_range: Tuple[float, float] = (0.0, 5000.0),
        fixed_latitude: float = 45.0,
        time_step: int = 0,
    ):
        """
        Build VisualizationSettings for regridding at fixed latitude.

        Parameters
        ----------
        width : int
            Image width (longitude bins)
        height : int
            Image height (depth bins)
        lon_range : Tuple[float, float]
            Longitude range in degrees
        depth_range : Tuple[float, float]
            Depth range in meters (positive downward)
        fixed_latitude : float
            Fixed latitude in degrees
        time_step : int
            Time step index
        """
        cfg = pyMOPS.VisualizationSettings()
        cfg.imageSize = (int(width), int(height))
        cfg.LonRange = (float(lon_range[0]), float(lon_range[1]))
        cfg.DepthRange = (float(depth_range[0]), float(depth_range[1]))
        cfg.FixedLatitude = float(fixed_latitude)
        cfg.TimeStep = int(time_step)
        return cfg

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(
        self,
        width: int = 720,
        height: int = 100,
        lon_range: Tuple[float, float] = (-180.0, 180.0),
        depth_range: Tuple[float, float] = (0.0, 5000.0),
        fixed_latitude: float = 45.0,
        time_step: int = 0,
        return_numpy: bool = True,
    ) -> np.ndarray:
        """
        Execute regridding at fixed latitude and return the image.

        Output image shape: (height, width, 4)
        Channels: [E, N, Vertical, Magnitude]

        Returns
        -------
        np.ndarray
            Image array of shape (H, W, 4)
        """
        if not self._initialized:
            raise RuntimeError(
                "MOPSReGrid is not initialized. Call .init(...) first."
            )

        cfg = self.build_config(
            width=width,
            height=height,
            lon_range=lon_range,
            depth_range=depth_range,
            fixed_latitude=fixed_latitude,
            time_step=time_step,
        )

        img = pyMOPS.MOPS_RunReGrid(cfg)

        if return_numpy:
            return np.asarray(img)
        return img

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------
    @staticmethod
    def save_to_binary(image: np.ndarray, filename: str):
        """
        Save regridded image as raw binary file (.bin).

        This matches the binary output format of tutorial/reGrid.cpp.

        Parameters
        ----------
        image : np.ndarray
            Image array of shape (H, W, 4)
        filename : str
            Output filename (e.g., "regrid_fixed_latitude.bin")
        """
        arr = np.asarray(image, dtype=np.float64)
        arr.tofile(filename)
        print(f"Saved raw binary file: {filename} "
              f"({arr.shape[1]} x {arr.shape[0]} x {arr.shape[2]} channels, "
              f"{arr.nbytes} bytes)")

    @staticmethod
    def save_to_png(
        image: np.ndarray,
        filename: str,
        channel: int = 3,
        cmap_name: str = "viridis",
    ):
        """
        Save one channel of the regridded image as a PNG with colormap.

        Parameters
        ----------
        image : np.ndarray
            Image array of shape (H, W, 4)
        filename : str
            Output filename (e.g., "regrid.png")
        channel : int
            Which channel to save (0=E, 1=N, 2=Vertical, 3=Magnitude)
        cmap_name : str
            Matplotlib colormap name
        """
        from PIL import Image
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        arr = np.asarray(image)
        if arr.ndim != 3 or channel >= arr.shape[2]:
            raise ValueError(f"Invalid channel {channel} for image shape {arr.shape}")

        channel_data = arr[:, :, channel]

        # Handle NaNs
        valid_mask = np.isfinite(channel_data)
        if not np.any(valid_mask):
            rgba_u8 = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
        else:
            valid_vals = channel_data[valid_mask]
            min_val = float(np.min(valid_vals))
            max_val = float(np.max(valid_vals))
            if min_val >= max_val:
                max_val = min_val + 1e-5

            norm = np.zeros_like(channel_data, dtype=np.float64)
            norm[valid_mask] = (channel_data[valid_mask] - min_val) / (max_val - min_val)
            norm = np.clip(norm, 0.0, 1.0)

            cmap = mpl.colormaps.get_cmap(cmap_name)
            rgba = cmap(norm)
            rgba_u8 = (rgba * 255.0).astype(np.uint8)

            # NaNs -> transparent black
            rgba_u8[~valid_mask, :] = 0
            rgba_u8[valid_mask, 3] = 255

        Image.fromarray(rgba_u8, mode="RGBA").save(filename)
        print(f"Saved PNG: {filename}")


class MOPSStreamline:
    """
    MOPSStreamline: a Python wrapper for single-timestep streamline tracing.

    Usage:
      1) init(device="gpu")
      2) set_time(start="0015-01-01", duration_seconds=..., or duration_ymd=(Y,M,D))
      3) set_seed(depth, lat_range=(lat_min,lat_max), lon_range=(lon_min,lon_max),
                  grid=(nx,ny), points=None, first_point=None)
      4) run(method="rk4", delta_minutes=1, record_every_minutes=6) -> list[dict]
    """

    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.grid = pyMOPS.MPASOGrid()
        self._seed_conf = None
        self._seed_points = None
        self._first_point = None
        self._follow_last = True
        self._first_round = True
        self._last_pt = None
        self._depth = None
        self._particle_depths = None
        self._one_min = 60

        self._start = None
        self._duration_seconds = None
        self._duration_ymd = None

        self._sol = pyMOPS.MPASOSolution()

    def reset_segments(self):
        """Reset continuation state so next run starts from original seeds."""
        self._first_round = True
        self._last_pt = None
        return self

    # ---------- Utilities ----------

    @staticmethod
    def _to_int_ymd(s: str) -> int:
        """Convert 'YYYY-MM-DD' to integer like 150101."""
        y, m, d = s.split("-")
        return int(y) * 10000 + int(m) * 100 + int(d)

    @staticmethod
    def _ymd_add(start: str, delta_ymd: tuple[int, int, int]) -> str:
        """Add (Y,M,D) offsets to a YYYY-MM-DD string."""
        y, m, d = map(int, start.split("-"))
        dy, dm, dd = delta_ymd
        y += dy
        m += dm
        # normalize month
        while m > 12:
            m -= 12
            y += 1
        while m < 1:
            m += 12
            y -= 1

        # normalize day
        mdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        def is_leap(yy): 
            return (yy % 4 == 0 and yy % 100 != 0) or (yy % 400 == 0)

        if m == 2 and is_leap(y):
            md = 29
        else:
            md = mdays[m - 1]

        d += dd
        while d > md:
            d -= md
            m += 1
            if m > 12:
                m = 1
                y += 1
            if m == 2 and is_leap(y):
                md = 29
            else:
                md = mdays[m - 1]
        while d < 1:
            m -= 1
            if m < 1:
                m = 12
                y -= 1
            if m == 2 and is_leap(y):
                md = 29
            else:
                md = mdays[m - 1]
            d += md
        return f"{y:04d}-{m:02d}-{d:02d}"

    # ---------- Step 1: Initialization ----------

    def init(self, device: str = "gpu"):
        """Initialize MOPS environment and load grid data."""
        pyMOPS.MOPS_Init(device)
        self.grid.init_from_reader(pyMOPS.MPASOReader.readGridData(self.yaml_path))
        return self

    # ---------- Step 2: Set time configuration ----------

    def set_time(self, *args,
                 start: str | tuple[int, int, int] | None = None,
                 duration_seconds: int | None = None,
                 duration_ymd: tuple[int, int, int] | None = None,
                 direction: str = "forward"):
        """
        Define the start date and duration for streamline integration.

        Compatible modes:
          1) Pathline-like: set_time(sy, sm, ey, em, direction="forward")
          2) Original    : set_time(start="YYYY-MM-DD", duration_seconds=...)
                         or set_time(start=(y,m,d), duration_ymd=(Y,M,D))

        Args:
            start: "YYYY-MM-DD" or (year, month, day)
            duration_seconds: integration duration in seconds
            duration_ymd: (years, months, days), optional alternative form
        """

        self.direction = direction.lower()

        # Pathline-like signature: (sy, sm, ey, em)
        if len(args) == 4:
            sy, sm, ey, em = map(int, args)
            self._start = f"{sy:04d}-{sm:02d}-01"
            start_dt = datetime.strptime(self._start, "%Y-%m-%d")
            end_dt = datetime.strptime(f"{ey:04d}-{em:02d}-01", "%Y-%m-%d")
            self._duration_seconds = int(abs((end_dt - start_dt).total_seconds()))
            self._duration_ymd = None
            if self._duration_seconds <= 0:
                raise ValueError("invalid time range: start and end month are identical")
            return self

        # Backward-compatible single positional argument for start
        if len(args) == 1 and start is None:
            start = args[0]
        elif len(args) > 0:
            raise ValueError("set_time accepts either (sy,sm,ey,em) or start=... form")

        if start is None:
            raise ValueError("start must be provided")

        if isinstance(start, tuple):
            self._start = f"{start[0]:04d}-{start[1]:02d}-{start[2]:02d}"
        else:
            self._start = str(start)

        self._duration_seconds = int(duration_seconds) if duration_seconds is not None else None
        self._duration_ymd = tuple(map(int, duration_ymd)) if duration_ymd is not None else None

        if self._duration_seconds is None and self._duration_ymd is None:
            raise ValueError("Provide either duration_seconds or duration_ymd.")
        return self

    # ---------- Step 3: Set seed configuration ----------

    def set_seed(self,
                 depth: float = None,
                 depths: np.ndarray | list | None = None,
                 lat_range: tuple = None,
                 lon_range: tuple = None,
                 grid: tuple = (2, 2),
                 points: np.ndarray | list | None = None,
                 follow_last: bool = True,
                 first_point: list | tuple | np.ndarray | None = None):
        """
        Define particle seed points.
        Two options:
            A) Provide 'points' as (N,3) Cartesian coordinates
            B) Provide 'lat_range', 'lon_range', and 'grid' for auto sampling
        """
        self._follow_last = bool(follow_last)
        self._first_point = np.array(first_point, float) if first_point is not None else None

        if depths is not None:
            self._particle_depths = np.asarray(depths, dtype=np.float32).flatten()
            self._depth = float(self._particle_depths[0])
        elif depth is not None:
            self._depth = float(depth)
            self._particle_depths = None
        else:
            raise ValueError("must provide either 'depth' (scalar) or 'depths' (array)")

        if points is not None:
            arr = np.asarray(points, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError("points must be a (N,3) array")
            self._seed_points = arr.copy()
            self._seed_conf = None
            if self._particle_depths is not None and len(self._particle_depths) != arr.shape[0]:
                raise ValueError(f"depths length ({len(self._particle_depths)}) must match points count ({arr.shape[0]})")
        else:
            if not (lat_range and lon_range):
                raise ValueError("When points is None, lat_range & lon_range must be provided.")
            if self._particle_depths is not None:
                raise ValueError("per-particle depths only supported when providing explicit points")
            nx, ny = grid
            conf = pyMOPS.SeedsSettings()
            conf.setSeedsRange((int(nx), int(ny)))
            conf.setGeoBox(tuple(map(float, lat_range)), tuple(map(float, lon_range)))
            conf.setDepth(self._depth)
            self._seed_conf = conf
            self._seed_points = None
        return self

    # ---------- Step 4: Run streamline tracing ----------

    def run(self, method: str = "rk4", delta_minutes: int = 1, record_every_minutes: int = 6):
        """
        Execute the streamline computation.
        Returns:
            A list[dict], each dict contains 'points', 'velocity', 'temperature', 'salinity', etc.
        """
        if self._start is None:
            raise RuntimeError("Call set_time(...) before run().")
        if self._depth is None:
            raise RuntimeError("Call set_seed(...) before run().")

        # Load single timestep solution
        self._sol.init_from_reader(pyMOPS.MPASOReader.readSolData(self.yaml_path, self._start, 0))
        self._sol.add_attribute("temperature", pyMOPS.AttributeFormat.kFloat)
        self._sol.add_attribute("salinity", pyMOPS.AttributeFormat.kFloat)

        # Register grid and attributes
        pyMOPS.MOPS_Begin()
        pyMOPS.MOPS_AddGridMesh(self.grid)
        pyMOPS.MOPS_AddAttribute(self._sol.getID(), self._sol)
        pyMOPS.MOPS_End()
        pyMOPS.MOPS_ActiveAttribute(self._sol.getID(), None)

        # Prepare seed points
        if (not self._first_round) and self._follow_last and self._last_pt is not None:
            seeds = np.asarray(self._last_pt, dtype=np.float64, order="C")
        elif self._seed_points is not None:
            seeds = np.asarray(self._seed_points, dtype=np.float64, order="C")
        else:
            seeds = pyMOPS.MOPS_GenerateSeedsPoints(self._seed_conf)
        if self._first_point is not None and seeds.shape[0] > 0:
            seeds[0] = self._first_point

        # Keep per-particle depths consistent with continuation seeds.
        if self._particle_depths is not None and (not self._first_round) and self._follow_last:
            radii = np.linalg.norm(seeds, axis=1)
            next_depths = EARTH_RADIUS_M - radii
            next_depths = np.clip(next_depths, 0.0, None)
            self._particle_depths = next_depths.astype(np.float32, copy=False)

        # Build trajectory configuration
        method_flag = pyMOPS.CalcMethodType.kRK4 if method.lower() == "rk4" else pyMOPS.CalcMethodType.kEuler

        cfg = pyMOPS.TrajectorySettings()
        cfg.depth = self._depth
        cfg.deltaT = int(delta_minutes) * self._one_min
        cfg.recordT = int(record_every_minutes) * self._one_min
        cfg.methodType = method_flag
        if self._particle_depths is not None:
            cfg.particle_depths = self._particle_depths.tolist()
        if self.direction == "forward":
            cfg.directionType = pyMOPS.CalcDirection.kForward
        else:
            cfg.directionType = pyMOPS.CalcDirection.kBackward

        # Duration
        if self._duration_seconds is not None:
            cfg.simulationDuration = int(self._duration_seconds)
        else:
            end = self._ymd_add(self._start, self._duration_ymd)
            start_dt = datetime.strptime(self._start, "%Y-%m-%d")
            end_dt = datetime.strptime(end, "%Y-%m-%d")
            cfg.simulationDuration = int(abs((end_dt - start_dt).total_seconds()))
            if cfg.simulationDuration <= 0:
                raise ValueError("duration_ymd results in zero duration")

        # Execute the streamline integration
        seg = pyMOPS.MOPS_RunStreamLine(cfg, seeds)

        # Update continuation state using the last point of each line.
        if len(seg) > 0:
            self._last_pt = np.stack([np.asarray(seg[i]["points"])[-1] for i in range(len(seg))]).astype(float, copy=False)
            if self._particle_depths is not None:
                radii = np.linalg.norm(self._last_pt, axis=1)
                next_depths = EARTH_RADIUS_M - radii
                next_depths = np.clip(next_depths, 0.0, None)
                self._particle_depths = next_depths.astype(np.float32, copy=False)
        self._first_round = False

        return seg


class MOPSPathline:
    """
      1) init(device="gpu")
      2) set_time(sy, sm, ey, em, direction="forward")
      3) set_seed(depth, lat_range=(lat_min,lat_max), lon_range=(lon_min,lon_max),
                  grid=(nx,ny), points=None, follow_last=True)
      4) run(method="rk4", delta_minutes=1, record_every_minutes=6) -> list[dict]
    
    Important stateful behavior:
      - This class is STATEFUL across consecutive run() calls and across month-pairs.
      - `_first_round` indicates whether we are at the very first month-pair segment.
      - `_last_pt` stores the last position of each particle from the previous segment.
      - If follow_last=True, subsequent segments will start from `_last_pt` (continuation).
      - When you change "depth groups", you usually want to reset `_first_round/_last_pt`
        so that each depth group starts from its own initial seeds (not from previous depth).
    """
    def __init__(self, yaml_path: str):
        # Path to the dataset yaml (grid + solution time series)
        self.yaml_path = yaml_path
        # Grid data (static mesh)
        self.grid = pyMOPS.MPASOGrid()
        # List of (start_month, end_month) time pairs to trace through
        self.pairs = None
        # "forward" or "backward"
        self.direction = "forward"
        self._seed_conf = None
        self._seed_points = None
        # If True, use last points of previous segment as the seeds for the next segment.
        # This is intended for continuing the SAME set of particles over multiple month pairs.
        self._follow_last = True
        self._first_round = True
        # The (scalar) depth used for trajectory configuration (meters, positive downward)
        self._depth = None
        self._particle_depths = None  # NEW: (N,) array for per-particle depths
        self._last_pt = None  # (N,3) np.ndarray, last point of each particle
        self._one_min = 60
        # Two solution buffers for time interpolation (A: current, B: next)
        self._solA = pyMOPS.MPASOSolution()
        self._solB = pyMOPS.MPASOSolution()

    def reset_segments(self):
        """
        Reset "segment continuation" state.

        Call this when you want a new independent run sequence,
        e.g., when you switch to a different depth group.

        After calling this:
          - The next run() will treat the next month-pair as the first segment.
          - Seeds will come from the user-provided points / generated seeds again.
          - `_last_pt` will be cleared, so follow_last continuation will not happen
            across different depth groups.
        """
        self._first_round = True
        self._last_pt = None
        
    @staticmethod
    def _month_pairs_forward(sy, sm, ey, em):
        """
        Generate monthly pairs for forward tracing:
        [('YYYY-MM-01', 'YYYY-(MM+1)-01'), ...] until end month.
        
        [('00sy-0sm-01','00sy-0sm+1-01'), ... , ('00ey-em-01','00ey-em+1-01')]
        example: month_pairs_forward(18,1,20,12)
        [('0018-01-01', '0018-02-01'), ..., ('0020-11-01', '0020-12-01')]
        """
        y, m = sy, sm
        out = []
        while (y < ey) or (y == ey and m <= em):
            by, bm = y, m+1
            if bm == 13:
                by, bm = y+1, 1
            if (by > ey) or (by == ey and bm > em):
                break
            a = f"{y:04d}-{m:02d}-01"
            b = f"{by:04d}-{bm:02d}-01"
            out.append((a, b))
            y, m = by, bm
        return out
    @staticmethod
    def _month_pairs_backward(sy, sm, ey, em):
        """
        [('00sy-0sm-01','00sy-0sm-1-01'), ... , ('00ey-em-01','00ey-em-1-01')]
        example: month_pairs_backward(20,12,18,1)
        [('0020-12-01', '0020-11-01'), ..., ('0018-02-01', '0018-01-01')]
        """
        y, m = sy, sm
        out = []
        while (y > ey) or (y == ey and m >= em):
            by, bm = y, m-1
            if bm == 0:
                by, bm = y-1, 1
                
                bm = 12
            if (by < ey) or (by == ey and bm < em):
                break
            a = f"{y:04d}-{m:02d}-01"
            b = f"{by:04d}-{bm:02d}-01"
            out.append((a, b))
            y, m = by, bm
        return out
    @staticmethod
    def _to_int_ymd(s):  # "0018-01-01" -> 180101
        y, mo, d = s.split("-")
        return int(y)*10000 + int(mo)*100 + int(d)
    @staticmethod
    def _time_gap_seconds(t1: str, t2: str, fmt: str = "%Y-%m-%d_%H:%M:%S") -> int:
        """
        Compute |t1 - t2| in seconds, where t1/t2 are timestamps embedded in solution objects.
        Some strings may contain trailing '\x00', so we strip them.
        """
        t1 = t1.split("\x00", 1)[0].strip()
        t2 = t2.split("\x00", 1)[0].strip()

        dt1 = datetime.strptime(t1, fmt)
        dt2 = datetime.strptime(t2, fmt)
        return int((dt1 - dt2).total_seconds())

    # -------------------------------------------------------------------------
    # (1) Initialize MOPS runtime and load grid
    # -------------------------------------------------------------------------
    def init(self, device: str = "gpu"):
        """Initialize MOPS and load the static grid mesh."""
        pyMOPS.MOPS_Init(device)
        # loading grid
        self.grid.init_from_reader(pyMOPS.MPASOReader.readGridData(self.yaml_path))
        return self

    # -------------------------------------------------------------------------
    # (2) Configure time range for pathline tracing
    # -------------------------------------------------------------------------
    def set_time(self, sy: int, sm: int, ey: int, em: int, direction: str = "forward"):
        """
        Set the tracing time range (month-based) and direction.

        direction:
          - "forward": month_pairs_forward
          - "backward": month_pairs_backward
        """
        self.direction = direction.lower()
        if self.direction == "forward":
            self.pairs = self._month_pairs_forward(sy, sm, ey, em)
        elif self.direction == "backward":
            self.pairs = self._month_pairs_backward(sy, sm, ey, em)
        else:
            raise ValueError("direction must be 'forward' or 'backward'")
        if not self.pairs:
            raise ValueError("no month pairs produced; check input range")
        return self

    # -------------------------------------------------------------------------
    # (3) Configure seeds: either explicit points or lat/lon range grid
    # -------------------------------------------------------------------------
    def set_seed(
        self,
        depth: float = None,
        depths: np.ndarray | list | None = None,  # NEW: per-particle depths (N,)
        lat_range: tuple = None,
        lon_range: tuple = None,
        grid: tuple = (2, 2),
        points: np.ndarray | list | None = None,   # directly provide (N,3) Cartesian coords
        follow_last: bool = True
    ):
        """
        Two usages:
        A) Give lat/lon + grid (internally use MOPS to generate sampling) - uses uniform depth
        B) Directly give the Cartesian coordinates of points (N,3) + optional per-particle depths
        
        Parameters:
          depth: scalar depth for uniform mode (meters, positive downward)
          depths: (N,) array of per-particle depths (meters, positive downward)
                  If provided, each particle uses its own depth for velocity interpolation.
          points: (N,3) Cartesian coordinates of seed points
          follow_last: If True, continue from last positions of previous segment
          
        Note: If both 'depth' and 'depths' are provided, 'depths' takes precedence.
        """
        self._follow_last = bool(follow_last)

        # Handle depth(s)
        if depths is not None:
            self._particle_depths = np.asarray(depths, dtype=np.float32).flatten()
            self._depth = float(self._particle_depths[0])  # fallback
        elif depth is not None:
            self._depth = float(depth)
            self._particle_depths = None
        else:
            raise ValueError("must provide either 'depth' (scalar) or 'depths' (array)")

        if points is not None:
            arr = np.asarray(points, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError("points must be a (N,3) array")
            self._seed_points = arr.copy()
            self._seed_conf = None
            
            # Validate depths array length if provided
            if self._particle_depths is not None:
                if len(self._particle_depths) != arr.shape[0]:
                    raise ValueError(f"depths length ({len(self._particle_depths)}) must match points count ({arr.shape[0]})")
        else:
            if not (lat_range and lon_range):
                raise ValueError("when points is None, must provide lat_range & lon_range")
            if self._particle_depths is not None:
                raise ValueError("per-particle depths only supported when providing explicit points")
            nx, ny = grid
            conf = pyMOPS.SeedsSettings()
            conf.setSeedsRange((int(nx), int(ny)))
            conf.setGeoBox(tuple(map(float, lat_range)), tuple(map(float, lon_range)))
            conf.setDepth(self._depth)
            self._seed_conf = conf
            self._seed_points = None
        return self

    # -------------------------------------------------------------------------
    # (4) Run pathline tracing through all month pairs and concatenate results
    # -------------------------------------------------------------------------
    def run(self, method: str = "rk4", delta_minutes: int = 1, record_every_minutes: int = 6):
        """
        Execute pathline integration across all configured month pairs, and concatenate
        per-segment results into a single long trajectory per particle.

        Returns:
          lines_acc: list[dict], where each dict corresponds to one particle line and contains:
            - "lineID" (int)
            - "points" (M,3)
            - "velocity" (M,3)
            - "temperature" (M,)
            - "salinity" (M,)
            - "lastPoint" (3,)
        """
        if self.pairs is None:
            raise RuntimeError("call set_time(...) before run()")
        if self._depth is None:
            raise RuntimeError("call set_seed(...) before run()")

        method_flag = pyMOPS.CalcMethodType.kRK4 if method.lower() == "rk4" else pyMOPS.CalcMethodType.kEuler
        dir_flag = pyMOPS.CalcDirection.kForward if self.direction == "forward" else pyMOPS.CalcDirection.kBackward

        lines_acc = None

        for start, end in self.pairs:
            # loading attributes
            if self._first_round:
                self._solA.init_from_reader(pyMOPS.MPASOReader.readSolData(self.yaml_path, start, 0))
                self._solB.init_from_reader(pyMOPS.MPASOReader.readSolData(self.yaml_path, end,   0))
            else:
                # swap
                self._solA, self._solB = self._solB, self._solA
                self._solB.init_from_reader(pyMOPS.MPASOReader.readSolData(self.yaml_path, end, 0))

            # loading temperature and salinity attributes
            for s in (self._solA, self._solB):
                s.add_attribute("temperature", pyMOPS.AttributeFormat.kFloat)
                s.add_attribute("salinity",    pyMOPS.AttributeFormat.kFloat)

            # registration
            pyMOPS.MOPS_Begin()
            pyMOPS.MOPS_AddGridMesh(self.grid)
            pyMOPS.MOPS_AddAttribute(self._solA.getID(), self._solA)
            pyMOPS.MOPS_AddAttribute(self._solB.getID(), self._solB)
            pyMOPS.MOPS_End()
            pyMOPS.MOPS_ActiveAttribute(self._solA.getID(), self._solB.getID())

            # Calculate time gap
            gap_sec = abs(self._time_gap_seconds(self._solB.getTimeStamp(), self._solA.getTimeStamp()))

            # Prepare seeds
            if self._first_round:
                if self._seed_points is not None:
                    seeds = self._seed_points.copy()
                else:
                    seeds = pyMOPS.MOPS_GenerateSeedsPoints(self._seed_conf)
            else:
                if self._follow_last:
                    if self._last_pt is None:
                        raise RuntimeError("follow_last=True but last_pts is None")
                    seeds = self._last_pt.copy()
                else:
                    if self._seed_points is not None:
                        seeds = self._seed_points.copy()
                    else:
                        seeds = pyMOPS.MOPS_GenerateSeedsPoints(self._seed_conf)

            # Keep per-particle depths consistent with continuation seeds.
            # This mirrors the C++ tutorial fix that recomputes depths from last valid XYZ points.
            if self._particle_depths is not None and self._follow_last and not self._first_round:
                radii = np.linalg.norm(seeds, axis=1)
                next_depths = EARTH_RADIUS_M - radii
                next_depths = np.clip(next_depths, 0.0, None)
                self._particle_depths = next_depths.astype(np.float32, copy=False)
                        
            # trajectory config
            cfg = pyMOPS.TrajectorySettings()
            cfg.depth        = self._depth  # fallback depth
            cfg.deltaT       = int(delta_minutes) * self._one_min
            cfg.simulationDuration = gap_sec
            cfg.recordT      = int(record_every_minutes) * self._one_min
            cfg.directionType= dir_flag
            cfg.methodType   = method_flag
            
            # Set per-particle depths if available
            if hasattr(self, '_particle_depths') and self._particle_depths is not None:
                cfg.particle_depths = self._particle_depths.tolist()

            # Run a segment
            seg = pyMOPS.MOPS_RunPathLine(cfg, seeds)

            # Update last_pt
            self._last_pt = np.stack([seg[i]["lastPoint"] for i in range(len(seg))]).astype(float, copy=False)

            # For per-particle depth mode, propagate evolved depths to next segment.
            if self._particle_depths is not None:
                radii = np.linalg.norm(self._last_pt, axis=1)
                next_depths = EARTH_RADIUS_M - radii
                next_depths = np.clip(next_depths, 0.0, None)
                self._particle_depths = next_depths.astype(np.float32, copy=False)

            if lines_acc is None:
                lines_acc = seg
                n = len(lines_acc)
                P_lists = [[np.asarray(lines_acc[i]["points"])] for i in range(n)]
                V_lists = [[np.asarray(lines_acc[i]["velocity"])] for i in range(n)]
                T_lists = [[np.asarray(lines_acc[i]["temperature"])] for i in range(n)]
                S_lists = [[np.asarray(lines_acc[i]["salinity"])] for i in range(n)]
            else:
                # Append new segment data into lists (skip first sample to avoid duplicates)
                for i in range(len(lines_acc)):
                    P_seg = np.asarray(seg[i]["points"])
                    V_seg = np.asarray(seg[i]["velocity"])
                    T_seg = np.asarray(seg[i]["temperature"])
                    S_seg = np.asarray(seg[i]["salinity"])

                    if P_seg.shape[0] > 1:
                        P_lists[i].append(P_seg[1:])
                        V_lists[i].append(V_seg[1:])
                        T_lists[i].append(T_seg[1:])
                        S_lists[i].append(S_seg[1:])

                    lines_acc[i]["lastPoint"] = seg[i]["lastPoint"]

            self._first_round = False

        # Finalize: concatenate lists into big arrays once
        if lines_acc is not None:
            for i in range(len(lines_acc)):
                lines_acc[i]["points"]      = np.vstack(P_lists[i]) if len(P_lists[i]) else np.empty((0, 3), float)
                lines_acc[i]["velocity"]    = np.vstack(V_lists[i]) if len(V_lists[i]) else np.empty((0, 3), float)
                lines_acc[i]["temperature"] = np.concatenate(T_lists[i]) if len(T_lists[i]) else np.empty((0,), float)
                lines_acc[i]["salinity"]    = np.concatenate(S_lists[i]) if len(S_lists[i]) else np.empty((0,), float)


        return lines_acc


