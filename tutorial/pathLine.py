import sys
sys.path.append("../tools/pyMOPS/pyMOPS/")
import pyMOPS
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

print("Available API:", dir(pyMOPS))
print("\n\n")
    




def xyz_to_latlon(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.where(r < 1e-8, np.nan, r)
    lat = np.arcsin(z / r) * 180 / np.pi
    lon = np.arctan2(y, x) * 180 / np.pi
    return lat, lon


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
            V = np.asarray(line.get("velocity", []))  # (N,3) 或空
            T = np.asarray(line.get("temperature", []))  # (N,) 或空
            S = np.asarray(line.get("salinity", []))     # (N,) 或空

        if P.shape[0] < 2:
            continue

        lat, lon = xyz_to_latlon(P[:, 0], P[:, 1], P[:, 2])
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

        if vals_seg is None:
            # When there is no scalar, draw a white line
            lc = LineCollection(segs, linewidths=linewidth, colors="white",
                                transform=ccrs.PlateCarree())
        else:
            lc = LineCollection(segs, linewidths=linewidth, cmap=cmap,
                                array=vals_seg, transform=ccrs.PlateCarree())
            # Update the global scalar range
            if vals_seg.size:
                scal_min = min(scal_min, float(np.nanmin(vals_seg)))
                scal_max = max(scal_max, float(np.nanmax(vals_seg)))

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

class MOPSPathline:
    """
      1) init(device="gpu")
      2) set_time(sy, sm, ey, em, direction="forward")
      3) set_seed(depth, lat_range=(lat_min,lat_max), lon_range=(lon_min,lon_max),
                  grid=(nx,ny), points=None, first_point=None, follow_last=True)
      4) run(method="rk4", delta_minutes=1, record_every_minutes=6) -> list[dict]
    """
    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.grid = pyMOPS.MPASOGrid()
        self.pairs = None
        self.direction = "forward"
        self._seed_conf = None
        self._seed_points = None
        self._first_point = None
        self._follow_last = True
        self._depth = None
        self._last_pt = None  # (3,) np.ndarray
        self._one_min = 60

        # solutions（复用，减少反复分配）
        self._solA = pyMOPS.MPASOSolution()
        self._solB = pyMOPS.MPASOSolution()

    @staticmethod
    def _month_pairs_forward(sy, sm, ey, em):
        """
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
        t1 = t1.split("\x00", 1)[0].strip()
        t2 = t2.split("\x00", 1)[0].strip()

        dt1 = datetime.strptime(t1, fmt)
        dt2 = datetime.strptime(t2, fmt)
        return int((dt1 - dt2).total_seconds())

    # ① init
    def init(self, device: str = "gpu"):
        pyMOPS.MOPS_Init(device)
        # loading grid
        self.grid.init_from_reader(pyMOPS.MPASOReader.readGridData(self.yaml_path))
        return self

    # ② set time range
    def set_time(self, sy: int, sm: int, ey: int, em: int, direction: str = "forward"):
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

    # ③ set seeds
    def set_seed(
        self,
        depth: float,
        lat_range: tuple = None,
        lon_range: tuple = None,
        grid: tuple = (2, 2),
        points: np.ndarray | list | None = None,   # directly provide (N,3) Cartesian coords
        first_point: list | tuple | np.ndarray | None = None,
        follow_last: bool = True
    ):
        """
        Two usages:
        A) Give lat/lon + grid (internally use MOPS to generate sampling)
        B) Directly give the Cartesian coordinates of points (N,3)
        first_point: If given, will override the first seed (e.g. single point tracking)
        follow_last: If True, the first seed will be replaced by the last point of the previous segment (default True)
        """
        self._depth = float(depth)
        self._first_point = np.array(first_point, float) if first_point is not None else None
        self._follow_last = bool(follow_last)

        if points is not None:
            arr = np.asarray(points, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError("points must be a (N,3) array")
            self._seed_points = arr.copy()
            self._seed_conf = None
        else:
            if not (lat_range and lon_range):
                raise ValueError("when points is None, must provide lat_range & lon_range")
            nx, ny = grid
            conf = pyMOPS.SeedsSettings()
            conf.setSeedsRange((int(nx), int(ny)))
            conf.setGeoBox(tuple(map(float, lat_range)), tuple(map(float, lon_range)))
            conf.setDepth(self._depth)
            self._seed_conf = conf
            self._seed_points = None
        return self

    # ④ run
    def run(self, method: str = "rk4", delta_minutes: int = 1, record_every_minutes: int = 6):
        if self.pairs is None:
            raise RuntimeError("call set_time(...) before run()")
        if self._depth is None:
            raise RuntimeError("call set_seed(...) before run()")

        method_flag = pyMOPS.CalcMethodType.kRK4 if method.lower() == "rk4" else pyMOPS.CalcMethodType.kEuler
        dir_flag = pyMOPS.CalcDirection.kForward if self.direction == "forward" else pyMOPS.CalcDirection.kBackward

        lines_acc = None
        first = True

        for start, end in self.pairs:
            # loading attributes
            if first:
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
            pyMOPS.MOPS_AddAttribute(self._to_int_ymd(start), self._solA)
            pyMOPS.MOPS_AddAttribute(self._to_int_ymd(end),   self._solB)
            pyMOPS.MOPS_End()
            pyMOPS.MOPS_ActiveAttribute(self._to_int_ymd(start), self._to_int_ymd(end))

            # Calculate time gap
            gap_sec = abs(self._time_gap_seconds(self._solB.getCurrentTime(), self._solA.getCurrentTime()))

            # Prepare seeds
            if self._seed_points is not None:
                seeds = self._seed_points.copy()
            else:
                seeds = pyMOPS.MOPS_GenerateSeedsPoints(self._seed_conf)

            # Override first point if needed
            if first and self._first_point is not None:
                seeds[0] = self._first_point
            elif (not first) and self._follow_last and (self._last_pt is not None):
                seeds[0] = self._last_pt

            # trajectory config
            cfg = pyMOPS.TrajectorySettings()
            cfg.depth        = self._depth
            cfg.deltaT       = int(delta_minutes) * self._one_min
            cfg.simulationDuration = gap_sec
            cfg.recordT      = int(record_every_minutes) * self._one_min
            cfg.directionType= dir_flag
            cfg.methodType   = method_flag

            # Run a segment
            seg = pyMOPS.MOPS_RunPathLine(cfg, seeds, [self._to_int_ymd(start), self._to_int_ymd(end)])

            # Update last_pt
            self._last_pt = np.asarray(seg[0]["lastPoint"], float)

            if lines_acc is None:
                lines_acc = seg
            else:
                for i in range(len(lines_acc)):
                    P_acc, V_acc = lines_acc[i]["points"],      lines_acc[i]["velocity"]
                    T_acc, S_acc = lines_acc[i]["temperature"], lines_acc[i]["salinity"]
                    P_seg, V_seg = seg[i]["points"],            seg[i]["velocity"]
                    T_seg, S_seg = seg[i]["temperature"],       seg[i]["salinity"]

                    if len(P_seg) > 1:
                        lines_acc[i]["points"]      = np.vstack([P_acc, P_seg[1:]])
                        lines_acc[i]["velocity"]    = np.vstack([V_acc, V_seg[1:]])
                        lines_acc[i]["temperature"] = np.concatenate([T_acc, T_seg[1:]])
                        lines_acc[i]["salinity"]    = np.concatenate([S_acc, S_seg[1:]])

                    lines_acc[i]["lastPoint"] = seg[i]["lastPoint"]

            first = False

        return lines_acc


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

    Vis_PathLines(lines, save_path="white.png")
    Vis_PathLines(lines, color_by="speed", cmap="turbo", linewidth=1.2, save_path="speed.png")
    