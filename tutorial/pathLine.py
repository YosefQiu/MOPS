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
    
yaml_path   = "/pscratch/sd/q/qiuyf/MOPS_Tutorial/test.yaml"
mpasoGrid   = pyMOPS.MPASOGrid()
solFront    = pyMOPS.MPASOSolution()
solBack     = pyMOPS.MPASOSolution()
last_pt     = None
ONE_MIN     = 60

def xyz_to_latlon(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.where(r < 1e-8, np.nan, r)
    lat = np.arcsin(z / r) * 180 / np.pi
    lon = np.arctan2(y, x) * 180 / np.pi
    return lat, lon

def _build_segments(lons, lats, values=None):
    """
    把一条轨迹拆成 line segments；剔除 NaN 和 180° 经线跨越造成的长跳变。
    如果传入 values (N,)，会返回每段的标量 (N-1,) （两端点取平均）。
    """
    lons = np.asarray(lons); lats = np.asarray(lats)
    ok = ~np.isnan(lons) & ~np.isnan(lats)
    lons = lons[ok]; lats = lats[ok]
    if len(lons) < 2:
        return np.empty((0, 2, 2)), np.empty((0,)) if values is not None else None

    # 处理经线差值（考虑环绕）：把差映射到 [-180, 180]
    dlon = ((lons[1:] - lons[:-1] + 180.0) % 360.0) - 180.0
    good = np.abs(dlon) < 170.0  # 阈值可调，避免跨国际日期变更线产生的“拉一条全球长线”

    segs = np.stack([
        np.column_stack([lons[:-1], lats[:-1]])[good],
        np.column_stack([lons[1:],  lats[1:]])[good]
    ], axis=1)

    if values is None:
        return segs, None
    values = np.asarray(values)
    values = values[ok]
    if len(values) != len(lons):
        # 长度不一致，直接返回 None（上层会处理）
        return segs, None
    vals_seg = 0.5 * (values[:-1] + values[1:])
    vals_seg = vals_seg[good]
    return segs, vals_seg

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
      - 新格式：来自 MOPS_RunPathLine 的 list[dict]，
                 dict 至少包含 'points'(N,3)，可选 'velocity'(N,3), 'temperature'(N,), 'salinity'(N,)
      - 旧格式：list[np.ndarray]，每个 (N,6) [x,y,z,vx,vy,vz]
    """

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
    lcs = []  # 收集所有 LineCollection 以统一归一化
    scal_min, scal_max = np.inf, -np.inf

    for line in trajectory_lines:
        # 兼容两种输入
        if isinstance(line, dict):
            P = np.asarray(line["points"])            # (N,3)
            V = np.asarray(line.get("velocity", []))  # (N,3) 或空
            T = np.asarray(line.get("temperature", []))  # (N,) 或空
            S = np.asarray(line.get("salinity", []))     # (N,) 或空
        else:
            arr = np.asarray(line)  # (N,6) = [x,y,z,vx,vy,vz]
            if arr.ndim != 2 or arr.shape[1] not in (3, 6):
                continue
            P = arr[:, :3]
            V = arr[:, 3:6] if arr.shape[1] == 6 else np.empty((0, 3))
            T = np.empty((0,))
            S = np.empty((0,))

        if P.shape[0] < 2:
            continue

        lat, lon = xyz_to_latlon(P[:, 0], P[:, 1], P[:, 2])
        all_lats.extend(lat[np.isfinite(lat)])
        all_lons.extend(lon[np.isfinite(lon)])

        # 选择着色数据
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
            # 未知关键字 → 当作 None
            values = None

        segs, vals_seg = _build_segments(lon, lat, values)

        if segs.shape[0] == 0:
            continue

        if vals_seg is None:
            # 无标量时，画白线
            lc = LineCollection(segs, linewidths=linewidth, colors="white",
                                transform=ccrs.PlateCarree())
        else:
            lc = LineCollection(segs, linewidths=linewidth, cmap=cmap,
                                array=vals_seg, transform=ccrs.PlateCarree())
            # 更新全局标量范围
            if vals_seg.size:
                scal_min = min(scal_min, float(np.nanmin(vals_seg)))
                scal_max = max(scal_max, float(np.nanmax(vals_seg)))

        ax.add_collection(lc)
        lcs.append(lc)

    # 自动设置范围
    if region_extent is None and len(all_lats) > 0:
        margin = 2.0
        ax.set_extent([
            float(np.nanmin(all_lons)) - margin, float(np.nanmax(all_lons)) + margin,
            float(np.nanmin(all_lats)) - margin, float(np.nanmax(all_lats)) + margin
        ], crs=ccrs.PlateCarree())

    # 统一归一化 & colorbar
    if color_by is not None and len(lcs) > 0 and np.isfinite([scal_min, scal_max]).all():
        if vmin is None: vmin = scal_min
        if vmax is None: vmax = scal_max
        if vmin == vmax:  # 防止常数场导致归一化异常
            vmin -= 1e-12
            vmax += 1e-12

        norm = Normalize(vmin=vmin, vmax=vmax)
        for lc in lcs:
            # 只有带 array 的集合才需要设置
            if lc.get_array() is not None:
                lc.set_norm(norm)
                lc.set_cmap(cmap)

        if show_colorbar:
            # 选一个带 array 的集合作为 colorbar 句柄
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


def month_pairs_forward(sy, sm, ey, em):
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

def month_pairs_backward(sy, sm, ey, em):
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

def to_int_ymd(s):  # "0018-01-01" -> 180101
    y, mo, d = s.split("-")
    return int(y)*10000 + int(mo)*100 + int(d)

def time_gap_seconds(t1: str, t2: str, fmt: str = "%Y-%m-%d_%H:%M:%S") -> int:
    t1 = t1.split("\x00", 1)[0].strip()
    t2 = t2.split("\x00", 1)[0].strip()

    dt1 = datetime.strptime(t1, fmt)
    dt2 = datetime.strptime(t2, fmt)
    return int((dt1 - dt2).total_seconds())

def generate_sample_points(depth: float, override_first=True):
    global last_pt
    conf = pyMOPS.SeedsSettings()
    conf.setSeedsRange((2, 2))
    conf.setGeoBox((35.0, 45.0), (-90.0, -65.0))
    conf.setDepth(depth)
    seeds = pyMOPS.MOPS_GenerateSeedsPoints(conf)

    if override_first:
        seeds[0] = [1908930.101867, -5174124.236251, 3189701.032088]
    else:
        if last_pt is None:
            raise ValueError("last_pt is None while override_first=False")
        seeds[0] = last_pt

    return seeds, conf

def runPathLine(depth: float, isFirst: bool, timeInterval: int, timestep_vec: list):
    global last_pt  
    seeds, conf = generate_sample_points(depth, isFirst)

    traj_conf = pyMOPS.TrajectorySettings()
    traj_conf.depth = conf.getDepth()
    traj_conf.deltaT = ONE_MIN
    traj_conf.simulationDuration = timeInterval
    traj_conf.recordT = ONE_MIN * 6

    lines = pyMOPS.MOPS_RunPathLine(traj_conf, seeds, timestep_vec)
    last_pt = np.asarray(lines[0]["lastPoint"], dtype=float)  # (3,)
    return lines




pyMOPS.MOPS_Init("gpu")

pairs = month_pairs_forward(18,1,20,12)
print("length:", len(pairs))

mpasoGrid.init_from_reader(pyMOPS.MPASOReader.readGridData(yaml_path))

bFirst = True
depth = 10.0
lines_acc = None
for idx, (start, end) in enumerate(pairs, 1):
    print(f"[{idx}/{len(pairs)}] {start} -> {end}")
    
    if bFirst:
        solFront.init_from_reader(pyMOPS.MPASOReader.readSolData(yaml_path, start, 0))
        solBack.init_from_reader(pyMOPS.MPASOReader.readSolData(yaml_path, end, 0))
    else:
        solFront, solBack = solBack, solFront
        solBack.init_from_reader(pyMOPS.MPASOReader.readSolData(yaml_path, end, 0))
    
    for s in (solFront, solBack):
        s.add_attribute("temperature", pyMOPS.AttributeFormat.kFloat)
        s.add_attribute("salinity", pyMOPS.AttributeFormat.kFloat)

    print("  solFront time:", solFront.getCurrentTime())
    print("  solBack time:", solBack.getCurrentTime())
    print("  Time gap (seconds):", time_gap_seconds(solBack.getCurrentTime(), solFront.getCurrentTime()))
    print()
    
    pyMOPS.MOPS_Begin()
    pyMOPS.MOPS_AddGridMesh(mpasoGrid)
    pyMOPS.MOPS_AddAttribute(to_int_ymd(start), solFront)
    pyMOPS.MOPS_AddAttribute(to_int_ymd(end), solBack)
    pyMOPS.MOPS_End()
    
    pyMOPS.MOPS_ActiveAttribute(to_int_ymd(start), to_int_ymd(end))
    
    timestep_vec = [to_int_ymd(start), to_int_ymd(end)]
    
    lines_seg = runPathLine(depth, bFirst, abs(time_gap_seconds(solBack.getCurrentTime(), solFront.getCurrentTime())), timestep_vec)
    print("  Pathline length:", len(lines_seg[0]["points"]))
    bFirst = False
    
    if lines_acc is None:
        lines_acc = lines_seg
    else:
        for i in range(len(lines_acc)):
            P_acc = lines_acc[i]["points"]
            V_acc = lines_acc[i]["velocity"]
            T_acc = lines_acc[i]["temperature"]
            S_acc = lines_acc[i]["salinity"]

            P_seg = lines_seg[i]["points"]
            V_seg = lines_seg[i]["velocity"]
            T_seg = lines_seg[i]["temperature"]
            S_seg = lines_seg[i]["salinity"]
            if len(P_seg) > 1:
                lines_acc[i]["points"]      = np.vstack([P_acc, P_seg[1:]])
                lines_acc[i]["velocity"]    = np.vstack([V_acc, V_seg[1:]])
                lines_acc[i]["temperature"] = np.concatenate([T_acc, T_seg[1:]])
                lines_acc[i]["salinity"]    = np.concatenate([S_acc, S_seg[1:]])

            lines_acc[i]["lastPoint"] = lines_seg[i]["lastPoint"]


Vis_PathLines(lines_acc, save_path="white.png")
Vis_PathLines(lines_acc, color_by="speed", cmap="turbo", linewidth=1.2, save_path="speed.png")
Vis_PathLines(lines_acc, color_by="temperature", cmap="inferno", vmin=0, vmax=30, save_path="temp.png")
Vis_PathLines(lines_acc, color_by="salinity", region_extent=[-90,-65,35,45], save_path="salinity.png")