#!/usr/bin/env python3
"""Task templates for generated MOPS jobs.

This module centralizes template rendering for:
- remapping
- streamline
- pathline
"""

import json
import re
import textwrap
from pathlib import Path
from string import Template

try:
    import yaml
except ImportError:
    yaml = None


DEFAULT_REMAPPING_CONFIG = {
    "yaml_path": "test_ab_climatology.yaml",
    "device": "gpu",
    "time_stamp": "0001-01-01",
    "time_step": 0,
    "width": 3601,
    "height": 1801,
    "lat_range": [-90.0, 90.0],
    "lon_range": [-180.0, 180.0],
    "fixed_depth": 10.0,
    "add_temperature": True,
    "add_salinity": True,
    "channels": [0, 1, 2, 3],
    "cmap_name": "coolwarm",
    "save_colorbar": True,
    "output_subdir": "Agent/outputs/remapping",
}


DEFAULT_STREAMLINE_CONFIG = {
    "yaml_path": "test_ab_climatology.yaml",
    "device": "gpu",
    "start_date": "0001-01-01",
    "duration_days": 5,
    "fixed_depth": 10.0,
    "lat_range": [35.0, 45.0],
    "lon_range": [-90.0, -65.0],
    "grid": [8, 8],
    "method": "rk4",
    "delta_minutes": 1,
    "record_every_minutes": 6,
    "color_by": "speed",
    "output_subdir": "Agent/outputs/streamline",
}


DEFAULT_PATHLINE_CONFIG = {
    "yaml_path": "test_ab_climatology.yaml",
    "device": "gpu",
    "start_year": 1,
    "start_month": 1,
    "end_year": 1,
    "end_month": 12,
    "direction": "forward",
    "fixed_depth": 10.0,
    "lat_range": [35.0, 45.0],
    "lon_range": [-90.0, -65.0],
    "grid": [8, 8],
    "method": "rk4",
    "delta_minutes": 60,        # 1 hour (changed from 1 min for longer simulations)
    "record_every_minutes": 360,  # 6 hours (changed from 6 min for longer simulations)
    "color_by": "velocity",
    "output_subdir": "Agent/outputs/pathline",
}


REMAPPING_YAML_TEMPLATE_TEXT = '''stream:
    name: mpas
    path_prefix: "/pscratch/sd/q/qiuyf/dataset/climatology"
    substreams:
        - name: mesh
          format: netcdf
          filenames: "ocean.EC30to60E2r2.210210.nc"
          static: true
          vars:
              - name: xCell
              - name: yCell
              - name: zCell
              - name: xEdge
              - name: yEdge
              - name: zEdge
              - name: xVertex
              - name: yVertex
              - name: zVertex
              - name: indexToCellID
              - name: indexToEdgeID
              - name: indexToVertexID
              - name: nEdgesOnCell
              - name: nEdgesOnEdge
              - name: cellsOnCell
              - name: cellsOnEdge
              - name: cellsOnVertex
              - name: edgesOnVertex
              - name: edgesOnCell
              - name: edgesOnEdge
              - name: verticesOnCell
              - name: verticesOnEdge
        - name: data
          format: netcdf
          filenames: "LowRes_0001-0010_climatology.*.nc"
          vars:
              - name: xtime
                possible_names:
                    - xtime
                    - xtime_startMonthly
                dimensions: auto
                optional: false
                multicomponents: false
              - name: normalVelocity
                possible_names:
                    - normalVelocity
                    - timeMonthly_avg_normalVelocity
                    - timeDaily_avg_normalVelocity
                dimensions: auto
                multicomponents: true
              - name: velocityMeridional
                possible_names:
                    - velocityMeridional
                    - timeMonthly_avg_velocityMeridional
                    - timeDaily_avg_velocityMeridional
                multicomponents: true
              - name: velocityZonal
                possible_names:
                    - velocityZonal
                    - timeMonthly_avg_velocityZonal
                    - timeDaily_avg_velocityZonal
                multicomponents: true
              - name: vertVelocityTop
                possible_names:
                    - vertVelocityTop
                    - timeMonthly_avg_vertVelocityTop
                multicomponents: true
              - name: salinity
                possible_names:
                    - salinity
                    - activeTracers_salinity
                    - timeDaily_avg_activeTracers_salinity
                    - timeMonthly_avg_activeTracers_salinity
                optional: true
                multicomponents: true
              - name: temperature
                possible_names:
                    - temperature
                    - timeDaily_avg_activeTracers_temperature
                    - activeTracers_temperature
                    - timeMonthly_avg_activeTracers_temperature
                optional: true
                multicomponents: true
              - name: zTop
                possible_names:
                    - zTop
                    - timeMonthly_avg_zTop
                optional: true
                multicomponents: true
              - name: zMid
                possible_names:
                    - zMid
                    - timeMonthly_avg_zMid
                optional: true
                multicomponents: true
              - name: layerThickness
                possible_names:
                    - layerThickness
                    - timeMonthly_avg_layerThickness
                    - timeDaily_avg_layerThickness
                optional: true
                multicomponents: true
              - name: bottomDepth
                possible_names:
                    - bottomDepth
              - name: seaSurfaceHeight
                possible_names:
                    - seaSurfaceHeight
                    - timeMonthly_avg_ssh
                    - timeDaily_avg_seaSurfaceHeight
                optional: true
                multicomponents: true
'''


REMAPPING_OUTPUT_SPECS = [
    {
        "output_index": 0,
        "channel": 0,
        "file": "output_0_ch0.png",
        "colorbar": "output_0_ch0_colorbar.png",
        "label": "Zonal Velocity",
        "quantity": "velocity_u",
    },
    {
        "output_index": 0,
        "channel": 1,
        "file": "output_0_ch1.png",
        "colorbar": "output_0_ch1_colorbar.png",
        "label": "Meridional Velocity",
        "quantity": "velocity_v",
    },
    {
        "output_index": 0,
        "channel": 2,
        "file": "output_0_ch2.png",
        "colorbar": "output_0_ch2_colorbar.png",
        "label": "Velocity Magnitude",
        "quantity": "velocity_speed",
    },
    {
        "output_index": 0,
        "channel": 3,
        "file": "output_0_ch3.png",
        "colorbar": "output_0_ch3_colorbar.png",
        "label": "Channel 3",
        "quantity": "channel_3",
    },
    {
        "output_index": 1,
        "channel": 0,
        "file": "output_1_ch0.png",
        "colorbar": "output_1_ch0_colorbar.png",
        "label": "Salinity",
        "quantity": "salinity",
    },
    {
        "output_index": 1,
        "channel": 1,
        "file": "output_1_ch1.png",
        "colorbar": "output_1_ch1_colorbar.png",
        "label": "Temperature",
        "quantity": "temperature",
    },
    {
        "output_index": 1,
        "channel": 2,
        "file": "output_1_ch2.png",
        "colorbar": "output_1_ch2_colorbar.png",
        "label": "Channel 2",
        "quantity": "channel_2",
    },
    {
        "output_index": 1,
        "channel": 3,
        "file": "output_1_ch3.png",
        "colorbar": "output_1_ch3_colorbar.png",
        "label": "Channel 3",
        "quantity": "channel_3",
    },
]


def _parse_common_params(user_request):
    req = user_request
    lower = req.lower()

    device = "gpu"
    if "cpu" in lower:
        device = "cpu"

    date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", req)
    date_str = date_match.group(1) if date_match else "0001-01-01"

    depth_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|meter|meters|米)", lower)
    fixed_depth = float(depth_match.group(1)) if depth_match else 10.0

    return {
        "device": device,
        "date": date_str,
        "fixed_depth": fixed_depth,
    }


def _find_nc_files_by_pattern(folder_path):
    folder = Path(folder_path)
    if not folder.is_dir():
        return {}

    nc_files = sorted(folder.glob("*.nc"))
    if not nc_files:
        return {}

    mesh_name = None
    for file_path in nc_files:
        name = file_path.name
        if name.startswith("ocean.") or "mesh" in name.lower():
            mesh_name = name
            break
    if mesh_name is None:
        mesh_name = nc_files[0].name

    data_names = [file_path.name for file_path in nc_files if file_path.name != mesh_name]
    if not data_names:
        return {"mesh": mesh_name}

    if len(data_names) == 1:
        data_pattern = data_names[0]
    else:
        prefix = data_names[0]
        for name in data_names[1:]:
            while prefix and not name.startswith(prefix):
                prefix = prefix[:-1]

        suffix = data_names[0]
        for name in data_names[1:]:
            while suffix and not name.endswith(suffix):
                suffix = suffix[1:]

        if prefix and suffix and len(prefix) + len(suffix) < len(data_names[0]):
            data_pattern = f"{prefix}*{suffix}"
        elif prefix:
            data_pattern = f"{prefix}*"
        elif suffix:
            data_pattern = f"*{suffix}"
        else:
            data_pattern = "*.nc"

    return {"mesh": mesh_name, "data": data_pattern}


def generate_remapping_yaml_config(data_folder):
    if yaml is None:
        return None

    base = yaml.safe_load(REMAPPING_YAML_TEMPLATE_TEXT)
    if not isinstance(base, dict):
        return None

    nc_patterns = _find_nc_files_by_pattern(data_folder)
    if not nc_patterns:
        return None

    stream = base.setdefault("stream", {})
    stream["path_prefix"] = str(Path(data_folder).resolve())

    for substream in stream.get("substreams", []):
        name = substream.get("name")
        if name == "mesh" and nc_patterns.get("mesh"):
            substream["filenames"] = nc_patterns["mesh"]
        elif name == "data" and nc_patterns.get("data"):
            substream["filenames"] = nc_patterns["data"]

    return base


def _render_remapping_job(user_request, config_path):
    template = Template(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            # Auto-generated by Agent/llm_task_agent.py
            # User request: $user_request

            import json
            import sys
            from pathlib import Path

            repo_root = Path(__file__).resolve().parents[2]
            tutorial_dir = repo_root / "tutorial"
            pymops_dir = repo_root / "tools" / "pyMOPS" / "pyMOPS"
            if pymops_dir.exists():
                sys.path.insert(0, str(pymops_dir))
            sys.path.insert(0, str(tutorial_dir))

            from pyMOPSAPI import MOPSRemapping


            def main():
                defaults = json.loads(r'''$defaults_json''')
                config_path = Path(r"$config_path")
                cfg = dict(defaults)
                if config_path.exists():
                    with open(str(config_path), "r") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        cfg.update(loaded)

                yaml_value = cfg.get("yaml_path", defaults["yaml_path"])
                if Path(str(yaml_value)).is_absolute():
                    yaml_path = str(Path(str(yaml_value)))
                else:
                    yaml_path = str(repo_root / str(yaml_value))

                lat_range = cfg.get("lat_range", defaults["lat_range"])
                lon_range = cfg.get("lon_range", defaults["lon_range"])
                if not isinstance(lat_range, (list, tuple)) or len(lat_range) != 2:
                    lat_range = defaults["lat_range"]
                if not isinstance(lon_range, (list, tuple)) or len(lon_range) != 2:
                    lon_range = defaults["lon_range"]

                channels = cfg.get("channels", defaults["channels"])
                if not isinstance(channels, (list, tuple)):
                    channels = defaults["channels"]

                out_subdir = str(cfg.get("output_subdir", defaults["output_subdir"]))
                out_dir = repo_root / out_subdir

                rm = MOPSRemapping(yaml_path).init(
                    device=str(cfg.get("device", defaults["device"])),
                    time_stamp=str(cfg.get("time_stamp", defaults["time_stamp"])),
                    time_step=int(cfg.get("time_step", defaults["time_step"])),
                    add_temperature=bool(cfg.get("add_temperature", defaults["add_temperature"])),
                    add_salinity=bool(cfg.get("add_salinity", defaults["add_salinity"])),
                )

                images = rm.run(
                    width=int(cfg.get("width", defaults["width"])),
                    height=int(cfg.get("height", defaults["height"])),
                    lat_range=(float(lat_range[0]), float(lat_range[1])),
                    lon_range=(float(lon_range[0]), float(lon_range[1])),
                    fixed_depth=float(cfg.get("fixed_depth", defaults["fixed_depth"])),
                    time_step=int(cfg.get("time_step", defaults["time_step"])),
                )

                MOPSRemapping.save_colormap_pngs(
                    images,
                    str(out_dir),
                    prefix="output",
                    channels=tuple(int(c) for c in channels),
                    cmap_name=str(cfg.get("cmap_name", defaults["cmap_name"])),
                    save_colorbar=bool(cfg.get("save_colorbar", defaults["save_colorbar"])),
                )

                manifest = {
                    "task": "remapping",
                    "yaml_path": yaml_path,
                    "device": str(cfg.get("device", defaults["device"])),
                    "time_stamp": str(cfg.get("time_stamp", defaults["time_stamp"])),
                    "time_step": int(cfg.get("time_step", defaults["time_step"])),
                    "width": int(cfg.get("width", defaults["width"])),
                    "height": int(cfg.get("height", defaults["height"])),
                    "lat_range": [float(lat_range[0]), float(lat_range[1])],
                    "lon_range": [float(lon_range[0]), float(lon_range[1])],
                    "fixed_depth": float(cfg.get("fixed_depth", defaults["fixed_depth"])),
                    "cmap_name": str(cfg.get("cmap_name", defaults["cmap_name"])),
                    "save_colorbar": bool(cfg.get("save_colorbar", defaults["save_colorbar"])),
                    "output_subdir": out_subdir,
                    "channel_map": [item for item in $channel_specs if int(item["output_index"]) < len(images)],
                }

                manifest_path = out_dir / "manifest.json"
                with open(str(manifest_path), "w") as f:
                    json.dump(manifest, f, indent=2, sort_keys=True)

                print("[GeneratedJob] remapping done ->", out_dir)
                print("[GeneratedJob] remapping manifest ->", manifest_path)


            if __name__ == "__main__":
                main()
            """
        )
    )
    return template.substitute(
        user_request=user_request.replace("\n", " "),
        defaults_json=json.dumps(DEFAULT_REMAPPING_CONFIG),
        channel_specs=json.dumps(REMAPPING_OUTPUT_SPECS),
        config_path=str(config_path),
    )


def _render_streamline_job(user_request, config_path):
    template = Template(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            # Auto-generated by Agent/llm_task_agent.py
            # User request: $user_request

            import json
            import sys
            from pathlib import Path

            repo_root = Path(__file__).resolve().parents[2]
            tutorial_dir = repo_root / "tutorial"
            pymops_dir = repo_root / "tools" / "pyMOPS" / "pyMOPS"
            if pymops_dir.exists():
                sys.path.insert(0, str(pymops_dir))
            sys.path.insert(0, str(tutorial_dir))

            from pyMOPSAPI import MOPSStreamline, Vis_PathLines


            def main():
                defaults = json.loads(r'''$defaults_json''')
                config_path = Path(r"$config_path")
                cfg = dict(defaults)
                if config_path.exists():
                    with open(str(config_path), "r") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        cfg.update(loaded)

                yaml_value = cfg.get("yaml_path", defaults["yaml_path"])
                if Path(str(yaml_value)).is_absolute():
                    yaml_path = str(Path(str(yaml_value)))
                else:
                    yaml_path = str(repo_root / str(yaml_value))

                lat_range = cfg.get("lat_range", defaults["lat_range"])
                lon_range = cfg.get("lon_range", defaults["lon_range"])
                grid = cfg.get("grid", defaults["grid"])

                if not isinstance(lat_range, (list, tuple)) or len(lat_range) != 2:
                    lat_range = defaults["lat_range"]
                if not isinstance(lon_range, (list, tuple)) or len(lon_range) != 2:
                    lon_range = defaults["lon_range"]
                if not isinstance(grid, (list, tuple)) or len(grid) != 2:
                    grid = defaults["grid"]

                out_subdir = str(cfg.get("output_subdir", defaults["output_subdir"]))
                out_dir = repo_root / out_subdir
                out_dir.mkdir(parents=True, exist_ok=True)

                sl = MOPSStreamline(yaml_path).init(str(cfg.get("device", defaults["device"])))
                sl.set_time(
                    start=str(cfg.get("start_date", defaults["start_date"])),
                    duration_ymd=(0, 0, int(cfg.get("duration_days", defaults["duration_days"])))
                )
                sl.set_seed(
                    depth=float(cfg.get("fixed_depth", defaults["fixed_depth"])),
                    lat_range=(float(lat_range[0]), float(lat_range[1])),
                    lon_range=(float(lon_range[0]), float(lon_range[1])),
                    grid=(int(grid[0]), int(grid[1])),
                )

                lines = sl.run(
                    method=str(cfg.get("method", defaults["method"])),
                    delta_minutes=int(cfg.get("delta_minutes", defaults["delta_minutes"])),
                    record_every_minutes=int(cfg.get("record_every_minutes", defaults["record_every_minutes"]))
                )

                out_file = out_dir / "streamline.png"
                Vis_PathLines(
                    lines,
                    save_path=str(out_file),
                    color_by=str(cfg.get("color_by", defaults["color_by"])),
                    title="Generated Streamline"
                )

                manifest = {
                    "task": "streamline",
                    "yaml_path": yaml_path,
                    "device": str(cfg.get("device", defaults["device"])),
                    "start_date": str(cfg.get("start_date", defaults["start_date"])),
                    "duration_days": int(cfg.get("duration_days", defaults["duration_days"])),
                    "fixed_depth": float(cfg.get("fixed_depth", defaults["fixed_depth"])),
                    "lat_range": [float(lat_range[0]), float(lat_range[1])],
                    "lon_range": [float(lon_range[0]), float(lon_range[1])],
                    "grid": [int(grid[0]), int(grid[1])],
                    "method": str(cfg.get("method", defaults["method"])),
                    "output_subdir": out_subdir,
                }

                manifest_path = out_dir / "manifest.json"
                with open(str(manifest_path), "w") as f:
                    json.dump(manifest, f, indent=2, sort_keys=True)

                print("[GeneratedJob] streamline done ->", out_file)
                print("[GeneratedJob] streamline manifest ->", manifest_path)


            if __name__ == "__main__":
                main()
            """
        )
    )
    return template.substitute(
        user_request=user_request.replace("\n", " "),
        defaults_json=json.dumps(DEFAULT_STREAMLINE_CONFIG),
        config_path=str(config_path),
    )


def _render_pathline_job(user_request, config_path):
    template = Template(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            # Auto-generated by Agent/llm_task_agent.py
            # User request: $user_request

            import json
            import sys
            from pathlib import Path

            repo_root = Path(__file__).resolve().parents[2]
            tutorial_dir = repo_root / "tutorial"
            pymops_dir = repo_root / "tools" / "pyMOPS" / "pyMOPS"
            if pymops_dir.exists():
                sys.path.insert(0, str(pymops_dir))
            sys.path.insert(0, str(tutorial_dir))

            from pyMOPSAPI import MOPSPathline, Vis_PathLines
            from export_pathline_binary import export_pathlines_to_binary, export_pathlines_to_json


            def main():
                defaults = json.loads(r'''$defaults_json''')
                config_path = Path(r"$config_path")
                cfg = dict(defaults)
                if config_path.exists():
                    with open(str(config_path), "r") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        cfg.update(loaded)

                yaml_value = cfg.get("yaml_path", defaults["yaml_path"])
                if Path(str(yaml_value)).is_absolute():
                    yaml_path = str(Path(str(yaml_value)))
                else:
                    yaml_path = str(repo_root / str(yaml_value))

                lat_range = cfg.get("lat_range", defaults["lat_range"])
                lon_range = cfg.get("lon_range", defaults["lon_range"])
                grid = cfg.get("grid", defaults["grid"])

                if not isinstance(lat_range, (list, tuple)) or len(lat_range) != 2:
                    lat_range = defaults["lat_range"]
                if not isinstance(lon_range, (list, tuple)) or len(lon_range) != 2:
                    lon_range = defaults["lon_range"]
                if not isinstance(grid, (list, tuple)) or len(grid) != 2:
                    grid = defaults["grid"]

                out_subdir = str(cfg.get("output_subdir", defaults["output_subdir"]))
                out_dir = repo_root / out_subdir
                out_dir.mkdir(parents=True, exist_ok=True)

                pl = MOPSPathline(yaml_path).init(str(cfg.get("device", defaults["device"])))
                pl.set_time(
                    int(cfg.get("start_year", defaults["start_year"])),
                    int(cfg.get("start_month", defaults["start_month"])),
                    int(cfg.get("end_year", defaults["end_year"])),
                    int(cfg.get("end_month", defaults["end_month"])),
                    direction=str(cfg.get("direction", defaults["direction"]))
                )
                pl.set_seed(
                    depth=float(cfg.get("fixed_depth", defaults["fixed_depth"])),
                    lat_range=(float(lat_range[0]), float(lat_range[1])),
                    lon_range=(float(lon_range[0]), float(lon_range[1])),
                    grid=(int(grid[0]), int(grid[1])),
                )

                lines = pl.run(
                    method=str(cfg.get("method", defaults["method"])),
                    delta_minutes=int(cfg.get("delta_minutes", defaults["delta_minutes"])),
                    record_every_minutes=int(cfg.get("record_every_minutes", defaults["record_every_minutes"]))
                )

                out_file = out_dir / "pathline.png"
                Vis_PathLines(
                    lines,
                    save_path=str(out_file),
                    color_by=str(cfg.get("color_by", defaults["color_by"])),
                    title="Generated Pathline"
                )

                # Export pathline data for web visualization
                bin_file = out_dir / "pathlines.bin"
                json_file = out_dir / "pathlines.json"

                export_pathlines_to_binary(
                    lines,
                    str(bin_file),
                    include_velocity=True,
                    include_scalars=True
                )

                export_pathlines_to_json(
                    lines,
                    str(json_file),
                    decimation_factor=1
                )

                manifest = {
                    "task": "pathline",
                    "yaml_path": yaml_path,
                    "device": str(cfg.get("device", defaults["device"])),
                    "start_year": int(cfg.get("start_year", defaults["start_year"])),
                    "start_month": int(cfg.get("start_month", defaults["start_month"])),
                    "end_year": int(cfg.get("end_year", defaults["end_year"])),
                    "end_month": int(cfg.get("end_month", defaults["end_month"])),
                    "direction": str(cfg.get("direction", defaults["direction"])),
                    "fixed_depth": float(cfg.get("fixed_depth", defaults["fixed_depth"])),
                    "lat_range": [float(lat_range[0]), float(lat_range[1])],
                    "lon_range": [float(lon_range[0]), float(lon_range[1])],
                    "grid": [int(grid[0]), int(grid[1])],
                    "method": str(cfg.get("method", defaults["method"])),
                    "output_subdir": out_subdir,
                    "output_files": {
                        "image": "pathline.png",
                        "binary": "pathlines.bin",
                        "json": "pathlines.json",
                        "binary_metadata": "pathlines.meta.json"
                    }
                }

                manifest_path = out_dir / "manifest.json"
                with open(str(manifest_path), "w") as f:
                    json.dump(manifest, f, indent=2, sort_keys=True)

                print("[GeneratedJob] pathline done ->", out_file)
                print("[GeneratedJob] pathline binary ->", bin_file)
                print("[GeneratedJob] pathline JSON ->", json_file)
                print("[GeneratedJob] pathline manifest ->", manifest_path)


            if __name__ == "__main__":
                main()
            """
        )
    )
    return template.substitute(
        user_request=user_request.replace("\n", " "),
        defaults_json=json.dumps(DEFAULT_PATHLINE_CONFIG),
        config_path=str(config_path),
    )


def render_job_script(task, user_request, config_path=""):
    if task == "remapping":
        return _render_remapping_job(user_request, config_path)
    if task == "streamline":
        return _render_streamline_job(user_request, config_path)
    return _render_pathline_job(user_request, config_path)
