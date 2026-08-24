from pyMOPSAPI import *
import os

def example_single_latitude():
    """Example: Regrid at a single fixed latitude"""
    yaml_path = "/pscratch/sd/q/qiuyf/MOPS_Tutorial/test_ab_climatology.yaml"

    rg = MOPSReGrid(yaml_path).init(
        device="gpu",
        time_stamp="0002-01-01",
        time_step=0,
        add_temperature=True,
        add_salinity=True,
    )

    # Run regridding at latitude 45°N
    image = rg.run(
        width=360 * 2,
        lat_range=(-90.0, 90.0),
        lon_range=(-180.0, 180.0),
        fixed_latitude=45.0,
        time_step=0,
    )

    print(f"Got regridded image shape: {np.asarray(image).shape}")

    # Save outputs
    MOPSReGrid.save_colormap_pngs(
        [image],
        "regrid_single_outputs",
        prefix="lat_45N",
        channels=(0, 1, 2, 3),
        cmap_name="coolwarm",
        save_colorbar=True,
    )


def example_multiple_latitudes():
    """Example: Regrid across multiple latitudes (Pacific focus)"""
    yaml_path = "/pscratch/sd/q/qiuyf/MOPS_Tutorial/test_ab_climatology.yaml"

    rg = MOPSReGrid(yaml_path).init(
        device="gpu",
        time_stamp="0002-01-01",
        time_step=0,
        add_temperature=True,
        add_salinity=True,
    )

    output_dir = "regrid_pacific_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Loop through latitudes with 2-degree increments
    # For Pacific focus: -60 to 60, or full range: -90 to 90
    for lat in range(-90, 91, 2):
        print(f"\n=== Processing latitude {lat}° ===")

        image = rg.run(
            width=360 * 2,
            lat_range=(-90.0, 90.0),
            lon_range=(-180.0, 180.0),
            fixed_latitude=float(lat),
            time_step=0,
        )

        # Create latitude-specific prefix
        lat_str = f"N{lat:03d}" if lat >= 0 else f"S{abs(lat):03d}"

        # Save with latitude in filename
        MOPSReGrid.save_colormap_pngs(
            [image],
            output_dir,
            prefix=f"lat_{lat_str}",
            channels=(0, 1, 2, 3),
            cmap_name="coolwarm",
            save_colorbar=False,  # Skip colorbar for batch processing
        )

        # Save as binary for later use
        MOPSReGrid.save_binary(
            [image],
            output_dir,
            prefix=f"lat_{lat_str}",
        )

    print(f"\n=== All latitudes processed! Files saved to {output_dir} ===")


def example_pacific_focus():
    """Example: Focus on Pacific latitudes only"""
    yaml_path = "/pscratch/sd/q/qiuyf/MOPS_Tutorial/test_ab_climatology.yaml"

    rg = MOPSReGrid(yaml_path).init(
        device="gpu",
        time_stamp="0002-01-01",
        time_step=0,
        add_temperature=True,
        add_salinity=True,
    )

    output_dir = "regrid_pacific_focus"
    os.makedirs(output_dir, exist_ok=True)

    # Pacific focus: -60 to 60 latitude
    pacific_lats = np.arange(-60, 62, 2)  # Every 2 degrees

    for lat in pacific_lats:
        print(f"Processing latitude {lat}°")

        image = rg.run(
            width=360 * 2,
            lat_range=(-90.0, 90.0),
            lon_range=(-180.0, 180.0),
            fixed_latitude=float(lat),
            time_step=0,
        )

        lat_str = f"N{int(lat):03d}" if lat >= 0 else f"S{int(abs(lat)):03d}"

        # Save PNG and binary
        MOPSReGrid.save_colormap_pngs(
            [image],
            output_dir,
            prefix=f"lat_{lat_str}",
            channels=(0, 1, 2, 3),
            cmap_name="coolwarm",
        )

    print(f"Done! {len(pacific_lats)} latitudes processed.")


if __name__ == "__main__":
    # Choose one of the examples:

    # example_single_latitude()
    example_multiple_latitudes()
    # example_pacific_focus()
