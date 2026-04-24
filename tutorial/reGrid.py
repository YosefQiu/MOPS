from pyMOPSAPI import *

def example():
    """
    Example: Regridding at fixed latitude.

    This mirrors the functionality of tutorial/reGrid.cpp.
    It creates a longitude-depth cross-section at a fixed latitude.
    """
    yaml_path = "/pscratch/sd/q/qiuyf/MOPS_Tutorial/bmoorema.yaml"

    # Initialize ReGrid
    rg = MOPSReGrid(yaml_path).init(
        device="gpu",
        time_stamp="0001-01-01",
        time_step=0,
        add_temperature=True,
        add_salinity=True,
    )

    # Run regridding at fixed latitude
    # This creates a longitude-depth cross-section
    image = rg.run(
        width=720,          # 360 * 2 (longitude bins)
        height=100,         # depth bins
        lon_range=(-180.0, 180.0),
        depth_range=(0.0, 5000.0),  # You can adjust based on your data
        fixed_latitude=45.0,         # Latitude of the cross-section
        time_step=0,
    )

    print(f"Regridded image shape: {image.shape}")
    print(f"Image channels: [E, N, Vertical, Magnitude]")

    # Save outputs
    # 1. Save as binary (matches C++ tutorial output format)
    MOPSReGrid.save_to_binary(image, "regrid_fixed_latitude.bin")

    # 2. Save individual channels as PNG
    MOPSReGrid.save_to_png(image, "regrid_E.png", channel=0, cmap_name="coolwarm")
    MOPSReGrid.save_to_png(image, "regrid_N.png", channel=1, cmap_name="coolwarm")
    MOPSReGrid.save_to_png(image, "regrid_Vertical.png", channel=2, cmap_name="coolwarm")
    MOPSReGrid.save_to_png(image, "regrid_Magnitude.png", channel=3, cmap_name="viridis")

    print("\nSaved outputs:")
    print("  - regrid_fixed_latitude.bin (binary)")
    print("  - regrid_E.png (East component)")
    print("  - regrid_N.png (North component)")
    print("  - regrid_Vertical.png (Vertical component)")
    print("  - regrid_Magnitude.png (Velocity magnitude)")


if __name__ == "__main__":
    example()
