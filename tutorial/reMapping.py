from pyMOPSAPI import *

def example():
    yaml_path = "/pscratch/sd/q/qiuyf/MOPS_Tutorial/test_ab_climatology.yaml"

    rm = MOPSRemapping(yaml_path).init(
        device="gpu",
        time_stamp="0002-01-01",
        time_step=0,
        add_temperature=True,
        add_salinity=True,
    )

    # Keep enum-related settings optional unless you are sure of the values.
    images = rm.run(
        width=3601,
        height=1801,
        lat_range=(-90.0, 90.0),
        lon_range=(-180.0, 180.0),
        fixed_depth=10.0,
        time_step=0,
    )

    print(f"Got {len(images)} remapped images")
    for i, img in enumerate(images):
        print(f"  image[{i}] shape = {np.asarray(img).shape}")

    MOPSRemapping.save_colormap_pngs(
        images,
        "remap_outputs",
        prefix="output",
        channels=(0, 1, 2, 3), 
        cmap_name="coolwarm",
        save_colorbar=True,
    )


if __name__ == "__main__":
    example()
