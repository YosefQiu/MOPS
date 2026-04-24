"""
Binary file reader for MOPS regridded data.

This utility reads .bin files saved by reGrid.cpp or reGrid.py
and provides functions for visualization and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from pathlib import Path


def read_regrid_binary(
    filename: str,
    width: int,
    height: int,
    channels: int = 4,
    dtype: np.dtype = np.float64
) -> np.ndarray:
    """
    Read a regridded binary file saved by MOPS.

    The binary file format is:
    - Raw doubles (8 bytes each)
    - Layout: [height, width, channels]
    - Channels: [E, N, Vertical, Magnitude]

    Parameters
    ----------
    filename : str
        Path to the .bin file
    width : int
        Image width (longitude bins)
    height : int
        Image height (depth bins)
    channels : int
        Number of channels (default: 4)
    dtype : np.dtype
        Data type (default: np.float64)

    Returns
    -------
    np.ndarray
        Image array of shape (height, width, channels)

    Examples
    --------
    >>> img = read_regrid_binary("regrid_fixed_latitude.bin", width=720, height=100)
    >>> print(img.shape)
    (100, 720, 4)
    """
    # Read raw binary data
    data = np.fromfile(filename, dtype=dtype)

    # Expected size
    expected_size = width * height * channels

    if data.size != expected_size:
        raise ValueError(
            f"File size mismatch: expected {expected_size} values "
            f"({width}x{height}x{channels}), got {data.size}"
        )

    # Reshape to (height, width, channels)
    image = data.reshape((height, width, channels))

    return image


def visualize_regrid(
    image: np.ndarray,
    lon_range: Tuple[float, float] = (-180.0, 180.0),
    depth_range: Tuple[float, float] = (0.0, 5000.0),
    fixed_latitude: float = 45.0,
    channel: int = 3,
    cmap: str = "viridis",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6),
):
    """
    Visualize a regridded image with proper axis labels.

    Parameters
    ----------
    image : np.ndarray
        Image array of shape (height, width, channels)
    lon_range : Tuple[float, float]
        Longitude range in degrees
    depth_range : Tuple[float, float]
        Depth range in meters
    fixed_latitude : float
        Latitude of the cross-section
    channel : int
        Which channel to visualize (0=E, 1=N, 2=Vertical, 3=Magnitude)
    cmap : str
        Matplotlib colormap name
    title : Optional[str]
        Plot title (auto-generated if None)
    save_path : Optional[str]
        Path to save the figure
    figsize : Tuple[float, float]
        Figure size in inches
    """
    channel_names = ["E (East)", "N (North)", "Vertical", "Magnitude"]

    if channel < 0 or channel >= image.shape[2]:
        raise ValueError(f"Invalid channel {channel}, image has {image.shape[2]} channels")

    data = image[:, :, channel]
    height, width = data.shape

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot image with correct extent
    # extent: [left, right, bottom, top]
    im = ax.imshow(
        data,
        aspect='auto',
        origin='upper',  # Depth increases downward
        extent=[lon_range[0], lon_range[1], depth_range[1], depth_range[0]],
        cmap=cmap,
        interpolation='bilinear'
    )

    # Labels
    ax.set_xlabel('Longitude (degrees)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)

    if title is None:
        title = f"Regridded {channel_names[channel]} at Latitude {fixed_latitude}°"
    ax.set_title(title, fontsize=14)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label(channel_names[channel], fontsize=12)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")

    plt.show()


def visualize_all_channels(
    image: np.ndarray,
    lon_range: Tuple[float, float] = (-180.0, 180.0),
    depth_range: Tuple[float, float] = (0.0, 5000.0),
    fixed_latitude: float = 45.0,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (16, 10),
):
    """
    Visualize all channels in a 2x2 grid.

    Parameters
    ----------
    image : np.ndarray
        Image array of shape (height, width, channels)
    lon_range : Tuple[float, float]
        Longitude range in degrees
    depth_range : Tuple[float, float]
        Depth range in meters
    fixed_latitude : float
        Latitude of the cross-section
    save_path : Optional[str]
        Path to save the figure
    figsize : Tuple[float, float]
        Figure size in inches
    """
    channel_names = ["E (East)", "N (North)", "Vertical", "Magnitude"]
    cmaps = ["coolwarm", "coolwarm", "coolwarm", "viridis"]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for i in range(min(4, image.shape[2])):
        ax = axes[i]
        data = image[:, :, i]

        im = ax.imshow(
            data,
            aspect='auto',
            origin='upper',
            extent=[lon_range[0], lon_range[1], depth_range[1], depth_range[0]],
            cmap=cmaps[i],
            interpolation='bilinear'
        )

        ax.set_xlabel('Longitude (degrees)')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f"{channel_names[i]} at Latitude {fixed_latitude}°")
        ax.grid(True, alpha=0.3, linestyle='--')

        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label(channel_names[i])

    plt.suptitle(f"Regridded Velocity Components at Latitude {fixed_latitude}°", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")

    plt.show()


def get_statistics(image: np.ndarray, channel: int = 3) -> dict:
    """
    Compute statistics for a specific channel.

    Parameters
    ----------
    image : np.ndarray
        Image array of shape (height, width, channels)
    channel : int
        Which channel to analyze

    Returns
    -------
    dict
        Dictionary with statistics (min, max, mean, std, valid_pixels)
    """
    data = image[:, :, channel]
    valid_mask = np.isfinite(data)
    valid_data = data[valid_mask]

    if valid_data.size == 0:
        return {
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "valid_pixels": 0,
            "total_pixels": data.size,
            "valid_fraction": 0.0
        }

    return {
        "min": float(np.min(valid_data)),
        "max": float(np.max(valid_data)),
        "mean": float(np.mean(valid_data)),
        "std": float(np.std(valid_data)),
        "valid_pixels": int(valid_data.size),
        "total_pixels": int(data.size),
        "valid_fraction": float(valid_data.size / data.size)
    }


def example():
    """Example usage of the binary reader."""
    # Read binary file
    filename = "regrid_fixed_latitude.bin"

    if not Path(filename).exists():
        print(f"Error: {filename} not found.")
        print("Run reGrid.py first to generate the binary file.")
        return

    print(f"Reading {filename}...")
    image = read_regrid_binary(
        filename,
        width=720,   # 360 * 2
        height=100,  # Adjust based on your depth range and resolution
        channels=4
    )

    print(f"Image shape: {image.shape}")
    print(f"Channels: [E, N, Vertical, Magnitude]")

    # Print statistics for each channel
    channel_names = ["E (East)", "N (North)", "Vertical", "Magnitude"]
    print("\nStatistics:")
    for i, name in enumerate(channel_names):
        stats = get_statistics(image, channel=i)
        print(f"\n{name}:")
        print(f"  Min: {stats['min']:.6f}")
        print(f"  Max: {stats['max']:.6f}")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std: {stats['std']:.6f}")
        print(f"  Valid pixels: {stats['valid_pixels']}/{stats['total_pixels']} "
              f"({stats['valid_fraction']*100:.1f}%)")

    # Visualize magnitude channel
    print("\nVisualizing magnitude channel...")
    visualize_regrid(
        image,
        lon_range=(-180.0, 180.0),
        depth_range=(0.0, 5000.0),
        fixed_latitude=45.0,
        channel=3,  # Magnitude
        cmap="viridis",
        save_path="regrid_magnitude_visualization.png"
    )

    # Visualize all channels
    print("\nVisualizing all channels...")
    visualize_all_channels(
        image,
        lon_range=(-180.0, 180.0),
        depth_range=(0.0, 5000.0),
        fixed_latitude=45.0,
        save_path="regrid_all_channels.png"
    )


if __name__ == "__main__":
    example()
