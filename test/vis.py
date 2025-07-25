import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def vis_image(imgs, output_file = "remap.png"):
    if not imgs or len(imgs) < 2:
        raise ValueError("imgs must be a list of at least 2 images")
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axs = axes.flatten()
    
    titles = [
        "Zonal Velocity",
        "Meridional Velocity",
        "Velocity Magnitude",
        "Temperature",
        "Salinity",
        "Unused"
    ]
    
    cmaps = [
        "coolwarm", "coolwarm", "coolwarm",
        "ocean", "ocean", None
    ]
    
    data_refs = [
        imgs[0][:, :, 0],
        imgs[0][:, :, 1],
        imgs[0][:, :, 2],
        imgs[1][:, :, 0], # temperature
        imgs[1][:, :, 1], # salinity
        None  
    ]
    
    vmins = [None, None, None, 0, 0, None]
    vmaxs = [None, None, None, 31, 40, None]

    extent = [-180, 180, -90, 90]
    for i, (ax, title, data) in enumerate(zip(axs, titles, data_refs)):
        if data is not None:
            # 复制 colormap，并设置 NaN 映射成黄色
            cmap = plt.get_cmap(cmaps[i]).copy()
            cmap.set_bad(color='yellow')

            im = im = ax.imshow(data, cmap=cmap, extent=extent, aspect='auto',
               vmin=vmins[i], vmax=vmaxs[i])
            ax.set_title(title)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()
    
def xyz_to_latlon(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.where(r < 1e-8, np.nan, r)
    lat = np.arcsin(z / r) * 180 / np.pi
    lon = np.arctan2(y, x) * 180 / np.pi
    return lat, lon

def plot_2d_latlon_cartopy(points, save_path="sample_points_cartopy.png"):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    lats, lons = xyz_to_latlon(x, y, z)   

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_global()
    ax.stock_img()  
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True)
    
    ax.scatter(lons, lats, color='red', s=1, transform=ccrs.PlateCarree())
    ax.set_title("Sample Points on Earth Map (Cartopy)")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_2d_trajectories(trajectory_lines, save_path="trajectories_2d.png"):
    
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_global()
    ax.stock_img()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True)

    for line_array in trajectory_lines:
        lats, lons = [], []
        for pt in line_array:
            lat, lon = xyz_to_latlon(pt[0], pt[1], pt[2])
            if np.isnan(lon) or np.isnan(lat):
                continue
            lats.append(lat)
            lons.append(lon)
            
        if len(lons) >= 2: 
            ax.plot(lons, lats, linewidth=0.8)

    plt.title("2D Trajectories (Lat/Lon)")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()