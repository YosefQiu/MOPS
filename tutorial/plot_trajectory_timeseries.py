#!/usr/bin/env python3
"""
Plot pathline trajectory velocity, temperature, salinity, and depth time series
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

# Configuration
pkl_path = "./LowRes_45N_backward_0_005/run_45N_negative_random_0_005Sv_3_dt_10m_save_1d.pkl"
output_prefix = "./LowRes_45N_backward_0_005/timeseries"

# Earth radius for depth calculation
R_EARTH = 6371000.0  # meters

# Load data
print(f"Loading file: {pkl_path}")
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

n_particles = len(data)
print(f"Number of particles: {n_particles}")

# Create figure: 4 subplots (velocity, temperature, salinity, depth)
fig, axes = plt.subplots(4, 1, figsize=(12, 14))

# Color map
colors = cm.tab10(np.linspace(0, 1, n_particles))

# Process each particle
for i, traj in enumerate(data):
    # Extract data
    positions = np.array(traj['points'])
    velocities = np.array(traj['velocity'])
    temps = np.array(traj.get('temperature', []))
    sals = np.array(traj.get('salinity', []))

    # Time steps (assuming each point is 1 day)
    n_points = len(velocities)
    time_days = np.arange(n_points)

    # Calculate velocity magnitude
    vel_mag = np.linalg.norm(velocities, axis=1)

    # Calculate depth from positions
    r = np.linalg.norm(positions, axis=1)
    depths = R_EARTH - r

    # Filter out NaN values
    valid_vel = ~np.isnan(vel_mag)
    valid_temp = ~np.isnan(temps) if len(temps) > 0 else np.array([])
    valid_sal = ~np.isnan(sals) if len(sals) > 0 else np.array([])

    # Plot velocity
    if np.any(valid_vel):
        axes[0].plot(time_days[valid_vel], vel_mag[valid_vel],
                     label=f'Particle {i}', color=colors[i], linewidth=1.5, alpha=0.8)

    # Plot temperature
    if len(temps) > 0 and np.any(valid_temp):
        axes[1].plot(time_days[valid_temp], temps[valid_temp],
                     label=f'Particle {i}', color=colors[i], linewidth=1.5, alpha=0.8)

    # Plot salinity
    if len(sals) > 0 and np.any(valid_sal):
        axes[2].plot(time_days[valid_sal], sals[valid_sal],
                     label=f'Particle {i}', color=colors[i], linewidth=1.5, alpha=0.8)

    # Plot depth
    axes[3].plot(time_days, depths, label=f'Particle {i}',
                 color=colors[i], linewidth=1.5, alpha=0.8)

# Configure velocity subplot
axes[0].set_xlabel('Time (days)', fontsize=12)
axes[0].set_ylabel('Velocity (m/s)', fontsize=12)
axes[0].set_title('Particle Velocity vs Time', fontsize=14, fontweight='bold')
axes[0].legend(loc='best', fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')  # Log scale for better velocity range display

# Configure temperature subplot
axes[1].set_xlabel('Time (days)', fontsize=12)
axes[1].set_ylabel('Temperature (°C)', fontsize=12)
axes[1].set_title('Particle Temperature vs Time', fontsize=14, fontweight='bold')
axes[1].legend(loc='best', fontsize=9)
axes[1].grid(True, alpha=0.3)

# Configure salinity subplot
axes[2].set_xlabel('Time (days)', fontsize=12)
axes[2].set_ylabel('Salinity (PSU)', fontsize=12)
axes[2].set_title('Particle Salinity vs Time', fontsize=14, fontweight='bold')
axes[2].legend(loc='best', fontsize=9)
axes[2].grid(True, alpha=0.3)

# Configure depth subplot
axes[3].set_xlabel('Time (days)', fontsize=12)
axes[3].set_ylabel('Depth (m)', fontsize=12)
axes[3].set_title('Particle Depth vs Time', fontsize=14, fontweight='bold')
axes[3].legend(loc='best', fontsize=9)
axes[3].grid(True, alpha=0.3)
axes[3].invert_yaxis()  # Invert y-axis so deeper is lower

# Adjust layout
plt.tight_layout()

# Save figure
output_file = f"{output_prefix}_all.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved figure: {output_file}")

# Show figure
plt.show()

# ============================================================================
# Generate detailed plots for each particle
# ============================================================================

for i, traj in enumerate(data):
    fig2, axes2 = plt.subplots(4, 1, figsize=(14, 14))

    # Extract data
    positions = np.array(traj['points'])
    velocities = np.array(traj['velocity'])
    temps = np.array(traj.get('temperature', []))
    sals = np.array(traj.get('salinity', []))

    n_points = len(velocities)
    time_days = np.arange(n_points)

    # Velocity: decompose into horizontal and vertical components
    vel_mag = np.linalg.norm(velocities, axis=1)
    vel_h = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
    vel_v = np.abs(velocities[:, 2])

    # Calculate depth
    r = np.linalg.norm(positions, axis=1)
    depths = R_EARTH - r

    # Filter NaN
    valid_vel = ~np.isnan(vel_mag)

    # Plot velocity
    axes2[0].plot(time_days[valid_vel], vel_mag[valid_vel],
                  label='Total velocity', color='blue', linewidth=2, alpha=0.8)
    axes2[0].plot(time_days[valid_vel], vel_h[valid_vel],
                  label='Horizontal velocity', color='green', linewidth=1.5, alpha=0.7, linestyle='--')
    axes2[0].plot(time_days[valid_vel], vel_v[valid_vel],
                  label='Vertical velocity', color='red', linewidth=1.5, alpha=0.7, linestyle=':')

    axes2[0].set_xlabel('Time (days)', fontsize=12)
    axes2[0].set_ylabel('Velocity (m/s)', fontsize=12)
    axes2[0].set_title(f'Particle {i} - Velocity Components', fontsize=14, fontweight='bold')
    axes2[0].legend(loc='best', fontsize=10)
    axes2[0].grid(True, alpha=0.3)
    axes2[0].set_yscale('log')

    # Add reference lines
    axes2[0].axhline(y=1e-3, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    axes2[0].axhline(y=1e-4, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Plot temperature
    if len(temps) > 0:
        valid_temp = ~np.isnan(temps)

        # Separate zero and non-zero values
        zero_temp = (np.abs(temps) < 1e-9)
        nonzero_temp = valid_temp & ~zero_temp

        # Plot non-zero values as line
        if np.any(nonzero_temp):
            axes2[1].plot(time_days[nonzero_temp], temps[nonzero_temp],
                          color='red', linewidth=2, alpha=0.8, label='Valid data')

        # Plot zero values as scatter
        if np.any(zero_temp):
            axes2[1].scatter(time_days[zero_temp], temps[zero_temp],
                            color='orange', s=30, alpha=0.6, label='Zero values (masked region)', zorder=5)

    axes2[1].set_xlabel('Time (days)', fontsize=12)
    axes2[1].set_ylabel('Temperature (°C)', fontsize=12)
    axes2[1].set_title(f'Particle {i} - Temperature', fontsize=14, fontweight='bold')
    axes2[1].grid(True, alpha=0.3)
    axes2[1].legend(loc='best', fontsize=10)

    # Plot salinity
    if len(sals) > 0:
        valid_sal = ~np.isnan(sals)

        # Separate zero and non-zero values
        zero_sal = (np.abs(sals) < 1e-9)
        nonzero_sal = valid_sal & ~zero_sal

        # Plot non-zero values as line
        if np.any(nonzero_sal):
            axes2[2].plot(time_days[nonzero_sal], sals[nonzero_sal],
                          color='blue', linewidth=2, alpha=0.8, label='Valid data')

        # Plot zero values as scatter
        if np.any(zero_sal):
            axes2[2].scatter(time_days[zero_sal], sals[zero_sal],
                            color='orange', s=30, alpha=0.6, label='Zero values (masked region)', zorder=5)

    axes2[2].set_xlabel('Time (days)', fontsize=12)
    axes2[2].set_ylabel('Salinity (PSU)', fontsize=12)
    axes2[2].set_title(f'Particle {i} - Salinity', fontsize=14, fontweight='bold')
    axes2[2].grid(True, alpha=0.3)
    axes2[2].legend(loc='best', fontsize=10)

    # Plot depth
    axes2[3].plot(time_days, depths, color='purple', linewidth=2, alpha=0.8)

    # Mark regions where temp/sal are zero
    if len(temps) > 0 and len(sals) > 0:
        both_zero = (np.abs(temps) < 1e-9) & (np.abs(sals) < 1e-9)
        if np.any(both_zero):
            # Shade regions where both are zero
            axes2[3].scatter(time_days[both_zero], depths[both_zero],
                           color='orange', s=30, alpha=0.5,
                           label='Masked region (temp=sal=0)', zorder=5)

    axes2[3].set_xlabel('Time (days)', fontsize=12)
    axes2[3].set_ylabel('Depth (m)', fontsize=12)
    axes2[3].set_title(f'Particle {i} - Depth', fontsize=14, fontweight='bold')
    axes2[3].grid(True, alpha=0.3)
    axes2[3].invert_yaxis()  # Invert y-axis
    if len(temps) > 0 and len(sals) > 0 and np.any(both_zero):
        axes2[3].legend(loc='best', fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Save
    output_file_single = f"{output_prefix}_particle{i}.png"
    plt.savefig(output_file_single, dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure: {output_file_single}")

    plt.close(fig2)

print("\n✅ All figures generated!")
print(f"   Overview: {output_prefix}_all.png")
print(f"   Individual: {output_prefix}_particle0.png ~ particle{n_particles-1}.png")

# ============================================================================
# Print summary statistics
# ============================================================================
print("\n" + "="*80)
print("SUMMARY: Temperature/Salinity Zero Values Analysis")
print("="*80)

for i, traj in enumerate(data):
    temps = np.array(traj.get('temperature', []))
    sals = np.array(traj.get('salinity', []))
    positions = np.array(traj['points'])

    if len(temps) > 0 and len(sals) > 0:
        r = np.linalg.norm(positions, axis=1)
        depths = R_EARTH - r

        zero_temp = (np.abs(temps) < 1e-9)
        zero_sal = (np.abs(sals) < 1e-9)
        both_zero = zero_temp & zero_sal

        if np.any(both_zero):
            depth_range = (depths[both_zero].min(), depths[both_zero].max())
            print(f"\nParticle {i}:")
            print(f"  Zero values: {both_zero.sum()}/{len(temps)} points ({100.0*both_zero.sum()/len(temps):.1f}%)")
            print(f"  Depth range when T=S=0: {depth_range[0]:.1f}m - {depth_range[1]:.1f}m")
            print(f"  Time indices: {np.where(both_zero)[0][:10].tolist()} ...")
