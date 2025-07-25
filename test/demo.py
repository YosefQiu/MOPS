import sys
sys.path.append("/pscratch/sd/q/qiuyf/MOPS_2/build_python/tools/pyMOPS")
import argparse
import pyMOPS

import vis


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", dest="yaml", type=str, default="../mpas.yaml")
    parser.add_argument("-t", "--timestep", dest="timestep", type=int, default="0")
    args = parser.parse_args()
    
    print("Available API:", dir(pyMOPS))
    print("\n\n")
    
    # Initialize MOPS with GPU support
    pyMOPS.MOPS_Init("gpu")
    
    # Create and initialize MPAS ocean grid from YAML configuration
    gridMesh = pyMOPS.MPASOGrid()
    gridMesh.init_from_yaml(args.yaml)
    
    # Create and initialize solution attributes from YAML
    attributes = pyMOPS.MPASOSolution()
    attributes.init_from_yaml(args.yaml, args.timestep)
    # Add temperature and salinity as float attributes
    attributes.add_attribute("temperature", pyMOPS.AttributeFormat.kFloat)
    attributes.add_attribute("salinity", pyMOPS.AttributeFormat.kFloat)
    
    # Set up MOPS simulation with grid and attributes
    pyMOPS.MOPS_Begin()
    pyMOPS.MOPS_AddGridMesh(gridMesh)
    pyMOPS.MOPS_AddAttribute(args.timestep, attributes)
    pyMOPS.MOPS_End()
    
    # Activate attributes for the specified timestep
    pyMOPS.MOPS_ActiveAttribute(args.timestep)
    
    # Configure visualization settings for remapping
    config = pyMOPS.VisualizationSettings()
    config.imageSize = (3601, 1801)  # High resolution output
    config.LatRange = (-90, 90)      # Full latitude range
    config.LonRange = (-180, 180)    # Full longitude range
    config.FixedDepth = 10.0         # Fixed depth for visualization
    config.TimeStep = args.timestep
    
    # Run remapping and visualize results
    imgs = pyMOPS.MOPS_RunRemapping(config)
    
    
    
    
    print(imgs[0].shape)  # [181, 361, 4]
    print(imgs[1].shape)  # [181, 361, 4]
    
    vis.vis_image(imgs)
    
    # Configure sampling settings for trajectory analysis
    conf = pyMOPS.SeedsSettings()
    conf.setSeedsRange((31, 31))  # Set sampling grid size
    conf.setGeoBox((35.0, 45.0), (-90.0, -45.0))  # Set geographical bounding box
    conf.setDepth(10.0)  # Set seeds' depth (unit: meters)

    # Generate sample points for trajectory analysis
    sample_pts = pyMOPS.MOPS_GenerateSamplePoints(conf)  # Returns np.ndarray
    
    
    vis.plot_2d_latlon_cartopy(sample_pts)
    
    # Configure trajectory settings
    traj_conf = pyMOPS.TrajectorySettings()
    traj_conf.depth = conf.getDepth()
    traj_conf.deltaT = 3600                     # Time step: 1 hour
    traj_conf.duration = 3600 * 24 * 30 * 2     # Time duration: 2 months
    traj_conf.recordT = 3600 * 2                # Record interval: every 2 hours
    
    # Input: sample_pts: np.ndarray, shape: (N, 3)
    lines = pyMOPS.MOPS_RunStreamLine(traj_conf, sample_pts) 
    print(f"Generated {len(lines)} trajectories")
    
    
    
    
    vis.plot_2d_trajectories(lines)

    
    
  
