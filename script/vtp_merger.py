#!/usr/bin/env python3
"""
VTP Merger - Merge multiple VTP files into a single file.

This script combines multiple VTK PolyData (VTP) files containing particle
trajectories into a single VTP file. It preserves all point data attributes
(temperature, salinity, velocity, etc.) and maintains line connectivity.
"""

import vtk
import os
import glob
import argparse
import re
import sys

try:
    import vtk
    HAS_VTK = True
except ImportError:
    HAS_VTK = False


def read_vtp(file_path):
    """
    Read a VTP file and return its vtkPolyData.

    Args:
        file_path: Path to VTP file

    Returns:
        vtkPolyData object
    """
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()


def merge_vtp_files(vtp_files):
    """
    Merge polydata from multiple VTP files without linking points across files.
    Preserves all point data attributes (temperature, salinity, velocity, etc.).

    Args:
        vtp_files: List of VTP file paths to merge

    Returns:
        Merged vtkPolyData object
    """
    merged_poly_data = vtk.vtkPolyData()
    merged_points = vtk.vtkPoints()
    merged_cells = vtk.vtkCellArray()

    # Dictionary to hold merged attribute arrays
    # Key: attribute name, Value: vtkDataArray
    merged_attributes = {}

    point_offset = 0  # Point index offset for merged points

    for file_idx, file_path in enumerate(vtp_files):
        print(f"Merging: {file_path}")
        poly_data = read_vtp(file_path)
        num_points = poly_data.GetNumberOfPoints()
        num_cells = poly_data.GetNumberOfCells()

        if num_points == 0 or num_cells == 0:
            print(f"  WARNING: File is empty, skipping")
            continue

        # Copy points
        for i in range(num_points):
            merged_points.InsertNextPoint(poly_data.GetPoint(i))

        # Copy lines (cells) with updated point indices
        for i in range(num_cells):
            cell = poly_data.GetCell(i)
            point_ids = cell.GetPointIds()

            new_cell = vtk.vtkPolyLine()
            new_cell.GetPointIds().SetNumberOfIds(point_ids.GetNumberOfIds())

            for j in range(point_ids.GetNumberOfIds()):
                new_cell.GetPointIds().SetId(j, point_offset + point_ids.GetId(j))

            merged_cells.InsertNextCell(new_cell)

        # Copy all point data attributes
        point_data = poly_data.GetPointData()
        num_arrays = point_data.GetNumberOfArrays()

        for i in range(num_arrays):
            array = point_data.GetArray(i)
            array_name = array.GetName()
            num_components = array.GetNumberOfComponents()
            num_tuples = array.GetNumberOfTuples()

            # Initialize merged array on first file
            if array_name not in merged_attributes:
                # Create new array with same type and components
                if num_components == 1:
                    merged_array = vtk.vtkDoubleArray()
                elif num_components == 3:
                    merged_array = vtk.vtkDoubleArray()
                else:
                    merged_array = vtk.vtkDoubleArray()

                merged_array.SetName(array_name)
                merged_array.SetNumberOfComponents(num_components)
                merged_attributes[array_name] = merged_array
                print(f"  Found attribute: {array_name} ({num_components} components)")

            # Copy all values from this array
            merged_array = merged_attributes[array_name]
            for j in range(num_tuples):
                if num_components == 1:
                    value = array.GetValue(j)
                    merged_array.InsertNextValue(value)
                elif num_components == 3:
                    value = array.GetTuple3(j)
                    merged_array.InsertNextTuple3(value[0], value[1], value[2])
                else:
                    # General case for any number of components
                    value = [array.GetComponent(j, c) for c in range(num_components)]
                    merged_array.InsertNextTuple(value)

        point_offset += num_points

    # Set merged polydata
    merged_poly_data.SetPoints(merged_points)
    merged_poly_data.SetLines(merged_cells)

    # Add all merged attributes to point data
    for attr_name, attr_array in merged_attributes.items():
        merged_poly_data.GetPointData().AddArray(attr_array)
        print(f"Added attribute: {attr_name} with {attr_array.GetNumberOfTuples()} values")

    # Set the first scalar attribute as active (for default coloring in ParaView)
    if merged_attributes:
        first_scalar = None
        for name in ['temperature', 'velocity_mag', 'salinity', 'depth']:
            if name in merged_attributes:
                first_scalar = name
                break
        if first_scalar:
            merged_poly_data.GetPointData().SetActiveScalars(first_scalar)
            print(f"Set default coloring: {first_scalar}")

    return merged_poly_data


def write_vtp(output_path, poly_data):
    """
    Write the merged poly_data to a VTP file.

    Args:
        output_path: Output VTP file path
        poly_data: vtkPolyData object to write
    """
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(poly_data)
    writer.Write()


def extract_start_step(filename):
    """
    Extract the starting step number from filenames like
    PathLine_10101_to_10201_FORWARD.vtp for sorting.

    Args:
        filename: VTP filename

    Returns:
        Starting step number, or 0 if not found
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    # Match the first number in PathLine_10101_to_10201_FORWARD
    m = re.search(r'PathLine_(\d+)_to_', base)
    if m:
        return int(m.group(1))
    # If no match is found, return 0 to avoid crashing during sort
    return 0


def main():
    """Main function with argument parsing."""
    if not HAS_VTK:
        print("ERROR: VTK is not installed.")
        print("Install with: pip install vtk")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description='Merge multiple VTP trajectory files into one',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./data output_merged.vtp
  %(prog)s /path/to/vtp/files result.vtp --pattern "PathLine_*.vtp"

Notes:
  - Merges all point data attributes (temperature, salinity, velocity, etc.)
  - Preserves line connectivity within each file
  - Files are sorted by starting step number if filename matches PathLine_*_to_* pattern
        """
    )

    parser.add_argument(
        'input_folder',
        help='Folder containing VTP files to merge'
    )

    parser.add_argument(
        'output_file',
        help='Output merged VTP filename (.vtp)'
    )

    parser.add_argument(
        '-p', '--pattern',
        default='PathLine_*.vtp',
        metavar='PATTERN',
        help='Glob pattern for VTP files (default: PathLine_*.vtp)'
    )

    args = parser.parse_args()

    # Match files in the input folder
    pattern = os.path.join(args.input_folder, args.pattern)
    vtp_files = glob.glob(pattern)

    if not vtp_files:
        print(f"ERROR: No VTP files found")
        print(f"Pattern: {pattern}")
        print(f"Folder: {args.input_folder}")
        sys.exit(1)

    # Sort by the starting step (e.g., 10101, 10201, ...)
    vtp_files.sort(key=extract_start_step)

    print("=" * 70)
    print("VTP FILE MERGER")
    print("=" * 70)
    print(f"\nFound {len(vtp_files)} file(s) matching pattern:")
    print(f"  Pattern: {args.pattern}")
    print(f"  Folder:  {args.input_folder}")
    print(f"\nFiles to merge (sorted by start step):")
    for f in vtp_files:
        print(f"  {os.path.basename(f)}")
    print()

    # Perform merge
    print("=" * 70)
    print("MERGING FILES")
    print("=" * 70)
    print()
    merged_poly_data = merge_vtp_files(vtp_files)

    # Save result
    print()
    print("=" * 70)
    print("SAVING OUTPUT")
    print("=" * 70)
    write_vtp(args.output_file, merged_poly_data)

    print(f"\nMerge complete!")
    print(f"  Input files:  {len(vtp_files)}")
    print(f"  Output file:  {args.output_file}")
    print(f"  Total points: {merged_poly_data.GetNumberOfPoints()}")
    print(f"  Total lines:  {merged_poly_data.GetNumberOfLines()}")


if __name__ == "__main__":
    main()
