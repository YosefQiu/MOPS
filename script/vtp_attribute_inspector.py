#!/usr/bin/env python3
"""
VTP Attribute Inspector - Detailed analysis of attributes in VTP files.

This script provides comprehensive statistics and visualizations for temperature,
salinity, and other attributes in VTK PolyData (VTP) files. Supports both VTK
library reading and XML parsing as a fallback.
"""

import sys
import os
import argparse
import numpy as np

try:
    import vtk
    HAS_VTK = True
except ImportError:
    HAS_VTK = False


def check_vtp_with_vtk(vtp_file):
    """
    Check VTP file using VTK library (recommended method).

    Args:
        vtp_file: Path to VTP file

    Returns:
        Dictionary with attribute statistics
    """
    print(f"\n{'='*70}")
    print(f"Reading VTP file with VTK: {vtp_file}")
    print(f"{'='*70}\n")

    # Read VTP file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_file)
    reader.Update()

    polydata = reader.GetOutput()

    # Get point data
    point_data = polydata.GetPointData()
    num_arrays = point_data.GetNumberOfArrays()
    num_points = polydata.GetNumberOfPoints()
    num_lines = polydata.GetNumberOfLines()

    print(f"File Statistics:")
    print(f"  - Number of points: {num_points}")
    print(f"  - Number of lines: {num_lines}")
    print(f"  - Number of point data arrays: {num_arrays}")
    print()

    # List all available arrays
    print(f"Available Point Data Arrays:")
    array_names = []
    for i in range(num_arrays):
        array = point_data.GetArray(i)
        array_name = array.GetName()
        num_components = array.GetNumberOfComponents()
        num_tuples = array.GetNumberOfTuples()
        array_names.append(array_name)
        print(f"  [{i}] {array_name}: {num_components} component(s), {num_tuples} values")
    print()

    # Check for temperature and salinity
    stats = {}

    for attr_name in ['temperature', 'salinity', 'Temperature', 'Salinity', 'temp', 'sal']:
        array = point_data.GetArray(attr_name)
        if array is not None:
            print(f"\n{'='*70}")
            print(f"Analyzing: {attr_name}")
            print(f"{'='*70}")

            # Convert VTK array to numpy
            num_tuples = array.GetNumberOfTuples()
            num_components = array.GetNumberOfComponents()

            # Extract data
            data = []
            for i in range(num_tuples):
                if num_components == 1:
                    data.append(array.GetValue(i))
                else:
                    data.append([array.GetComponent(i, j) for j in range(num_components)])

            data = np.array(data)

            # Calculate statistics
            if num_components == 1:
                min_val = np.min(data)
                max_val = np.max(data)
                mean_val = np.mean(data)
                std_val = np.std(data)

                # Count special values
                nan_count = np.sum(np.isnan(data))
                inf_count = np.sum(np.isinf(data))
                zero_count = np.sum(data == 0.0)
                negative_count = np.sum(data < 0.0)

                print(f"\n  Components: {num_components} (scalar)")
                print(f"  Data points: {num_tuples}")
                print(f"\n  Statistics:")
                print(f"    Min:    {min_val:.6f}")
                print(f"    Max:    {max_val:.6f}")
                print(f"    Mean:   {mean_val:.6f}")
                print(f"    Std:    {std_val:.6f}")
                print(f"\n  Special Values:")
                print(f"    NaN:      {nan_count:6d} ({100*nan_count/num_tuples:.2f}%)")
                print(f"    Inf:      {inf_count:6d} ({100*inf_count/num_tuples:.2f}%)")
                print(f"    Zero:     {zero_count:6d} ({100*zero_count/num_tuples:.2f}%)")
                print(f"    Negative: {negative_count:6d} ({100*negative_count/num_tuples:.2f}%)")

                # Histogram
                print(f"\n  Distribution (10 bins):")
                if not np.all(np.isnan(data)) and not np.all(np.isinf(data)):
                    valid_data = data[np.isfinite(data)]
                    if len(valid_data) > 0:
                        hist, bin_edges = np.histogram(valid_data, bins=10)
                        for i in range(len(hist)):
                            bar = '█' * int(50 * hist[i] / np.max(hist))
                            print(f"    [{bin_edges[i]:8.2f} - {bin_edges[i+1]:8.2f}]: {hist[i]:6d} {bar}")

                stats[attr_name] = {
                    'min': min_val,
                    'max': max_val,
                    'mean': mean_val,
                    'std': std_val,
                    'nan_count': nan_count,
                    'inf_count': inf_count,
                    'zero_count': zero_count,
                    'negative_count': negative_count
                }
            else:
                # Vector data
                print(f"\n  Components: {num_components} (vector)")
                print(f"  Data points: {num_tuples}")
                for comp in range(num_components):
                    comp_data = data[:, comp]
                    print(f"\n  Component {comp}:")
                    print(f"    Min:  {np.min(comp_data):.6f}")
                    print(f"    Max:  {np.max(comp_data):.6f}")
                    print(f"    Mean: {np.mean(comp_data):.6f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")

    if stats:
        for name, stat in stats.items():
            print(f"\n{name}:")
            print(f"  Range: [{stat['min']:.6f}, {stat['max']:.6f}]")
            print(f"  Mean:  {stat['mean']:.6f} ± {stat['std']:.6f}")
            if stat['nan_count'] > 0 or stat['inf_count'] > 0:
                print(f"  WARNING: Contains {stat['nan_count']} NaN and {stat['inf_count']} Inf values")
    else:
        print("\nWARNING: No temperature or salinity data found!")
        print(f"Available arrays: {', '.join(array_names)}")

    print()
    return stats


def check_vtp_with_xml(vtp_file):
    """
    Check VTP file using XML parsing (fallback method).

    Args:
        vtp_file: Path to VTP file

    Returns:
        Dictionary with attribute statistics
    """
    print(f"\n{'='*70}")
    print(f"Reading VTP file with XML parser: {vtp_file}")
    print(f"{'='*70}\n")

    import xml.etree.ElementTree as ET

    tree = ET.parse(vtp_file)
    root = tree.getroot()

    # Find PolyData -> Piece -> PointData
    polydata = root.find('PolyData')
    if polydata is None:
        print("ERROR: Not a valid VTP file (no PolyData element)")
        return {}

    piece = polydata.find('Piece')
    if piece is None:
        print("ERROR: No Piece element found")
        return {}

    num_points = int(piece.get('NumberOfPoints', 0))
    num_lines = int(piece.get('NumberOfLines', 0))

    print(f"File Statistics:")
    print(f"  - Number of points: {num_points}")
    print(f"  - Number of lines: {num_lines}")
    print()

    point_data = piece.find('PointData')
    if point_data is None:
        print("ERROR: No PointData element found")
        return {}

    # List all arrays
    print(f"Available Point Data Arrays:")
    arrays = point_data.findall('DataArray')
    for i, array in enumerate(arrays):
        name = array.get('Name', 'Unknown')
        num_components = array.get('NumberOfComponents', '1')
        print(f"  [{i}] {name}: {num_components} component(s)")
    print()

    # Find temperature and salinity
    stats = {}
    for attr_name in ['temperature', 'salinity', 'Temperature', 'Salinity', 'temp', 'sal']:
        for array in arrays:
            if array.get('Name') == attr_name:
                print(f"\n{'='*70}")
                print(f"Analyzing: {attr_name}")
                print(f"{'='*70}")

                # Check for metadata in XML attributes
                range_min = array.get('RangeMin')
                range_max = array.get('RangeMax')
                data_format = array.get('format', 'ascii')
                data_type = array.get('type', 'Unknown')
                num_components = int(array.get('NumberOfComponents', '1'))

                print(f"\n  Data Type: {data_type}")
                print(f"  Format: {data_format}")
                print(f"  Components: {num_components}")

                # If data is in appended format, use metadata from XML
                if data_format == 'appended' and range_min and range_max:
                    min_val = float(range_min)
                    max_val = float(range_max)

                    print(f"\n  Range (from VTP metadata):")
                    print(f"    Min:    {min_val:.10f}")
                    print(f"    Max:    {max_val:.10f}")
                    print(f"    Spread: {max_val - min_val:.10f}")

                    # Warning for unusual values
                    if min_val < 0:
                        print(f"\n  WARNING: Minimum value is NEGATIVE: {min_val:.10f}")
                    if max_val == 0 and min_val < 0:
                        print(f"  WARNING: All values are NON-POSITIVE (max = 0)")

                    stats[attr_name] = {
                        'min': min_val,
                        'max': max_val,
                        'format': data_format
                    }

                # Try to parse inline ASCII data
                elif data_format == 'ascii':
                    data_text = array.text
                    if data_text:
                        values = [float(x) for x in data_text.strip().split()]
                        data = np.array(values)

                        # Calculate statistics
                        min_val = np.min(data)
                        max_val = np.max(data)
                        mean_val = np.mean(data)
                        std_val = np.std(data)

                        nan_count = np.sum(np.isnan(data))
                        inf_count = np.sum(np.isinf(data))
                        zero_count = np.sum(data == 0.0)
                        negative_count = np.sum(data < 0.0)

                        print(f"\n  Statistics:")
                        print(f"    Min:    {min_val:.10f}")
                        print(f"    Max:    {max_val:.10f}")
                        print(f"    Mean:   {mean_val:.10f}")
                        print(f"    Std:    {std_val:.10f}")
                        print(f"\n  Special Values:")
                        print(f"    NaN:      {nan_count:6d}")
                        print(f"    Inf:      {inf_count:6d}")
                        print(f"    Zero:     {zero_count:6d}")
                        print(f"    Negative: {negative_count:6d}")

                        stats[attr_name] = {
                            'min': min_val,
                            'max': max_val,
                            'mean': mean_val,
                            'std': std_val,
                            'format': data_format
                        }
                    else:
                        print(f"  WARNING: Array '{attr_name}' has no inline data")
                else:
                    print(f"  INFO: Data stored in {data_format} format (cannot parse without VTK)")
                    if range_min and range_max:
                        print(f"  Range (from metadata): [{range_min}, {range_max}]")

    # Summary
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")

    if stats:
        for name, stat in stats.items():
            print(f"\n{name}:")
            print(f"  Range: [{stat['min']:.10f}, {stat['max']:.10f}]")
            if 'mean' in stat:
                print(f"  Mean:  {stat['mean']:.10f} ± {stat['std']:.10f}")
            print(f"  Format: {stat['format']}")
    else:
        print("\nWARNING: No temperature or salinity data extracted!")

    print()
    return stats


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Inspect detailed attributes in VTP files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s trajectory.vtp
  %(prog)s data.vtp --xml-only

Notes:
  - Uses VTK library by default for full analysis
  - Falls back to XML parsing if VTK is unavailable
  - Analyzes temperature, salinity, and other attributes
        """
    )

    parser.add_argument(
        'vtp_file',
        help='Path to VTP file to inspect'
    )

    parser.add_argument(
        '--xml-only',
        action='store_true',
        help='Force XML parsing even if VTK is available'
    )

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.vtp_file):
        print(f"ERROR: File not found: {args.vtp_file}")
        sys.exit(1)

    # Check file size
    file_size = os.path.getsize(args.vtp_file)
    print(f"File: {args.vtp_file}")
    print(f"Size: {file_size / 1024:.2f} KB")

    # Use VTK if available and not forced to XML mode
    if HAS_VTK and not args.xml_only:
        try:
            check_vtp_with_vtk(args.vtp_file)
        except Exception as e:
            print(f"\nERROR using VTK: {e}")
            print("\nTrying XML parser instead...\n")
            check_vtp_with_xml(args.vtp_file)
    else:
        if args.xml_only:
            print("INFO: XML-only mode requested\n")
        else:
            print("WARNING: VTK not available, using XML parser\n")
        check_vtp_with_xml(args.vtp_file)


if __name__ == "__main__":
    main()
