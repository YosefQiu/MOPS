#!/usr/bin/env python3
"""
VTP Batch Checker - Analyze temperature and salinity ranges across multiple VTP files.

This script scans a directory for VTP (VTK PolyData) files and extracts statistical
information about temperature and salinity attributes, providing an overview of
data quality and value ranges across all files.
"""

import sys
import os
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def extract_ranges(vtp_file):
    """
    Extract temperature and salinity ranges from a VTP file.

    Args:
        vtp_file: Path to VTP file

    Returns:
        Dictionary with extracted statistics, or None if extraction fails
    """
    try:
        tree = ET.parse(vtp_file)
        root = tree.getroot()

        polydata = root.find('PolyData')
        if polydata is None:
            return None

        piece = polydata.find('Piece')
        if piece is None:
            return None

        point_data = piece.find('PointData')
        if point_data is None:
            return None

        num_points = int(piece.get('NumberOfPoints', 0))
        arrays = point_data.findall('DataArray')

        results = {'num_points': num_points}

        for array in arrays:
            name = array.get('Name', '')
            if name in ['temperature', 'salinity']:
                range_min = array.get('RangeMin')
                range_max = array.get('RangeMax')
                if range_min and range_max:
                    results[name] = {
                        'min': float(range_min),
                        'max': float(range_max)
                    }

        return results

    except Exception as e:
        print(f"Error processing {vtp_file}: {e}")
        return None


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Batch analyze temperature and salinity ranges in VTP files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/vtp/directory
  %(prog)s .                          # Check current directory
  %(prog)s --output report.txt ./data # Save report to file
        """
    )

    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='Directory containing VTP files (default: current directory)'
    )

    parser.add_argument(
        '-o', '--output',
        metavar='FILE',
        help='Save report to file instead of printing to console'
    )

    args = parser.parse_args()

    target_dir = args.directory

    if not os.path.exists(target_dir):
        print(f"Error: Directory not found: {target_dir}")
        sys.exit(1)

    # Find all VTP files
    vtp_files = sorted(Path(target_dir).glob("*.vtp"))

    if not vtp_files:
        print(f"Error: No VTP files found in {target_dir}")
        sys.exit(1)

    # Prepare output
    output_lines = []

    output_lines.append("=" * 80)
    output_lines.append(f"VTP File Analysis: {target_dir}")
    output_lines.append("=" * 80)
    output_lines.append(f"Found {len(vtp_files)} VTP file(s)")
    output_lines.append("")

    # Process each file
    all_results = []
    for vtp_file in vtp_files:
        results = extract_ranges(vtp_file)
        if results:
            all_results.append((vtp_file.name, results))

    if not all_results:
        print("Error: No valid data extracted from VTP files")
        sys.exit(1)

    # Print results in table format
    header = f"{'File':<50} {'Points':>8} {'Temp Min':>12} {'Temp Max':>12} {'Sal Min':>12} {'Sal Max':>12}"
    separator = f"{'-'*50} {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12}"

    output_lines.append(header)
    output_lines.append(separator)

    for filename, data in all_results:
        num_points = data['num_points']
        temp_min = data.get('temperature', {}).get('min', float('nan'))
        temp_max = data.get('temperature', {}).get('max', float('nan'))
        sal_min = data.get('salinity', {}).get('min', float('nan'))
        sal_max = data.get('salinity', {}).get('max', float('nan'))

        output_lines.append(
            f"{filename:<50} {num_points:>8} {temp_min:>12.6f} {temp_max:>12.6f} "
            f"{sal_min:>12.6f} {sal_max:>12.6f}"
        )

    # Calculate overall ranges
    output_lines.append("")
    output_lines.append("=" * 80)
    output_lines.append("Overall Ranges Across All Files:")
    output_lines.append("=" * 80)
    output_lines.append("")

    all_temp_min = min([d.get('temperature', {}).get('min', float('inf')) for _, d in all_results])
    all_temp_max = max([d.get('temperature', {}).get('max', float('-inf')) for _, d in all_results])
    all_sal_min = min([d.get('salinity', {}).get('min', float('inf')) for _, d in all_results])
    all_sal_max = max([d.get('salinity', {}).get('max', float('-inf')) for _, d in all_results])

    output_lines.append("Temperature:")
    output_lines.append(f"  Min:   {all_temp_min:.10f}")
    output_lines.append(f"  Max:   {all_temp_max:.10f}")
    output_lines.append(f"  Range: {all_temp_max - all_temp_min:.10f}")

    output_lines.append("")
    output_lines.append("Salinity:")
    output_lines.append(f"  Min:   {all_sal_min:.10f}")
    output_lines.append(f"  Max:   {all_sal_max:.10f}")
    output_lines.append(f"  Range: {all_sal_max - all_sal_min:.10f}")

    # Data quality warnings
    output_lines.append("")
    output_lines.append("=" * 80)
    output_lines.append("Data Quality Warnings:")
    output_lines.append("=" * 80)
    output_lines.append("")

    warnings = []
    if all_temp_min < 0:
        warnings.append(f"WARNING: Temperature has NEGATIVE values (min = {all_temp_min:.6f})")
    if all_sal_min < 0:
        warnings.append(f"WARNING: Salinity has NEGATIVE values (min = {all_sal_min:.6f})")
    if all_sal_max <= 0:
        warnings.append(f"WARNING: Salinity has NO POSITIVE values (max = {all_sal_max:.6f})")

    if warnings:
        for warning in warnings:
            output_lines.append(warning)
    else:
        output_lines.append("No data quality issues detected.")

    output_lines.append("")

    # Output results
    report = "\n".join(output_lines)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
