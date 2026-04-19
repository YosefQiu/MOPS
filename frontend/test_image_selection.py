#!/usr/bin/env python3
"""
Test image selection logic to ensure correct 5 images are chosen.
"""

# Mock manifest with 8 channel_map items
mock_manifest = {
    "channel_map": [
        {
            "output_index": 0,
            "channel": 0,
            "file": "output_0_ch0.png",
            "colorbar": "output_0_ch0_colorbar.png",
            "label": "Zonal Velocity",
            "quantity": "velocity_u",
        },
        {
            "output_index": 0,
            "channel": 1,
            "file": "output_0_ch1.png",
            "colorbar": "output_0_ch1_colorbar.png",
            "label": "Meridional Velocity",
            "quantity": "velocity_v",
        },
        {
            "output_index": 0,
            "channel": 2,
            "file": "output_0_ch2.png",
            "colorbar": "output_0_ch2_colorbar.png",
            "label": "Velocity Magnitude",
            "quantity": "velocity_speed",
        },
        {
            "output_index": 0,
            "channel": 3,
            "file": "output_0_ch3.png",
            "colorbar": "output_0_ch3_colorbar.png",
            "label": "Channel 3",
            "quantity": "channel_3",
        },
        {
            "output_index": 1,
            "channel": 0,
            "file": "output_1_ch0.png",
            "colorbar": "output_1_ch0_colorbar.png",
            "label": "Salinity",
            "quantity": "salinity",
        },
        {
            "output_index": 1,
            "channel": 1,
            "file": "output_1_ch1.png",
            "colorbar": "output_1_ch1_colorbar.png",
            "label": "Temperature",
            "quantity": "temperature",
        },
        {
            "output_index": 1,
            "channel": 2,
            "file": "output_1_ch2.png",
            "colorbar": "output_1_ch2_colorbar.png",
            "label": "Channel 2",
            "quantity": "channel_2",
        },
        {
            "output_index": 1,
            "channel": 3,
            "file": "output_1_ch3.png",
            "colorbar": "output_1_ch3_colorbar.png",
            "label": "Channel 3",
            "quantity": "channel_3",
        },
    ]
}


def test_image_selection():
    """Test the image selection logic"""
    channel_map = mock_manifest['channel_map']

    print("=" * 70)
    print("Testing Image Selection Logic")
    print("=" * 70)

    print(f"\nTotal images in manifest: {len(channel_map)}")

    # Method 1: By quantity (preferred)
    print("\n--- Method 1: Select by quantity name ---")
    useful_quantities = ['velocity_u', 'velocity_v', 'velocity_speed', 'salinity', 'temperature']
    selected_images = []

    for quantity in useful_quantities:
        for item in channel_map:
            if item.get('quantity') == quantity:
                selected_images.append(item)
                break

    print(f"Selected {len(selected_images)} images:")
    for i, item in enumerate(selected_images):
        print(f"  {i+1}. {item['file']:20s} - {item['label']:25s} ({item['quantity']})")

    # Verify correctness
    print("\n--- Verification ---")
    expected_files = [
        "output_0_ch0.png",
        "output_0_ch1.png",
        "output_0_ch2.png",
        "output_1_ch0.png",
        "output_1_ch1.png",
    ]

    expected_labels = [
        "Zonal Velocity",
        "Meridional Velocity",
        "Velocity Magnitude",
        "Salinity",
        "Temperature",
    ]

    all_correct = True
    for i, (expected_file, expected_label) in enumerate(zip(expected_files, expected_labels)):
        actual_file = selected_images[i]['file']
        actual_label = selected_images[i]['label']

        if actual_file == expected_file and actual_label == expected_label:
            print(f"✓ Image {i+1}: {actual_file} ({actual_label})")
        else:
            print(f"✗ Image {i+1}: Expected {expected_file}, got {actual_file}")
            all_correct = False

    print("\n" + "=" * 70)
    if all_correct:
        print("✅ All images selected correctly!")
    else:
        print("❌ Image selection has errors!")
    print("=" * 70)

    return all_correct


if __name__ == "__main__":
    success = test_image_selection()
    exit(0 if success else 1)
