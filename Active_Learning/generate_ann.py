import os
import csv
import re
 # Script to extract the filename correspondences from the dataset
def get_sort_key(filename):
    # Extract the last numerical value from the filename
    match = re.search(r'_(\d+)\.png$', filename)
    if match:
        return int(match.group(1))
    return -1

def collect_segmentation_maps(root_dir, csv_file):
    # List to store the paths of segmentation maps
    segmentation_maps = []

    # Traverse the root directory
    for sequence in sorted(os.listdir(root_dir)):
        sequence_path = os.path.join(root_dir, sequence)
        segmentation_maps_dir = os.path.join(sequence_path, 'segm_maps')

        # Check if the segmentation maps directory exists
        if os.path.exists(segmentation_maps_dir):
            images = sorted(os.listdir(segmentation_maps_dir), key=get_sort_key)
            for img_file in images:
                img_path = os.path.join(segmentation_maps_dir, img_file)
                if os.path.isfile(img_path):  # Ensure it's a file
                    segmentation_maps.append(img_path)

    # Write the paths to a CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image_Path'])  # Write header
        for img_path in segmentation_maps:
            writer.writerow([img_path])

# Specify the root directory and the CSV file name
root_dir = 'valid/'  # Change this to your root directory
csv_file = 'valid_segmentation_maps.csv'

# Run the function
collect_segmentation_maps(root_dir, csv_file)

print(f"CSV file '{csv_file}' created with paths of segmentation maps.")
