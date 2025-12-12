"""
This script renames ImageNet class folders using a 5-digit numeric index
(e.g., 00000, 00001, ...). It reads a mapping file containing WNIDs,
then renames each corresponding folder under the dataset root directory.
"""

import os

# Set the ImageNet dataset root directory
imagenet_root = ""  # Your ImageNet dataset root path
mapping_file = ""   # Your class-mapping file path

# Read the mapping file to load WNID → index order
wnid_list = []
with open(mapping_file, "r") as f:
    for line in f:
        wnid = line.strip().split()[0]  # Extract the WNID part
        wnid_list.append(wnid)

# Iterate through dataset folders and rename them
for i, wnid in enumerate(wnid_list):
    old_path = os.path.join(imagenet_root, wnid)  # Original folder path
    new_name = f"{i:05d}"  # New 5-digit class name (e.g., 00000, 00001)
    new_path = os.path.join(imagenet_root, new_name)  # Renamed folder path
    
    if os.path.isdir(old_path):  # Ensure the class folder exists
        os.rename(old_path, new_path)
        print(f"✅ {wnid} -> {new_name}")
    else:
        print(f"⚠️ Skipped {wnid}, folder not found")

print("🎉 All class folders have been successfully renamed!")
