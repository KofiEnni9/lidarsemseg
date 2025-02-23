import open3d as o3d
import numpy as np  # Use np for consistency with numpy

# Read the PCD file
pcd = o3d.io.read_point_cloud('/home/kofi/Desktop/lidata/lidar_data/raw_data/3d_url/1574174675.pcd')

# To see all available attributes/columns
print("Available attributes:")
print("Points (x, y, z):")
print(pcd.points)  # This will show x, y, z coordinates

if pcd.colors:
    print("Colors (RGB):")
    print(pcd.colors)  # RGB values if present

if pcd.normals:
    print("Normals:")
    print(pcd.normals)  

output = open('/home/kofi/Desktop/lidarsemseg/lidarsemseg/output.txt', 'a')

with open('/home/kofi/Desktop/lidata/lidar_data/raw_data/3d_url/1574174728.pcd', 'r') as f:
    for line in f:
        output.write(line)

output.close()