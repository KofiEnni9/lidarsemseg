"""
Visualization Utils

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import open3d as o3d
import numpy as np
import torch

def save_txt_and_pcd(points, colors, txt_path, pcd_path):
            # First save as txt
            point_cloud_data = np.hstack((points, colors))
            np.savetxt(txt_path, point_cloud_data, fmt='%.6f', 
                    header='x y z r g b', comments='')
            
            # Convert txt to pcd
            with open(txt_path, 'r') as txt_file:
                # Skip header line
                next(txt_file)
                
                # Write PCD header
                with open(pcd_path, 'w') as pcd_file:
                    # Count lines in txt file
                    num_points = sum(1 for line in txt_file)
                    
                    # Write PCD header
                    pcd_file.write("# .PCD v0.7 - Point Cloud Data file format\n")
                    pcd_file.write("VERSION 0.7\n")
                    pcd_file.write("FIELDS x y z rgb\n")
                    pcd_file.write("SIZE 4 4 4 4\n")
                    pcd_file.write("TYPE F F F F\n")
                    pcd_file.write("COUNT 1 1 1 1\n")
                    pcd_file.write(f"WIDTH {num_points}\n")
                    pcd_file.write("HEIGHT 1\n")
                    pcd_file.write("VIEWPOINT 0 0 0 1 0 0 0\n")
                    pcd_file.write(f"POINTS {num_points}\n")
                    pcd_file.write("DATA ascii\n")
                    
                    # Reset file pointer
                    txt_file.seek(0)
                    next(txt_file)  # Skip header again
                    
                    # Write points and colors
                    for line in txt_file:
                        x, y, z, r, g, b = map(float, line.strip().split())
                        # Convert RGB to float
                        rgb = int(r * 255) << 16 | int(g * 255) << 8 | int(b * 255)
                        pcd_file.write(f"{x} {y} {z} {rgb}\n")

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x


# def save_point_cloud(coord, color=None, file_path="pc.ply", logger=None):
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     coord = to_numpy(coord)
#     if color is not None:
#         color = to_numpy(color)
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(coord)
#     pcd.colors = o3d.utility.Vector3dVector(
#         np.ones_like(coord) if color is None else color
#     )
#     o3d.io.write_point_cloud(file_path, pcd)
#     if logger is not None:
#         logger.info(f"Save Point Cloud to: {file_path}")


