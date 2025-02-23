import os
from pathlib import Path
import shutil
import numpy as np

import numpy as np
import open3d as o3d
import json
import os
from pathlib import Path

class LidarDataProcessor:
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.categories = [
            "Grass",
            "Low Vegetation | Shrubs",
            "Rough Trail | Bumpy Road | Gravel",
            "Smooth Trail | Grass Trail",
            "High Vegetation | Foliage",
            "Obstacle |Fallen Tree Trunks, Rocks / boulders, etc.on the trail",
        ]
    

    def load_point_cloud(self, pcd_path):
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        # Extract x, y, z coordinates
        return np.asarray(pcd.points)
    
    def load_pc_intensity(self, pcd_path):
        # Read the PCD file manually to get intensity (column 4)
        with open(pcd_path, 'r') as f:
            # Skip header lines
            for _ in range(11):
                next(f)
            # Read data lines
            points = []
            for line in f:
                if line.strip():
                    values = line.strip().split()
                    points.append(float(values[3]))  # intensity is 4th column
        return np.array(points, dtype=np.float32)
    
    def load_pc_reflectivity(self, pcd_path):
        # Read the PCD file manually to get reflectivity (column 6)
        with open(pcd_path, 'r') as f:
            # Skip header lines
            for _ in range(11):
                next(f)
            # Read data lines
            points = []
            for line in f:
                if line.strip():
                    values = line.strip().split()
                    points.append(float(values[5]))  # reflectivity is 6th column
        return np.array(points, dtype=np.float32)
    
    def load_labels(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data['result']['data']
    
    def create_label_mask(self, points, segments):
        labels = np.zeros(points.shape[0], dtype=np.int32)
        # Initialize counter dictionary
        segment_counts = {name: 0 for name in self.categories}
        
        for segment in segments:
            category = segment['attr']['category'][0]
            # print(category)
            label_id = self.categories.index(category) if category in self.categories else -1 
            if label_id != -1:
                indices = segment['indexs']
                labels[indices] = label_id
                # Count points for this segment
                segment_counts[category] += len(indices)
        
        # Print the counts
        print("\nPoint distribution across segments:")
        total_points = points.shape[0]
        for category, count in segment_counts.items():
            percentage = (count / total_points) * 100
            print(f"{category}: {count} points ({percentage:.2f}%)")
        
        return labels
    
    def label_color(self, points, segments):
        # Initialize colors array with zeros
        colors = np.zeros((len(points), 3), dtype=np.float32)  # Change to 3 channels for RGB
        
        for segment in segments:
            color_hex = segment['attr']['color']
            indices = segment['indexs']
            
            # Convert hex color to RGB
            if isinstance(color_hex, str) and color_hex.startswith('#'):
                color_rgb = [int(color_hex[i:i+2], 16) / 255.0 for i in (1, 3, 5)]  # Normalize to [0, 1]
                colors[indices] = color_rgb
                
        return colors
    
    def normalize_points(self, points):
        # Center the points
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # Scale to unit sphere
        max_distance = np.max(np.linalg.norm(points, axis=1))
        points = points / max_distance
        
        return points
    
    def process_single_sample(self, pcd_path, json_path):
        # Load point cloud
        points = self.load_point_cloud(pcd_path)
        
        # Load and process labels
        segments = self.load_labels(json_path)
        labels = self.create_label_mask(points, segments)
        
        # Get color labels
        colors = self.label_color(points, segments)
        
        # Load intensity and reflectivity
        intensity = self.load_pc_intensity(pcd_path)
        reflectivity = self.load_pc_reflectivity(pcd_path)
        
        # Normalize points
        points_normalized = self.normalize_points(points)
        
        return {
            'coord': points_normalized.astype(np.float32),
            'segment': labels.astype(np.int32),
            # 'origin_coord': points.astype(np.float32),
            'color': colors.astype(np.int32),
            'intensity': intensity.astype(np.float32),
            'reflectivity': reflectivity.astype(np.float32),
        }
    
    def save_processed_data(self, output_dir, data_dict, name):
        output_path = Path(output_dir) / name
        output_path.mkdir(parents=True, exist_ok=True)
        
        for key, value in data_dict.items():
            np_path = output_path / f"{key}.npy"
            np.save(str(np_path), value) 




def process_dataset(labeled_data_dir, raw_data_dir, output_dir):
    processor = LidarDataProcessor(labeled_data_dir)
    
    # Create output directories
    output_dir = Path(output_dir)
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    
    for directory in [train_dir, val_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Fix the glob pattern
    label_files = list(Path(labeled_data_dir).glob('**/*.json'))
    
    # Process each file
    for idx, label_file in enumerate(label_files):
        try:
            # Simplified path handling
            json_path = label_file

            pcd_name = json_path.stem
            pcd_path = Path(raw_data_dir) / f"{pcd_name}.pcd"

            if not pcd_path.exists():
                print(f"Skipping {pcd_path.name} - no corresponding PCD file")
                continue

            # Process the sample
            data_dict = processor.process_single_sample(pcd_path, json_path)

            # Split into train/val (80/20 split)
            output_subdir = train_dir if idx % 5 != 0 else val_dir
            processor.save_processed_data(output_subdir, data_dict, pcd_path.stem)
            
            print(f"Processed {idx+1}/{len(label_files)}: {pcd_path.name}")
            
        except Exception as e:
            print(f"Error processing {label_file.name}: {str(e)}")
            continue

if __name__ == "__main__":

    # raw pcd data
    # PCD file columns: FIELDS x y z intensity t reflectivity ring noise range    
    raw_data_dir = "/home/kofi/Desktop/lidata/lidar_data/raw_data/3d_url"

    labeled_data_dirs = [
        '/home/kofi/Desktop/lidata/lidar_data/labeled_data/Set2-3D.zip/Set2/3d_url',
        '/home/kofi/Desktop/lidata/lidar_data/labeled_data/Set3-3D.zip/Set3/3d_url',
        '/home/kofi/Desktop/lidata/lidar_data/labeled_data/Set1-3D.zip/Set1/3d_url'
    ]

    output_dir = "/home/kofi/Desktop/lidarsemseg/configs/CAVS/data/cavs"

    for labeled_data_dir in labeled_data_dirs:
        process_dataset(labeled_data_dir, raw_data_dir, output_dir) 