from mmdet3d.datasets.transforms import LoadPointsFromFile, LoadAnnotations3D
from mmdet3d.registry import TRANSFORMS
import json
import numpy as np
import open3d as o3d

@TRANSFORMS.register_module()
class LoadCAVSlidPoints(LoadPointsFromFile):
    """Custom point cloud loader for PCD files"""
    def _load_points(self, pts_filename):
        pcd = o3d.io.read_point_cloud(pts_filename)
        points = np.asarray(pcd.points)  # [N, 3] for xyz
        
        # Get intensity and ring from custom fields if they exist
        # Note: These might be stored in different ways depending on how the PCD was saved
        intensity = np.asarray(pcd.colors)[:, 0] if pcd.has_colors() else np.zeros(len(points))
        ring = np.asarray(pcd.colors)[:, 1] if pcd.has_colors() else np.zeros(len(points))
        
        # Combine all features: xyz + intensity + ring
        points_with_features = np.column_stack([points, intensity, ring])
        return points_with_features

@TRANSFORMS.register_module()
class LoadCAVSlidAnnotations3D(LoadAnnotations3D):
    """Custom annotation loader for JSON files"""
    
    # Define the category to label mapping
    CATEGORY_MAP = {
        'Unlabeled': 0,
        'Low Vegetation | Shrubs': 1,
        'High Vegetation | Foliage': 2,
        'Smooth Trail': 3,
        'Rough Trail | Bumpy Road | Gravel': 4,
        'Grass': 5,
        'Obstacle |Fallen Tree Trunks, Rocks / boulders, etc.on the trail': 6
    }
    
    def get_num_points(self, results):
        """Get number of points in the point cloud."""
        if 'points' in results:
            return len(results['points'])
        # If points haven't been loaded yet, get it from the loaded point cloud
        pts_bytes = results['points_bytes']
        points = np.frombuffer(pts_bytes, dtype=np.float32)
        return len(points) // 5  # Divide by 5 since each point has 5 values (x,y,z,intensity,ring)
    
    def _load_semantic_seg_3d(self, results):
        pts_semantic_mask_path = results['pts_semantic_mask_path']
        
        # If it's a dict, extract the actual path
        if isinstance(pts_semantic_mask_path, dict):
            pts_semantic_mask_path = pts_semantic_mask_path.get('pts_semantic_mask_path')
        
        with open(pts_semantic_mask_path, 'r') as f:
            data = json.load(f)

        # Initialize labels array with zeros (unlabeled)
        num_points = self.get_num_points(results)
        labels = np.zeros(num_points, dtype=np.int32)
        
        # Fill in labels using the category mapping
        for segment in data['data']['result']:
            category = segment['attr']['category'][0]
            label = self.CATEGORY_MAP.get(category, 0)  # Default to 0 (Unlabeled) if category not found
            indices = segment['indexs']
            labels[indices] = label
            
        results['pts_semantic_mask'] = labels
        return results
