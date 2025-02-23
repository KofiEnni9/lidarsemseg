import os
import sys
from typing import Dict, Any

import numpy as np

# Assuming the CAVSDataset is in a module called datasets
from cavslid import CAVSDataset

def test_dataset(data_root: str):
    """
    Comprehensive test of the CAVSDataset class
    
    Args:
        data_root (str): Root directory of your point cloud dataset
    """
    print("Testing CAVSDataset...")
    
    # Test basic initialization
    try:
        dataset = CAVSDataset(
            data_root=data_root,
            split=['train'],  # or whatever split you have
            timestamp=(0,)
        )
        print(f"Total number of data points: {len(dataset.data_list)}")
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # Test data retrieval
    def print_data_info(data: Dict[str, Any]):
        print("\nData Point Information:")
        for key, value in data.items():
            if isinstance(value, (np.ndarray, list)):
                print(f"{key}: shape {value.shape if hasattr(value, 'shape') else len(value)}")
            else:
                print(f"{key}: {value}")

    # Test single frame retrieval
    try:
        print("\nTesting single frame retrieval...")
        single_frame = dataset.get_single_frame(0)
        print_data_info(single_frame)
    except Exception as e:
        print(f"Single frame retrieval failed: {e}")

    # Test multi-frame retrieval
    try:
        print("\nTesting multi-frame retrieval...")
        multi_frame = dataset.get_data(0)
        print_data_info(multi_frame)
    except Exception as e:
        print(f"Multi-frame retrieval failed: {e}")

    # Test data name retrieval
    try:
        print("\nTesting data name retrieval...")
        data_name = dataset.get_data_name(0)
        print(f"Data Name: {data_name}")
    except Exception as e:
        print(f"Data name retrieval failed: {e}")

    # Optional: Iterate through a few data points
    try:
        print("\nIterating through first 5 data points:")
        for i in range(min(5, len(dataset.data_list))):
            print(f"\nData point {i}:")
            data = dataset.get_data(i)
            print_data_info(data)
    except Exception as e:
        print(f"Iteration failed: {e}")

def main():
    # Modify this path to your actual dataset root
    data_root = r"C:\Users\kofie\Desktop\lidata\lidar_data\labeled_data\Set1-3D.zip\Set1\3d_url\1574174638.json"
    
    # Check if path exists
    if not os.path.exists(data_root):
        print(f"Error: Dataset root path does not exist: {data_root}")
        print("Please provide the correct path to your dataset")
        return

    test_dataset(data_root)

if __name__ == "__main__":
    main()