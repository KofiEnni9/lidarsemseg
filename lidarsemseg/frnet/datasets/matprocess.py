# 0	0	0	0
# 0	0	0	0
# -14.906012	1.6530291	-2.2951910	1
# -15.988307	1.1794561	-2.6049051	1
# -18.929214	0.69710088	-3.2568810	1
# -1.3706257	1.4909458e-06	-0.24866483	4
# -1.3130785	0.14561622	-0.25228152	4
# -1.0580093	0.078049257	-0.21271738	4
# -1.1139100	0.041021649	-0.23418573	4
# 0	0	0	0
# -1.0704627	0.11871090	-0.24706735	4
# 0	0	0	0
# 0	0	0	0
# 0	0	0	0

# >> gTruth.LabelDefinitions

# ans =

#   6×6 table

#           Name          Type            LabelColor            Group      Description    VoxelLabelID
#     ________________    _____    ________________________    ________    ___________    ____________

#     {'low_veg'     }    Voxel    {[     0.6314 1 0.6314]}    {'None'}    {0×0 char}        {[1]}    
#     {'high_veg'    }    Voxel    {[0.1373 0.4392 0.1373]}    {'None'}    {0×0 char}        {[2]}    
#     {'smooth_trail'}    Voxel    {[0.6510 0.6510 0.6510]}    {'None'}    {0×0 char}        {[3]}    
#     {'rough_trail' }    Voxel    {[0.1804 0.2627 0.6000]}    {'None'}    {0×0 char}        {[4]}    
#     {'grass'       }    Voxel    {[               0 1 1]}    {'None'}    {0×0 char}        {[5]}    
#     {'obstacles'   }    Voxel    {[     1 0.0745 0.6510]}    {'None'}    {0×0 char}        {[6]}    

# >> gTruth.DataSource

#   PointCloudSequenceSource with properties:

#             Name: "Point Cloud Sequence"
#      Description: "A Point Cloud sequence reader"
#       SourceName: "C:\Users\kofie\Desktop\pendrive\ple\m_point"
#     SourceParams: [1×1 struct]
#       SignalName: "m_point"
#       SignalType: PointCloud
#        Timestamp: {[1133×1 duration]}
#       NumSignals: 1

# >> 

import os
from os.path import dirname, join as pjoin
import scipy.io as sio
import json
import uuid
import time

def convert_mat_to_json(mat_file_path, output_file_path):
    # Load the .mat file
    data = sio.loadmat(mat_file_path)
    
    # Create the base JSON structure
    json_output = {
        "data": {
            "result": []
        }
    }
    
    # Define label categories and their colors
    label_mapping = {
        0: {"category": "Unlabeled", "color": "#000000"},
        1: {"category": "Low Vegetation | Shrubs", "color": "#ECFA14"},
        2: {"category": "High Vegetation | Foliage", "color": "#FAAD14"},
        3: {"category": "Smooth Trail", "color": "#A6A6A6"},
        4: {"category": "Rough Trail | Bumpy Road | Gravel", "color": "#2E439A"},
        5: {"category": "Grass", "color": "#00FFFF"},
        6: {"category": "Obstacle |Fallen Tree Trunks, Rocks \/ boulders, etc.on the trail", "color": "#FF13A6"}
    }
    
    # Group points by their label (last column)
    point_groups = {}
    for i, row in enumerate(data['L']):
        label = int(row[3])

        if label not in point_groups:
            point_groups[label] = []
        
        point_groups[label].append(i)
    # print(point_groups)


    for label, points in point_groups.items():
            segment = {
                "index": str(uuid.uuid4()),
                "id": str(uuid.uuid4()),
                "attr": {
                    "category": [label_mapping[label]["category"]],
                    "color": label_mapping[label]["color"],
                    "label": [label_mapping[label]["category"]],
                    "code": [""]
                },
                "type": "pcl_segment",
                "indexs": list(points),

            }
            json_output["data"]["result"].append(segment)
    
    # Write to JSON file
    with open(output_file_path, 'w') as f:
        json.dump(json_output, f, indent=2)

# Process all .mat files in the directory
input_directory = r'C:\Users\kofie\Desktop\pendrive\VoxelLabelData'
for filename in sorted(os.listdir(input_directory)):
    if filename.endswith('.mat'):
        mat_file_path = os.path.join(input_directory, filename)
        # Create output filename by replacing .mat extension with .json

        output_filename = filename.replace('.mat', '.json')
        dir = os.path.join(input_directory, 'output')
        os.makedirs(dir, exist_ok=True)
        output_file_path = os.path.join(dir, output_filename)
        
        try:
            convert_mat_to_json(mat_file_path, output_file_path)
            print(f"Processed {filename} successfully")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
