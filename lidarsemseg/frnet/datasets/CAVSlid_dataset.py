from typing import Callable, List, Optional, Union

import numpy as np
from mmdet3d.datasets import Seg3DDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class CAVSlidSegDataset(Seg3DDataset):
    """NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        data_root (str, optional): Path of dataset root. Defaults to None.
        ann_file (str): Path of annotation file. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_prefix (dict): Prefix for training data. Defaults to
            dict(pts='', img='', pts_instance_mask='', pts_semantic_mask='').
        pipeline (List[dict or Callable]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input, it
            usually has following keys:

            - use_camera: bool
            - use_lidar: bool

            Defaults to dict(use_lidar=True, use_camera=False).
        ignore_index (int, optional): The label index to be ignored, e.g.
            unannotated points. If None is given, set to len(self.classes) to
            be consistent with PointSegClassMapping function in pipeline.
            Defaults to None.
        scene_idxs (str or np.ndarray, optional): Precomputed index to load
            data. For scenes with many points, we may sample it several times.
            Defaults to None.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
    """
    METAINFO = {
        'classes': [
            'Unlabeled',
            'Low Vegetation | Shrubs',
            'High Vegetation | Foliage',
            'Smooth Trail',
            'Rough Trail | Bumpy Road | Gravel',
            'Grass',
            'Obstacle |Fallen Tree Trunks, Rocks / boulders, etc.on the trail'
        ],
        'label_mapping': {
            0: {"category": "Unlabeled", "color": "#000000"},
            1: {"category": "Low Vegetation | Shrubs", "color": "#ECFA14"},
            2: {"category": "High Vegetation | Foliage", "color": "#FAAD14"},
            3: {"category": "Smooth Trail", "color": "#A6A6A6"},
            4: {"category": "Rough Trail | Bumpy Road | Gravel", "color": "#2E439A"},
            5: {"category": "Grass", "color": "#00FFFF"},
            6: {"category": "Obstacle |Fallen Tree Trunks, Rocks / boulders, etc.on the trail", "color": "#FF13A6"}
        },
        'palette': [  # RGB values for each class
            [0, 0, 0],        # Unlabeled - Black
            [236, 250, 20],   # Low Vegetation - Yellow
            [250, 173, 20],   # High Vegetation - Orange
            [166, 166, 166],  # Smooth Trail - Gray
            [46, 67, 154],    # Rough Trail - Blue
            [0, 255, 255],    # Grass - Cyan
            [255, 19, 166]    # Obstacle - Pink
        ],
        'seg_valid_class_ids': tuple(range(7)),
        'seg_all_class_ids': tuple(range(7))
    }

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     pts='',
                     img='',
                     pts_instance_mask='',
                     pts_semantic_mask=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 ignore_index: Optional[int] = None,
                 scene_idxs: Optional[Union[str, np.ndarray]] = None,
                 test_mode: bool = False,
                 **kwargs) -> None:
            super(CAVSlidSegDataset, self).__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            ignore_index=ignore_index,
            scene_idxs=scene_idxs,
            test_mode=test_mode,
            **kwargs)

    def get_seg_label_mapping(self, metainfo):
        seg_label_mapping = np.zeros(metainfo['max_label'] + 1, dtype=np.int64)
        for idx in metainfo['seg_label_mapping']:
            seg_label_mapping[idx] = metainfo['seg_label_mapping'][idx]
        return seg_label_mapping
