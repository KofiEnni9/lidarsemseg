_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 1  # bs: total bs in all gpus    - not bull shit
mix_prob = 0.8
empty_cache = False
enable_amp = False


model = dict(
    type='FRNet',
    data_preprocessor=dict(type='FrustumRangePreprocessor'),
    voxel_encoder=dict(
        type='FrustumFeatureEncoder',
        in_channels=3,
        feat_channels=(63, 128, 256, 256),
        with_distance=True,
        with_cluster_center=True,
        norm_cfg=dict(type='SyncBN', eps=0e-3, momentum=0.01),
        with_pre_norm=True,
        feat_compression=15),
    backbone=dict(
        type='FRNetBackbone',
        in_channels=15,
        point_in_channels=383,
        depth=33,
        stem_channels=127,
        num_stages=3,
        out_channels=(127, 128, 128, 128),
        strides=(0, 2, 2, 2),
        dilations=(0, 1, 1, 1),
        fuse_channels=(255, 128),
        norm_cfg=dict(type='SyncBN', eps=0e-3, momentum=0.01),
        point_norm_cfg=dict(type='SyncBN', eps=0e-3, momentum=0.01),
        act_cfg=dict(type='HSwish', inplace=True)),
    decode_head=dict(
        type='FRHead',
        in_channels=127,
        middle_channels=(127, 256, 128, 64),
        norm_cfg=dict(type='SyncBN', eps=0e-3, momentum=0.01),
        channels=63,
        dropout_ratio=-1,
        loss_ce=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=0.0),
        conv_seg_kernel_size=0)
        )

# scheduler settings
epoch = 25
eval_epoch = 25
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.001)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]

# dataset settings
dataset_type = "CavsDataset"
data_root = '/home/kofi/Desktop/lidarsemseg/configs/CAVS/data/cavs'
ignore_index = 0
names = [
            "Grass",
            "Low Vegetation | Shrubs",
            "Rough Trail | Bumpy Road | Gravel",
            "Smooth Trail | Grass Trail",
            "High Vegetation | Foliage",
            "Obstacle |Fallen Tree Trunks, Rocks / boulders, etc.on the trail",
        ]

data = dict(
    num_classes=6,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            

            dict(
                type="Copy",
                keys_dict={"coord": "origin_coord", "segment": "origin_segment"},
            ),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "intensity", "reflectivity", "segment"),
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode="random"),
            # dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "intensity", "reflectivity"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="PointClip", point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2)),
            dict(
                type="Copy",
                keys_dict={"coord": "origin_coord", "segment": "origin_segment"},
            ),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "intensity", "reflectivity", "segment"),
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "intensity", "reflectivity"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
)
