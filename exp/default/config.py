weight = None
resume = False
evaluate = True
test_only = False
seed = 6549682
save_path = 'exp/default'
num_worker = 6
batch_size = 2
batch_size_val = None
batch_size_test = None
epoch = 25
eval_epoch = 25
clip_grad = 1.0
sync_bn = False
enable_amp = False
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
mix_prob = 0.8
param_dicts = [dict(keyword='block', lr=0.0002)]
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='SemSegEvaluator'),
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='PreciseEvaluator', test_last=False)
]
train = dict(type='DefaultTrainer')
test = dict(type='SemSegTester', verbose=True)
model = dict(
    type='DefaultSegmentorV2',
    num_classes=6,
    backbone_out_channels=64,
    backbone=dict(
        type='PT-v3m1',
        in_channels=5,
        order=['z', 'z-trans', 'hilbert', 'hilbert-trans'],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1, ignore_index=-1),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1,
            ignore_index=-1)
    ])
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.001)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.002, 0.0002],
    pct_start=0.04,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=100.0)
dataset_type = 'CavsDataset'
data_root = '/home/kofi/Desktop/lidarsemseg/configs/CAVS/data/cavs'
ignore_index = -1
names = [
    'Grass', 'Low Vegetation | Shrubs', 'Rough Trail | Bumpy Road | Gravel',
    'Smooth Trail | Grass Trail', 'High Vegetation | Foliage',
    'Obstacle |Fallen Tree Trunks, Rocks / boulders, etc.on the trail'
]
data = dict(
    num_classes=6,
    ignore_index=-1,
    names=[
        'Grass', 'Low Vegetation | Shrubs',
        'Rough Trail | Bumpy Road | Gravel', 'Smooth Trail | Grass Trail',
        'High Vegetation | Foliage',
        'Obstacle |Fallen Tree Trunks, Rocks / boulders, etc.on the trail'
    ],
    train=dict(
        type='CavsDataset',
        split='train',
        data_root='/home/kofi/Desktop/lidarsemseg/configs/CAVS/data/cavs',
        transform=[
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(
                type='PointClip',
                point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2)),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            dict(
                type='Copy',
                keys_dict=dict(coord='origin_coord',
                               segment='origin_segment')),
            dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'intensity', 'reflectivity', 'segment'),
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=('coord', 'intensity', 'reflectivity'))
        ],
        test_mode=False,
        ignore_index=-1,
        loop=1),
    val=dict(
        type='CavsDataset',
        split='val',
        data_root='/home/kofi/Desktop/lidarsemseg/configs/CAVS/data/cavs',
        transform=[
            dict(
                type='PointClip',
                point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2)),
            dict(
                type='Copy',
                keys_dict=dict(coord='origin_coord',
                               segment='origin_segment')),
            dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'intensity', 'reflectivity', 'segment'),
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=('coord', 'intensity', 'reflectivity'))
        ],
        test_mode=False,
        ignore_index=-1),
    test=dict(
        type='CavsDataset',
        split='val',
        data_root='/home/kofi/Desktop/lidarsemseg/configs/CAVS/data/cavs',
        transform=[
            dict(
                type='PointClip',
                point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2)),
            dict(
                type='Copy',
                keys_dict=dict(coord='origin_coord',
                               segment='origin_segment')),
            dict(
                type='GridSample',
                grid_size=0.025,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'intensity', 'reflectivity', 'segment'),
                return_inverse=True)
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='test',
                return_grid_coord=True,
                keys=('coord', 'intensity', 'reflectivity')),
            crop=None,
            post_transform=[
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'grid_coord', 'index', 'origin_coord'),
                    feat_keys=('coord', 'intensity', 'reflectivity'))
            ],
            aug_transform=[[{
                'type': 'RandomRotateTargetAngle',
                'angle': [0],
                'axis': 'z',
                'center': [0, 0, 0],
                'p': 1
            }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [0.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }]]),
        ignore_index=-1))
