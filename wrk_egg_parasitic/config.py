
img_scale = (640, 640)
model = dict(
    type='YOLOX',
    input_size=(640, 640),
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=1.33, widen_factor=1.25),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[320, 640, 1280],
        out_channels=320,
        num_csp_blocks=4),
    bbox_head=dict(
        type='YOLOXHead', num_classes=80, in_channels=320, feat_channels=320),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))
data_root = '/home/dgxuser27/icip/Chula-Egg/'
dataset_type = 'CocoDataset'
train_pipeline = [
    dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
    dict(
        type='RandomAffine', scaling_ratio_range=(0.1, 2),
        border=(-320, -320)),
    dict(
        type='MixUp',
        img_scale=(640, 640),
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
classes = ('Ascaris lumbricoides', 'Capillaria philippinensis',
           'Enterobius vermicularis', 'Fasciolopsis buski', 'Hookworm egg',
           'Hymenolepis diminuta', 'Hymenolepis nana',
           'Opisthorchis viverrine', 'Paragonimus spp', 'Taenia spp. egg',
           'Trichuris trichiura')
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type='CocoDataset',
        ann_file='/home/dgxuser27/icip/Chula-Egg/labeled/train/labels.json',
        img_prefix='/home/dgxuser27/icip/Chula-Egg/labeled/data/',
        classes=('Ascaris lumbricoides', 'Capillaria philippinensis',
                 'Enterobius vermicularis', 'Fasciolopsis buski',
                 'Hookworm egg', 'Hymenolepis diminuta', 'Hymenolepis nana',
                 'Opisthorchis viverrine', 'Paragonimus spp',
                 'Taenia spp. egg', 'Trichuris trichiura'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False),
    pipeline=[
        dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
        dict(
            type='RandomAffine',
            scaling_ratio_range=(0.1, 2),
            border=(-320, -320)),
        dict(
            type='MixUp',
            img_scale=(640, 640),
            ratio_range=(0.8, 1.6),
            pad_val=114.0),
        dict(type='YOLOXHSVRandomAug'),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
        dict(
            type='Pad',
            pad_to_square=True,
            pad_val=dict(img=(114.0, 114.0, 114.0))),
        dict(
            type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=6,
    persistent_workers=True,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            ann_file='/home/dgxuser27/icip/Chula-Egg/labeled/train/labels.json',
            img_prefix='/home/dgxuser27/icip/Chula-Egg/labeled/data/',
            classes=('Ascaris lumbricoides', 'Capillaria philippinensis',
                     'Enterobius vermicularis', 'Fasciolopsis buski',
                     'Hookworm egg', 'Hymenolepis diminuta',
                     'Hymenolepis nana', 'Opisthorchis viverrine',
                     'Paragonimus spp', 'Taenia spp. egg',
                     'Trichuris trichiura'),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False),
        pipeline=[
            dict(
                type='RandomAffine',
                scaling_ratio_range=(0.1, 2),
                border=(-320, -320)),
            dict(
                type='MixUp',
                img_scale=(640, 640),
                ratio_range=(0.8, 1.6),
                pad_val=114.0),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(
                type='FilterAnnotations',
                min_gt_bbox_wh=(1, 1),
                keep_empty=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        ann_file='/home/dgxuser27/icip/Chula-Egg/labeled/test/test.json',
        img_prefix='/home/dgxuser27/icip/Chula-Egg/labeled/data/',
        classes=('Ascaris lumbricoides', 'Capillaria philippinensis',
                 'Enterobius vermicularis', 'Fasciolopsis buski',
                 'Hookworm egg', 'Hymenolepis diminuta', 'Hymenolepis nana',
                 'Opisthorchis viverrine', 'Paragonimus spp',
                 'Taenia spp. egg', 'Trichuris trichiura'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        pad_to_square=True,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file=
        '/home/dgxuser27/icip/Chula-Egg/unlabeled/annotations/test.json',
        img_prefix='/home/dgxuser27/icip/Chula-Egg/unlabeled/data/',
        classes=('Ascaris lumbricoides', 'Capillaria philippinensis',
                 'Enterobius vermicularis', 'Fasciolopsis buski',
                 'Hookworm egg', 'Hymenolepis diminuta', 'Hymenolepis nana',
                 'Opisthorchis viverrine', 'Paragonimus spp',
                 'Taenia spp. egg', 'Trichuris trichiura'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        pad_to_square=True,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
max_epochs = 200
num_last_epochs = 15
interval = 10
evaluation = dict(
    save_best='auto', interval=10, dynamic_intervals=[(285, 1)], metric='bbox')
work_dir = 'wrk_egg_parasatic'
auto_resume = False
gpu_ids = range(0, 4)
