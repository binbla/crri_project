# ========================================================================
datasets_root = '/data2/datasets/'
classwise = True
format_only = False
backend_args = None
use_mp_eval = True

# ========================================================================

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', direction='horizontal', prob=0.5),
    dict(type='RetinexNorm', prob=0.5, sigma_list=[15, 80, 250]),

    dict(type='RandomChoiceResize', scales=[(1333, 720), (1333, 760), (1333, 800)],
         keep_ratio=True),

    dict(type='RandomApply',
         transforms=[dict(type='SCGTransform', v1_scale=0.005, v2_scale=0.7)],
         prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(type='Contrast', min_mag=0.6, max_mag=1.4),
                dict(type='Brightness', min_mag=0.6, max_mag=1.4),
                dict(type='Saturation', min_mag=0.6, max_mag=1.4)
            ],
            [dict(type='Saturation', min_mag=0, max_mag=0)],
            [],
        ],
        prob=[0.8, 0.1, 0.1]),
    dict(
        type='RandomChoice',
        transforms=[

            [dict(type='Blur', sigma=(0.1, 2.0))],

            [dict(type='StructureAugment', prob=1.0, alpha_range=(0.5, 1.0), mode='sobel')],

            [],
        ],
        prob=[0.4, 0.4, 0.2]),
    dict(
        type='RandomErasing',
        n_patches=(1, 3),
        ratio=(0.02, 0.2),
        squared=True,
        bbox_erased_thr=0.9,
        img_border_value=128),
    dict(type='PackDetInputs'),
]

train_pipeline_clean = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', direction='horizontal', prob=0.5),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# ========================================================================

voc2007 = dict(type='VOCDataset',
               data_root=datasets_root + '/PASCAL_VOC/',
               ann_file='VOC2007/ImageSets/Main/trainval.txt',
               data_prefix=dict(sub_data_root='VOC2007/'),
               filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
               pipeline=train_pipeline,
               backend_args=backend_args)
voc2012 = dict(type='VOCDataset',
               data_root=datasets_root + '/PASCAL_VOC/',
               ann_file='VOC2012/ImageSets/Main/trainval.txt',
               data_prefix=dict(sub_data_root='VOC2012/'),
               filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
               pipeline=train_pipeline,
               backend_args=backend_args)

voc = dict(
    type='ConcatDataset',
    ignore_keys=['dataset_type'],
    datasets=[voc2007, voc2012])

clipart_dataset = dict(type='VOCDataset',
                       data_root=datasets_root + '/clipart/',
                       ann_file='ImageSets/Main/trainval.txt',
                       data_prefix=dict(sub_data_root=''),
                       filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
                       pipeline=test_pipeline,
                       backend_args=None)
clipart_metric = dict(
    type='VOCMetric',
    metric='mAP',
    eval_mode='area',
    prefix='clipart',
)

watercolor_dataset = dict(type='VOCDataset',
                          data_root=datasets_root + '/watercolor/',
                          ann_file='ImageSets/Main/trainval.txt',
                          data_prefix=dict(sub_data_root=''),
                          filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
                          pipeline=test_pipeline,
                          backend_args=backend_args)
watercolor_metric = dict(
    type='VOCMetric',
    metric='mAP',
    eval_mode='area',
    prefix='watercolor',
)

comic_dataset = dict(type='VOCDataset',
                     data_root=datasets_root + '/comic/',
                     ann_file='ImageSets/Main/trainval.txt',
                     data_prefix=dict(sub_data_root=''),
                     filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
                     pipeline=test_pipeline,
                     backend_args=backend_args)
comic_metric = dict(
    type='VOCMetric',
    metric='mAP',
    eval_mode='area',
    prefix='comic',
)

val_dataset = dict(type='ConcatDataset',
                   ignore_keys=['dataset_type'],
                   datasets=[clipart_dataset, watercolor_dataset, comic_dataset])
val_dataset_prefix = ['r2a', 'r2a', 'r2a']
val_dataset_metric = [clipart_metric, watercolor_metric, comic_metric]

train_dataloader = dict(batch_size=32,
                        num_workers=16,
                        persistent_workers=True,
                        sampler=dict(type='InfiniteSampler', shuffle=True),
                        batch_sampler=dict(type='AspectRatioBatchSampler'),
                        dataset=voc)

val_dataloader = dict(batch_size=16,
                      num_workers=16,
                      persistent_workers=True,
                      drop_last=False,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=val_dataset)
test_dataloader = val_dataloader

val_evaluator = dict(type='MultiDatasetsEvaluator',
                     metrics=val_dataset_metric,
                     dataset_prefixes=val_dataset_prefix)
test_evaluator = val_evaluator
