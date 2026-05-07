dataset_type = 'CocoDataset'
data_root = '/data2/datasets/coco/'
classwise = True
format_only = False
backend_args = None
use_mp_eval = True

# ========================================================================

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', direction='horizontal', prob=0.5),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomApply',
         transforms=[dict(type='SCGTransform', v1_scale=0.005, v2_scale=0.7)],
         prob=0.5),
    dict(type='RandomApply',
         transforms=[
             dict(type='Contrast', min_mag=0.6, max_mag=1.4),
             dict(type='Brightness', min_mag=0.6, max_mag=1.4),
             dict(type='Saturation', min_mag=0.6, max_mag=1.4),
         ],
         prob=0.8),
    dict(type='RandomApply', transforms=[dict(type='Saturation', min_mag=0, max_mag=0)], prob=0.2),
    dict(type='RandomApply', transforms=[dict(type='Blur', sigma=(0.1, 2.0))], prob=0.5),
    dict(type='RandomApply',
         transforms=[
             dict(type='RandomEraseTransform', sl=0.05, sh=0.15, r1=0.3, r2=3.3, value='random'),
         ],
         prob=0.7),
    dict(type='RandomApply',
         transforms=[
             dict(type='RandomEraseTransform', sl=0.02, sh=0.15, r1=0.1, r2=6, value='random'),
         ],
         prob=0.5),
    dict(type='RandomApply',
         transforms=[
             dict(type='RandomEraseTransform', sl=0.02, sh=0.15, r1=0.05, r2=8, value='random'),
         ],
         prob=0.3),

    dict(type='PackDetInputs'),
]


train_pipeline_clean = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', direction='horizontal', prob=0.5),
    dict(type='Resize', scale=(1280, 720), keep_ratio=True),
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



coco_train = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='annotations/instances_train2017.json',
    data_prefix=dict(img='train2017/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args,
)
coco_val = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='annotations/instances_val2017.json',
    data_prefix=dict(img='val2017/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    test_mode=True,
    pipeline=test_pipeline,
    backend_args=backend_args,
)
coco_train_metric = dict(type='CocoMetric',
                         ann_file=data_root + 'annotations/instances_train2017.json',
                         metric=['bbox'],
                         format_only=format_only,
                         classwise=classwise,
                         use_mp_eval=use_mp_eval,
                         prefix='coco_train',
                         backend_args=backend_args)
coco_val_metric = dict(type='CocoMetric',
                       ann_file=data_root + 'annotations/instances_val2017.json',
                       metric=['bbox'],
                       format_only=format_only,
                       classwise=classwise,
                       use_mp_eval=use_mp_eval,
                       prefix='coco_val',
                       backend_args=backend_args)

# ========================================================================

train_dataloader = dict(batch_size=16,
                        num_workers=16,
                        persistent_workers=True,
                        sampler=dict(type='InfiniteSampler', shuffle=True),
                        batch_sampler=dict(type='AspectRatioBatchSampler'),
                        dataset=coco_train)

val_dataloader = dict(batch_size=16,
                      num_workers=16,
                      persistent_workers=True,
                      drop_last=False,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=coco_val)

test_dataloader = val_dataloader

val_evaluator = coco_val_metric
test_evaluator = val_evaluator
