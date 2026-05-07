# ========================================================================


dataset_type = 'DiverseWeatherDatasetVOC'
data_root = '/data2/datasets/diverse_weather/xmldataset/'
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
    dict(type='RetinexNorm', prob=0.5, sigma_list=[15, 80, 250]),
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


# DC train
daytime_clear_train = dict(
    type=dataset_type,
    data_root=data_root + 'daytime_clear/',
    ann_file='VOC2007/ImageSets/Main/train.txt',
    data_prefix=dict(sub_data_root='VOC2007/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args,
)

# DC test
daytime_clear_test = dict(
    type=dataset_type,
    data_root=data_root + 'daytime_clear/',
    ann_file='VOC2007/ImageSets/Main/test.txt',
    data_prefix=dict(sub_data_root='VOC2007/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
    pipeline=test_pipeline,
    backend_args=backend_args,
)
daytime_clear_metric = dict(
    type='VOCMetric',
    metric='mAP',
    eval_mode='area',
    prefix='daytime_clear',
)
# DF
daytime_foggy = dict(
    type=dataset_type,
    data_root=data_root + 'daytime_foggy/',
    ann_file='VOC2007/ImageSets/Main/train.txt',
    data_prefix=dict(sub_data_root='VOC2007/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
    pipeline=test_pipeline,
    backend_args=backend_args,
)
daytime_foggy_metric = dict(
    type='VOCMetric',
    metric='mAP',
    eval_mode='area',
    prefix='daytime_foggy',
)
# DR
dusk_rainy = dict(
    type=dataset_type,
    data_root=data_root + 'dusk_rainy/',
    ann_file='VOC2007/ImageSets/Main/train.txt',
    data_prefix=dict(sub_data_root='VOC2007/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
    pipeline=test_pipeline,
    backend_args=backend_args,
)
dusk_rainy_metric = dict(
    type='VOCMetric',
    metric='mAP',
    eval_mode='area',
    prefix='dusk_rainy',
)
# NR
night_rainy = dict(
    type=dataset_type,
    data_root=data_root + 'night_rainy/',
    ann_file='VOC2007/ImageSets/Main/train.txt',
    data_prefix=dict(sub_data_root='VOC2007/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
    pipeline=test_pipeline,
    backend_args=backend_args,
)
night_rainy_metric = dict(
    type='VOCMetric',
    metric='mAP',
    eval_mode='area',
    prefix='night_rainy',
)
# NS
night_sunny = dict(
    type=dataset_type,
    data_root=data_root + 'Night-Sunny',
    ann_file='ImageSets/Main/train.txt',
    data_prefix=dict(sub_data_root=''),
    filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
    pipeline=test_pipeline,
    backend_args=backend_args,
)
night_sunny_metric = dict(
    type='VOCMetric',
    metric='mAP',
    eval_mode='area',
    prefix='night_sunny',
)

# ========================================================================





val_datasets = dict(
    type='ConcatDataset',
    datasets=[daytime_clear_test, daytime_foggy, dusk_rainy, night_rainy, night_sunny])
val_datasets_prefixes = ['dw', 'dw', 'dw', 'dw', 'dw']
val_datasets_metrics = [
    daytime_clear_metric, daytime_foggy_metric, dusk_rainy_metric, night_rainy_metric,
    night_sunny_metric
]

test_datasets = val_datasets
test_datasets_prefixes = val_datasets_prefixes
test_datasets_metrics = val_datasets_metrics
# ========================================================================


train_dataloader = dict(batch_size=32,
                        num_workers=16,
                        persistent_workers=True,
                        sampler=dict(type='InfiniteSampler', shuffle=True),
                        batch_sampler=dict(type='AspectRatioBatchSampler'),
                        dataset=daytime_clear_train)

val_dataloader = dict(batch_size=32,
                      num_workers=16,
                      persistent_workers=True,
                      drop_last=False,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=val_datasets)

test_dataloader = dict(batch_size=32,
                       num_workers=16,
                       persistent_workers=True,
                       drop_last=False,
                       sampler=dict(type='DefaultSampler', shuffle=False),
                       dataset=test_datasets)

# ========================================================================

val_evaluator = dict(type='MultiDatasetsEvaluator',
                     metrics=val_datasets_metrics,
                     dataset_prefixes=val_datasets_prefixes)

test_evaluator = dict(type='MultiDatasetsEvaluator',
                      metrics=test_datasets_metrics,
                      dataset_prefixes=test_datasets_prefixes)

# ========================================================================
