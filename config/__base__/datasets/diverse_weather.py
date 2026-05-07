# ========================================================================

dataset_type = 'DiverseWeatherDataset'
data_root = '/data2/datasets/diverse_weather/'
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

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),

    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# ========================================================================

daytime_clear = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='annotations/daytime_clear_annotations_train.json',
    data_prefix=dict(img='daytime_clear_train/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args,
)
# DC
daytime_clear_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='annotations/daytime_clear_annotations_test.json',
    data_prefix=dict(img='daytime_clear_train/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=test_pipeline,
    backend_args=backend_args,
)
daytime_clear_metric = dict(type='CocoMetric',
                            ann_file=data_root + 'annotations/daytime_clear_annotations_test.json',
                            metric=['bbox'],
                            format_only=format_only,
                            classwise=classwise,
                            use_mp_eval=use_mp_eval,
                            prefix='daytime_clear',
                            backend_args=backend_args)
# DF
daytime_foggy = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='annotations/daytime_foggy_annotations_f.json',
    data_prefix=dict(img='daytime_foggy_test/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    test_mode=True,
    pipeline=test_pipeline,
    backend_args=backend_args,
)
daytime_foggy_metric = dict(type='CocoMetric',
                            ann_file=data_root + 'annotations/daytime_foggy_annotations_f.json',
                            metric=['bbox'],
                            format_only=format_only,
                            classwise=classwise,
                            use_mp_eval=use_mp_eval,
                            prefix='daytime_foggy',
                            backend_args=backend_args)
# DR
dusk_rainy = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='annotations/dusk_rainy_annotations_f.json',
    data_prefix=dict(img='dusk_rainy_test/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    test_mode=True,
    pipeline=test_pipeline,
    backend_args=backend_args,
)
dusk_rainy_metric = dict(type='CocoMetric',
                         ann_file=data_root + 'annotations/dusk_rainy_annotations_f.json',
                         metric=['bbox'],
                         format_only=format_only,
                         classwise=classwise,
                         use_mp_eval=use_mp_eval,
                         prefix='dusk_rainy',
                         backend_args=backend_args)
# NR
night_rainy = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='annotations/night_rainy_annotations_f.json',
    data_prefix=dict(img='night_rainy_test/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    test_mode=True,
    pipeline=test_pipeline,
    backend_args=backend_args,
)
night_rainy_metric = dict(type='CocoMetric',
                          ann_file=data_root + 'annotations/night_rainy_annotations_f.json',
                          metric=['bbox'],
                          format_only=format_only,
                          classwise=classwise,
                          use_mp_eval=use_mp_eval,
                          prefix='night_rainy',
                          backend_args=backend_args)
# NS
night_sunny = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='annotations/night_sunny_annotations_f.json',
    data_prefix=dict(img='night_sunny_test/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    test_mode=True,
    pipeline=test_pipeline,
    backend_args=backend_args,
)
night_sunny_metric = dict(type='CocoMetric',
                          ann_file=data_root + 'annotations/night_sunny_annotations_f.json',
                          metric=['bbox'],
                          format_only=format_only,
                          classwise=classwise,
                          use_mp_eval=use_mp_eval,
                          prefix='night_sunny',
                          backend_args=backend_args)

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
                        num_workers=8,
                        persistent_workers=True,
                        sampler=dict(type='InfiniteSampler', shuffle=True),
                        batch_sampler=dict(type='AspectRatioBatchSampler'),
                        dataset=daytime_clear)

val_dataloader = dict(batch_size=16,
                      num_workers=8,
                      persistent_workers=True,
                      drop_last=False,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=val_datasets)

test_dataloader = dict(batch_size=16,
                       num_workers=8,
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
