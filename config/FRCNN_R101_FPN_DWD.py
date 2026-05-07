# import
custom_imports = dict(
    imports=[
        'projects.crri_project.components',
        'projects.crri_project.models',
        'projects.crri_project.hooks'
    ],
    allow_failed_imports=False)

# overall
_base_ = [
    '__base__/datasets/diverse_weather.py',
    '__base__/models/faster_rcnn_r101_fpn_hook.py',
    '__base__/schedules/schedule_30k_amp_stage.py',
    '__base__/default_runtime_iter.py'
]
# modify
num_classes = 7
work_dir = 'runtime/FRCNN_R101_FPN_DWD'
show_dir = 'runtime/FRCNN_R101_FPN_DWD/show_dir'
# load_from = "runtime/FRCNN_R101_FPN_DWD/iter_5000.pth"
resume = False

randomness = dict(
    seed=42,
    deterministic=False,
    diff_rank_seed=True
)

_base_.optim_wrapper.accumulative_counts = 1
_base_.train_dataloader.batch_size = 30
_base_.model.roi_head.bbox_head.num_classes = num_classes
_base_.model.res1_adv_cfg = dict(
    type='CRRI',
    in_channels=256,
    bank_size=512,
    momentum=0.01,
    eps=1e-6,
    init_cfg=None)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=30000, val_interval=1000, val_begin=5000)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExponentialMovingAverage',
        strict_load=False,
        begin_iter=2000,
        momentum=0.0001,          interval=1),  ]
# val_cfg = dict(type='TeacherStudentValLoop')
