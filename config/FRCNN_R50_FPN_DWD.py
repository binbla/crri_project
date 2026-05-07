# import
custom_imports = dict(
    imports=[
        'projects.crri_project.components',  #
        'projects.crri_project.models',
        'projects.crri_project.hooks'
    ],
    allow_failed_imports=False)

# overall
_base_ = [
    '__base__/datasets/diverse_weather.py',  #
    '__base__/models/faster_rcnn_r50_fpn_hook.py',
    '__base__/schedules/schedule_30k_amp_stage.py',
    '__base__/default_runtime_iter.py'
]
# modify
num_classes = 7
work_dir = 'runtime/FRCNN_R50_FPN_DWD'
show_dir = 'runtime/FRCNN_R50_FPN_DWD/show_dir'
# load_from = "runtime/FRCNN_R50_FPN_DWD/iter_5000.pth"
resume = False

randomness = dict(
    seed=42,
    deterministic=False,
    diff_rank_seed=True
)

_base_.optim_wrapper.accumulative_counts = 1
# _base_.model.backbone.norm_cfg = dict(type='SyncBN', requires_grad=True)
_base_.train_dataloader.batch_size = 32
_base_.model.roi_head.bbox_head.num_classes = num_classes

_base_.model.res1_adv_cfg = dict(
    type='CRRI',
    in_channels=256,
    bank_size=512,
    momentum=0.01,
    eps=1e-6,
    init_cfg=None)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExponentialMovingAverage',
        strict_load=False,
        begin_iter=5000,
        momentum=0.0001,
        interval=1),
]
# val_cfg = dict(type='TeacherStudentValLoop')# mean-teacher
# custom_hooks = [dict(type='MeanTeacherHook', momentum=0.0004, interval=1, skip_buffer=True)]
