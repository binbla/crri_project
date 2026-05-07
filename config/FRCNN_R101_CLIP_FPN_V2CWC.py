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
    '__base__/datasets/voc2cwc_coco.py',  #
    '__base__/models/faster_rcnn_CLIP_r101_fpn_hook.py',
    '__base__/schedules/schedule_30k_amp_cosine_adamw.py',
    '__base__/default_runtime_iter.py'
]
# _base_.model.backbone.pretrained = "/data2/shared/clip/RN101.pt"
# modify
num_classes = 20
work_dir = 'runtime/FRCNN_R101_CLIP_FPN_V2CWC'
show_dir = 'runtime/FRCNN_R101_CLIP_FPN_V2CWC/show_dir'
# load_from = "runtime/FRCNN_R101_CLIP_FPN_V2CWC/iter_5000.pth"
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

_base_.model.res2_adv_cfg = dict(
    type='CRRI',
    in_channels=512,
    bank_size=512,
    momentum=0.01,
    eps=1e-6,
    init_cfg=None)

_base_.param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
         end=1000),
    # dict(type='MultiStepLR', by_epoch=False, milestones=[21000, 24000, 27000], gamma=0.8)
    dict(type='CosineAnnealingLR',
         begin=10000,
         end=30000,
         T_max=20000,
         eta_min=1e-6,
         by_epoch=False),
]
_base_.optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-2, betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        },
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
    ),
    accumulative_counts=1,
    clip_grad=dict(max_norm=5, norm_type=2),
)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=30000, val_interval=1000, val_begin=1000)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExponentialMovingAverage',
        strict_load=False,
        begin_iter=2000,
        momentum=0.0004,
        interval=1),
]
# val_cfg = dict(type='TeacherStudentValLoop')# mean-teacher
# custom_hooks = [dict(type='MeanTeacherHook', momentum=0.0004, interval=1, skip_buffer=True)]
