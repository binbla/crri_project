default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1000,
        by_epoch=False,
        max_keep_ckpts=40,
        save_optimizer=True,
        save_param_scheduler=True,
        #  save_best=['dw/night_rainy/bbox_mAP_50'],
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='DetLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer',
                  line_width=5,
                  alpha=0.7)

log_processor = dict(type='LogProcessor', window_size=10, by_epoch=False)

log_level = 'INFO'
load_from = None
