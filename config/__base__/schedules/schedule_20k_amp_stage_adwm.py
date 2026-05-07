# training schedule for 20k iterations
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate scheduler
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
         end=500),  # 500 iterations for linear warm-up
    dict(type='ConstantLR', factor=1.0, begin=500, end=15000, by_epoch=False),
    dict(type='CosineAnnealingLR', begin=15000, end=20000, T_max=5000, eta_min=1e-6,
         by_epoch=False),
]

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=1e-4, betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }),
    accumulative_counts=1,
    #  clip_grad=dict(max_norm=5, norm_type=2)
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
