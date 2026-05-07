# training schedule for 20k iterations
train_cfg = dict(type='IterBasedTrainLoop', max_iters=30000, val_interval=1000, val_begin=10000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate scheduler
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
         end=500),  # 100 iterations for linear warm-up
    dict(type='MultiStepLR', by_epoch=False, milestones=[21000, 24000, 27000], gamma=0.8)
]

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    accumulative_counts=1,
    #  clip_grad=dict(max_norm=5, norm_type=2)
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
