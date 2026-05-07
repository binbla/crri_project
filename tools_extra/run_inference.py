from mmdet.apis import DetInferencer

custom_imports = dict(
    imports=[
        'projects.chr_project/.components',  #
        'projects.chr_project/.models',
        'projects.chr_project/.hooks'
    ],
    allow_failed_imports=False)



config = f'path/to/config.py'
checkpoint = f'path/to/checkpoint.pth'

input_dir = f'path/to/images/'
out_dir = f'path/to/output/'

device = 'cuda:0'

inferencer = DetInferencer(
    model=config,
    weights=checkpoint,
    # palette='DiverseWeatherDataset',
    device=device)

inferencer(inputs=input_dir, out_dir=out_dir, no_save_pred=True, no_save_vis=True)
