import torch

from dotmap import DotMap
from torch.utils.data import DataLoader
from tqdm import tqdm
from visdom import Visdom

import H36M
from demo import draw_merged_image

viz = Visdom()

config = DotMap({
    "annotation_path": "/home/grayish/Downloads/Human3.6M/annot",
    "image_path": "/home/grayish/Downloads/Human3.6M",
    "pretrained_path": "./pretrained/",
    "subjects": [1, 5, 6, 7, 8, 9, 11],
    "task": "train",
    "num_parts": 17,
    "heatmap_xy_coefficient": 2,
    "voxel_xy_resolution": 64,
    "voxel_z_resolutions": [1, 2, 4, 64],
    "batch": 12,
    "workers": 8,
    "epoch": 5
})

loader = DataLoader(
    H36M.Data(
        annotation_path=config.annotation_path,
        image_path=config.image_path,
        subjects=config.subjects,
        task='train',
        heatmap_xy_coefficient=config.heatmap_xy_coefficient,
        voxel_xy_resolution=config.voxel_xy_resolution,
        voxel_z_resolutions=config.voxel_z_resolutions,
    ),
    config.batch, shuffle=True, pin_memory=True,
    num_workers=config.workers,
)

pretrained_epoch = 0
torch.set_num_threads(8)
for epoch in range(pretrained_epoch + 1, pretrained_epoch + 1 + config.epoch):
    with tqdm(total=len(loader), unit=' iter', unit_scale=False) as progress:
        progress.set_description('Epoch %d' % epoch)

        for image, vo in loader:
            # viz.images(image.numpy(), win='image')
            # gt_image_window = draw_merged_image(voxels[:, :, -config.voxel_z_resolutions[-1]:, :, :], image.numpy(), 'gt')

            progress.update(1)
