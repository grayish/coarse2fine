import os

import numpy as np
import torch
import torch.nn as nn
from dotmap import DotMap
from torch.utils.data import DataLoader
from tqdm import tqdm
from visdom import Visdom

import H36M
from demo import draw_merged_image
from hourglass import StackedHourglass
from log import log

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

log.info('Reboot ')

log.debug(config)

log.info('Loading Human3.6M data...')

loader = DataLoader(
    H36M.Data(
        annotation_path=config.annotation_path,
        image_path=config.image_path,
        subjects=config.subjects,
        task='train',
        heatmap_xy_coefficient=config.heatmap_xy_coefficient,
        voxel_xy_resolution=config.voxel_xy_resolution,
        voxel_z_resolutions=config.voxel_z_resolutions,
        joints=config.num_parts
    ),
    config.batch, shuffle=True, pin_memory=True,
    num_workers=config.workers,
)
log.info('load complete.')

log.info('Creating model...')
model = StackedHourglass(config.voxel_z_resolutions, 256, config.num_parts)
optimizer = torch.optim.RMSprop(model.parameters(), lr=2.5e-4)
step = np.zeros([1], dtype=np.uint32)
log.info('Done')

log.info('Searching for the pretrained model...')
pretrained_epoch = 0
for _, _, files in os.walk(config.pretrained_path):
    for file in files:
        name, extension = file.split('.')
        epoch = int(name)
        if epoch > pretrained_epoch:
            pretrained_epoch = epoch
if pretrained_epoch > 0:
    pretrained_model = os.path.join(config.pretrained_path, '%d.save' % pretrained_epoch)
    pretrained_model = torch.load(pretrained_model)

    log.info('Loading the pretrained model (%d epoch, %d step)... ' %
             (pretrained_model['epoch'], pretrained_model['step']))

    model.load_state_dict(pretrained_model['state'])
    step[0] = pretrained_model['step']
    optimizer.load_state_dict(pretrained_model['optimizer'])
    log.info('Done')
else:
    pretrained_model = None
    log.info('No pretrained model found.')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log.info('set device: %s' % device)
model = model.to(device)

# guassian_voxel = GaussianVoxel(config.voxel_z_resolutions,
#                                config.heatmap_xy_coefficient,
#                                config.batch,
#                                config.num_parts,
#                                config.voxel_xy_resolution)
# guassian_voxel.to(device)

criterion = nn.MSELoss()

torch.set_num_threads(4)
for epoch in range(pretrained_epoch + 1, pretrained_epoch + 1 + config.epoch):
    with tqdm(total=len(loader), unit=' iter', unit_scale=False) as progress:
        progress.set_description('Epoch %d' % epoch)

        for images, voxels in loader:
            images_cpu = images
            voxel_cpu = voxels[-1]
            images = images.to(device)
            for idx, voxel in enumerate(voxels):
                voxels[idx] = voxel.to(device).view(config.batch,
                                                    -1,
                                                    config.voxel_xy_resolution,
                                                    config.voxel_xy_resolution)

            optimizer.zero_grad()
            outputs = model(images)

            loss = sum([criterion(out, voxel) for out, voxel in zip(outputs, voxels)])
            loss.backward()

            optimizer.step()
            progress.set_postfix(loss=float(loss.item()))

            if step % 100 == 0:
                output_cpu = outputs[-1].view(config.batch, config.num_parts, config.voxel_z_resolutions[-1],
                                              config.voxel_xy_resolution,
                                              config.voxel_xy_resolution).cpu().detach()
                draw_merged_image(output_cpu, images_cpu.numpy(), 'train')
                draw_merged_image(voxel_cpu, images_cpu.numpy(), 'gt')

            step = step + 1
            progress.update(1)

    log.info('Saving the trained model... (%d epoch, %d step)' % (epoch, step))
    pretrained_model = os.path.join(config.pretrained_path, '%d.save' % epoch)
    torch.save({
        'epoch': epoch,
        'step': step,
        'state': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, pretrained_model)
    log.info('Done')
