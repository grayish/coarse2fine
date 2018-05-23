import json
import H36M
import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm as tqdm
from hourglass import StackedHourglass
from dotmap import DotMap
from log import log
from H36M.task import Task
from torch.utils.data import DataLoader
from torch.autograd import Variable
from visdom import Visdom

viz = Visdom()

with open('config.json') as fd:
    config = DotMap(json.load(fd))


def to_var(x, is_volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=is_volatile)


if __name__ == "__main__":

    log.info('Reboot ')

    log.debug(config)

    log.info('Loading Human3.6M data...')
    loader = DataLoader(
        H36M.Data(
            annotation_path=config.annotation_path,
            image_path=config.image_path,
            subjects=config.subjects,
            task=Task.from_str(config.task),
            heatmap_xy_coefficient=config.heatmap_xy_coefficient,
            voxel_xy_resolution=config.voxel_xy_resolution,
            voxel_z_resolutions=config.voxel_z_resolutions,
        ),
        config.batch, shuffle=True, pin_memory=True,
        # num_workers=config.workers,
    )
    log.info('Done')

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

    if torch.cuda.is_available():
        model.cuda()
        log.info('CUDA enabled.')

    criterion = nn.MSELoss()
    loss_window, gt_image_window, out_image_window = None, None, None

    for epoch in range(pretrained_epoch + 1, pretrained_epoch + 1 + config.epoch):
        with tqdm(total=len(loader), unit=' iter', unit_scale=False) as progress:
            progress.set_description('Epoch %d' % epoch)

            for raw_data, image, voxels in loader:

                images = to_var(image).to(torch.float)
                for idx, voxel in enumerate(voxels):
                    voxels[idx] = to_var(voxel).to(torch.float)

                optimizer.zero_grad()
                outputs = model(images)

                loss = sum([criterion(out, voxel) for out, voxel in zip(outputs, voxels)])
                loss.backward()

                optimizer.step()

                progress.set_postfix(loss=float(loss.data))
                progress.update(config.batch)

                step = step + config.batch

                assert viz.check_connection()
                loss_window =  viz.line(X=step,
                    Y=np.array([float(loss.data)]),
                    win=loss_window,
                    update='append' if loss_window is not None else None)

        log.info('Saving the trained model... (%d epoch, %d step)' % (epoch, step))
        pretrained_model = os.path.join(config.pretrained_path, '%d.save' % epoch)
        torch.save({
            'epoch': epoch,
            'step': step,
            'state': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, pretrained_model)
        log.info('Done')
