import json
import H36M
import torch
import os
import numpy as np
from hourglass import StackedHourglass
from dotmap import DotMap
from log import log
from H36M.task import Task
from torch.utils.data import DataLoader

with open('config.json') as fd:
    config = DotMap(json.load(fd))

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
        config.batch, shuffle=True, num_workers=config.workers, pin_memory=True,
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

    '''
    log.info('Saving the trained model... (%d epoch, %d step)' % (pretrained_epoch, step))
    pretrained_model = os.path.join(config.pretrained_path, '%d.save' % pretrained_epoch)
    torch.save({
        'epoch': pretrained_epoch,
        'step': step,
        'state': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, pretrained_model)
    log.info('Done')
    '''