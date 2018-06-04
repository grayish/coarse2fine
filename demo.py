import json
import os

import numpy as np
import skimage.transform
import torch
import torch.nn as nn
from dotmap import DotMap
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm as tqdm
from visdom import Visdom

import H36M
from H36M.task import Task
from hourglass import StackedHourglass
from log import log

Color = torch.FloatTensor(
    [[0, 0, 0.5],
     [0, 0, 1],
     [0, 1, 0],
     [1, 1, 0],
     [1, 0, 0]]
)


def merge_to_color_heatmap(batch_heatmaps):
    batch, joints, depth, height, width = batch_heatmaps.size()

    batch_heatmaps_flat = batch_heatmaps.view(batch, joints, depth, -1)
    max_depth_idx = batch_heatmaps_flat.max(-1)[0].max(-1)[1]

    test = list()
    for b_idx in range(batch):
        test2 = list()
        for j_idx in range(joints):
            test2.append(batch_heatmaps[b_idx, j_idx, max_depth_idx[b_idx, j_idx], :, :].view(1, 1, height, width))

        test.append(torch.cat(test2, dim=1))

    batch_heatmaps = torch.cat(test, dim=0)
    # batch_heatmaps = torch.cat(
    #     [torch.cat(
    #         [batch_heatmaps[b_idx, j_idx, max_depth_idx[b_idx, j_idx], :, :] for j_idx in range(joints)], dim=2)
    #      for b_idx in range(batch)], dim=0)

    heatmaps = batch_heatmaps.clamp(0, 1.).view(-1)

    frac = torch.div(heatmaps, 0.25)
    lower_indices, upper_indices = torch.floor(frac).long(), torch.ceil(frac).long()

    t = frac - torch.floor(frac)
    t = t.view(-1, 1)

    k = Color.index_select(0, lower_indices)
    k_1 = Color.index_select(0, upper_indices)

    color_heatmap = (1.0 - t) * k + t * k_1
    color_heatmap = color_heatmap.view(batch, joints, height, width, 3)
    color_heatmap = color_heatmap.permute(0, 4, 2, 3, 1)  # B3HWC
    color_heatmap, _ = torch.max(color_heatmap, 4)  # B3HW

    return color_heatmap


def draw_merged_image(heatmaps, images, window):
    heatmaps = merge_to_color_heatmap(heatmaps)
    heatmaps = heatmaps.permute(0, 2, 3, 1)  # NHWC

    resized_heatmaps = list()
    for idx, ht in enumerate(heatmaps):
        color_ht = skimage.transform.resize(ht.numpy(), (256, 256), mode='constant')
        resized_heatmaps.append(color_ht)

    resized_heatmaps = np.stack(resized_heatmaps, axis=0)

    images = images.transpose(0, 2, 3, 1) * 0.6
    overlayed_image = np.clip(images + resized_heatmaps * 0.4, 0, 1.)

    overlayed_image = overlayed_image.transpose(0, 3, 1, 2)

    return viz.images(tensor=overlayed_image, nrow=3, win=window)


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
        config.batch, shuffle=True, pin_memory=False,
        num_workers=config.workers,
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

            xx = torch.ones(5)
            for image in loader:
                xx = xx + 1
                # images = to_var(image).to(torch.float)
                # for idx, voxel in enumerate(voxels):
                #     voxels[idx] = to_var(voxel).to(torch.float)
                #
                # optimizer.zero_grad()
                # outputs = model(images)
                #
                # loss = sum([criterion(out, voxel) for out, voxel in zip(outputs, voxels)])
                # loss.backward()
                #
                # optimizer.step()
                #
                # progress.set_postfix(loss=float(loss.data))
                # progress.update(1)
                #
                # step = step + config.batch
                #
                # assert viz.check_connection()
                # loss_window = viz.line(X=step,
                #                        Y=np.array([float(loss.data)]),
                #                        win=loss_window,
                #                        update='append' if loss_window is not None else None)
                #
                # out = outputs[-1].squeeze().contiguous()
                # viz.images(image.numpy(), win='image')
                # out_image_window = draw_merged_image(voxels[-1], images.copy(), out_image_window)
                gt_image_window = draw_merged_image(out, images.copy(), gt_image_window)

        log.info('Saving the trained model... (%d epoch, %d step)' % (epoch, step))
        pretrained_model = os.path.join(config.pretrained_path, '%d.save' % epoch)
        torch.save({
            'epoch': epoch,
            'step': step,
            'state': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, pretrained_model)
        log.info('Done')
