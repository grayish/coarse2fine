import h5py
import os

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from dotmap import DotMap
from torch.utils.data import DataLoader
from tqdm import tqdm
from visdom import Visdom

import H36M
from H36M.task import Task
from H36M.annotation import Annotation
from demo import draw_merged_image
from hourglass import StackedHourglass
from log import log

viz = Visdom()

config = DotMap({
    "annotation_path": "/media/nulledge/2nd/data/Human3.6M/converted/annot",
    "image_path": "/media/nulledge/2nd/data/Human3.6M/converted/",
    "pretrained_path": "./pretrained/",
    "subjects": [1, 5, 6, 7, 8, 9, 11],
    "task": str(Task.Valid),
    "num_parts": 17,
    "heatmap_xy_coefficient": 2,
    "voxel_xy_resolution": 64,
    "voxel_z_resolutions": [1, 2, 4, 64],
    "batch": 12,
    "workers": 8,
    "epoch": 100
})

log.info('Reboot ')

log.debug(config)

log.info('Loading Human3.6M data...')

loader = DataLoader(
    H36M.Data(
        image_path=config.image_path,
        subjects=config.subjects,
        task=str(Task.Valid),
        heatmap_xy_coefficient=config.heatmap_xy_coefficient,
        voxel_xy_resolution=config.voxel_xy_resolution,
        voxel_z_resolutions=config.voxel_z_resolutions,
        joints=config.num_parts,
        augment=False,
    ),
    config.batch, shuffle=(config.task == str(Task.Train)), pin_memory=True,
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
    log.info('Done')
else:
    pretrained_model = None
    log.info('No pretrained model found.')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log.info('set device: %s' % device)
model = model.to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=2.5e-4)
# optimizer.load_state_dict(pretrained_model['optimizer'])

criterion = nn.MSELoss()

# Reconstructed voxel z-value.
z_boundary = np.squeeze(
    scipy.io.loadmat('/media/nulledge/2nd/data/Human3.6M/converted/annot/data/voxel_limits.mat')['limits'])
z_reconstructed = (z_boundary[1:65] + z_boundary[0:64]) / 2
z_delta = z_boundary[32]

# Camera intrinsics.
cam_intrinsics = {'f': {}, 'c': {}, 'k': {}, 'p': {}}
for path, _, files in os.walk('calibration'):
    for file in files:
        name, _ = file.split('.')
        serial, param = name.split('_')

        cam_intrinsics[param][serial] = np.loadtxt(os.path.join(path, file))

JPE = 0.0 # Joint Position Error.
num = 1
num_sequence = dict()
error_sequence = dict()

torch.set_num_threads(4)
for epoch in range(pretrained_epoch + 1, pretrained_epoch + 1 + config.epoch):
    with tqdm(total=len(loader), unit=' iter', unit_scale=False) as progress:
        progress.set_description('Epoch %d' % epoch)

        with torch.set_grad_enabled(config.task == str(Task.Train)):

            for images, voxels, cameras, raw_data, sequence in loader:
                images_cpu = images
                images = images.to(device)

                optimizer.zero_grad()
                outputs = model(images)

                # Parameter optimization.
                if config.task == str(Task.Train):

                    voxel_cpu = voxels[-1]
                    for idx, voxel in enumerate(voxels):
                        voxels[idx] = voxel.to(device).view(-1,
                                                            config.num_parts * config.voxel_z_resolutions[idx],
                                                            config.voxel_xy_resolution,
                                                            config.voxel_xy_resolution)

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

                # 3D pose reconstruction.
                elif config.task == str(Task.Valid):

                    fine_results = outputs[-1]
                    z_res = config.voxel_z_resolutions[-1]
                    x_res = y_res = config.voxel_xy_resolution
                    n_batch, channel, height, width = fine_results.shape

                    for batch, fine_result in enumerate(fine_results):

                        f, c, k, p = [cam_intrinsics[param][cameras[batch]] for param in ['f', 'c', 'k', 'p']]

                        num = num + 1

                        for joint in range(config.num_parts):
                            joint_prediction = fine_result[joint * z_res:(joint + 1) * z_res, :, :]
                            joint_prediction = joint_prediction.view(-1)  # flatten

                            _, part = joint_prediction.max(0)
                            z = part / (x_res * y_res)
                            y = (part % (x_res * y_res)) / y_res
                            x = (part % (x_res * y_res)) % y_res

                            x, y, z = [int(x), int(y), int(z)]
                            in_volume_space = [x, y, z]

                            center = raw_data[str(Annotation.Center)][batch]
                            scale = raw_data[str(Annotation.Scale)][batch]
                            x_coord, y_coord = [0, 1, ]
                            x = (x - x_res/2) * (scale*200/x_res) + center[x_coord]
                            y = (y - y_res/2) * (scale*200/y_res) + center[y_coord]

                            in_image_space = [x, y, 1]

                            S, root, z_coord = [raw_data[str(Annotation.S)][batch], 0, 2, ]
                            z_root = S[root, z_coord] + z_delta
                            z_relative = z_reconstructed[z]
                            z = z_root + z_relative

                            x = (x - c[0]) * z / f[0]
                            y = (y - c[1]) * z / f[1]

                            in_camera_space = [x, y, z]

                            reconstructed = np.asarray(in_camera_space)
                            S_joint = S[joint]
                            error = np.linalg.norm(reconstructed - S_joint)
                            JPE = JPE + error

                            if sequence[batch] not in num_sequence.keys():
                                num_sequence[sequence[batch]] = 0
                                error_sequence[sequence[batch]] = 0

                            error_sequence[sequence[batch]] += error
                            num_sequence[sequence[batch]] += 1

                    progress.update(1)
                    progress.set_postfix(MPJPE='%fmm' % (JPE / ((num-1) * config.num_parts)))

                else:
                    log.error('Wrong task: %s' % str(config.task))
                    raise Exception('Wrong task!')

    log.info(error_sequence)
    log.info(num_sequence)
    for key in error_sequence.keys():
        log.info('%s: %f' % (key, error_sequence[key] / num_sequence[key]))

    if config.task == str(Task.Valid):
        break

    log.info('Saving the trained model... (%d epoch, %d step)' % (epoch, step))
    pretrained_model = os.path.join(config.pretrained_path, '%d.save' % epoch)
    torch.save({
        'epoch': epoch,
        'step': step,
        'state': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, pretrained_model)
    log.info('Done')
