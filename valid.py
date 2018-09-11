import numpy as np
import torch
import scipy.io
import os
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
from H36M import OldHuman36m as Human36m
from H36M import OldTask as Task
from H36M import OldAnnotation as Annotation
from demo import get_merged_image

from util import get_coord_3d
from visdom import Visdom

viz = Visdom(env='result')


def validate(model, cfg):
    valid_batch_size = 1

    valid_dataset = Human36m(
        task=Task.Valid,
        augment=False,
        image_path=cfg.image_path,
        subjects=cfg.subjects,
        heatmap_xy_coefficient=cfg.heatmap_xy_coefficient,
        voxel_xy_resolution=cfg.voxel_xy_res,
        voxel_z_resolutions=cfg.voxel_z_res,
        joints=cfg.num_parts)

    valid_loader = DataLoader(valid_dataset, valid_batch_size,
                              shuffle=False, pin_memory=True, num_workers=cfg.workers)

    # reset running statistics in BatchNorm layers with 0 mean and 1 variance
    # for key, value in model.state_dict().items():
    #     if 'running_mean' in key:
    #         layer = model
    #         modules = key.split('.')[:-1]
    #         for module in modules:
    #             if module.isdigit():
    #                 layer = layer[int(module)]
    #             else:
    #                 layer = getattr(layer, module)
    #         layer.reset_running_stats()
    #         layer.momentum = None

    # # update input statistics
    # with tqdm(total=len(valid_loader), unit=' iter', unit_scale=False) as progress:
    #     progress.set_description('Reset BatchNorm')
    #     for images, voxels, cameras, raw_data, sequence in valid_loader:
    #         with torch.no_grad():
    #             model(images.to(cfg.device))
    #         progress.update(1)

    # file_name = 'reset_statistics.save.backup_grad_batch_norm'
    # torch.save({'state': model.state_dict(), }, file_name)
    # backup = torch.load(file_name)
    # model.load_state_dict(backup['state'])

    # Reconstructed voxel z-value.
    z_boundary = np.squeeze(
        scipy.io.loadmat('./Human3.6M/annot/voxel_limits.mat')['limits'])
    z_reconstructed = (z_boundary[1:65] + z_boundary[0:64]) / 2
    z_delta = z_boundary[32]
    z_reconstructed = torch.tensor(z_reconstructed).to(cfg.device).float()
    z_delta = torch.tensor(z_delta).to(cfg.device)

    # Camera intrinsics.
    cam_intrinsics = {'f': {}, 'c': {}, 'k': {}, 'p': {}}
    for path, _, files in os.walk('calibration'):
        for file in files:
            name, _ = file.split('.')
            serial, param = name.split('_')

            cam_intrinsics[param][serial] = np.loadtxt(os.path.join(path, file))

    # change model mode
    # model = model.eval()
    mpjpe = 0
    action_error = dict()
    # valid_loader.num_workers = cfg.workers
    with tqdm(total=len(valid_loader), unit=' iter', unit_scale=False) as progress:
        # for images, voxels, raw_data in valid_loader:
        for images, voxels, cameras, raw_data, sequence in valid_loader:
            batch_size, _, _, _ = images.shape

            with torch.no_grad():
                imgs = images.to(cfg.device)
                flip_imgs = images.flip(3).to(cfg.device)

                outputs = model(imgs)
                flip_outputs = model(flip_imgs)

            out = outputs[-1].view(-1, cfg.num_parts, cfg.voxel_z_res[-1],
                                   cfg.voxel_xy_res, cfg.voxel_xy_res)
            flip_out = flip_outputs[-1].view(-1, cfg.num_parts, cfg.voxel_z_res[-1],
                                             cfg.voxel_xy_res, cfg.voxel_xy_res)

            # swap left right joints
            swap_indices = torch.LongTensor([0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]).to(cfg.device)
            flip_out = torch.index_select(flip_out, 1, swap_indices)
            flip_out = flip_out.flip(4)

            out = (out + flip_out) / 2.0  # normal and flip sum and div 2
            pred = get_coord_3d(out, num_parts=cfg.num_parts)

            gt_coord = raw_data[str(Annotation.S)].float().to(cfg.device)
            # f_intrinsic = raw_data[Annotation.INTRINSIC][:, 0:2].to(cfg.device).unsqueeze(1)
            # c_intrinsic = raw_data[Annotation.INTRINSIC][:, 2:4].to(cfg.device).unsqueeze(1)
            f_list = list()
            c_list = list()
            for cam in cameras:
                f, c, k, p = [cam_intrinsics[param][cam] for param in ['f', 'c', 'k', 'p']]
                f_list.append(torch.tensor(f))
                c_list.append(torch.tensor(c))

            centers = raw_data[str(Annotation.Center)].float().to(cfg.device)
            centers = centers.unsqueeze(1).expand(batch_size, cfg.num_parts, 2)
            scales = raw_data[str(Annotation.Scale)].float().to(cfg.device)
            scales = scales.unsqueeze(-1).unsqueeze(-1).expand(batch_size, cfg.num_parts, 2)

            f_intrinsic = torch.stack(f_list, dim=0).float().cuda().unsqueeze(1)
            c_intrinsic = torch.stack(c_list, dim=0).float().cuda().unsqueeze(1)

            xy_res = float(cfg.voxel_xy_res)
            xy_image = (pred[:, :, 0:2] - xy_res / 2.0) * (scales * 200.0 / xy_res) + centers  # image space
            z_idx = pred[:, :, 2]

            z_pelvis = gt_coord[:, 0, 2] + z_delta
            z_reletive = z_reconstructed[z_idx.long()]
            z_coord_abs = (z_reletive + z_pelvis.unsqueeze(-1)).unsqueeze(-1)
            xy_camera = (xy_image - c_intrinsic) * z_coord_abs / f_intrinsic
            camera_coord = torch.cat([xy_camera, z_coord_abs], dim=2)

            delta = camera_coord - gt_coord
            dists = delta.pow(2).sum(-1).sqrt()
            batch_mpjpe = dists.mean()  # mean per joint position error (per batch)

            mpjpe += batch_mpjpe.item()

            # output_cpu = outputs[-1].view(batch_size, 17, 64, 64, 64).cpu().detach()
            # viz.images(tensor=get_merged_image(output_cpu, images.numpy()), nrow=3, win='train')
            # viz.images(tensor=get_merged_image(voxels[-1], images.numpy()), nrow=3, win='gt')

            for idx, action in enumerate(sequence):
                if action not in action_error.keys():
                    action_error[action] = [0, 0]  # count and error

                action_error[action][0] += 1
                action_error[action][1] += dists[idx].cpu()

            # for idx, action in enumerate(raw_data[Annotation.ACTION]):
            #     if action not in action_error.keys():
            #         action_error[action] = [0, 0]  # count and error
            #
            #     action_error[action][0] += 1
            #     action_error[action][1] += dists[idx].cpu()

            progress.set_postfix(MPJPE=batch_mpjpe.item())
            progress.update(1)

    mpjpe = mpjpe / len(valid_loader)

    with open('results.bin', 'wb')as f:
        pickle.dump(action_error, f)

    for k, v in action_error.items():
        print(k, (v[1] / v[0]).mean())

    return mpjpe
