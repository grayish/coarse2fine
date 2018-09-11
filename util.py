import torch
import torch.nn as nn


def get_coord_3d(heatmap, num_parts):
    batch, _, _, height, width = heatmap.shape  # BJDHW
    heatmap = heatmap.view(batch, num_parts, -1)
    _, idx = heatmap.max(2)

    coords = idx.to(torch.float).unsqueeze(-1).repeat(1, 1, 3)
    coords[:, :, 0] = (coords[:, :, 0] % (height * width)) % height  # x
    coords[:, :, 1] = (coords[:, :, 1] % (height * width)) / height  # y
    coords[:, :, 2] = (coords[:, :, 2] / (height * width))  # z
    # coords = coords.floor()

    return coords.floor_()


def cal_pck_acc(preds, label, res, factor=0.05):
    delta = preds - label
    dists = delta.pow(2).sum(-1).sqrt() / (res * factor)
    dists_count = (dists >= 0).sum()

    outside_idx = (label[:, :, 0] <= 1) | (label[:, :, 1] <= 1)
    outside_count = outside_idx.sum()

    hit_count = (dists <= 1).sum()

    acc = (hit_count - outside_count).float() / (dists_count - outside_count).float()

    return acc


def get_accuracy(heatmap, label, config):
    preds = get_coord_3d(heatmap, num_parts=config.num_parts)
    gt = get_coord_3d(label, num_parts=config.num_parts)
    acc = cal_pck_acc(preds, gt, config.voxel_xy_res, factor=0.05)

    return acc

