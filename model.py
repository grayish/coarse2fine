import numpy as np
import torch
import torch.nn as nn


def gaussian_3d(size, depth, sigma=0.25, mean=0.5, amplitude=1.0):
    width = size
    height = size

    sigma_u = sigma
    sigma_v = sigma
    sigma_r = sigma

    mean_u = mean * width + 0.5
    mean_v = mean * height + 0.5
    mean_r = mean * depth + 0.5

    over_sigma_u = 1.0 / (sigma_u * width)
    over_sigma_v = 1.0 / (sigma_v * height)
    over_sigma_r = 1.0 / (sigma_r * depth)

    x = np.arange(0, width, 1, dtype=np.float32)
    y = x[:, np.newaxis]
    z = np.arange(0, depth, 1, dtype=np.float32)
    z = z[:, np.newaxis, np.newaxis]

    du = (x + 1 - mean_u) * over_sigma_u
    dv = (y + 1 - mean_v) * over_sigma_v
    dr = (z + 1 - mean_r) * over_sigma_r

    gau = amplitude * np.exp(-0.5 * (du * du + dv * dv + dr * dr))

    return torch.tensor(gau  # .transpose(1, 2, 0)

class GaussianVoxel(nn.Module):
    def __init__(self, voxel_z_res_list, heatmap_xy_coeff, batch, part, size):
        super(GaussianVoxel, self).__init__()
        self.voxel_z_res_list = torch.FloatTensor(voxel_z_res_list)
        self.heatmap_xy_coeff = torch.tensor(heatmap_xy_coeff, dtype=torch.float32)
        self.batch = batch
        self.part = part
        self.size = size
        self.width = size
        self.height = size

        self.z_res_cumsum = torch.cumsum(torch.FloatTensor([0] + voxel_z_res_list), dim=0)
        self.heatmap_z_coeff = 2 * torch.floor(
            (6 * self.heatmap_xy_coeff * self.voxel_z_res_list / self.voxel_z_res_list[-1] + 1) / 2) + 1

        self.gaussian_voxel = torch.zeros(batch, part, self.z_res_cumsum[-1], size, size)

        self.gaussian = [gaussian_3d(3 * 2 * self.heatmap_xy_coeff + 1, z_coeff) for z_coeff  in self.heatmap_z_coeff]


    def forward(self, coords):
        self.gaussian_voxel.zero_()

        zind = coords[:, :, 2]
        z = torch.ceil(zind.unsqeeze(-1) * self.voxel_z_res_list / self.voxel_z_res_list[-1]) - 1


        return coords
