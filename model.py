import itertools

import torch
import torch.nn as nn


def gaussian_3d_dd(ksize, z_ksize,
                res, z_res,
                center_x, center_y, center_z,
                sigma=0.25, amplitude=1.0, device='cuda'):
    over_sigma_u = 1.0 / (sigma * ksize)
    over_sigma_v = 1.0 / (sigma * ksize)
    over_sigma_r = 1.0 / (sigma * z_ksize)

    x = torch.arange(0, res, 1, dtype=torch.float32).to(device)
    y = torch.arange(0, res, 1, dtype=torch.float32).to(device)
    y = y.view(-1, 1)
    z = torch.arange(0, z_res, 1, dtype=torch.float32).to(device)
    z = z.view(-1, 1, 1)

    du = (x + 1 - center_x) * over_sigma_u
    dv = (y + 1 - center_y) * over_sigma_v
    dr = (z + 1 - center_z) * over_sigma_r

    gau = amplitude * torch.exp(-0.5 * (du * du + dv * dv + dr * dr))

    return gau  # .transpose(1, 2, 0)


def gaussian_3d(size, depth, sigma=0.25, mean=0.5, amplitude=1.0, device='cpu'):
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

    x = torch.arange(0, width, 1, dtype=torch.float32).to(device)
    y = x.view(-1, 1)
    z = torch.arange(0, depth, 1, dtype=torch.float32).to(device)
    z = z.view(-1, 1, 1)

    du = (x + 1 - mean_u) * over_sigma_u
    dv = (y + 1 - mean_v) * over_sigma_v
    dr = (z + 1 - mean_r) * over_sigma_r

    gau = amplitude * torch.exp(-0.5 * (du * du + dv * dv + dr * dr))

    return gau  # .transpose(1, 2, 0)


class GaussianVoxel(nn.Module):
    def __init__(self, voxel_z_res_list, heatmap_xy_coeff, batch, part, size):
        super(GaussianVoxel, self).__init__()
        self.voxel_z_res_list = nn.Parameter(torch.FloatTensor(voxel_z_res_list), requires_grad=False)
        self.heatmap_xy_coeff = nn.Parameter(torch.tensor(heatmap_xy_coeff, dtype=torch.float32), requires_grad=False)
        self.batches = batch
        self.joints = part
        self.size = size
        self.width = size
        self.height = size

        # self.z_res_cumsum = nn.Parameter(torch.cumsum(torch.FloatTensor([0] + voxel_z_res_list), dim=0), False)
        self.heatmap_z_coeff = nn.Parameter(2 * torch.floor(
            (6 * self.heatmap_xy_coeff * self.voxel_z_res_list / self.voxel_z_res_list[-1] + 1) / 2) + 1, False)

        self.gaussians = nn.ParameterList()
        for z_coeff in self.heatmap_z_coeff:
            gau = gaussian_3d(3 * 2 * self.heatmap_xy_coeff + 1, z_coeff)
            self.gaussians.append(nn.Parameter(gau, False))

        self.voxels = nn.ParameterList()
        for z_res in self.voxel_z_res_list:
            vx = torch.zeros(batch, part, z_res, size, size)
            self.voxels.append(nn.Parameter(vx, False))

    def forward(self, coords):
        for vx in self.voxels:
            vx.zero_()

        zidx = coords[:, :, 2]
        zidx = torch.ceil(zidx.unsqueeze(-1) * self.voxel_z_res_list / self.voxel_z_res_list[-1]) - 1
        zidx = zidx.short().transpose(1, 2).transpose(0, 1)
        zpad = torch.floor(self.heatmap_z_coeff / 2).short()

        coords = coords.short()
        xy = coords[:, :, 0:2]
        pad = 3 * self.heatmap_xy_coeff.short()

        # for b, j in itertools.product(range(self.batches), range(self.joints)):
        #     for z_res, z_coeff, z in zip(self.voxel_z_res_list, self.heatmap_z_coeff, zidx):
        #         vx = gaussian_3d(2 * pad + 1, z_coeff,
        #                          self.size, z_res,
        #                          xy[b, j, 0], xy[b, j, 1], z[b, j])

        dst = [torch.clamp(xy - pad, min=0), torch.clamp(xy + pad + 1, max=self.size, min=0)]
        src = [torch.clamp(pad - xy, min=0), pad + 1 + torch.clamp(self.size - xy - 1, max=pad)]

        zdsts, zsrcs = list(), list()
        for z, z_res, pad in zip(zidx, self.voxel_z_res_list.short(), zpad):
            zdst = [torch.clamp(z - pad, min=0), torch.clamp(z + pad + 1, max=z_res, min=0)]  # BJ
            zsrc = [torch.clamp(pad - z, min=0), pad + 1 + torch.clamp(z_res - z - 1, max=pad)]  # BJ
            zdsts.append(zdst)
            zsrcs.append(zsrc)

        for (vx, zdst, zsrc, g), b, j in itertools.product(zip(self.voxels, zdsts, zsrcs, self.gaussians),
                                                           range(self.batches),
                                                           range(self.joints)):
            z_dst_slice = slice(zdst[0][b, j], zdst[1][b, j])
            y_dst_slice = slice(dst[0][b, j, 1], dst[1][b, j, 1])
            x_dst_slice = slice(dst[0][b, j, 0], dst[1][b, j, 0])

            z_src_slice = slice(zsrc[0][b, j], zsrc[1][b, j])
            y_src_slice = slice(src[0][b, j, 1], src[1][b, j, 1])
            x_src_slice = slice(src[0][b, j, 0], src[1][b, j, 0])

            vx[b, j, z_dst_slice, y_dst_slice, x_dst_slice] = g[z_src_slice, y_src_slice, x_src_slice]

        # for idx in range(len(self.voxel_z_res_list)):
        #     voxel = self.gaussian_voxel[:, self.z_res_cumsum[idx]:self.z_res_cumsum[idx + 1], :, :]
        #     for part_idx in range(len(self.joints)):
        #         vx = voxel[]
        #
        # self.gaussian_voxel

        return self.voxels

    def set_voxel(self, volume, voxel_xy_res, voxel_z_res, xy, z, heatmap_xy_coeff, heatmap_z_coeff):
        pad = 3 * heatmap_xy_coeff
        zpad = math.floor(heatmap_z_coeff / 2)
        y0, x0 = int(xy[1]), int(xy[0])
        dst = [max(0, y0 - pad), max(0, min(voxel_xy_res, y0 + pad + 1)),
               max(0, x0 - pad), max(0, min(voxel_xy_res, x0 + pad + 1)),
               max(0, z - zpad), max(0, min(voxel_z_res, z + zpad + 1))]
        src = [max(0, pad - y0), pad + min(pad, voxel_xy_res - y0 - 1) + 1,
               max(0, pad - x0), pad + min(pad, voxel_xy_res - x0 - 1) + 1,
               max(0, zpad - z), zpad + min(zpad, voxel_z_res - z - 1) + 1]

        g = gaussian_3d(3 * 2 * heatmap_xy_coeff + 1, heatmap_z_coeff)
        volume[dst[4]:dst[5], dst[0]:dst[1], dst[2]:dst[3]] = g[src[4]:src[5], src[0]:src[1], src[2]:src[3]]
