import itertools
import math
import os
import pickle
import random

import numpy as np
import torch
import torch.utils.data as torch_data
from PIL import Image
from torchvision import transforms as T
from vectormath import Vector2
from random import shuffle

from H36M import Task
from H36M.util import rand, gaussian_3d
from .annotation import Annotation


class Human36m(torch_data.Dataset):
    def __init__(self, image_path, subjects, task, joints,
                 heatmap_xy_coefficient,
                 voxel_xy_resolution,
                 voxel_z_resolutions,
                 augment,
                 num_split=None):
        self.SCALE_FACTOR = 0.20
        self.ROTATE_DEGREE = 30
        self.ROTATE_PROB = 0.4
        self.HORIZONTAL_FLIP_PROB = 0.5

        self.voxel_z_res_list = torch.IntTensor(voxel_z_resolutions)
        self.voxel_xy_res = voxel_xy_resolution
        self.heatmap_xy_coeff = heatmap_xy_coefficient
        self.task = task
        self.subjects = subjects
        self.joints = joints
        self.image_path = image_path
        self.augment = augment
        self.num_split = num_split
        self.current_split_set = 0

        # initialize
        self.z_list = self.voxel_z_res_list.to(torch.float)
        self.heatmap_z_coeff = 2 * torch.floor(
            (6 * self.heatmap_xy_coeff * self.z_list / self.z_list[-1] + 1) / 2) + 1

        self.gaussians = list()
        for z_coeff in self.heatmap_z_coeff:
            gau = gaussian_3d(3 * 2 * self.heatmap_xy_coeff + 1, z_coeff)
            self.gaussians.append(gau)

        self.voxels = list()
        for z_res in self.voxel_z_res_list:
            # z_res = z_res.to(torch.int)
            vx = torch.zeros(self.joints, z_res, self.voxel_xy_res, self.voxel_xy_res)
            self.voxels.append(vx)

        self.annot_data = pickle.load(open("./180809_%s_list_fix.bin" % task, 'rb'))

        if self.num_split is None or self.task is Task.Valid:
            self.data = [self.annot_data]
        else:
            # split whole data into n-sized chunks
            print("Data split mode is enabled. You should call change_current_split_set after epoch")
            self._shuffle_and_split()

    def _shuffle_and_split(self):
        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]

        shuffle(self.annot_data)
        num_ = int(len(self.annot_data) / self.num_split) + 1
        self.data = list(chunks(self.annot_data, num_))
        print("Human3.6m dataset is shuffled and splited into %d-sized %d chunks"
              % (num_, len(self.data)))

    def __len__(self):
        return len(self.data[self.current_split_set])

    def __getitem__(self, index):
        raw_data = self.data[self.current_split_set][index]

        image, voxels = self._preprocess(raw_data, augment=self.augment)

        return image, voxels, raw_data

    def __add__(self, item):
        pass

    def _preprocess(self, raw_data, augment=False):
        # Common annotations for training and validation.
        image_name = raw_data[Annotation.IMG]
        center = raw_data[Annotation.CENTER]
        zind = np.clip(raw_data[Annotation.ZIDX], 1, 64)
        part = raw_data[Annotation.PART]
        scale = raw_data[Annotation.SCALE]  # * 1.25

        # Calculate augmentation params.
        angle, is_hflip, transform = 0, False, list()
        if augment:
            scale = scale * random.uniform(1 - self.SCALE_FACTOR, 1 + self.SCALE_FACTOR) * 1.25
            angle = rand(self.ROTATE_DEGREE) if random.random() < self.ROTATE_PROB else 0
            if random.random() < self.HORIZONTAL_FLIP_PROB:
                is_hflip = True
                # calculate flipped positions
                for jt in part:
                    jt[0] = jt[0] + 2 * (center[0] - jt[0])

                # swap the index of joint, pelvis[0], spine[7,8,9,10]
                swap_indices = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]
                part = np.take(part, indices=swap_indices, axis=0)
                zind = np.take(zind, indices=swap_indices, axis=0)
                # swap_part = np.copy(part)
                # swap_part[1:4] = part[4:7]
                # swap_part[4:7] = part[1:4]
                # swap_part[11:14] = part[14:17]
                # swap_part[14:17] = part[11:14]
                # part = swap_part

            transform.append(T.ColorJitter(0.3, 0.3, 0.3, 0.3))

        transform.append(T.ToTensor())
        transform = T.Compose(transform)
        image_xy_res = 200 * scale

        # get image and labels (voxels)
        image_path = os.path.join(self.image_path, image_name)
        image = self._get_augment_image(image_path, center, scale,
                                        angle, is_hflip, transform)
        voxels = self._get_voxels(part, center, image_xy_res, angle, zind)

        return image, voxels

    @staticmethod
    def _get_augment_image(image_path, center, scale,
                           angle=0, hflip=False, transform=None, target_res=256):
        image = Image.open(image_path)

        width, height = image.size
        center = Vector2(center)  # assign new array

        crop_ratio = 200 * scale / target_res

        if crop_ratio >= 2:  # if box size is greater than two time of resolution px
            # scale down image
            height = math.floor(height / crop_ratio)
            width = math.floor(width / crop_ratio)

            if max([height, width]) < 2:
                # Zoomed out so much that the image is now a single pixel or less
                raise ValueError("Width or height is invalid!")

            image = image.resize((width, height), Image.BILINEAR)

            center /= crop_ratio
            scale /= crop_ratio

        ul = (center - 200 * scale / 2).astype(int)
        br = (center + 200 * scale / 2).astype(int)  # Vector2

        if crop_ratio >= 2:  # force image size 256 x 256
            br -= (br - ul - target_res)

        pad_length = math.ceil(((ul - br).length - (br.x - ul.x)) / 2)

        if angle != 0:
            ul -= pad_length
            br += pad_length

        crop_src = [max(0, ul.x), max(0, ul.y), min(width, br.x), min(height, br.y)]
        crop_dst = [max(0, -ul.x), max(0, -ul.y), min(width, br.x) - ul.x, min(height, br.y) - ul.y]
        crop_image = image.crop(crop_src)

        new_image = Image.new("RGB", (br.x - ul.x, br.y - ul.y))
        new_image.paste(crop_image, box=crop_dst)

        if hflip:
            new_image = new_image.transpose(Image.FLIP_LEFT_RIGHT)

        if angle != 0:
            new_image = new_image.rotate(angle, resample=Image.BILINEAR)
            new_image = new_image.crop(box=(pad_length, pad_length,
                                            new_image.width - pad_length,
                                            new_image.height - pad_length))

        if crop_ratio < 2:
            new_image = new_image.resize((target_res, target_res), Image.BILINEAR)

        if transform is not None:
            new_image = transform(new_image)

        return new_image

    def _get_voxels(self, part, center, image_xy_res, angle, zidx):
        part = torch.from_numpy(part).float()
        center = torch.from_numpy(center).float()
        zidx = torch.from_numpy(zidx).float()

        # for vx in self.voxels:
        #     vx.zero_()
        voxels = list()
        for z_res in self.voxel_z_res_list:
            vx = torch.zeros(self.joints, z_res, self.voxel_xy_res, self.voxel_xy_res)
            voxels.append(vx)

        xy = self.voxel_xy_res * (part - center) / image_xy_res + self.voxel_xy_res * 0.5

        if angle != 0.0:
            xy = xy - self.voxel_xy_res / 2
            cos = math.cos(angle * math.pi / 180)
            sin = math.sin(angle * math.pi / 180)
            x = sin * xy[:, 1] + cos * xy[:, 0]
            y = cos * xy[:, 1] - sin * xy[:, 0]
            xy[:, 0] = x
            xy[:, 1] = y
            xy = xy + self.voxel_xy_res / 2

        zidx = torch.ceil(zidx.unsqueeze(-1) * self.z_list / self.z_list[-1]) - 1
        zidx = zidx.short().t()
        zpad = torch.floor(self.heatmap_z_coeff / 2).short()

        xy = xy.short()
        pad = 3 * self.heatmap_xy_coeff

        dst = [torch.clamp(xy - pad, min=0), torch.clamp(xy + pad + 1, max=self.voxel_xy_res, min=0)]
        src = [torch.clamp(pad - xy, min=0), pad + 1 + torch.clamp(self.voxel_xy_res - xy - 1, max=pad)]

        # z_res_cumsum = np.insert(np.cumsum(self.voxel_z_res_list), 0, 0)
        # voxels = np.zeros((len(part), z_res_cumsum[-1], self.voxel_xy_res, self.voxel_xy_res), dtype=np.float32)  # JVHW

        zdsts, zsrcs = list(), list()
        for z, z_res, pad in zip(zidx, self.voxel_z_res_list.short(), zpad):
            zdst = [torch.clamp(z - pad, min=0), torch.clamp(z + pad + 1, max=z_res, min=0)]  # BJ
            zsrc = [torch.clamp(pad - z, min=0), pad + 1 + torch.clamp(z_res - z - 1, max=pad)]  # BJ
            zdsts.append(zdst)
            zsrcs.append(zsrc)

        for (vx, zdst, zsrc, g), jt in itertools.product(zip(voxels, zdsts, zsrcs, self.gaussians),
                                                         range(self.joints)):
            if xy[jt, 0] < 0 or self.voxel_xy_res <= xy[jt, 0] or \
                    xy[jt, 1] < 0 or self.voxel_xy_res <= xy[jt, 1]:
                continue

            z_dst_slice = slice(zdst[0][jt], zdst[1][jt])
            y_dst_slice = slice(dst[0][jt, 1], dst[1][jt, 1])
            x_dst_slice = slice(dst[0][jt, 0], dst[1][jt, 0])

            z_src_slice = slice(zsrc[0][jt], zsrc[1][jt])
            y_src_slice = slice(src[0][jt, 1], src[1][jt, 1])
            x_src_slice = slice(src[0][jt, 0], src[1][jt, 0])

            vx[jt, z_dst_slice, y_dst_slice, x_dst_slice] = g[z_src_slice, y_src_slice, x_src_slice]

        return voxels

    def change_current_split_set(self):
        self.current_split_set = self.current_split_set + 1
        if self.current_split_set >= len(self.data):
            self._shuffle_and_split()
            self.current_split_set = 0
