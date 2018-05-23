import h5py
import torch.utils.data as torch_data
from .util import *
from .task import tasks, Task
from .annotation import annotations, Annotation
from os.path import join
from functools import lru_cache


class Data(torch_data.Dataset):

    def __init__(self, annotation_path, image_path, subjects, task,
                 heatmap_xy_coefficient, voxel_xy_resolution, voxel_z_resolutions):

        self.voxel_z_resolutions = voxel_z_resolutions
        self.voxel_xy_resolution = voxel_xy_resolution
        self.heatmap_xy_coefficient = heatmap_xy_coefficient
        self.task = task
        self.subjects = subjects
        self.image_path = image_path
        self.data = dict()

        for task in tasks:
            annotation_file = join(annotation_path, '%s.h5' % task)
            file = h5py.File(annotation_file, 'r')
            self.data[str(task)] = dict()
            for annotation in annotations[str(task)]:
                self.data[str(task)][str(annotation)] = file[str(annotation)]

            image_name_file = join(annotation_path, '%s_images.txt' % task)
            self.data[str(task)][str(Annotation.Image)] = np.genfromtxt(image_name_file, dtype=str)

    def __len__(self):
        return len(self.data[str(self.task)][str(Annotation.Image)])

    def __getitem__(self, index):
        raw_data = (self.data[str(self.task)][str(Annotation.Image)][index],)
        for annotation in annotations[str(self.task)]:
            raw_data = raw_data + (self.data[str(self.task)][str(annotation)][index],)

        image = None
        voxels = None

        if self.task == Task.Train:
            image, voxels = self.preprocess(raw_data)

        return raw_data, image, voxels

    def __add__(self, item):
        pass

    def preprocess(self, raw_data):
        image_name, S, center, part, scale, zind = raw_data

        voxels = list()

        # Extract subject and camera name from an image name.
        subject, _, camera, _ = decode_image_name(image_name)

        # Pre-calculate constants.
        image_xy_resolution = 200 * scale
        scale_factor = 2 ** rand(0.25)
        rotate_factor = rand(30) if random.random() <= 0.4 else 0

        # Crop RGB image.
        image = skimage.img_as_float(skimage.io.imread('%s/%s/%s' % (self.image_path, subject, image_name)))
        image = crop_image(image, center, scale, 0, 256)
        # imageio.imwrite('rgb.png', image)

        # Build voxel.
        voxel_z_fine_resolution = self.voxel_z_resolutions[-1]
        for voxel_z_coarse_resolution in self.voxel_z_resolutions:
            # heatmap_z_coefficient is 1, 1, 1, 3, 5, 7, 13 for 1, 2, 4, 8, 16, 32, 64.
            heatmap_z_coefficient = 2 * math.floor(
                (6 * self.heatmap_xy_coefficient * voxel_z_coarse_resolution / voxel_z_fine_resolution + 1) / 2) + 1

            # Convert the coordinate from a RGB image to a cropped RGB image.
            xy = self.voxel_xy_resolution * (part - center) / image_xy_resolution + self.voxel_xy_resolution * 0.5

            voxel = np.zeros(shape=(self.voxel_xy_resolution, self.voxel_xy_resolution,
                                    len(part) * voxel_z_coarse_resolution))
            for part_idx in range(len(part)):
                # zind range (1, 64)
                # z range (0, 63)
                z = math.ceil(zind[part_idx] * voxel_z_coarse_resolution / voxel_z_fine_resolution) - 1
                voxel[:, :, part_idx * voxel_z_coarse_resolution: (part_idx + 1) * voxel_z_coarse_resolution] \
                    = generate_voxel(
                    self.voxel_xy_resolution, voxel_z_coarse_resolution,
                    xy[part_idx], z,
                    self.heatmap_xy_coefficient, heatmap_z_coefficient)

            if self.task == Task.Train:
                voxel = crop_image(
                    voxel,
                    [(self.voxel_xy_resolution - 1) / 2, (self.voxel_xy_resolution - 1) / 2],
                    self.voxel_xy_resolution * scale * scale_factor / 200,
                    rotate_factor,
                    self.voxel_xy_resolution)
            voxel = voxel.transpose(2, 0, 1)
            voxels.append(voxel)

        if self.task == Task.Train:
            image = crop_image(
                image,
                [(256 - 1) / 2, (256 - 1) / 2],
                256 * scale * scale_factor / 200,
                rotate_factor,
                256)
        image = image.transpose(2, 0, 1)
        return image, voxels
