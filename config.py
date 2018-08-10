import torch

from H36M import Task


class Config(object):
    def __init__(self):
        self.annotation_path = "./Human3.6M/annot"
        self.image_path = "./Human3.6M/images"
        self.pretrained_path = "./pretrained/"
        self.subjects = [1, 5, 6, 7, 8, 9, 11]
        self.task = Task.Train
        self.num_parts = 17
        self.heatmap_xy_coefficient = 2
        self.voxel_xy_res = 64
        self.voxel_z_res = [1, 2, 4, 64]
        self.batch = 4
        self.workers = 8
        self.epoch = 10
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
