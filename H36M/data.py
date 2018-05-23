import h5py
import torch.utils.data as torch_data
from .task import tasks
from .annotation import annotations, Annotation
from os.path import join
from functools import lru_cache


class Data(torch_data.Dataset):

    def __init__(self, annotation_path, image_path, subjects, task):

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
            file = open(image_name_file, 'r')
            self.data[str(task)][str(Annotation.Image)] = list()
            for line in file:
                self.data[str(task)][str(Annotation.Image)].append(line)

    @lru_cache(maxsize=1)
    def __len__(self):
        return len(self.data[str(self.task)][str(Annotation.Image)])

    def __getitem__(self, index):
        raw_data = ()
        for annotation in annotations[str(self.task)]:
            raw_data = raw_data + (self.data[str(self.task)][str(annotation)][index],)
        return raw_data

    def __add__(self, item):
        pass
