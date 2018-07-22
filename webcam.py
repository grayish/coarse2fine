import cv2 as cv

import os
import numpy as np
import torch
import torch.nn as nn
import scipy.io
from dotmap import DotMap
from torch.utils.data import DataLoader
from tqdm import tqdm
from visdom import Visdom

import H36M
from H36M.annotation import Annotation
from H36M.task import Task
from demo import get_merged_image
from hourglass import StackedHourglass
from log import log

config = DotMap({
    "annotation_path": "./Human3.6M/converted/annot",
    "image_path": "./Human3.6M/converted",
    "pretrained_path": "./pretrained/",
    "subjects": [1, 5, 6, 7, 8, 9, 11],
    "task": str(Task.Valid),
    "num_parts": 17,
    "heatmap_xy_coefficient": 2,
    "voxel_xy_resolution": 64,
    "voxel_z_resolutions": [1, 2, 4, 64],
    "batch": 6,
    "workers": 8,
    "epoch": 100
})

model = StackedHourglass(config.voxel_z_resolutions, 256, config.num_parts)
step = np.zeros([1], dtype=np.uint32)

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
    model.load_state_dict(pretrained_model['state'])
    step[0] = pretrained_model['step']
else:
    pretrained_model = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

cap = cv.VideoCapture(0)

with torch.set_grad_enabled(False):
    while (True):
        ret, frame = cap.read()

        image = cv.resize(frame, (256, 256))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = np.asarray(image / 255).astype(np.float32)
        # image[:, :, 2] += 0.4
        # image[:, :, 2] = np.clip(image[:, :, 2], 0, 1)
        image = image.transpose(2, 0, 1)
        image = torch.Tensor([image])
        image = image.to(device)

        output = model(image)

        image = image[0]
        output = output[-1].view(-1, config.num_parts, config.voxel_z_resolutions[-1],
                              config.voxel_xy_resolution,
                              config.voxel_xy_resolution).cpu().detach()[0]

        z_res = config.voxel_z_resolutions[-1]
        x_res = y_res = config.voxel_xy_resolution
        joint_dict = list()
        for joint in range(config.num_parts):
            joint_prediction = output[joint, :, :, :]
            joint_prediction = joint_prediction.view(-1)  # flatten

            _, part = joint_prediction.max(0)
            z = part / (x_res * y_res)
            y = (part % (x_res * y_res)) / y_res
            x = (part % (x_res * y_res)) % y_res

            x, y, z = [int(x), int(y), int(z)]
            in_volume_space = [(x + 1) * 4 - 4, (y + 1) * 4 - 4]

            joint_dict.append(np.asarray(in_volume_space).astype(np.int32))

        image = np.asarray(image.data * 255).astype(np.uint8).transpose(1, 2, 0)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_CV = image.copy()

        # Left leg
        left_leg_color = (0, 0, 255)
        cv.line(image_CV, tuple(joint_dict[0]), tuple(joint_dict[1]), left_leg_color, 5)
        cv.line(image_CV, tuple(joint_dict[1]), tuple(joint_dict[2]), left_leg_color, 5)
        cv.line(image_CV, tuple(joint_dict[2]), tuple(joint_dict[3]), left_leg_color, 5)

        # Right leg
        right_leg_color = (255, 0, 0)
        cv.line(image_CV, tuple(joint_dict[0]), tuple(joint_dict[4]), right_leg_color, 5)
        cv.line(image_CV, tuple(joint_dict[4]), tuple(joint_dict[5]), right_leg_color, 5)
        cv.line(image_CV, tuple(joint_dict[5]), tuple(joint_dict[6]), right_leg_color, 5)

        # Spine
        spine_color = (0, 255, 0)
        cv.line(image_CV, tuple(joint_dict[0]), tuple(joint_dict[7]), spine_color, 5)
        cv.line(image_CV, tuple(joint_dict[7]), tuple(joint_dict[8]), spine_color, 5)
        cv.line(image_CV, tuple(joint_dict[8]), tuple(joint_dict[9]), spine_color, 5)
        cv.line(image_CV, tuple(joint_dict[9]), tuple(joint_dict[10]), spine_color, 5)

        # Left arm
        left_arm_color = (0, 0, 255)
        cv.line(image_CV, tuple(joint_dict[8]), tuple(joint_dict[11]), left_arm_color, 5)
        cv.line(image_CV, tuple(joint_dict[11]), tuple(joint_dict[12]), left_arm_color, 5)
        cv.line(image_CV, tuple(joint_dict[12]), tuple(joint_dict[13]), left_arm_color, 5)

        # Right arm
        right_arm_color = (255, 0, 0)
        cv.line(image_CV, tuple(joint_dict[8]), tuple(joint_dict[14]), right_arm_color, 5)
        cv.line(image_CV, tuple(joint_dict[14]), tuple(joint_dict[15]), right_arm_color, 5)
        cv.line(image_CV, tuple(joint_dict[15]), tuple(joint_dict[16]), right_arm_color, 5)

        image_CV = cv.resize(image_CV, (1024, 1024))
        cv.imshow('frame', image_CV)

        if cv.waitKey(1) and 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
