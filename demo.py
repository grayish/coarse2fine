import numpy as np
import torch
from torchvision import transforms

Color = torch.FloatTensor(
    [[0, 0, 0.5],
     [0, 0, 1],
     [0, 1, 0],
     [1, 1, 0],
     [1, 0, 0]]
)


def merge_to_color_heatmap(batch_heatmaps):
    batch, joints, depth, height, width = batch_heatmaps.size()

    batch_heatmaps_flat = batch_heatmaps.view(batch, joints, depth, -1)
    max_depth_idx = batch_heatmaps_flat.max(-1)[0].max(-1)[1]

    test = list()
    for b_idx in range(batch):
        test2 = list()
        for j_idx in range(joints):
            test2.append(batch_heatmaps[b_idx, j_idx, max_depth_idx[b_idx, j_idx], :, :].view(1, 1, height, width))

        test.append(torch.cat(test2, dim=1))

    batch_heatmaps = torch.cat(test, dim=0)
    # batch_heatmaps = torch.cat(
    #     [torch.cat(
    #         [batch_heatmaps[b_idx, j_idx, max_depth_idx[b_idx, j_idx], :, :] for j_idx in range(joints)], dim=2)
    #      for b_idx in range(batch)], dim=0)

    heatmaps = batch_heatmaps.clamp(0, 1.).view(-1)

    frac = torch.div(heatmaps, 0.25)
    lower_indices, upper_indices = torch.floor(frac).long(), torch.ceil(frac).long()

    t = frac - torch.floor(frac)
    t = t.view(-1, 1)

    k = Color.index_select(0, lower_indices)
    k_1 = Color.index_select(0, upper_indices)

    color_heatmap = (1.0 - t) * k + t * k_1
    color_heatmap = color_heatmap.view(batch, joints, height, width, 3)
    color_heatmap = color_heatmap.permute(0, 4, 2, 3, 1)  # B3HWC
    color_heatmap, _ = torch.max(color_heatmap, 4)  # B3HW

    return color_heatmap


T = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.ToTensor()
])


def get_merged_image(heatmaps, images):
    heatmaps = merge_to_color_heatmap(heatmaps)
    # heatmaps = heatmaps.permute(0, 2, 3, 1)  # NHWC

    resized_heatmaps = list()
    for idx, ht in enumerate(heatmaps):
        color_ht = T(ht)
        # color_ht = skimage.transform.resize(ht.numpy(), (256, 256), mode='constant')
        resized_heatmaps.append(color_ht)

    resized_heatmaps = np.stack(resized_heatmaps, axis=0)

    # images = images.transpose(0, 2, 3, 1) * 0.6
    images = images * 0.6
    overlayed_image = np.clip(images + resized_heatmaps * 0.4, 0, 1.)

    # overlayed_image = overlayed_image.transpose(0, 3, 1, 2)

    return overlayed_image
    # return viz.images(tensor=overlayed_image, nrow=3, win=window)
