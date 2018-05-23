import json
import H36M
import imageio
import numpy as np
from dotmap import DotMap
from H36M.task import Task

with open('config.json') as fd:
    config = DotMap(json.load(fd))

data = H36M.Data(
    annotation_path=config.annotation_path,
    image_path=config.image_path,
    subjects=config.subjects,
    task=Task.from_str(config.task),
    heatmap_xy_coefficient=config.heatmap_xy_coefficient,
    voxel_xy_resolution=config.voxel_xy_resolution,
    voxel_z_resolutions=config.voxel_z_resolutions,
)


print(len(data))
raw_data, image, voxels = data[22783]
print(raw_data)
imageio.imwrite('tmp/test.png', image)
voxel_idx = -1
for z in range(config.voxel_z_resolutions[voxel_idx]):
    for x in range(config.voxel_xy_resolution):
        for y in range(config.voxel_xy_resolution):
            voxels[voxel_idx][y, x, z] = np.max(voxels[voxel_idx][y, x, [config.voxel_z_resolutions[voxel_idx] * part + z for part in range(config.num_parts)]])
for z in range(config.voxel_z_resolutions[voxel_idx]):
    imageio.imwrite('tmp/%02d_%02d.png' % (config.voxel_z_resolutions[voxel_idx], z), voxels[voxel_idx][:, :, z])
    