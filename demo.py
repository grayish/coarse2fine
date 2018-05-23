import json
import H36M
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