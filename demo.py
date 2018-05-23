import json
import H36M
from dotmap import DotMap
from log import log
from H36M.task import Task
from torch.utils.data import DataLoader


with open('config.json') as fd:
    config = DotMap(json.load(fd))

if __name__ == "__main__":
    log.info('Reboot ')

    log.debug(config)

    log.info('Loading Human3.6M data...')
    loader = DataLoader(
        H36M.Data(
            annotation_path=config.annotation_path,
            image_path=config.image_path,
            subjects=config.subjects,
            task=Task.from_str(config.task),
            heatmap_xy_coefficient=config.heatmap_xy_coefficient,
            voxel_xy_resolution=config.voxel_xy_resolution,
            voxel_z_resolutions=config.voxel_z_resolutions,
        ),
        config.batch, shuffle=True, num_workers=config.workers, pin_memory=True,
    )

