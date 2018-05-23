import json
from dotmap import DotMap
import H36M
from H36M.task import Task

with open('config.json') as fd:
    config = DotMap(json.load(fd))

data = H36M.Data(
    annotation_path=config.annotation_path,
    image_path=config.image_path,
    subjects=config.subjects,
    task=Task.from_str(config.task),
)

print(len(data))
print(data[0])
