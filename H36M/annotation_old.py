from enum import Enum
from .task_old import OldTask


class OldAnnotation(Enum):
    S = 'S'
    Center = 'center'
    Part = 'part'
    Scale = 'scale'
    Z = 'zind'
    Image = 'image'

    def to_str(self):
        return str(self)

    def __str__(self):
        return self.value


annotations = dict()
annotations[str(OldTask.Train)] = [
    OldAnnotation.S,
    OldAnnotation.Center,
    OldAnnotation.Part,
    OldAnnotation.Scale,
    OldAnnotation.Z,
]
annotations[str(OldTask.Valid)] = [
    OldAnnotation.S,
    OldAnnotation.Center,
    OldAnnotation.Part,
    OldAnnotation.Scale,
    OldAnnotation.Z,
]