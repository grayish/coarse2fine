from enum import Enum
from .task import Task


class Annotation(Enum):
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
annotations[str(Task.Train)] = [
    Annotation.S,
    Annotation.Center,
    Annotation.Part,
    Annotation.Scale,
    Annotation.Z,
]
annotations[str(Task.Valid)] = [
    Annotation.Center,
    Annotation.Scale,
]