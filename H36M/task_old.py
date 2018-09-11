from enum import Enum


class OldTask(Enum):
    Train = 'train'
    Valid = 'valid'

    def to_str(self):
        return str(self)

    def __str__(self):
        return self.value

    @staticmethod
    def from_str(string):
        for task in OldTask:
            if str(task) == string:
                return task
        return None


tasks = [
    OldTask.Train,
    OldTask.Valid,
]
