from enum import Enum , auto

class RequestState(Enum):
    WAITING = auto()
    RUNNING = auto()
    PREEMPTED = auto()
    FINISHED = auto()
    FAILED = auto()
