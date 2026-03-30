from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class TaskWindow:
    """Represents the range of task indices assigned to a worker."""
    start: int
    end: int


class SchedulerABC(ABC):
    def __init__(self, rank: int, world_size: int, num_tasks: int) -> None:
        self.rank = rank
        self.world_size = world_size
        self.num_tasks = num_tasks
        self.task_window = TaskWindow(0, 0)
        self.drained = False

    @abstractmethod
    def update_task_window(self) -> None:
        """Update the task window for the current rank."""
        pass

    def is_drained(self) -> bool:
        """Check if all tasks have been scheduled."""
        return self.drained
