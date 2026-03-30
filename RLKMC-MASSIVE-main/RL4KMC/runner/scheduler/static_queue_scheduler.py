from .scheduler import SchedulerABC


class StaticQueueScheduler(SchedulerABC):
    def update_task_window(self) -> None:
        """Update the task window for the current rank."""
        if self.drained:
            return

        tasks_per_rank = (self.num_tasks + self.world_size - 1) // self.world_size
        self.task_window.start = self.rank * tasks_per_rank
        self.task_window.end = min((self.rank + 1) * tasks_per_rank, self.num_tasks)
        self.drained = True
