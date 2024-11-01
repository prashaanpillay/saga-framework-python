import datetime
from typing import Callable, Dict, List, Optional
from source.context import Context
from source.errors.task_execution_exception import TaskExecutionException
from source.logging.saga_logging import log_rollback_task_execution_error, log_task_execution_progress
from source.task import Task
from source.task_status import TaskStatus


class RollbackTask(Task):
    def __init__(
        self,
        name: str,
        compensation: Optional['Task'] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize a Task instance.

        Args:
            name (str): The name of the task.
            compensation (Optional[Task]): A compensating task in case of failure.
            metadata (Optional[Dict[str, str]]): Additional metadata for the task.
        """
        self.name = name
        self.metadata = metadata or {}
        self.compensation = compensation
        self.status = TaskStatus.PENDING
        self.updated_time_utc = self._current_utc_time()
        self.task_attributes = ["compensation"]

        # Initialize state durations with zero timedelta for each possible TaskStatus
        self.state_durations: Dict[TaskStatus, datetime.timedelta] = {
            status: datetime.timedelta() for status in TaskStatus
        }

        # Record the time when the task was initialized
        self.last_status_update_time: datetime.datetime = self.updated_time_utc

        # Hook points for before and after execution
        self.on_before_execution: List[Callable[[], None]] = []
        self.on_after_execution: List[Callable[[], None]] = []

    def execute(self, context: Context):
        """
        Execute the task, updating its status accordingly.

        Args:
            context (Context): The context in which the task is executed.
        """
        # Execute all 'before' hooks
        self._execute_before_hooks()

        self._update_status(TaskStatus.IN_PROGRESS)
        try:
            self._run(context)
            self._update_status(TaskStatus.COMPLETED)
        except TaskExecutionException as e:
            if self.compensation is not None:
                log_rollback_task_execution_error(self.name, self.compensation.name, e)
            self._update_status(TaskStatus.FAILED)
            self._handle_failure(context)

        # Execute all 'after' hooks
        self._execute_after_hooks()

    def compensate(self, context: Context):
        """
        Execute the compensating task if available.

        Args:
            context (Context): The context in which the compensation is executed.
        """
        if self.compensation is not None:
            self.compensation.execute(context)

    def _handle_failure(self, context: Context):
        """
        Handle task failure by triggering compensation.

        Args:
            context (Context): The context in which the task failed.
        """
        self.compensate(context)
