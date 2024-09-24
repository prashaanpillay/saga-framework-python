import datetime
from typing import Dict, Optional
from abc import ABC, abstractmethod

from source.context import Context
from source.errors.task_execution_exception import TaskExecutionException
from source.logging.saga_logging import log_task_execution_error, log_task_execution_progress
from source.task_status import TaskStatus


class Task(ABC):
    def __init__(
            self,
            name: str,
            inputs: Optional[Dict[str, str]] = None,
            compensation: Optional['Task'] = None,
            metadata: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize a Task instance.

        Args:
            name (str): The name of the task.
            inputs (Optional[Dict[str, str]]): Input parameters for the task.
            compensation (Optional[Task]): A compensating task in case of failure.
            metadata (Optional[Dict[str, str]]): Additional metadata for the task.
        """
        self.name = name
        self.inputs = inputs or {}
        self.metadata = metadata or {}
        self.compensation = compensation
        self.status = TaskStatus.PENDING
        self.updated_time_utc = self._current_utc_time()

        # Initialize state durations with zero timedelta for each possible TaskStatus
        self.state_durations: Dict[TaskStatus, datetime.timedelta] = {
            status: datetime.timedelta() for status in TaskStatus
        }

        # Record the time when the task was initialized
        self.last_status_update_time: datetime.datetime = self.updated_time_utc

    def execute(self, context: Context):
        """
        Execute the task, updating its status accordingly.

        Args:
            context (Context): The context in which the task is executed.
        """
        self._update_status(TaskStatus.IN_PROGRESS)
        try:
            self._run(context)
            self._update_status(TaskStatus.COMPLETED)
        except TaskExecutionException as e:
            log_task_execution_error(self.name, self.compensation.name, e)
            self._update_status(TaskStatus.FAILED)
            self._handle_failure(context)

    def compensate(self, context: Context):
        """
        Execute the compensating task if available.

        Args:
            context (Context): The context in which the compensation is executed.
        """
        if self.compensation is not None:
            self.compensation.execute(context)

    def __str__(self) -> str:
        """
        Return a string representation of the task.

        Returns:
            str: The name and status of the task.
        """
        return f"{self.name} : {self.status}"

    @abstractmethod
    def _run(self, context: Context):
        """
        The core logic of the task to be implemented by subclasses.

        Args:
            context (Context): The context in which the task is executed.
        """
        pass

    def _update_status(self, new_status: TaskStatus):
        """
        Update the status and track the duration spent in the previous status.

        Args:
            new_status (TaskStatus): The new status of the task.
        """
        current_time = self._current_utc_time()
        duration = current_time - self.last_status_update_time

        # Add the duration to the previous status
        self.state_durations[self.status] += duration

        # Update the status and the last status update time
        self.status = new_status
        self.updated_time_utc = current_time
        self.last_status_update_time = current_time

    def _handle_failure(self, context: Context):
        """
        Handle task failure by triggering compensation.

        Args:
            context (Context): The context in which the task failed.
        """
        self.compensate(context)

    @staticmethod
    def _current_utc_time() -> datetime.datetime:
        """
        Get the current UTC time.

        Returns:
            datetime.datetime: The current time in UTC.
        """
        return datetime.datetime.now(datetime.timezone.utc)

    def get_state_durations(self) -> Dict[str, str]:
        """
        Retrieve the durations spent in each state formatted as strings.

        Returns:
            Dict[str, str]: A dictionary with state names as keys and durations as values.
        """
        return {status.name: str(duration) for status, duration in self.state_durations.items()}

    def log_state_durations(self):
        """
        Log the durations spent in each state.
        """
        durations = self.get_state_durations()
        log_task_execution_progress(self.name, durations)
