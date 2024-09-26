from typing import List, Optional

from source.context import Context
from source.errors.saga_compensation_execution_exception import SagaCompensationExecutionException
from source.errors.saga_execution_exception import SagaExecutionException
from source.logging.saga_logging import log_saga_execution_error, log_saga_compensation_execution_error
from source.task import Task


class Saga:
    def __init__(self, context: Optional[Context] = None):
        self._tasks: List[Task] = []
        self._completed_tasks: List[Task] = []
        self._context: Context = context or Context()

    def add_task(self, task: Task) -> None:
        """
        Adds a task to the saga.

        Args:
            task (Task): The task to be added.
        """
        self._tasks.append(task)

    def execute(self) -> None:
        """
        Executes all tasks in the saga sequentially.
        If a task fails, compensates all previously completed tasks.

        Raises:
            SagaExecutionException: If any task fails during execution.
        """
        executing_task: Optional[Task] = None
        try:
            for task in self._tasks:
                executing_task = task
                task.execute(self._context)
                self._completed_tasks.append(task)
                executing_task = None
        except SagaExecutionException as error:
            task_name = executing_task.name
            log_saga_execution_error(task_name, error)
            self._compensate()
            raise

    def _compensate(self) -> None:
        """
        Compensates all completed tasks in reverse order.
        Logs any errors encountered during compensation.
        """
        for task in reversed(self._completed_tasks):
            try:
                task.compensate(self._context)
            except SagaCompensationExecutionException as compensation_error:
                log_saga_compensation_execution_error(task.name, compensation_error)
