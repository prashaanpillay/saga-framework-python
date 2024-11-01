from typing import Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from source.context import Context
from source.errors.parallel_task_requires_tasks_exception import ParallelTaskRequiresTasksException
from source.errors.task_execution_exception import TaskExecutionException
from source.logging.saga_logging import log_rollback_task_execution_error
from source.rollback_task import RollbackTask
from source.task import Task
from source.task_status import TaskStatus


class ParallelTask(RollbackTask):
    def __init__(
        self,
        name: str,
        tasks: List[Task],
        compensation: Optional[Task] = None,
        metadata: Optional[Dict[str, str]] = None,
        max_workers: Optional[int] = None
    ):
        """
        Initialize a ParallelTask instance.

        Args:
            name (str): The name of the parallel task.
            tasks (List[Task]): A list of Task instances to execute in parallel.
            compensation (Optional[Task]): A compensating task in case of failure.
            metadata (Optional[Dict[str, str]]): Additional metadata for the task.
            max_workers (Optional[int]): The maximum number of threads to use for parallel execution.
        """
        super().__init__(name, compensation, metadata)
        if not tasks:
            raise ParallelTaskRequiresTasksException()
        self.tasks = tasks
        self.max_workers = max_workers or len(tasks)  # Default to number of tasks if not specified
        self.task_attributes.append("parallel")

    def _run(self, context: Context):
        """
        Execute all child tasks in parallel.

        Args:
            context (Context): The context in which the tasks are executed.
        """
        # Dictionary to keep track of task futures
        futures_to_task = {}
        # To track successfully completed tasks for compensation in case of failure
        completed_tasks = []

        # Using ThreadPoolExecutor to run tasks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks to the executor
            for task in self.tasks:
                # Add before execution hooks specific to child tasks
                for hook in task.on_before_execution:
                    hook()
                future = executor.submit(task.execute, context)
                futures_to_task[future] = task

            # As each task completes, check for exceptions
            for future in as_completed(futures_to_task):
                task = futures_to_task[future]
                try:
                    future.result()
                    if task.status == TaskStatus.COMPLETED:
                        completed_tasks.append(task)
                    else:
                        raise TaskExecutionException(f"Task {task.name} failed with status {task.status}.")
                except TaskExecutionException as e:
                    log_rollback_task_execution_error(task.name, self.compensation.name if self.compensation else "No Compensation", e)
                    self._update_status(TaskStatus.FAILED)
                    self._handle_failure_parallel(context, completed_tasks)
                    # for f in futures_to_task:
                    #     f.cancel()
                    raise TaskExecutionException(f"ParallelTask {self.name} failed due to task {task.name}.") from e

        # If all tasks completed successfully
        self._update_status(TaskStatus.COMPLETED)

    def _handle_failure_parallel(self, context: Context, completed_tasks: List[Task]):
        """
        Handle failure in ParallelTask by compensating all successfully completed child tasks.

        Args:
            context (Context): The context in which the tasks were executed.
            completed_tasks (List[Task]): The list of tasks that completed successfully before failure.
        """
        # Trigger compensation for each completed task
        for task in reversed(completed_tasks):  # Reverse to compensate in LIFO order
            if task.compensation:
                try:
                    task.compensation.execute(context)
                except TaskExecutionException as e:
                    log_rollback_task_execution_error(task.compensation.name, "No Compensation", e)
        # Trigger the compensation for the ParallelTask itself, if any
        super()._handle_failure(context)

    def compensate(self, context: Context):
        """
        Execute the compensating task for ParallelTask if available.

        Args:
            context (Context): The context in which the compensation is executed.
        """
        # Compensate for ParallelTask's own compensation
        super().compensate(context)
        # Additionally, compensate each child task
        for task in self.tasks:
            if task.compensation and task.status == TaskStatus.COMPLETED:
                task.compensation.execute(context)

    def __str__(self) -> str:
        """
        Return a string representation of the ParallelTask.

        Returns:
            str: The name and status of the ParallelTask.
        """
        return f"{self.name} : {self.status} (ParallelTask with {len(self.tasks)} tasks)"
