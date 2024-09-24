from loguru import logger
from typing import Optional, Dict
import sys

from source.errors.saga_compensation_execution_exception import SagaCompensationExecutionException
from source.errors.saga_execution_exception import SagaExecutionException
from source.errors.task_execution_exception import TaskExecutionException

logger.remove()
logger.add(sys.stdout, format="{time} {level} {message}", level="DEBUG")


def log_task_execution_error(
        task_name: str,
        compensation_name: Optional[str],
        exception: Optional[TaskExecutionException] = None
):
    """
    Logs an error when executing a task.

    :param task_name: The name of the task.
    :param compensation: The compensation task.
    :param exception: The exception that was raised.
    """
    message = f"Code: {TaskExecutionException.ERROR_CODE}. Task Failed: {task_name}."
    if compensation_name:
        message = f"{message} Starting Compensation Task: {compensation_name}."
    if exception:
        logger.error(message, exc_info=exception)
    else:
        logger.error(message)


def log_saga_execution_error(failure_task: str, exception: SagaExecutionException):
    """
    Logs an error when executing a saga.

    :param failure_task: The name of the task that failed.
    :param exception: The exception that was raised.
    """
    message = f"Code: {SagaExecutionException.ERROR_CODE}. Saga Failed on Task: {failure_task}."
    logger.error(message, exc_info=exception)


def log_saga_compensation_execution_error(failure_compensation_task: str,
                                          exception: SagaCompensationExecutionException):
    """
    Logs an error when executing a saga compensation.

    :param failure_task: The name of the task that failed.
    :param exception: The exception that was raised.
    """
    message = f"Code: {SagaCompensationExecutionException.ERROR_CODE}. Saga Failed on Compensation Task: {failure_compensation_task}."
    logger.error(message, exc_info=exception)


def log_task_execution_progress(task_name: str, state_durations: Dict[str, str]):
    """
    Logs the progress of a task through its states.

    :param task_name: The name of the task.
    :param state_durations: A dictionary of state durations.
    """
    for state, duration in state_durations.items():
        print(f"Task '{task_name}' spent {duration} in state '{state}'")
