import pytest
from unittest.mock import MagicMock, patch
from source.saga import Saga
from source.task import Task
from source.context import Context
from source.errors.saga_execution_exception import SagaExecutionException
from source.errors.saga_compensation_execution_exception import SagaCompensationExecutionException


class TestTask(Task):
    def _run(self, context: Context):
        pass  # Simple implementation for testing


@pytest.fixture
def context():
    return Context()


@pytest.fixture
def saga():
    return Saga()


@pytest.fixture
def task():
    return TestTask(name="TestTask")


@pytest.fixture
def failing_task():
    failing_task = TestTask(name="FailingTask")
    failing_task._run = MagicMock(side_effect=SagaExecutionException)
    return failing_task


def test_saga_initialization(saga):
    assert saga._tasks == []
    assert saga._completed_tasks == []
    assert isinstance(saga._context, Context)


def test_add_task(saga, task):
    saga.add_task(task)
    assert len(saga._tasks) == 1
    assert saga._tasks[0] == task


def test_execute_saga_success(saga, task, context):
    with patch.object(task, 'execute') as mock_execute:
        saga.add_task(task)
        saga.execute()
        mock_execute.assert_called_once_with(saga._context)
        assert len(saga._completed_tasks) == 1
        assert saga._completed_tasks[0] == task


def test_execute_saga_failure(saga, failing_task, task):
    # Add one successful task and one failing task
    saga.add_task(task)
    saga.add_task(failing_task)

    with patch.object(failing_task, 'execute', side_effect=SagaExecutionException), \
            patch('source.saga.log_saga_execution_error') as mock_log_error, \
            patch.object(saga, '_compensate') as mock_compensate:

        with pytest.raises(SagaExecutionException):
            saga.execute()

        # Verify compensation was triggered after failure
        mock_compensate.assert_called_once()

        # Verify logging for the failed task
        mock_log_error.assert_called_once()


def test_compensation_on_failure(saga, task, failing_task):
    compensating_task = MagicMock()
    task.compensation = compensating_task

    saga.add_task(task)
    saga.add_task(failing_task)

    with patch('source.saga.log_saga_execution_error'), \
            patch('source.saga.log_saga_compensation_execution_error') as mock_log_compensation_error:

        with pytest.raises(SagaExecutionException):
            saga.execute()

        # Ensure that the compensation task was executed
        compensating_task.execute.assert_called_once()

        # Ensure no compensation logging error (since compensation should succeed)
        mock_log_compensation_error.assert_not_called()


def test_compensation_failure(saga, task, failing_task):
    # Mock compensation task to fail
    compensating_task = MagicMock()
    compensating_task.execute.side_effect = SagaCompensationExecutionException
    task.compensation = compensating_task

    saga.add_task(task)
    saga.add_task(failing_task)

    with patch('source.saga.log_saga_execution_error'), \
            patch('source.saga.log_saga_compensation_execution_error') as mock_log_compensation_error:

        with pytest.raises(SagaExecutionException):
            saga.execute()

        # Ensure compensation was attempted but failed
        compensating_task.execute.assert_called_once()
        mock_log_compensation_error.assert_called_once()
