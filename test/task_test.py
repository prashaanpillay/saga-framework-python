import pytest
from unittest.mock import MagicMock, patch
from source.context import Context
from source.errors.task_execution_exception import TaskExecutionException
from source.task_status import TaskStatus
from source.task import Task
import datetime


class TestTask(Task):
    def _run(self, context: Context):
        pass


class ContextUsingTask(Task):
    def _run(self, context: Context):
        value = context.get('key')
        context.set('key', value + 1)

class CompensationTask(Task):
    def _run(self, context: Context):
        value = context.get('key')
        context.set('key', value - 1)

@pytest.fixture
def context():
    return Context()


@pytest.fixture
def task():
    return TestTask(name="TestTask")


def test_task_initialization(task):
    assert task.name == "TestTask"
    assert task.status == TaskStatus.PENDING
    assert isinstance(task.state_durations, dict)
    assert all(isinstance(duration, datetime.timedelta) for duration in task.state_durations.values())


def test_task_execute_success(task, context):
    with patch.object(task, '_run') as mock_run:
        task.execute(context)
        assert task.status == TaskStatus.COMPLETED
        mock_run.assert_called_once()


def test_task_execute_failure(task, context):
    task.compensation = MagicMock()
    with patch.object(task, '_run', side_effect=TaskExecutionException):
        with patch('source.task.log_task_execution_error') as mock_log_error:
            with patch.object(task, '_handle_failure') as mock_handle_failure:
                task.execute(context)
                assert task.status == TaskStatus.FAILED
                mock_log_error.assert_called_once()
                mock_handle_failure.assert_called_once()


def test_task_compensate(task, context):
    compensation_task = MagicMock()
    task.compensation = compensation_task
    task.compensate(context)
    compensation_task.execute.assert_called_once_with(context)


def test_execute_failing_executes_compensation(task, context):
    compensation_task = MagicMock()
    task = TestTask(name="TestTask", compensation=compensation_task)

    with patch.object(task, '_run', side_effect=TaskExecutionException):
        with patch('source.task.log_task_execution_error') as mock_log_error:
            task.execute(context)
            assert task.status == TaskStatus.FAILED
            mock_log_error.assert_called_once()
            compensation_task.execute.assert_called_once_with(context)


def test_update_status_updates_durations(task):
    initial_time = task.last_status_update_time
    with patch('source.task.Task._current_utc_time', return_value=initial_time + datetime.timedelta(seconds=5)):
        task._update_status(TaskStatus.IN_PROGRESS)
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.state_durations[TaskStatus.PENDING] == datetime.timedelta(seconds=5)


def test_get_state_durations(task):
    durations = task.get_state_durations()
    assert isinstance(durations, dict)
    assert all(isinstance(value, str) for value in durations.values())


def test_log_state_durations(task):
    with patch('source.task.log_task_execution_progress') as mock_log_progress:
        task.log_state_durations()
        mock_log_progress.assert_called_once()

def test_context_using_task(context):
    compensate = CompensationTask(name="CompensationTask")
    task = ContextUsingTask(name="ContextUsingTask", compensation = compensate)
    context.set('key', 1)
    task.execute(context)
    assert context.get('key') == 2
    task.compensate(context)
    assert context.get('key') == 1