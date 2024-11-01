import pytest
from unittest.mock import MagicMock, patch
from source.context import Context
from source.errors.task_execution_exception import TaskExecutionException
from source.rollback_task import RollbackTask
from source.task_status import TaskStatus
import datetime


class TestTask(RollbackTask):
    def _run(self, context: Context):
        pass


class ContextUsingTask(RollbackTask):
    def _run(self, context: Context):
        value = context.get('key')
        context.set('key', value + 1)


class CompensationTask(RollbackTask):
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
        with patch('source.rollback_task.log_rollback_task_execution_error') as mock_log_error:
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
        with patch('source.rollback_task.log_rollback_task_execution_error') as mock_log_error:
            task.execute(context)
            assert task.status == TaskStatus.FAILED
            mock_log_error.assert_called_once()
            compensation_task.execute.assert_called_once_with(context)


def test_update_status_updates_durations(task):
    initial_time = task.last_status_update_time
    with patch('source.rollback_task.RollbackTask._current_utc_time', return_value=initial_time + datetime.timedelta(seconds=5)):
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
    task = ContextUsingTask(name="ContextUsingTask", compensation=compensate)
    context.set('key', 1)
    task.execute(context)
    assert context.get('key') == 2
    task.compensate(context)
    assert context.get('key') == 1


def test_add_before_execution_hook(task, context):
    # Define a mock hook function
    before_hook = MagicMock()

    # Add the before execution hook
    task.add_before_execution_hook(before_hook)

    # Execute the task
    with patch.object(task, '_run') as mock_run:
        task.execute(context)

        # Verify that the before execution hook was called
        before_hook.assert_called_once()

        # Ensure the task's _run method was executed
        mock_run.assert_called_once()


def test_add_multiple_before_execution_hooks(task, context):
    # Define multiple mock hook functions
    before_hook1 = MagicMock()
    before_hook2 = MagicMock()

    # Add the before execution hooks
    task.add_before_execution_hook(before_hook1)
    task.add_before_execution_hook(before_hook2)

    # Execute the task
    with patch.object(task, '_run') as mock_run:
        task.execute(context)

        # Verify that both before execution hooks were called
        before_hook1.assert_called_once()
        before_hook2.assert_called_once()

        # Ensure the task's _run method was executed
        mock_run.assert_called_once()


def test_add_after_execution_hook(task, context):
    # Define a mock hook function
    after_hook = MagicMock()

    # Add the after execution hook
    task.add_after_execution_hook(after_hook)

    # Execute the task
    with patch.object(task, '_run') as mock_run:
        task.execute(context)

        # Verify that the after execution hook was called
        after_hook.assert_called_once()

        # Ensure the task's _run method was executed
        mock_run.assert_called_once()


def test_add_multiple_after_execution_hooks(task, context):
    # Define multiple mock hook functions
    after_hook1 = MagicMock()
    after_hook2 = MagicMock()

    # Add the after execution hooks
    task.add_after_execution_hook(after_hook1)
    task.add_after_execution_hook(after_hook2)

    # Execute the task
    with patch.object(task, '_run') as mock_run:
        task.execute(context)

        # Verify that both after execution hooks were called
        after_hook1.assert_called_once()
        after_hook2.assert_called_once()

        # Ensure the task's _run method was executed
        mock_run.assert_called_once()


def test_hooks_execution_order(task, context):
    call_order = []

    # Define hook functions that record their execution order
    def before_hook():
        call_order.append('before')

    def after_hook():
        call_order.append('after')

    # Define a mock _run method that records its execution
    def run(context):
        call_order.append('run')

    # Add the hooks to the task
    task.add_before_execution_hook(before_hook)
    task.add_after_execution_hook(after_hook)

    # Patch the _run method with our custom run function
    with patch.object(task, '_run', side_effect=run) as mock_run:
        task.execute(context)

    # Verify that the hooks and _run were called in the correct order
    assert call_order == ['before', 'run', 'after'], f"Call order was {call_order}"

    # Additionally, verify that each was called exactly once
    assert mock_run.call_count == 1, f"_run was called {mock_run.call_count} times"
    assert 'before' in call_order, "before_hook was not called"
    assert 'after' in call_order, "after_hook was not called"
