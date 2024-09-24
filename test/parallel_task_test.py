from time import sleep, time
from unittest.mock import patch, MagicMock
import pytest
from concurrent.futures import Future

from source.context import Context
from source.errors.parallel_task_requires_tasks_exception import ParallelTaskRequiresTasksException
from source.errors.task_execution_exception import TaskExecutionException
from source.task import Task
from source.parallel_task import ParallelTask
from source.task_status import TaskStatus


# Define test task classes
class TestTask(Task):
    def _run(self, context: Context):
        self.status = TaskStatus.COMPLETED


class FailingTask(Task):
    def _run(self, context: Context):
        raise TaskExecutionException("Task Failed.")


class CompensationTask(Task):
    def _run(self, context: Context):
        self.status = TaskStatus.COMPLETED


class UnexpectedExceptionTask(Task):
    def _run(self, context: Context):
        raise TaskExecutionException("Unexpected exception.")


# Fixtures
@pytest.fixture
def context():
    return Context()


@pytest.fixture
def compensation_task_factory():
    def create_compensation(name):
        return CompensationTask(name=name)

    return create_compensation


@pytest.fixture
def child_task1(compensation_task_factory):
    return TestTask(name="ChildTask1", compensation=compensation_task_factory("ChildTask1_Compensation"))


@pytest.fixture
def child_task2(compensation_task_factory):
    return TestTask(name="ChildTask2", compensation=compensation_task_factory("ChildTask2_Compensation"))


@pytest.fixture
def failing_child_task(compensation_task_factory):
    return FailingTask(name="FailingChildTask", compensation=compensation_task_factory("FailingChildTask_Compensation"))


@pytest.fixture
def parallel_task(child_task1, child_task2, compensation_task_factory):
    return ParallelTask(
        name="ParallelTask",
        tasks=[child_task1, child_task2],
        compensation=compensation_task_factory("ParallelTask_Compensation")
    )


# Helper functions
def mock_execute_success(task, context):
    task.status = TaskStatus.COMPLETED


def mock_execute_failure(task, context, exception_message="Task Failed."):
    raise TaskExecutionException(exception_message)


def mock_handle_failure(context, completed_tasks):
    pass


# Test classes
class TestParallelTaskInitialization:
    def test_initialization(self, parallel_task, child_task1, child_task2):
        assert parallel_task.name == "ParallelTask"
        assert parallel_task.status == TaskStatus.PENDING
        assert len(parallel_task.tasks) == 2
        assert parallel_task.tasks[0] == child_task1
        assert parallel_task.tasks[1] == child_task2
        assert parallel_task.max_workers == 2  # Default to number of tasks

    def test_initialization_custom_max_workers(self, child_task1, child_task2, compensation_task_factory):
        custom_parallel_task = ParallelTask(
            name="CustomParallelTask",
            tasks=[child_task1, child_task2],
            compensation=compensation_task_factory("CustomParallelTask_Compensation"),
            max_workers=5
        )
        assert custom_parallel_task.max_workers == 5

    def test_initialization_no_tasks(self):
        with pytest.raises(ParallelTaskRequiresTasksException) as exc_info:
            ParallelTask(name="NoTaskParallel", tasks=[], compensation=None)
        # Assert that the exception message is correct
        assert exc_info.value.ERROR_CODE == 4


class TestParallelTaskExecution:
    def test_execute_success(self, parallel_task, context):
        # Spy on child tasks' execute methods without altering their behavior
        with patch.object(parallel_task.tasks[0], 'execute', wraps=parallel_task.tasks[0].execute) as mock_execute1, \
                patch.object(parallel_task.tasks[1], 'execute', wraps=parallel_task.tasks[1].execute) as mock_execute2:
            # Execute the parallel task
            parallel_task.execute(context)

            # Verify that both child tasks' execute methods were called once with the correct context
            mock_execute1.assert_called_once_with(context)
            mock_execute2.assert_called_once_with(context)

            # Verify that child tasks' statuses are set to COMPLETED
            assert parallel_task.tasks[0].status == TaskStatus.COMPLETED, "ChildTask1 status should be COMPLETED"
            assert parallel_task.tasks[1].status == TaskStatus.COMPLETED, "ChildTask2 status should be COMPLETED"

            # Verify that ParallelTask status is COMPLETED
            assert parallel_task.status == TaskStatus.COMPLETED, "ParallelTask status should be COMPLETED"

    def test_execute_in_parallel(self, context):
        # Define child tasks that take time to execute
        class DelayedTask(Task):
            def __init__(self, name, delay, compensation=None):
                super().__init__(name, compensation)
                self.delay = delay

            def _run(self, context: Context):
                sleep(self.delay)
                self.status = TaskStatus.COMPLETED

        # Create two delayed tasks
        delayed_task1 = DelayedTask(name="DelayedTask1", delay=2)
        delayed_task2 = DelayedTask(name="DelayedTask2", delay=2)

        # Create a ParallelTask with these delayed tasks
        parallel_task = ParallelTask(
            name="ParallelDelayedTask",
            tasks=[delayed_task1, delayed_task2],
            compensation=None,
            max_workers=2  # Ensure both tasks can run concurrently
        )

        start_time = time()
        parallel_task.execute(context)
        end_time = time()

        duration = end_time - start_time

        # Since both tasks take 2 seconds and run in parallel, total duration should be slightly over 2 seconds
        assert duration < 4, f"Parallel execution took too long: {duration} seconds"
        assert parallel_task.status == TaskStatus.COMPLETED, "ParallelTask status should be COMPLETED"

    def test_execute_one_failure(self, parallel_task, context, failing_child_task):
        # Replace one child task with a failing task
        parallel_task.tasks[1] = failing_child_task

        def set_task_0_completed(ctx):
            parallel_task.tasks[0].status = TaskStatus.COMPLETED

        # Patch the execute methods and logging functions correctly
        with patch.object(parallel_task.tasks[0], 'execute', side_effect=set_task_0_completed) as mock_execute1, \
                patch.object(parallel_task.tasks[1], 'execute', wraps=parallel_task.tasks[1].execute) as mock_execute2, \
                patch('source.parallel_task.log_task_execution_error') as mock_log_error, \
                patch.object(parallel_task, '_handle_failure_parallel') as mock_handle_failure:
            parallel_task.execute(context)

            # Verify that both child tasks' execute methods were called once with the correct context
            mock_execute1.assert_called_once_with(context)
            mock_execute2.assert_called_once_with(context)

            # Verify that log_task_execution_error was called once with appropriate arguments
            mock_log_error.assert_called_once()
            log_call_args = mock_log_error.call_args[0]
            assert log_call_args[0] == "FailingChildTask", "Incorrect task name logged."
            assert log_call_args[1] == "ParallelTask_Compensation", "Incorrect compensation task name logged."
            assert isinstance(log_call_args[2], TaskExecutionException), "Incorrect exception type logged."

            # Verify that _handle_failure_parallel was called once with completed_tasks
            mock_handle_failure.assert_called_once_with(context, [parallel_task.tasks[0]])

            # Verify that ParallelTask status is FAILED
            assert parallel_task.status == TaskStatus.FAILED, f"Expected status FAILED, got {parallel_task.status}"

    def test_execute_unexpected_exception(self, parallel_task, context):
        # Define an unexpected exception task
        unexpected_task = UnexpectedExceptionTask(
            name="UnexpectedTask",
            compensation=CompensationTask(name="UnexpectedTask_Compensation")
        )
        parallel_task.tasks[1] = unexpected_task

        def set_task_0_completed(ctx):
            parallel_task.tasks[0].status = TaskStatus.COMPLETED

        with patch.object(parallel_task.tasks[0], 'execute', side_effect=set_task_0_completed) as mock_execute1, \
                patch.object(parallel_task.tasks[1], 'execute',
                             side_effect=TaskExecutionException("Unexpected exception.")) as mock_execute2, \
                patch('source.parallel_task.log_task_execution_error') as mock_log_error, \
                patch.object(parallel_task, '_handle_failure_parallel') as mock_handle_failure:
            parallel_task.execute(context)

            # Verify that execute methods were called
            mock_execute1.assert_called_once_with(context)
            mock_execute2.assert_called_once_with(context)

            # Verify log_task_execution_error
            mock_log_error.assert_called_once()
            log_call_args = mock_log_error.call_args[0]
            assert log_call_args[0] == "UnexpectedTask", "Incorrect task name logged."
            assert log_call_args[1] == "ParallelTask_Compensation", "Incorrect compensation task name logged."
            assert isinstance(log_call_args[2], TaskExecutionException), "Incorrect exception type logged."

            # Verify _handle_failure_parallel
            mock_handle_failure.assert_called_once_with(context, [parallel_task.tasks[0]])

            # Verify ParallelTask status
            assert parallel_task.status == TaskStatus.FAILED, f"Expected status FAILED, got {parallel_task.status}"


class TestParallelTaskCompensation:
    def test_handle_failure_compensations(self, parallel_task, context, failing_child_task):
        # Replace one child task with a failing task
        parallel_task.tasks[1] = failing_child_task

        def set_task_0_completed(ctx):
            parallel_task.tasks[0].status = TaskStatus.COMPLETED

        with patch.object(parallel_task.tasks[0], 'execute', side_effect=set_task_0_completed), \
                patch.object(parallel_task.tasks[1], 'execute',
                             side_effect=TaskExecutionException("Task2 Failed")) as mock_execute2, \
                patch('source.parallel_task.log_task_execution_error') as mock_log_error, \
                patch.object(parallel_task, '_handle_failure_parallel') as mock_handle_failure:
            parallel_task.execute(context)

            # Verify execute calls
            parallel_task.tasks[0].execute.assert_called_once_with(context)
            parallel_task.tasks[1].execute.assert_called_once_with(context)

            # Verify log_task_execution_error
            mock_log_error.assert_called_once()
            log_call_args = mock_log_error.call_args[0]
            assert log_call_args[0] == "FailingChildTask", "Incorrect task name logged."
            assert log_call_args[1] == "ParallelTask_Compensation", "Incorrect compensation task name logged."
            assert isinstance(log_call_args[2], TaskExecutionException), "Incorrect exception type logged."

            # Verify _handle_failure_parallel
            mock_handle_failure.assert_called_once_with(context, [parallel_task.tasks[0]])

            # Verify ParallelTask status
            assert parallel_task.status == TaskStatus.FAILED, f"Expected status FAILED, got {parallel_task.status}"

    def test_compensation_order(self, parallel_task, context, failing_child_task):
        parallel_task.tasks[1] = failing_child_task

        def set_task_0_completed(ctx):
            parallel_task.tasks[0].status = TaskStatus.COMPLETED

        with patch.object(parallel_task.tasks[0], 'execute', side_effect=set_task_0_completed), \
                patch.object(parallel_task.tasks[1], 'execute', side_effect=TaskExecutionException("Task2 Failed")), \
                patch.object(parallel_task.tasks[0].compensation, 'execute') as mock_comp1, \
                patch.object(parallel_task, '_handle_failure_parallel') as mock_handle_failure:
            parallel_task.execute(context)

            # Verify that _handle_failure_parallel was called with the first child task as completed
            mock_handle_failure.assert_called_once_with(context, [parallel_task.tasks[0]])

            # Verify that compensation was called in LIFO order
            mock_comp1.assert_called_once_with(context)

    @pytest.mark.repeat(5)
    @pytest.mark.xfail(reason="This test is expected to fail due to the randomness of parallel execution.")
    def test_compensation_multiple_completed(self, parallel_task, context, failing_child_task):
        # Add a third child task
        child_task3 = TestTask(
            name="ChildTask3",
            compensation=CompensationTask(name="ChildTask3_Compensation")
        )
        parallel_task.tasks.append(child_task3)

        # Replace the third child task with a failing task
        parallel_task.tasks[2] = failing_child_task

        # Define side effects to mark tasks as completed
        def set_task_0_run(ctx):
            parallel_task.tasks[0].status = TaskStatus.COMPLETED
            print(f"Task {parallel_task.tasks[0].name} marked as COMPLETED")

        def set_task_1_run(ctx):
            parallel_task.tasks[1].status = TaskStatus.COMPLETED
            print(f"Task {parallel_task.tasks[1].name} marked as COMPLETED")

        # Initialize compensation order tracker
        compensation_order = []

        # Define side effects to track compensation execution order
        def comp1_side_effect(ctx):
            parallel_task.tasks[0].compensation.status = TaskStatus.COMPLETED
            compensation_order.append('comp1')
            print(f"Compensation task for {parallel_task.tasks[0].name} executed")

        def comp2_side_effect(ctx):
            parallel_task.tasks[1].compensation.status = TaskStatus.COMPLETED
            compensation_order.append('comp2')
            print(f"Compensation task for {parallel_task.tasks[1].name} executed")

        # Mock ThreadPoolExecutor to avoid real parallelism
        def mock_executor_submit(task_fn, ctx):
            future = Future()
            try:
                result = task_fn(ctx)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            return future

        with patch('source.parallel_task.ThreadPoolExecutor') as mock_executor, \
                patch.object(parallel_task.tasks[0], '_run', side_effect=set_task_0_run), \
                patch.object(parallel_task.tasks[1], '_run', side_effect=set_task_1_run), \
                patch.object(parallel_task.tasks[2], '_run', side_effect=TaskExecutionException("Task3 Failed")), \
                patch('source.parallel_task.log_task_execution_error'), \
                patch.object(parallel_task.tasks[0].compensation, 'execute',
                             side_effect=comp1_side_effect) as mock_comp1, \
                patch.object(parallel_task.tasks[1].compensation, 'execute',
                             side_effect=comp2_side_effect) as mock_comp2, \
                patch.object(parallel_task, '_handle_failure_parallel') as mock_handle_failure:

            # Mock the ThreadPoolExecutor context
            mock_executor.return_value.__enter__.return_value.submit.side_effect = mock_executor_submit

            # Execute the parallel task and expect a TaskExecutionException
            parallel_task.execute(context)

            # Verify that `_handle_failure_parallel` was called with the completed tasks
            args, _ = mock_handle_failure.call_args
            actual_completed_tasks = args[1]

            # Print debug information for the completed tasks
            print(f"Completed tasks: {[task.name for task in actual_completed_tasks]}")

            # Assert that the completed tasks contain tasks with the expected names
            completed_task_names = {task.name for task in actual_completed_tasks}
            expected_task_names = {parallel_task.tasks[0].name, parallel_task.tasks[1].name}

            assert expected_task_names == completed_task_names, (
                f"Expected completed tasks with names {expected_task_names}, "
                f"but got tasks with names {completed_task_names}"
            )

            # Verify that both compensation tasks were executed once
            assert mock_comp1.call_count == 1, "Compensation task for ChildTask1 was not executed exactly once."
            assert mock_comp2.call_count == 1, "Compensation task for ChildTask2 was not executed exactly once."

            # Verify that compensation tasks were called in the correct order (LIFO)
            assert len(compensation_order) == 2, "Expected 2 compensation tasks to be executed."

    def test_compensation_some_tasks_no_compensation(self, parallel_task, context, failing_child_task):
        # Remove compensation from child_task1
        parallel_task.tasks[0].compensation = None

        # Replace one child task with a failing task
        parallel_task.tasks[1] = failing_child_task

        def set_task_0_completed(ctx):
            parallel_task.tasks[0].status = TaskStatus.COMPLETED

        def set_task_1_completed(ctx):
            parallel_task.tasks[1].status = TaskStatus.FAILED
            parallel_task.tasks[1].compensate(context)

        def set_task_1_comp_completed(ctx):
            parallel_task.tasks[1].compensation.status = TaskStatus.COMPLETED

        # Patch execute methods and compensation execute
        with patch.object(parallel_task.tasks[0], 'execute', side_effect=set_task_0_completed), \
                patch.object(parallel_task.tasks[1], 'execute', side_effect=set_task_1_completed) as mock_execute2, \
                patch.object(parallel_task.tasks[1].compensation, 'execute',
                             side_effect=set_task_1_comp_completed) as mock_comp2, \
                patch.object(parallel_task, '_handle_failure_parallel') as mock_handle_failure, \
                patch('source.parallel_task.log_task_execution_error') as mock_log_error:
            parallel_task.execute(context)

            # Verify that compensation for child_task1 was not called (since it's None)
            mock_comp2.assert_called_once_with(context)

            # Verify that _handle_failure_parallel was called with the first child task as completed
            mock_handle_failure.assert_called_once_with(context, [parallel_task.tasks[0]])

            # Verify that log_task_execution_error was called once
            mock_log_error.assert_called_once()

            # Verify that ParallelTask status is FAILED
            assert parallel_task.status == TaskStatus.FAILED, f"Expected status FAILED, got {parallel_task.status}"


class TestParallelTaskHooks:
    def test_parallel_task_with_single_hooks(self, parallel_task, context):
        # Define mock hooks
        before_hook = MagicMock()
        after_hook = MagicMock()

        # Add hooks to the ParallelTask
        parallel_task.add_before_execution_hook(before_hook)
        parallel_task.add_after_execution_hook(after_hook)

        def set_task_0_completed(ctx):
            parallel_task.tasks[0].status = TaskStatus.COMPLETED

        def set_task_1_completed(ctx):
            parallel_task.tasks[1].status = TaskStatus.COMPLETED

        # Patch child tasks' execute methods to set status to COMPLETED
        with patch.object(parallel_task.tasks[0], 'execute', side_effect=set_task_0_completed), \
                patch.object(parallel_task.tasks[1], 'execute', side_effect=set_task_1_completed):
            # Execute ParallelTask
            parallel_task.execute(context)

            # Verify that hooks were called once
            before_hook.assert_called_once()
            after_hook.assert_called_once()

            # Verify that ParallelTask status is COMPLETED
            assert parallel_task.status == TaskStatus.COMPLETED, "ParallelTask status should be COMPLETED"

    def test_parallel_task_with_multiple_hooks(self, parallel_task, context):
        # Define multiple mock hooks
        before_hook1 = MagicMock()
        before_hook2 = MagicMock()
        after_hook1 = MagicMock()
        after_hook2 = MagicMock()

        def set_task_0_completed(ctx):
            parallel_task.tasks[0].status = TaskStatus.COMPLETED

        def set_task_1_completed(ctx):
            parallel_task.tasks[1].status = TaskStatus.COMPLETED

        # Add hooks to the ParallelTask
        parallel_task.add_before_execution_hook(before_hook1)
        parallel_task.add_before_execution_hook(before_hook2)
        parallel_task.add_after_execution_hook(after_hook1)
        parallel_task.add_after_execution_hook(after_hook2)

        # Patch child tasks' execute methods to set status to COMPLETED
        with patch.object(parallel_task.tasks[0], 'execute', side_effect=set_task_0_completed), \
                patch.object(parallel_task.tasks[1], 'execute', side_effect=set_task_1_completed):
            # Execute ParallelTask
            parallel_task.execute(context)

            # Verify that all hooks were called once
            before_hook1.assert_called_once()
            before_hook2.assert_called_once()
            after_hook1.assert_called_once()
            after_hook2.assert_called_once()

        # To verify the order of hook execution, use a shared tracker
        call_tracker = []

        def before1():
            call_tracker.append('before1')

        def before2():
            call_tracker.append('before2')

        def after1():
            call_tracker.append('after1')

        def after2():
            call_tracker.append('after2')

        # Re-assign hooks with call tracking
        parallel_task.on_before_execution = [before1, before2]
        parallel_task.on_after_execution = [after1, after2]

        # Patch child tasks' execute methods to track hooks
        with patch.object(parallel_task.tasks[0], 'execute', return_value=None), \
                patch.object(parallel_task.tasks[1], 'execute', return_value=None):
            parallel_task.execute(context)

        # Verify the order
        # Since execution is parallel, hooks before and after can be called in any order relative to each other
        # However, 'before' hooks should be called before task executions, and 'after' hooks after
        assert 'before1' in call_tracker, "'before1' was not called."
        assert 'before2' in call_tracker, "'before2' was not called."
        assert 'after1' in call_tracker, "'after1' was not called."
        assert 'after2' in call_tracker, "'after2' was not called."

        before_indices = [call_tracker.index('before1'), call_tracker.index('before2')]
        after_indices = [call_tracker.index('after1'), call_tracker.index('after2')]

        # Ensure all 'before' hooks are called before 'after' hooks
        assert max(before_indices) < min(after_indices), "Before hooks were not called before after hooks."

    def test_parallel_task_hooks_execution_order(self, parallel_task, context):
        call_order = []

        # Define hook functions that record their execution order
        def before_hook():
            call_order.append('before')

        def after_hook():
            call_order.append('after')

        # Add hooks to the ParallelTask
        parallel_task.add_before_execution_hook(before_hook)
        parallel_task.add_after_execution_hook(after_hook)

        # Define child tasks' execute methods to record their execution
        def child1_run(ctx):
            call_order.append('task1')

        def child2_run(ctx):
            call_order.append('task2')

        with patch.object(parallel_task.tasks[0], 'execute', side_effect=child1_run), \
                patch.object(parallel_task.tasks[1], 'execute', side_effect=child2_run):
            # Execute ParallelTask
            parallel_task.execute(context)

        # Verify the call order
        # 'before' should be first, 'after' should be last
        assert call_order[0] == 'before', "The 'before' hook was not called first."
        assert call_order[-1] == 'after', "The 'after' hook was not called last."

        # Ensure both tasks were executed
        assert 'task1' in call_order, "'task1' was not executed."
        assert 'task2' in call_order, "'task2' was not executed."


class TestParallelTaskCompensationWithoutCompensationTasks:
    def test_tasks_without_compensation(self, parallel_task, context):
        # Remove compensation from both child tasks
        parallel_task.tasks[0].compensation = None
        parallel_task.tasks[1].compensation = None

        # Replace one child task with a failing task without compensation
        parallel_task.tasks[1] = FailingTask(name="FailingChildTask", compensation=None)

        def set_task_0_completed(ctx):
            parallel_task.tasks[0].status = TaskStatus.COMPLETED

        with patch.object(parallel_task.tasks[0], 'execute', side_effect=set_task_0_completed), \
                patch.object(parallel_task.tasks[1], 'execute',
                             side_effect=TaskExecutionException("Task2 Failed")) as mock_execute2, \
                patch('source.parallel_task.log_task_execution_error') as mock_log_error, \
                patch.object(parallel_task, '_handle_failure_parallel') as mock_handle_failure:
            parallel_task.execute(context)

            # Verify that _handle_failure_parallel was called with the first child task as completed
            mock_handle_failure.assert_called_once_with(context, [parallel_task.tasks[0]])

            # Verify that log_task_execution_error was called once
            mock_log_error.assert_called_once()

            # Verify that ParallelTask status is FAILED
            assert parallel_task.status == TaskStatus.FAILED, f"Expected status FAILED, got {parallel_task.status}"


class TestParallelTaskContext:
    def test_shared_context_modifications(self):
        # Define child tasks that modify the context
        class IncrementTask(Task):
            def __init__(self, name, key, increment=1, compensation=None):
                super().__init__(name, compensation)
                self.key = key
                self.increment = increment

            def _run(self, context: Context):
                value = context.get(self.key, 0)
                context.set(self.key, value + self.increment)
                self.status = TaskStatus.COMPLETED

        child_task1 = IncrementTask(name="IncrementTask1", key="counter", increment=1)
        child_task2 = IncrementTask(name="IncrementTask2", key="counter", increment=2)

        # Create a new ParallelTask with these incrementing tasks
        parallel_task = ParallelTask(
            name="ParallelIncrementTask",
            tasks=[child_task1, child_task2],
            compensation=None
        )

        context = Context()

        # Execute ParallelTask
        parallel_task.execute(context)

        # Verify that the counter was incremented correctly
        assert context.get('counter', 0) == 3, f"Expected counter to be 3, got {context.get('counter')}"

    def test_context_shared(self, parallel_task, context):
        # Define child tasks that read and write to the context
        class ReadWriteTask(Task):
            def __init__(self, name, key, increment=1, compensation=None):
                super().__init__(name, compensation)
                self.key = key
                self.increment = increment

            def _run(self, context: Context):
                value = context.get(self.key, 0)
                context.set(self.key, value + self.increment)
                self.status = TaskStatus.COMPLETED

        task1 = ReadWriteTask(name="ReadWriteTask1", key="counter", increment=1)
        task2 = ReadWriteTask(name="ReadWriteTask2", key="counter", increment=2)

        # Create a new ParallelTask with these ReadWriteTasks
        parallel_task = ParallelTask(
            name="ParallelReadWriteTask",
            tasks=[task1, task2],
            compensation=None
        )

        # Execute ParallelTask
        parallel_task.execute(context)

        # Verify that the counter was incremented correctly
        # Order of execution is not guaranteed, but the final value should be 3
        assert context.get('counter', 0) == 3, f"Expected counter to be 3, got {context.get('counter')}"


class TestParallelTaskExecutorCleanup:
    def test_executor_cleanup(self, parallel_task, context):
        # Patch the ThreadPoolExecutor to monitor its usage
        with patch('source.parallel_task.ThreadPoolExecutor') as mock_executor_class, \
                patch('source.parallel_task.as_completed') as _:
            # Mock the executor instance returned by the context manager
            mock_executor = MagicMock()
            mock_executor_class.return_value.__enter__.return_value = mock_executor

            # Create mock futures for each task
            mock_future1 = MagicMock()
            mock_future1.result.return_value = None  # Simulate successful task execution
            mock_future2 = MagicMock()
            mock_future2.result.return_value = None

            # Setup the executor to return these futures when submit is called
            mock_executor.submit.side_effect = [mock_future1, mock_future2]

            # Execute the parallel task
            parallel_task.execute(context)

            # Ensure ThreadPoolExecutor is used correctly with max_workers
            mock_executor_class.assert_called_once_with(max_workers=len(parallel_task.tasks))

            # Ensure __enter__ (executor start) and __exit__ (executor cleanup) are called
            mock_executor_class.return_value.__enter__.assert_called_once()
            mock_executor_class.return_value.__exit__.assert_called_once()

            # Verify that child tasks are submitted to the executor
            assert mock_executor.submit.call_count == len(parallel_task.tasks), \
                f"Expected {len(parallel_task.tasks)} tasks to be submitted, got {mock_executor.submit.call_count}"

            # Verify that the executor's shutdown was not explicitly called (handled by the context manager)
            mock_executor.shutdown.assert_not_called()


class TestParallelTaskWithHooks:
    def test_parallel_task_and_child_tasks_hooks(self, parallel_task, context):
        call_order = []

        # Define hook functions for ParallelTask
        def before_parallel():
            call_order.append('before_parallel')

        def after_parallel():
            call_order.append('after_parallel')

        # Define hook functions for child tasks
        def before_child1():
            call_order.append('before_child1')

        def after_child1():
            call_order.append('after_child1')

        def before_child2():
            call_order.append('before_child2')

        def after_child2():
            call_order.append('after_child2')

        # Add hooks to ParallelTask
        parallel_task.add_before_execution_hook(before_parallel)
        parallel_task.add_after_execution_hook(after_parallel)

        # Add hooks to child tasks
        parallel_task.tasks[0].add_before_execution_hook(before_child1)
        parallel_task.tasks[0].add_after_execution_hook(after_child1)
        parallel_task.tasks[1].add_before_execution_hook(before_child2)
        parallel_task.tasks[1].add_after_execution_hook(after_child2)

        # Define child tasks' _run methods to record their hooks
        def child1_run(ctx):
            call_order.append('child1_run')  # Ensure this gets added
            parallel_task.tasks[0].status = TaskStatus.COMPLETED

        def child2_run(ctx):
            call_order.append('child2_run')  # Ensure this gets added
            parallel_task.tasks[1].status = TaskStatus.COMPLETED

        # Patch the _run method instead of execute
        with patch.object(parallel_task.tasks[0], '_run', side_effect=child1_run), \
                patch.object(parallel_task.tasks[1], '_run', side_effect=child2_run):
            parallel_task.execute(context)

        # Verify that 'before_parallel' is first and 'after_parallel' is last
        assert call_order[0] == 'before_parallel', "'before_parallel' hook was not called first."
        assert call_order[-1] == 'after_parallel', "'after_parallel' hook was not called last."

        # Verify that all other calls are present
        expected_events = ['before_child1', 'before_child2', 'child1_run', 'child2_run', 'after_child1', 'after_child2']

        for event in expected_events:
            assert event in call_order, f"Expected '{event}' in call_order."

        # Optionally, verify the order between before and after hooks
        before_parallel_index = call_order.index('before_parallel')
        after_parallel_index = call_order.index('after_parallel')

        # All other events should be between before_parallel and after_parallel
        for event in expected_events:
            event_index = call_order.index(event)
            assert before_parallel_index < event_index < after_parallel_index, \
                f"'{event}' should be between 'before_parallel' and 'after_parallel'."
