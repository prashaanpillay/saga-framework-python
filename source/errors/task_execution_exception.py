class TaskExecutionException(Exception):
    ERROR_CODE = 1

    def __init__(self, *args):
        super().__init__(*args)
