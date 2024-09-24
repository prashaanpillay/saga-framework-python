class TaskExecutionException(Exception):
    ERROR_CODE = 1
    def __init__(self):
        super().__init__()
