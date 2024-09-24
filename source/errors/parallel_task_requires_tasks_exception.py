class ParallelTaskRequiresTasksException(Exception):
    ERROR_CODE = 4
    def __init__(self, *args):
        super().__init__(*args)
