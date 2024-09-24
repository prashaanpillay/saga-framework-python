class SagaExecutionException(Exception):
    ERROR_CODE = 2
    def __init__(self):
        super().__init__()
