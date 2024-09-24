class SagaCompensationExecutionException(Exception):
    ERROR_CODE = 3
    def __init__(self):
        super().__init__()
