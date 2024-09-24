class SagaCompensationExecutionException(Exception):
    ERROR_CODE = 3
    def __init__(self, *args):
        super().__init__(*args)
