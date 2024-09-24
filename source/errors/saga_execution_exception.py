class SagaExecutionException(Exception):
    ERROR_CODE = 2
    def __init__(self, *args):
        super().__init__(*args)
