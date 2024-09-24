# Saga Framework

## Overview
The Saga Pattern Implementation in Python is a robust and flexible library that facilitates managing complex transactions in distributed systems using the Saga pattern. It ensures data consistency and reliability by handling long-running transactions through a sequence of local transactions, each with corresponding compensating actions in case of failures.

## Features
- **Task Management**: Easily add and manage a sequence of tasks that constitute a saga.
- **Compensation Mechanism**: Automatically compensates completed tasks in reverse order if a failure occurs.
- **Context Handling**: Share contextual information across tasks seamlessly.
- **Comprehensive Logging**: Detailed logging of saga execution, task progress, and error handling.
- **Customizable Logging**: Inject custom loggers to tailor logging to your needs.
- **Exception Handling**: Robust error handling with custom exceptions for clear error propagation.
- **Extensible Design**: Modular architecture allowing for easy extension and customization.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation
You can install the Saga library using pip:

```bash
pip install saga-framework
```

Usage
Here's a simple example demonstrating how to define and execute a saga with tasks and compensations:

```python
from saga import Saga, Task, Context
from saga.logging import SagaLogger
from saga.errors import SagaExecutionException

# Define your tasks
class CreateOrderTask(Task):
    def execute(self, context: Context):
        # Logic to create an order
        print("Order created.")

    def compensate(self, context: Context):
        # Logic to compensate order creation
        print("Order creation compensated.")

class ChargePaymentTask(Task):
    def execute(self, context: Context):
        # Logic to charge payment
        print("Payment charged.")
        # Simulate failure
        raise SagaExecutionException("Payment failed.")

    def compensate(self, context: Context):
        # Logic to refund payment
        print("Payment refunded.")

# Initialize Saga with a logger
logger = SagaLogger()
saga = Saga(logger=logger)

# Add tasks to the saga
saga.add_task(CreateOrderTask())
saga.add_task(ChargePaymentTask())

# Execute the saga
try:
    saga.execute()
except SagaExecutionException as e:
    print(f"Saga failed: {e}")
```