import uuid
from random import random
from typing import List

from source.context import Context
from source.errors.task_execution_exception import TaskExecutionException
from source.saga import Saga
from source.task import Task


class Item:
    def __init__(self):
        self.item_id = uuid.uuid4()
        self.name = f"Item_{self.item_id}"
        self.price = random() * 100


class ShopperContext(Context):
    def __init__(self):
        self.balance_context = BalanceContext()
        self.cart_context = ShoppingCartContext()
        self.payment_status = "pending"


class BalanceContext(Context):
    def __init__(self):
        self.balance = 0


class ShoppingCartContext(Context):
    def __init__(self):
        self.user_id = uuid.uuid4()
        self.cart_id = uuid.uuid4()
        self.cart_items: List[Item] = []

    def add_item(self, item: str):
        self.cart_items.append(item)

    def remove_item(self, item: str):
        self.cart_items.remove(item)

    def total_price(self):
        return sum(item.price for item in self.cart_items)


class ProcessPaymentTask(Task):
    def _run(self, context: ShopperContext):
        # Process payment
        context.balance_context.balance -= context.cart_context.total_price()
        context.payment_status = "success"
        raise TaskExecutionException("Payment failed")

    def compensate(self, context: ShopperContext):
        # Compensate payment
        context.balance_context.balance += context.cart_context.total_price()
        context.payment_status = "failed"


def __main__() -> None:
    # Create a new shopper context
    shopper_context = ShopperContext()
    shopper_context.balance_context.balance = 100
    shopper_context.cart_context.add_item(Item())

    # Create a new saga
    saga = Saga(shopper_context)

    # Create a new task to process payment
    process_payment_task = ProcessPaymentTask("ProcessPayment", compensation=None)

    # Add the task to the saga
    saga.add_task(process_payment_task)

    # Execute the saga
    saga.execute()

    # Check the payment status
    print(f"Payment status: {shopper_context.payment_status}")

    # Check the balance
    print(f"Balance: {shopper_context.balance_context.balance}")


if __name__ == "__main__":
    __main__()
