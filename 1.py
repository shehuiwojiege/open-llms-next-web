from typing import Union


def foo() -> Union[None, str]:
    return "1"


print(foo())


a = {"a": 1}
print("b" in a)