from collections.abc import Callable


def add_decorator(cls, methods: list[str], deco: Callable):
    for method in methods: setattr(cls, method, deco(getattr(cls, method)))