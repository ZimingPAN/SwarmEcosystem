__all__ = ["Leader", "Worker"]


def __getattr__(name: str):
    if name == "Leader":
        from .leader import Leader

        return Leader
    if name == "Worker":
        from .worker import Worker

        return Worker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
