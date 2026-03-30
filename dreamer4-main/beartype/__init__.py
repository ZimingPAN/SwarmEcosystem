def beartype(fn=None, *args, **kwargs):
    del args, kwargs
    if fn is None:
        def decorator(inner):
            return inner
        return decorator
    return fn

