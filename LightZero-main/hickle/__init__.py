import pickle


def dump(obj, file_obj, mode='w', **kwargs):
    del mode, kwargs
    return pickle.dump(obj, file_obj)


def load(file_obj, **kwargs):
    del kwargs
    return pickle.load(file_obj)

