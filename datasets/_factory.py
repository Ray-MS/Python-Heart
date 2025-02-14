_dataset_entrypoints: dict[str, callable] = dict()


def register_dataset(fn: callable) -> callable:
    dataset_name = fn.__name__
    _dataset_entrypoints[dataset_name] = fn


def dataset_entypoints(dataset_name: str) -> callable:
    return _dataset_entrypoints[dataset_name]


def create_dataset(
    dataset_name: str,
    root: str = './data',
    **kwargs
) -> object:
    return dataset_entypoints(dataset_name.lower())(root, **kwargs)
