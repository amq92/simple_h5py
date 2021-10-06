import os
from typing import Union

import h5py
import numpy as np

BIG_ATTRS = 'big_attrs'


class BasicPoint(dict):

    def __init__(self, path: str, route: Union[str, tuple], **kwargs):

        # H5 file path
        self.path = path

        # H5 point route
        self.route = route_h5(route)

        # populate the dictionary
        super().__init__(**kwargs)


class Attributes(BasicPoint):

    def __init__(self, path: str, route: str, **kwargs):

        super().__init__(path, route)

        # read attributes from file
        attrs = dict()
        if os.path.exists(self.path):
            with h5py.File(self.path, 'r') as f:
                if self.route in f:
                    for k, v in f[self.route].attrs.items():

                        # return the dataset to which the big_attrs points to
                        if isinstance(v, h5py.Reference):
                            v = f[v][:]

                        # populate the dict
                        attrs[k] = v

        # populate the dict object
        super().__init__(self.path, self.route, **attrs)

    def __setitem__(self, key, value) -> None:

        with h5py.File(self.path, 'a') as f:
            assert self.route in f, 'cannot set attribute'

        # add to file
        write_to_h5(self.path, self.route, {key: value})

        # add to object
        super().__setitem__(key, value)


class BasicElement(BasicPoint):

    def __init__(self, path: str, route: str, attrs: Attributes,
                 attrs_required: tuple = (), **kwargs):

        super().__init__(path, route, **kwargs)
        self.__attrs = attrs
        self.__attrs_required = attrs_required

    def __repr__(self) -> str:
        return '\n> '.join([
            type(self).__name__,
            f'path: {self.path}',
            f'route: {self.route}',
            f'attrs: {self.attrs}',
        ])

    def __str__(self) -> str:
        return '\n'.join([
            ' '.join([
                type(self).__name__,
                f'"{self.route}"',
            ]),
            *('> ' + str(sub_el) for sub_el in self.values()),
        ])

    def is_valid(self) -> bool:
        for attr in self.__attrs_required:
            location = ':'.join((self.path, self.route))
            assert attr in self.attrs, f'missing attr {attr} in {location}'
            assert self.attrs[attr] is not None, \
                f'found empty attr {attr} in {location}'
        return True

    @property
    def attrs(self) -> Attributes:
        return self.__attrs

    @attrs.setter
    def attrs(self, value: dict) -> None:
        write_to_h5(self.path, self.route, value)
        if self.__attrs:
            self.__attrs.update(value)
        else:
            self.__attrs = value


class Group(BasicElement):

    def __init__(self, path: str, route: str, attrs: Attributes, **kwargs):
        super().__init__(path, route, attrs, **kwargs)

    def __repr__(self) -> str:
        return '\n> '.join([
            super().__repr__(),
            f'datasets: {list(self.keys())}',
        ])

    def __setitem__(self, key: str, value: np.ndarray) -> None:

        # retrieve attributes from existing dataset
        attrs = self[key].attrs if key in self.keys() else None

        # either over-write dataset or make a new one
        write_to_h5(self.path, (self.route, key), value)

        # get the object that points to the dataset
        dataset = Dataset(self.path, (self.route, key), attrs, value.shape)

        super().__setitem__(key, dataset)

    def __delitem__(self, key: object) -> None:
        # remove dataset from dict object
        super().__delitem__(key)

        # remove dataset from file
        remove_from_h5(self.path, (self.route, key))

    def is_valid(self) -> bool:
        return all((dataset.is_valid() for dataset in self.values())) and \
            super().is_valid()


class Dataset(BasicElement):

    def __init__(self, path: str, route: str, attrs: Attributes,
                 shape: tuple, **kwargs):
        super().__init__(path, route, attrs, **kwargs)
        self.shape = shape

    def __getitem__(self, key) -> np.ndarray:
        with h5py.File(self.path, 'r') as f:
            return f[self.route][key]

    def __repr__(self) -> str:
        return '\n> '.join([
            super().__repr__(),
            f'shape: {self.shape}',
        ])

    def __str__(self) -> str:
        return ' '.join([
            super().__str__(),
            str(self.shape),
        ])


class BasicH5File(dict):
    def __init__(self, path: str, group_attrs_required: tuple = (),
                 dataset_attrs_required: tuple = (), **kwargs):

        assert '.h5' in os.path.splitext(os.path.basename(path))[1],\
            f'wrong filename {path}'

        self.path = path

        self.__group_attrs_required = group_attrs_required
        self.__dataset_attrs_required = dataset_attrs_required

        if os.path.exists(self.path):
            self.read()
            assert self.is_valid(), 'wrong h5 structure'

    def read(self):
        with h5py.File(self.path, 'r') as f:
            for group_name, group in f.items():
                if group_name != BIG_ATTRS:
                    self[group_name] = Group(
                        path=self.path,
                        route=group_name,
                        attrs=Attributes(
                            path=self.path,
                            route=group_name,
                            **dict(group.attrs)),
                        attrs_required=self.__group_attrs_required,
                        **{
                            dataset_name:
                            Dataset(
                                path=self.path,
                                route=(group_name, dataset_name),
                                attrs=Attributes(
                                    path=self.path,
                                    route=(group_name, dataset_name),
                                    **dict(dataset.attrs)),
                                shape=dataset.shape,
                                attrs_required=self.__dataset_attrs_required
                            )
                            for dataset_name, dataset in group.items()
                        }
                    )

    def __repr__(self) -> str:
        return '\n'.join([
            type(self).__name__,
            f'path: {self.path}',
            f'groups: {list(self.keys())}',
        ])

    def __str__(self) -> str:
        return '\n'.join([
            ' '.join([
                type(self).__name__,
                f'({self.path})',
            ]),
            *('> ' + str(group).replace('>', '  >')
              for group in self.values()),
        ])

    def __setitem__(self, key: str, value: object):
        super().__setitem__(key, value if value is not None else Group(
            self.path, key, Attributes(self.path, key)))

    def __delitem__(self, key: object) -> None:
        # remove dataset from dict object
        super().__delitem__(key)

        # remove dataset from file
        remove_from_h5(self.path, key)

    def is_valid(self) -> bool:
        return all((group.is_valid() for group in self.values()))


def route_h5(route: Union[str, tuple]) -> str:
    # must start with '/' for consistency
    route = route if isinstance(route, str) else '/'.join(route)
    return route if route[0] == '/' else f'/{route}'


def remove_from_h5(path: str, route: Union[str, tuple]) -> None:

    # H5 point route
    route = route_h5(route)

    # remove from file
    with h5py.File(path, 'a') as f:
        del f[route]


def write_to_h5(path: str, route: Union[str, tuple],
                data_or_attrs: Union[np.ndarray, Attributes]) -> None:

    # create directory if necessary
    path_dir = os.path.normpath(os.path.realpath(os.path.dirname(path)))

    if not os.path.isdir(path_dir):
        os.makedirs(path_dir, exist_ok=True)

    # H5 point route
    route = route_h5(route)

    # write data or attributes to file
    if isinstance(data_or_attrs, np.ndarray):
        write_to_h5_data(path, route, data_or_attrs)

    elif isinstance(data_or_attrs, (dict, Attributes)):
        write_to_h5_attrs(path, route, data_or_attrs)


def write_to_h5_data(path: str, route: str, data: np.ndarray = None) -> None:

    # open filename in "all" mode
    with h5py.File(path, 'a') as f:

        if route in f and f[route].shape == data.shape:
            # simply over-write existing dataset
            f[route][...] = data
        else:
            # remove previous dataset (inconsistent size)
            if route in f:
                del f[route]
            # add new dataset
            f.create_dataset(route, data=data)


def write_to_h5_attrs(path: str, route: str, attrs: Attributes = None) -> None:

    # open filename in "all" mode
    with h5py.File(path, 'a') as f:

        # create route if needed
        if route not in f:
            f.create_group(route)

        # set attributes
        for k, v in attrs.items():
            # get possible big route
            k_big = route_h5((route, 'attrs', k))[1:].replace('/', '.')
            route_big = route_h5((BIG_ATTRS, k_big))

            # verify if attribute exists and if it is a big attribute
            if k in f[route].attrs and \
                    isinstance(f[route].attrs[k], h5py.Reference):
                # delete the attribute and the dataset it points to
                # since the new attribute may be small enough to fit
                del f[route].attrs[k]
                del f[route_big]

            # HDF5 format imposes a limit of 64 Kb for attributes
            try:
                f[route].attrs[k] = v
            except BaseException:
                # verify if 'orphan' dataset exists, if so delete it
                if route_big in f:
                    del f[route_big]
                f.create_dataset(route_big, data=v)
                f[route].attrs[k] = f[route_big].ref
