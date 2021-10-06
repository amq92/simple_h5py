<h1 align="center">simple h5py</h1>
<p align="center">A simple wrapper for the h5py library<p>


```python
import numpy as np
from simple_h5py import BasicH5File

# Creating some data
# >> notice the "huge" attribute !
group_attrs = dict(a=1, b=2)
dataset = np.ones((5, 4, 3))
dataset_attrs = dict(new=5, huge=np.ones((1000000, 3)))

# Write contents to file
obj = BasicH5File('demo.h5')
obj['my_group'] = None
obj['my_group'].attrs = group_attrs.copy()
obj['my_group']['my_dataset'] = dataset
obj['my_group']['my_dataset'].attrs = dataset_attrs.copy()

# Read contents from file
obj = BasicH5File('demo.h5')
print(obj['my_group'].attrs)
print(obj['my_group']['my_dataset'][0])
print(obj['my_group']['my_dataset'].attrs)
```

The above snippet creates a HDF5 file with the following "content tree".

```bash
demo.h5
│── my_group
│   ├── .attrs
│   │   ├── a (1)
│   │   └── b (2)
│   └── my_dataset (array)
│       └── .attrs
│           ├── new (5)
│           └── huge (array ref)
└── big_attrs
    └── my_group.my_dataset.attrs.huge (array)
```

> See [below](#equivalent-snippet-using-h5py) for the equivalent snippet using *vanilla* `h5py`.


## Contents
+ [Installation instructions](#installation-instructions)
+ [Library Features](#library-features)
+ [Additional notes](#additional-notes)

## Installation instructions

From PyPI

```bash
pip install simple_h5py
```

From Conda

```bash
conda install -c arturo.mendoza.quispe simple_h5py
```

From git

```bash
pip install git+https://github.com/amq92/simple_h5py.git
```


## Library Features

### :rocket: Full Python experience 

While [`h5py`](http://www.h5py.org/) does provide a *high-level interface to the HDF5 library using established Python and NumPy concepts*, it purposely does not go the extra mile to provide a full Python experience to the user.
Of course, this design choice allows for great flexibility such as enabling chunked storage, allowing to store and manipulate the data in memory (using the `core` driver) and much more ! 

The goal of `simple_h5py` is to allow easier creation and handling of HDF5 files using the fabulous `h5py` library as a support :+1:

The following example highlights the similarity between creating a python-only object and creating a HDF5 file using `simple_h5py` !

```python
obj = dict()
obj['group'] = dict()
obj['group']['attrs'] = dict(a=1, b=2)
obj['group']['dataset'] = dict(contents=dataset)
obj['group']['dataset']['attrs'] = dict(c=3, d=4)

obj = BasicH5File('myfile.h5')
obj['group'] = None
obj['group'].attrs = dict(a=1, b=2)
obj['group']['dataset'] = dataset
obj['group']['dataset'].attrs = dict(c=3, d=4)
```

### :zap: Intelligent open/close of file

If the library is used for **creating** a new HDF5 file, it will open the file stream whenever a new value is given (either to a group, a dataset or an attribute instance) and subsequently closes it.
This is done only once for each `__setitem__` call.
Hence, setting an entire attribute group with a single dictionary will only require one open/close directive.
However, setting each attribute group entry will require multiple ones.

It the library is used for **reading** a new HDF5 file, it will parse the complete content-tree and load it into memory.
Additionally, it will load all group and dataset attributes too.
However, the **datasets will not be loaded** since they are assumed to be heavy.
Since the returned object contains the correct HDF5 references, the datasets can be loaded at any time (either completely or partially).

```python
obj = BasicH5File('myfile.h5')           # Load content-tree & attrs from disk
v1 = obj['group'].attrs                  # Inspect object in memory
v2 = obj['group'].attrs['a']             # Inspect object in memory
v3 = obj['group']['dataset'].attrs       # Inspect object in memory
v4 = obj['group']['dataset'].attrs['c']  # Inspect object in memory
v5 = obj['group']['dataset'][:]          # Load complete dataset from disk
v6 = obj['group']['dataset'][:10]        # Partially load dataset from disk
```

This strategy should allow a more *fluid* interaction with the HDF5 file since it can be fully inspected at any time, without requiring multiple open/close directives !

### :earth_americas: Handling of BIG ATTRIBUTES

The section of HDF5 User's Guide dedicated to the case of [Large Attributes](https://support.hdfgroup.org/HDF5/doc1.6/UG/13_Attributes.html).
Since, *Attributes are intended to be small objects*, most implementations limit the size of these meta-data (`h5py` will throw a RuntimeError).
The User's Guide proposes to point the attribute to another supplemental dataset.

`simple_h5py` implements this and makes the issue completely transparent to the user.
Every large attribute will be stored into a dataset with full path `/big_attrs/<dataset_name>.<group_name>.attrs.<attribute_name>`.

```python
obj = BasicH5File('myfile.h5')
dst = obj['group']['dataset']
dst.attrs['e'] = np.ones((1000000, 3))  # silently creates large dataset in
                                        # '/big_attrs/group.dataset.attrs.e'
dst.attrs['f'] = np.ones((10, 3))       # normal attribute creation but with
                                        # identical syntax ! 
```

Note that `simple_h5py` automatically handles the `huge` attribute during both, reading and writing the contents !

```python
obj = BasicH5File('myfile.h5')
v7 = obj['group']['dataset'].attrs['e']  # they are identical for the user !
v8 = obj['group']['dataset'].attrs['f']
```

### :pencil2: Nice `print` of the content-tree

Since the `BasicH5File` contains the entire content-tree at all times, displaying the object (either `__repr__` or `__str__`) allows for fast inspection of the file contents.

```python
print(obj)
# BasicH5File (myfile.h5)
# > Group "/group"
#   > Dataset "/group/dataset" (20, 30, 10)

display(obj['group'])
# Group
# > path: myfile.h5
# > route: /group
# > attrs: {'a': 1, 'b': 2}
# > datasets: ['dataset']

obj['group']['dataset']
# Dataset
# > path: myfile.h5
# > route: /group/dataset
# > attrs: {'c': 3, 'd': 4, 'e': array([[1., 1., 1.],
#        ...,
#        [1., 1., 1.]]), 'f': array([[1., 1., 1.],
#        ...,
#        [1., 1., 1.]])}
# > shape: (20, 30, 10)
```

### :cake: **[EXTRA]** Define required attributes

Define required attributes for all groups and all datasets:

```python
obj = BasicH5File('myfile.h5',
                  group_attrs_required=('a', 'b'),
                  dataset_attrs_required=('c', 'd'))
```

As such, if a file does not comply, an assertion error is raised.
A file may have more attributes than those required, but no less.

This feature is useful for ensuring that the HDF5 files to be read comply with the desired criteria.
One can even subclass the `BasicH5File` for easier use:

```python
class StrictH5File(BasicH5File):
    def __init__(self, path: str):
        super().__init__(path,
                         group_attrs_required=('a', 'b'),
                         dataset_attrs_required=('c', 'd'))

obj = StrictH5File('myfile.h5')
```

## Additional notes

`simple_h5py` is not meant to be `h5py` replacement but a useful sidekick.
Indeed, for "simple" use-cases, such as those shown here, `simple_h5py` allows faster development by *hiding* many of the implementation details.
As such, `h5py` should be employed for more advanced or custom needs.
Moroever, a file can be written with `simple_h5py` and then be read using `h5py`, and viceversa.


## Equivalent snippet using `h5py`

The following code performs has the same effect as the sample snippet.

> Note the simpler syntax that allows `simple_h5py` !

```python
import h5py
import numpy as np

group_attrs = dict(a=1, b=2)
dataset = np.ones((5, 4, 3))
dataset_attrs = dict(new=5, huge=np.ones((1000000, 3)))

# Use context manager to avoid open/close
with h5py.File('demo.h5', 'w') as obj:
    # Create group
    obj.create_group(name='my_group')

    # Add attributes to group one at a time
    for k, v in group_attrs.items():
        obj['my_group'].attrs[k] = v

    # Create dataset
    obj['my_group'].create_dataset('my_dataset', data=dataset)

    # Add attributes to dataset one at a time
    for k, v in dataset_attrs.items():

        # Use try/except for capturing the "large attributes"
        try:
            obj['my_group']['my_dataset'].attrs[k] = v
        except BaseException:
            # Create an auxiliary dataset called in a helper group 'big_attrs'
            if 'big_attrs' not in obj:
                obj.create_group(name='big_attrs')
            obj['big_attrs'].create_dataset('huge_attr',
                                            data=dataset_attrs['huge'])

            # Store the reference to the auxiliary dataset
            obj['my_group']['my_dataset'].attrs[k] = \
                obj['big_attrs']['huge_attr'].ref

# Use context manager to avoid open/close
with h5py.File('demo.h5', 'r') as obj:

    # Read attributes and convert to dictionary for further use
    read_group_attrs = dict(obj['my_group'].attrs)
    read_dataset_attrs = dict(obj['my_group']['my_dataset'].attrs)

    # Read entire dataset
    read_dataset = obj['my_group']['my_dataset'][:]

    # Verify if any reference is present in the attributes (i.e. big attribute)
    for k, v in read_dataset_attrs.items():
        if isinstance(v, h5py.Reference):
            read_dataset_attrs[k] = obj[v][:]

# Display contents
print(read_group_attrs)
print(read_dataset_attrs)
print(read_dataset[0])
```
