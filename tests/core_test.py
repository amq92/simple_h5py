import os
import unittest

import h5py
import numpy as np
from simple_h5py import BasicH5File


class TestCase(unittest.TestCase):

    fname = 'test.h5'

    group_name = 'my_group'
    dataset_name = 'my_dataset'

    group_attrs = dict(a=1, b=2)
    dataset = np.arange(600).reshape((20, 10, 3))
    dataset_attrs = dict(type='new', category='none',
                         huge=np.ones((1000000, 3)))

    def assertIsFile(self, path, msg=None):
        msg = msg if msg else 'File does not exist: {}'
        if not os.path.isfile(path):
            raise AssertionError(msg.format(path))

    def assertArrayEqual(self, first, second):
        self.assertIsInstance(first, np.ndarray)
        self.assertIsInstance(second, np.ndarray)
        if not np.all(first.ravel() == second.ravel()):
            raise AssertionError('Arrays are not equal')

    def assertEqual(self, first, second, msg=None):

        if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
            return self.assertArrayEqual(first, second)
        else:
            return super().assertEqual(first, second, msg)

    def tearDown(self):
        self.assertIsFile(self.fname)
        os.remove(self.fname)


class ReadTesting(TestCase):

    def setUp(self):

        GN, DN = self.group_name, self.dataset_name
        BA = 'big_attrs'

        with h5py.File(self.fname, 'w') as obj:
            obj.create_group(name=GN)
            for k, v in self.group_attrs.items():
                obj[GN].attrs[k] = v
            obj.create_group(name=BA)
            obj[GN].create_dataset(DN, data=self.dataset)
            obj[BA].create_dataset('huge_attr',
                                   data=self.dataset_attrs['huge'])
            for k, v in self.dataset_attrs.items():
                if 'huge' in k:
                    obj[GN][DN].attrs[k] = obj[BA]['huge_attr'].ref
                else:
                    obj[GN][DN].attrs[k] = v

        self.assertIsFile(self.fname)

    def test_01_read_group(self):
        obj = BasicH5File(self.fname)
        self.assertIn(self.group_name, obj)
        group = obj[self.group_name]
        self.assertIn(self.dataset_name, group)

    def test_02_read_group_attrs(self):
        obj = BasicH5File(self.fname)
        group_attrs = obj[self.group_name].attrs
        self.assertDictEqual(self.group_attrs, group_attrs)

    def test_03_read_dataset(self):
        obj = BasicH5File(self.fname)
        dataset = obj[self.group_name][self.dataset_name][:]
        self.assertEqual(self.dataset, dataset)

    def test_04_read_dataset_attrs(self):
        obj = BasicH5File(self.fname)
        dataset_attrs = obj[self.group_name][self.dataset_name].attrs
        for k, v in self.dataset_attrs.items():
            self.assertEqual(v, dataset_attrs[k])


class WriteTesting(TestCase):
    def test_01_write_group_with_attributes(self):
        GN = self.group_name

        obj = BasicH5File(self.fname)
        obj[GN] = None
        obj[GN].attrs = self.group_attrs.copy()

        with h5py.File(self.fname, 'r') as obj:
            self.assertIn(self.group_name, obj.keys())
            for k, v in self.group_attrs.items():
                self.assertEqual(v, obj[GN].attrs[k])

    def test_02_write_dataset_with_attributes(self):
        GN, DN = self.group_name, self.dataset_name

        obj = BasicH5File(self.fname)
        obj[GN] = None
        obj[GN].attrs = self.group_attrs.copy()
        obj[GN][DN] = self.dataset
        obj[GN][DN].attrs = self.dataset_attrs.copy()

        with h5py.File(self.fname, 'r') as obj:
            self.assertIn(self.dataset_name, obj[GN].keys())
            self.assertEqual(self.dataset, obj[GN][DN][:])
            for k, v in self.dataset_attrs.items():
                attr = obj[GN][DN].attrs[k]
                if isinstance(attr, h5py.Reference):
                    attr = obj[attr][:]
                self.assertEqual(v, attr)

    def test_03_write_multiple_big_attributes(self):
        GN1 = 'group_1'
        GN2 = 'group_2'
        DN1 = 'dataset_1'
        DN2 = 'dataset_2'
        BA = 'big_attrs'
        N = 1000000

        obj = BasicH5File(self.fname)
        obj[GN1] = None
        obj[GN1].attrs = dict()
        obj[GN2] = None
        obj[GN2].attrs = dict()
        obj[GN1][DN1] = np.ones((2, 2)) * 1
        obj[GN1][DN1].attrs = dict(a=1, b=np.ones((N, 3)) * 1)
        obj[GN1][DN2] = np.ones((2, 2)) * 2
        obj[GN1][DN2].attrs = dict(a=2, b=np.ones((N, 3)) * 2)
        obj[GN2][DN1] = np.ones((2, 2)) * 3
        obj[GN2][DN1].attrs = dict(a=3, b=np.ones((N, 3)) * 3)

        with h5py.File(self.fname, 'r') as obj:
            r11 = '.'.join((GN1, DN1, 'attrs', 'b'))
            r12 = '.'.join((GN1, DN2, 'attrs', 'b'))
            r21 = '.'.join((GN2, DN1, 'attrs', 'b'))
            self.assertIn(BA, obj.keys())
            self.assertIn(r11, obj[BA].keys())
            self.assertIn(r12, obj[BA].keys())
            self.assertIn(r21, obj[BA].keys())
            self.assertEqual(obj[BA][r11][:][0, 0], 1)
            self.assertEqual(obj[BA][r12][:][0, 0], 2)
            self.assertEqual(obj[BA][r21][:][0, 0], 3)


class AttrsTesting(TestCase):

    def setUp(self):
        GN, DN = self.group_name, self.dataset_name

        obj = BasicH5File(self.fname)
        obj[GN] = None
        obj[GN].attrs = self.group_attrs.copy()
        obj[GN][DN] = self.dataset
        obj[GN][DN].attrs = self.dataset_attrs.copy()

    def tearDown(self):
        self.assertIsFile(self.fname)
        os.remove(self.fname)

    def test_01_read_with_group_attrs_required(self):
        GR = list(self.group_attrs.keys())
        BasicH5File(self.fname, group_attrs_required=GR)

    def test_02_read_with_group_attrs_required(self):
        GR = list(self.group_attrs.keys())
        GR.pop(0)
        BasicH5File(self.fname, group_attrs_required=GR)

    def test_03_read_with_group_attrs_required(self):
        GR = list(self.group_attrs.keys())
        GR.append('extra_attr')
        self.assertRaises(AssertionError, BasicH5File, self.fname,
                          group_attrs_required=GR)

    def test_04_read_with_dataset_attrs_required(self):
        DR = list(self.dataset_attrs.keys())
        BasicH5File(self.fname, dataset_attrs_required=DR)

    def test_05_read_with_dataset_attrs_required(self):
        DR = list(self.dataset_attrs.keys())
        DR.pop(0)
        BasicH5File(self.fname, dataset_attrs_required=DR)

    def test_06_read_with_dataset_attrs_required(self):
        DR = list(self.dataset_attrs.keys())
        DR.append('extra_attr')
        self.assertRaises(AssertionError, BasicH5File, self.fname,
                          dataset_attrs_required=DR)

    def test_07_read_with_group_and_dataset_attrs_required(self):
        GR = list(self.group_attrs.keys())
        DR = list(self.dataset_attrs.keys())
        BasicH5File(self.fname, group_attrs_required=GR,
                    dataset_attrs_required=DR)

    def test_08_read_with_group_and_dataset_attrs_required(self):
        GR = list(self.group_attrs.keys())
        DR = list(self.dataset_attrs.keys())
        GR.pop(0)
        DR.pop(0)
        BasicH5File(self.fname, group_attrs_required=GR,
                    dataset_attrs_required=DR)

    def test_09_read_with_group_and_dataset_attrs_required(self):
        GR = list(self.group_attrs.keys())
        DR = list(self.dataset_attrs.keys())
        GR.append('extra_attr')
        DR.append('extra_attr')
        self.assertRaises(AssertionError, BasicH5File, self.fname,
                          group_attrs_required=GR,
                          dataset_attrs_required=DR)


if __name__ == '__main__':
    unittest.main()
