# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import numpy as np
from zhusuan.invertible.scaling import Scaling
from zhusuan.invertible.coupling import Coupling, MaskCoupling, get_coupling_mask
from zhusuan.invertible.sequential import RevSequential
from zhusuan.invertible.made import MADE
import unittest


def assert_invertible(tester, flow, x):
    y, log_det_J = flow(x)
    x_rec, log_det_J = flow.forward(y, reverse=True)
    delta = (x - x_rec).abs_().sum().detach().numpy()
    tester.assertLess(delta, 1e-4)


class TestScaling(unittest.TestCase):
    def test_init(self):
        Scaling(2)

    def test_forward(self):
        flow = Scaling(200)
        x = torch.rand([1, 200])
        assert_invertible(self, flow, x)



class TestCoupling(unittest.TestCase):
    def test_init(self):
        Coupling(20, 40, 2, 1)

    def test_forward(self):
        flow = Coupling(200, 200, 2, 1)
        x = torch.rand([1, 200])
        assert_invertible(self, flow, x)

    def test_get_mask(self):
        mask1 = get_coupling_mask(10, 1, 2, split_type="OddEven")
        mask2 = get_coupling_mask(10, 1, 2, split_type="Half")
        mask3 = get_coupling_mask(20, 1, 2, split_type="RandomHalf")
        self.assertEqual(mask1[0].dtype, torch.float32)
        self.assertEqual(mask2[0].dtype, torch.float32)
        self.assertEqual(mask3[0].dtype, torch.float32)
        delta1 = (mask1[0] - torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])).abs_().sum().detach().numpy()
        delta2 = (mask2[0] - torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])).abs_().sum().detach().numpy()
        self.assertLess(delta1, 1e-5)
        self.assertLess(delta2, 1e-5)



class TestMaskCoupling(unittest.TestCase):

    def test_init(self):
        mask = get_coupling_mask(200, 1, 4)
        MaskCoupling(200, 200, 4, mask)

    def test_forward(self):
        mask = get_coupling_mask(200, 1, 1)
        flow = MaskCoupling(200, 200, 4, mask[0])
        x = torch.rand([1, 200])
        assert_invertible(self, flow, x)


class TestRevSequential(unittest.TestCase):

    def test_init(self):
        RevSequential([Coupling(20, 40, 2, 1)])


    def test_forward(self):
        seqs = [Coupling(200, 200, 2, 1) for i in range(4)]
        flow = RevSequential(seqs)
        x = torch.rand([1, 200])
        assert_invertible(self, flow, x)


class TestMade(unittest.TestCase):

    def test_init(self):
        MADE(200, 200, 3)

    def test_forward(self):
        flow = MADE(200, 200, 3)
        x = torch.rand([1, 200])
        assert_invertible(self, flow, x)