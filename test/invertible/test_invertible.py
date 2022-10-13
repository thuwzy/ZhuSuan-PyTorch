# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
from scipy import stats
import numpy as np
from zhusuan.invertible.scaling import Scaling
import unittest


class TestScaling(unittest.TestCase):

    def test_forward(self):
        scal = Scaling(2)
        x = torch.ones([1])
        self.assertEqual(x[0], 1)
