from __future__ import division, absolute_import, print_function

import platform
import pytest

import numpy as np
from numpy.testing import assert_


class TestErrstate(object):
    def test_divide(self):
        with np.errstate(all='raise', under='ignore'):
            a = -np.arange(3)
            # This should work
            with np.errstate(divide='ignore'):
                a // 0
            # While this should fail!
            try:
                a // 0
            except FloatingPointError:
                pass
            else:
                self.fail("Did not raise divide by zero error")

    def test_errcall(self):
        def foo(*args):
            print(args)

        olderrcall = np.geterrcall()
        with np.errstate(call=foo):
            assert_(np.geterrcall() is foo, 'call is not foo')
            with np.errstate(call=None):
                assert_(np.geterrcall() is None, 'call is not None')
        assert_(np.geterrcall() is olderrcall, 'call is not olderrcall')
