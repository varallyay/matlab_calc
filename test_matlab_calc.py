"""Tests for matlab_calc.py.   Run:  python3 test_matlab_calc.py

The engine tests need only NumPy. The GUI and icon tests need a display (they are
skipped otherwise); the anti-aliased icon test also needs Pillow.
The installation tests work in temporary folders: they never touch your home folder,
/Applications or /opt. The update tests use a local stand-in for GitHub, not the internet.
"""
import gc
import http.server
import importlib.util
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import tkinter.font  # noqa: F401  (tk.font)
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matlab_calc as mc  # noqa: E402

ICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), mc.APP_ICON_FILE)

HAVE_PIL = importlib.util.find_spec('PIL') is not None


class EngineCase(unittest.TestCase):
    def setUp(self):
        self.eng = mc.MatlabEngine()

    def out(self, line):
        return ''.join(c for c in self.eng.execute(line) if isinstance(c, str))

    def ans(self, line, var='ans'):
        text = self.out(line)
        self.assertIsNone(self.eng.last_error, f"{line!r} -> {text}")
        return self.eng.variables[var]

    def assertValue(self, line, expected, var='ans'):
        got = self.ans(line, var)
        expected = mc.to_matrix(expected)
        self.assertEqual(got.shape, expected.shape, line)
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12, err_msg=line)

    def assertError(self, line, fragment):
        text = self.out(line)
        self.assertIsNotNone(self.eng.last_error, f"{line!r} should fail -> {text}")
        self.assertIn(fragment, text, f"{line!r} -> {text}")


class TestScalarAndElementwise(EngineCase):
    def test_elementwise_on_scalars(self):
        # The original translator gave 0, 2, 2 and a TypeError for these
        self.assertValue('2 ./ 4', 0.5)
        self.assertValue('5 ./ 2', 2.5)
        self.assertValue('2 .^ 3', 8)
        self.assertValue('3 .* 4', 12)
        self.assertValue('2 .\\ 4', 2)
        self.assertValue('pi .^ 2', np.pi ** 2)

    def test_scalars_from_results_keep_working(self):
        self.assertValue('s = [1 2] * [3; 4]; s ./ 2', 5.5)
        self.assertValue('sin(1) ./ [1 2]', [np.sin(1), np.sin(1) / 2])
        self.assertValue("x = 4; x'", 4)

    def test_precedence(self):
        self.out('A = [1 2];')
        self.assertValue('2 * A .^ 2', [2, 8])
        self.assertValue('-A .^ 2', [-1, -4])
        self.assertValue('-2^2', -4)
        self.assertValue('2^-1', 0.5)
        self.assertValue('2^3^2', 64)          # left-associative, as in MATLAB
        self.assertValue('1 + 2 * 3 - 4 / 2', 5)
        self.assertValue('(1 + 2) * 3', 9)
        self.assertValue('1:3 + 1', [1, 2, 3, 4])   # ':' binds looser than '+'

    def test_implicit_expansion_and_division_by_zero(self):
        self.assertValue('[1 2 3] + [10; 20]', [[11, 12, 13], [21, 22, 23]])
        self.assertValue('1/0', np.inf)
        self.assertValue('[1 -1] / 0', [np.inf, -np.inf])
        self.assertTrue(np.isnan(self.ans('0/0')).all())


class TestMatrixOperations(EngineCase):
    A = np.array([[4.0, 1.0], [2.0, 3.0]])

    def setUp(self):
        super().setUp()
        self.out('A = [4 1; 2 3]; b = [1; 2];')

    def test_products_and_division(self):
        self.assertValue('A * A', self.A @ self.A)
        self.assertValue('A .* A', self.A * self.A)
        self.assertValue('A \\ b', np.linalg.solve(self.A, [[1], [2]]))
        self.assertValue("b' / A", np.linalg.solve(self.A.T, [[1], [2]]).T)
        self.assertValue('A / 2', self.A / 2)
        self.assertValue('[1 2 3] / [4 5 6]', 32 / 77)          # least squares, as in MATLAB

    def test_powers(self):
        self.assertValue('A ^ 3', np.linalg.matrix_power(self.A, 3))
        self.assertValue('A ^ -1', np.linalg.inv(self.A))
        self.assertValue('R = A ^ 0.5; R * R', self.A)
        vals, vecs = np.linalg.eig(self.A)
        self.assertValue('2 ^ A', vecs @ np.diag(2 ** vals) @ np.linalg.inv(vecs))
        self.assertValue('A .^ 2', self.A ** 2)

    def test_transposes(self):
        self.assertValue("A'", self.A.T)
        self.assertValue("[1+2i 3]'", [[1 - 2j], [3]])
        self.assertValue("[1+2i 3].'", [[1 + 2j], [3]])

    def test_functions(self):
        self.assertValue('det(A)', 10)
        self.assertValue('inv(A)', np.linalg.inv(self.A))
        self.assertValue('eye(2)', np.eye(2))
        self.assertValue('n = 6/2; zeros(n, n)', np.zeros((3, 3)))   # float sizes used to crash
        self.assertValue('ones(2, 3)', np.ones((2, 3)))
        self.assertValue('ones([2 3])', np.ones((2, 3)))
        self.assertValue('abs([-3 4])', [3, 4])
        self.assertValue('log10(1000)', 3)
        self.assertValue('exp(1)', np.e)

    def test_power_and_times_functions(self):
        self.assertValue('power(A, 2)', self.A ** 2)
        self.assertValue('power(A, b)', self.A ** np.array([[1], [2]]))     # implicit expansion
        self.assertValue('power(2, [1 2 3])', [2, 4, 8])
        self.assertValue('power(-8, 1/3)', complex(-8) ** (1 / 3))
        self.assertValue('times(A, A)', self.A * self.A)
        self.assertValue('times([1 2 3], [4; 5])', [[4, 8, 12], [5, 10, 15]])
        self.assertValue('times(2, 3)', 6)
        for a, b in (('A', 'b'), ('[1 2 3]', '[1; 2]'), ('2', 'A'), ('A', "A'")):
            self.assertValue(f'power({a}, {b}) - {a}.^({b})', np.zeros_like(self.ans(f'{a}.^({b})')))
            self.assertValue(f'times({a}, {b}) - {a}.*({b})', np.zeros_like(self.ans(f'{a}.*({b})')))
        self.assertError('power([1 2 3], [1 2])', 'Arrays have incompatible sizes')
        self.assertError('times(1)', 'Not enough input arguments')
        self.assertError('power(1, 2, 3)', 'Too many input arguments')

    def test_constants(self):
        self.assertValue('pi', np.pi)
        self.assertValue('e', np.e)
        self.assertValue('[Inf inf]', [np.inf, np.inf])
        self.assertValue('i + j', 2j)
        self.assertTrue(np.isnan(self.ans('NaN')).all())
        self.assertValue('pi()', np.pi)
        self.assertError('pi(2)', 'Too many input arguments')
        self.assertValue('i = 5; i', 5, var='i')                  # a variable hides the constant

    def test_rounding(self):
        # MATLAB rounds halves away from zero (NumPy's np.round would give -2 -2 0 0 2 2)
        self.assertValue('round([-2.5 -1.5 -0.5 0.5 1.5 2.5])', [-3, -2, -1, 1, 2, 3])
        self.assertValue('round(0.49999999999999994)', 0)
        self.assertValue('round(pi, 2)', 3.14)
        self.assertValue('round(0.125, 2)', 0.13)
        self.assertValue('round(1234.5678, -2)', 1200)
        self.assertValue('round(2.5 + 1.5i)', 3 + 2j)          # real and imaginary parts separately
        self.assertValue('floor([-2.5 2.5])', [-3, 2])
        self.assertValue('ceil([-2.5 2.5])', [-2, 3])
        self.assertValue('floor(-0.5 + 2.7i)', -1 + 2j)
        self.assertValue('round([Inf -Inf])', [np.inf, -np.inf])
        self.assertValue('round(5, -400)', 0)
        self.assertValue('round(123, 400)', 123)
        self.assertEqual(self.out('round(-0.4)'), 'ans =\n    0\n')  # no '-0'
        self.assertError('round(1, 1.5)', 'must be an integer scalar')
        self.assertError('round(1, 2, 3)', 'Too many input arguments')
        self.assertError('ceil()', 'Not enough input arguments')

    def test_complex_results(self):
        self.assertValue('sqrt(-4)', 2j)
        self.assertValue('(-8)^(1/3)', complex(-8) ** (1 / 3))
        self.assertValue('abs(3 + 4i)', 5)
        self.assertValue('(1 + 2j) * i', -2 + 1j)
        self.assertValue('log(-1)', np.pi * 1j)

    def test_comparisons_and_logic(self):
        self.assertValue('a = 3; a == 3', 1)
        self.assertValue('a ~= 3', 0)
        self.assertValue('[1 2 3] > 1', [0, 1, 1])
        self.assertValue('~[0 2]', [1, 0])
        self.assertValue('a > 2 & a < 4', 1)
        self.assertValue('a < 2 | a > 4', 0)
        self.assertValue('a > 2 && a < 4', 1)

    def test_ranges(self):
        self.assertValue('1:5', [1, 2, 3, 4, 5])
        self.assertValue('5:-2:1', [5, 3, 1])
        self.assertValue('0:0.1:0.3', [0, 0.1, 0.2, 0.3])
        self.assertEqual(self.ans('1:0').shape, (1, 0))

    def test_dimension_errors(self):
        self.assertError('[1 2 3] * [4 5 6]', 'Incorrect dimensions for matrix multiplication')
        self.assertError('[1 2 3] ^ 2', 'Incorrect dimensions for raising a matrix to a power')
        self.assertError('[1 2 3] + [1 2]', 'Arrays have incompatible sizes')
        self.assertError('[1 2; 3]', 'Dimensions of arrays being concatenated are not consistent')
        self.assertError('inv([1 2; 2 4])', 'singular')
        self.assertError('det([1 2 3])', 'Matrix must be square')


class TestParsing(EngineCase):
    def test_matrix_literals(self):
        self.assertValue('[1 -2]', [1, -2])
        self.assertValue('[1 - 2]', -1)
        self.assertValue('[1 +2]', [1, 2])
        self.assertValue('[1, 2; 3, 4]', [[1, 2], [3, 4]])
        self.assertValue('[1 .5]', [1, 0.5])
        self.assertValue('[1 .* 2]', 2)
        self.assertValue("x = [1 2]; [x' x']", [[1, 1], [2, 2]])
        self.assertValue('[[1 2]; [3 4]]', [[1, 2], [3, 4]])
        self.assertValue('[1 2\n3 4]', [[1, 2], [3, 4]])
        self.assertValue('[1 2 ...\n 3]', [1, 2, 3])
        self.assertValue('[[] 1 2]', [1, 2])
        self.assertEqual(self.ans('[]').shape, (0, 0))

    def test_number_literals(self):
        self.assertValue('1e-3', 0.001)
        self.assertValue('1.5e3', 1500)
        self.assertValue('.5', 0.5)
        self.assertValue('1.', 1)
        self.assertValue('1./[1 2]', [1, 0.5])
        self.assertValue('2.^[1 2]', [2, 4])
        self.assertValue('4i', 4j)

    def test_statements_and_comments(self):
        self.assertEqual(self.out('B = [1 2 3];   % Semicolon suppresses output'), '')
        self.assertValue('B', [1, 2, 3], var='B')
        text = self.out('x = 5, y = x ./ 2; z = y * 2')
        self.assertIn('x =\n    5\n', text)
        self.assertNotIn('y =', text)
        self.assertIn('z =\n    5\n', text)
        self.assertValue('a = 1; b = 2', 2, var='b')

    def test_names(self):
        self.assertValue('clearance = 5', 5, var='clearance')   # used to be treated as 'clear'
        self.assertValue('pi = 3; pi', 3, var='pi')             # variables shadow constants
        self.assertValue('pi()', 3)                            # x() is just x
        self.assertError('foo + 1', "Unrecognized function or variable 'foo'")
        self.assertError('_mat = 3', 'Invalid text character')

    def test_syntax_errors(self):
        self.assertError('y = [1 2', 'Unbalanced')
        self.assertError('(1 + 2))', 'Unbalanced')
        self.assertError('y = 1 +', 'incomplete')
        self.assertError('1 2', "Invalid expression near '2'")
        self.assertError("'abc", 'not terminated')
        self.assertError('x = 1 = 2', "Invalid use of '='")
        self.assertError('sin(1, 2)', 'Too many input arguments')
        self.assertError('sin()', 'Not enough input arguments')
        self.assertError('2 € 3', 'Invalid text character')

    def test_error_stops_rest_of_line(self):
        text = self.out('a = 1, foo, b = 2')
        self.assertIn('a =', text)
        self.assertNotIn('b =', text)
        self.assertNotIn('b', self.eng.variables)


class TestIndexing(EngineCase):
    A = np.arange(1, 13, dtype=float).reshape(3, 4)   # [1 2 3 4; 5 6 7 8; 9 10 11 12]
    COLUMN_MAJOR = A.ravel(order='F')                  # MATLAB numbers elements down the columns

    def setUp(self):
        super().setUp()
        self.out('A = [1 2 3 4; 5 6 7 8; 9 10 11 12]; v = [10 20 30 40]; c = [10; 20; 30];')

    def test_elements(self):
        self.assertValue('A(2,3) + A(1,4)', 7 + 4)
        self.assertValue('A(2,3)', 7)
        self.assertValue('A(3, 1)', 9)
        self.assertValue('A(5)', self.COLUMN_MAJOR[4])         # 6
        self.assertValue('v(3)', 30)
        self.assertValue('c(2)', 20)
        self.assertValue('A(2,3) * v(4) - c(1) ./ A(1)', 7 * 40 - 10)
        self.assertValue('A(A(1,2), A(1,3))', 7)              # subscripts can be expressions
        self.assertValue('sqrt(A(3,4) + A(1,4))', 4)
        self.assertValue('[A(1,1) A(3,4); v(1:2)]', [[1, 12], [10, 20]])
        self.assertValue('x = 5; x(1)', 5)
        self.assertValue('ans(1)', 5)

    def test_rows_columns_and_ranges(self):
        self.assertValue('A(2, :)', self.A[1:2, :])
        self.assertValue('A(:, 3)', self.A[:, 2:3])
        self.assertValue('A(1:2, [1 3])', self.A[np.ix_([0, 1], [0, 2])])
        self.assertValue('A([3 1], 2)', self.A[np.ix_([2, 0], [1])])
        self.assertValue('A(:, :)', self.A)
        self.assertValue('A()', self.A)
        self.assertValue('A(:)', self.COLUMN_MAJOR.reshape(-1, 1))
        self.assertValue('v(2:3)', [20, 30])
        self.assertValue('v([1 1 2])', [10, 10, 20])

    def test_end(self):
        self.assertValue('A(end)', 12)
        self.assertValue('A(end, 1)', 9)
        self.assertValue('A(1, end)', 4)
        self.assertValue('A(end-1, end-2:end)', [6, 7, 8])
        self.assertValue('v(2:end)', [20, 30, 40])
        self.assertValue('v(end:-1:1)', [40, 30, 20, 10])
        self.assertValue('A(round(end/2))', self.COLUMN_MAJOR[5])  # end inside a function call
        self.assertValue('A(v(end) / 20, end)', 8)             # inner end belongs to v: A(40/20, 4)
        self.assertError('A(v(end) / 10, end)', 'Index in position 1 exceeds array bounds')  # A(4, 4)

    def test_result_shapes(self):
        self.assertValue('v([1; 3])', [10, 30])                # a vector keeps its orientation
        self.assertValue('c([1 3])', [[10], [30]])
        self.assertValue('A([1 2; 3 4])', self.COLUMN_MAJOR[np.array([[1, 2], [3, 4]]) - 1])  # [A(1) A(2); A(3) A(4)]
        self.assertValue('A([2 3])', [5, 9])                   # matrix(row vector) is a row
        self.assertValue('x = 7; x([1 1 1])', [7, 7, 7])
        self.assertEqual(self.ans('A([])').shape, (0, 0))
        self.assertEqual(self.ans('v([])').size, 0)

    def test_variables_shadow_functions(self):
        self.assertValue('sin = [5 6 7]; sin(2)', 6)
        self.assertValue('clear sin; sin(0)', 0)

    def test_errors(self):
        self.assertError('A(3, 5)', 'Index in position 2 exceeds array bounds. Index must not exceed 4.')
        self.assertError('A(4, 1)', 'Index in position 1 exceeds array bounds. Index must not exceed 3.')
        self.assertError('A(13)', 'Index exceeds the number of array elements. Index must not exceed 12.')
        self.assertError('A(end + 1)', 'Index must not exceed 12')
        self.assertError('A(0)', 'Array indices must be positive integers')
        self.assertError('A(1.5, 1)', 'Index in position 1 is invalid.')
        self.assertError('A(-1)', 'Array indices must be positive integers')
        self.assertError('A(NaN)', 'Array indices must be positive integers')
        self.assertError('A(1, 2, 3)', 'Only one or two subscripts')
        self.assertError('A([1 0 1])', 'Array indices must be positive integers or logical values')
        self.assertError('end + 1', "'end' can only be used inside the subscript")
        self.assertError('sin(end)', "'end' can only be used inside the subscript")
        self.assertError('sin(:)', 'A colon on its own can only be used as a subscript')
        self.assertError('end = 3', "'end' is a keyword")
        self.assertError('foo(1)', "Unrecognized function or variable 'foo'")

    def test_display(self):
        self.assertEqual(self.out('A(2,3) + A(1,4)'), 'ans =\n    11\n')
        self.assertEqual(self.out('A(:, end)'), 'ans =\n     4\n     8\n    12\n')
        self.assertIn('A(i, j)', self.out('help end'))


class TestAssignment(EngineCase):
    def setUp(self):
        super().setUp()
        self.out('A = [1 2 3 4; 5 6 7 8; 9 10 11 12]; v = [10 20 30 40]; c = [1; 2; 3];')

    def A(self):
        return np.arange(1, 13, dtype=float).reshape(3, 4)

    def test_change_elements(self):
        expected = self.A()
        expected[1, 2] = 5
        self.assertValue('A(2,3) = 5', expected, var='A')
        expected[0, 1] = -1                                                # A(4) is A(1,2): down the columns
        self.assertValue('A(4) = -1', expected, var='A')
        expected[:, 0] = 0
        self.assertValue('A(:, 1) = 0', expected, var='A')
        expected[2, :] = [7, 8, 9, 10]
        self.assertValue('A(3, :) = [7 8 9 10]', expected, var='A')
        expected[np.ix_([0, 1], [2, 3])] = [[1, 2], [3, 4]]
        self.assertValue('A(1:2, end-1:end) = [1 2; 3 4]', expected, var='A')
        expected[:, 1] = [100, 200, 300]
        self.assertValue('A(:, 2) = [100 200 300]', expected, var='A')   # a row fits a column
        self.assertValue('v(end) = v(1) + v(2)', [10, 20, 30, 30], var='v')
        self.assertValue('c(2) = 5', [[1], [5], [3]], var='c')

    def test_output_and_suppression(self):
        self.assertEqual(self.out('v(2) = 0'), 'v =\n    10   0   30   40\n')
        self.assertEqual(self.out('v(1) = 0;'), '')
        self.assertValue('v', [0, 0, 30, 40], var='v')

    def test_growing(self):
        self.assertValue('v(6) = 60', [10, 20, 30, 40, 0, 60], var='v')
        self.assertValue('v(end+1) = 70', [10, 20, 30, 40, 0, 60, 70], var='v')
        self.assertValue('c(5) = 9', [[1], [2], [3], [0], [9]], var='c')
        grown = np.zeros((4, 5))
        grown[:3, :4] = self.A()
        grown[3, 4] = 1
        self.assertValue('A(4, 5) = 1', grown, var='A')                   # new rows/columns are 0
        self.assertValue('w(3) = 7', [0, 0, 7], var='w')                  # creates a new variable
        self.assertValue('M(2, 3) = 1', [[0, 0, 0], [0, 0, 1]], var='M')
        self.assertValue('col(:, 1) = [1 2 3]', [[1], [2], [3]], var='col')
        self.assertValue('s = 5; s(3) = 1', [5, 0, 1], var='s')
        self.assertError('B = [1 2; 3 4]; B(5) = 1', 'Attempt to grow array along ambiguous dimension')

    def test_conditions(self):
        expected = self.A()
        expected[expected > 6] = 0
        self.assertValue('A(A > 6) = 0', expected, var='A')
        self.assertValue('v(v > 25) = v(v > 25) / 10', [10, 20, 3, 4], var='v')
        self.assertValue('v(v == 20) = []', [10, 3, 4], var='v')

    def test_deleting(self):
        self.assertValue('v([1 3]) = []', [20, 40], var='v')
        self.assertValue('c(2) = []', [[1], [3]], var='c')
        self.assertValue('A(2, :) = []', self.A()[[0, 2], :], var='A')
        self.assertValue('A(:, [1 end]) = []', self.A()[np.ix_([0, 2], [1, 2])], var='A')
        self.assertValue('B = [1 2; 3 4]; B([1 4]) = []', [3, 2], var='B')   # a matrix becomes a row
        self.assertError('A(1, 2) = []', 'A null assignment can have only one non-colon index')
        self.assertError('v(9) = []', 'Index exceeds the number of array elements')

    def test_kinds_of_values(self):
        self.assertValue('v(2) = 1i', [10, 1j, 30, 40], var='v')           # becomes complex
        self.assertEqual(self.ans('b = [true false]; b(3) = 5;', var='b').dtype, bool)  # stays logical
        self.assertError('b(1) = NaN', 'logical array')

    def test_errors(self):
        self.assertError('v(2) = [1 2]', 'different number of elements')
        self.assertError('A(1, :) = [1 2 3]', 'the size of the left side is 1-by-4 and the size of the right side is 1-by-3')
        self.assertError('A(0) = 1', 'Array indices must be positive integers')
        self.assertError('A(1.5) = 1', 'Array indices must be positive integers')
        self.assertError('A() = 1', 'needs a subscript')
        self.assertError('end(1) = 3', "'end' is a keyword")
        self.assertValue('A(2,3) == 7', True)                              # '==' compares, it does not assign


class TestConditionsAndFind(EngineCase):
    def setUp(self):
        super().setUp()
        self.out('A = [1 2 3 4; 5 6 7 8; 9 10 11 12]; v = [10 20 30 40];')

    def test_find_examples_from_the_request(self):
        self.assertValue('A = [1 2 2.4 2.3]; find(A >= 2)', [2, 3, 4])
        self.assertValue("A = A'; find(A >= 2)", [[2], [3], [4]])

    def test_find(self):
        A = np.arange(1, 13, dtype=float).reshape(3, 4)
        self.assertValue('find(A > 6)', (np.flatnonzero(A.ravel(order='F') > 6) + 1).reshape(-1, 1))
        self.assertValue('find(eye(3))', [[1], [5], [9]])
        self.assertValue('find([0 3 0 5])', [2, 4])
        self.assertValue('find([1 0 1 1], 2)', [1, 3])
        self.assertValue('v(find(v > 15))', [20, 30, 40])
        self.assertEqual(self.ans('find([0 0])').shape, (1, 0))
        self.assertEqual(self.ans('find([0; 0])').shape, (0, 1))
        self.assertEqual(self.ans('find([])').shape, (0, 0))
        self.assertError('find(v, 0)', 'positive scalar integer')
        self.assertIn('find(A >= 2)', self.out('help find'))

    def test_selecting_by_condition(self):
        A = np.arange(1, 13, dtype=float).reshape(3, 4)
        self.assertValue('A(A > 6)', A.ravel(order='F')[A.ravel(order='F') > 6].reshape(-1, 1))
        self.assertValue('v(v > 15)', [20, 30, 40])                        # a row stays a row
        self.assertValue('v(v > 15 & v < 35)', [20, 30])
        self.assertValue('v(~(v == 20))', [10, 30, 40])
        self.assertValue('A(A(:, 1) > 4, :)', A[1:, :])                    # rows chosen by a condition
        self.assertValue('A(:, A(1, :) >= 3)', A[:, 2:])
        self.assertValue('v(logical([1 0 1 0]))', [10, 30])
        self.assertEqual(self.ans('v(v > 100)').size, 0)
        self.assertError('v([1 0 1])', 'Array indices must be positive integers or logical values')

    def test_logical_values(self):
        self.assertEqual(self.ans('A > 6').dtype, bool)
        self.assertValue('true + true', 2)
        self.assertValue('sum = (v > 15) * 1; sum', [0, 1, 1, 1], var='sum')
        self.assertValue('-true', -1)
        self.assertValue('[true 2]', [1, 2])
        self.assertValue('true(2)', np.ones((2, 2)))
        self.assertValue('false(1, 3)', np.zeros((1, 3)))
        self.assertValue('logical([2 0 -1])', [1, 0, 1])
        self.assertError('logical(NaN)', 'Only real numbers other than NaN can be converted to logical')
        self.assertEqual(self.out('v > 15'), 'ans =\n  1x4 logical array\n    0   1   1   1\n')
        self.assertEqual(self.out('3 > 2'), 'ans =\n  logical\n    1\n')
        self.out('t = v > 15;')
        self.assertIn('t          1x4        logical', self.out('whos'))


def text_of(value):
    return mc.chars_to_text(value)


class TestText(EngineCase):
    def test_literals(self):
        self.assertEqual(text_of(self.ans("s = 'hello'", var='s')), 'hello')
        self.assertEqual(text_of(self.ans("'it''s'")), "it's")
        self.assertEqual(text_of(self.ans('"say ""hi"""')), 'say "hi"')
        self.assertEqual(text_of(self.ans("'Árvíztűrő'")), 'Árvíztűrő')        # not only ASCII
        self.assertEqual(self.ans("''").shape, (0, 0))
        self.assertError("'abc", 'not terminated')

    def test_text_inside_lines(self):
        self.assertEqual(text_of(self.ans("a = '%d; not, a comment' % but this is", var='a')),
                         '%d; not, a comment')
        self.assertEqual(self.out("b = 'x;y'; c = 2"), 'c =\n    2\n')
        self.assertValue("x = [1 2]'; y = x'", [1, 2], var='y')                 # quotes still transpose
        self.assertEqual(text_of(self.ans("[x' 'ab']")), '\x01\x02ab')

    def test_operations(self):
        self.out("s = 'hello';")
        self.assertEqual(text_of(self.ans('s(end:-1:1)')), 'olleh')
        self.assertEqual(text_of(self.ans("[s ' world']")), 'hello world')
        self.assertEqual(text_of(self.ans("['abc', 65]")), 'abcA')              # text with numbers is text
        self.assertValue("'a' + 1", 98)                                          # arithmetic uses the codes
        self.assertValue("s == 'hello'", [True] * 5)
        self.assertEqual(text_of(self.ans("s(1) = 'J'; s", var='s')), 'Jello')
        self.assertEqual(self.ans("['ab'; 'cd']").shape, (2, 2))
        self.assertError("['abc'; 'de']", 'Dimensions of arrays being concatenated are not consistent')

    def test_display(self):
        self.assertEqual(self.out("s = 'hi'"), "s =\n    'hi'\n")
        self.assertEqual(self.out("['ab'; 'cd']"), "ans =\n  2x2 char array\n    'ab'\n    'cd'\n")
        self.assertEqual(self.out("''"), "ans =\n    ''\n")
        self.out("t = 'x';")
        self.assertIn('t          1x1        char', self.out('whos'))


class TestSeveralOutputs(EngineCase):
    def setUp(self):
        super().setUp()
        self.out('A = [3 1 4; 1 5 9; 2 6 5];')

    def test_assigning_several_outputs(self):
        self.assertEqual(self.out('[m, n] = size(A)'), 'm =\n    3\nn =\n    3\n')  # the window adds blank lines
        self.assertValue('[r, c] = find(A > 5);', [[3], [2]], var='r')
        self.assertValue('c', [[2], [3]], var='c')
        self.assertValue('[r c v] = find(A > 5);', [[1], [1]], var='v')         # v: values of A > 5 (true)
        self.assertValue('[r c v] = find(A .* (A > 5));', [[6], [9]], var='v')  # spaces separate names too
        self.assertValue('[~, i] = max([4 9 2]);', 2, var='i')                   # ~ skips an output
        self.assertNotIn('~', self.eng.variables)
        self.assertEqual(self.out('[p, q] = size(A);'), '')

    def test_errors(self):
        self.assertError('[a, b] = sin(1)', 'Error using sin\nToo many output arguments.')
        self.assertError('[a, b] = A', 'Too many output arguments')
        self.assertError('[a, b, c, d] = find(A)', 'Too many output arguments')
        self.assertError('disp(1) + 1', 'Error using disp\nToo many output arguments.')
        self.assertError('x = rng(1)', 'Too many output arguments')

    def test_functions_without_result_keep_ans(self):
        self.out('7;')
        self.assertEqual(self.out("disp('x')"), 'x\n')
        self.assertEqual(self.out("fprintf('y\\n')"), 'y\n')
        self.assertValue('ans', 7)


class TestNewMath(EngineCase):
    def test_remainders(self):
        self.assertValue('rem([7 -7 7], [3 3 -3])', [1, -1, 1])
        self.assertValue('mod([7 -7 7], [3 3 -3])', [1, 2, -2])
        self.assertValue('mod(5.5, 2)', 1.5)
        self.assertValue('mod(7, 0)', 7)
        self.assertTrue(np.isnan(self.ans('rem(7, 0)')).all())
        self.assertValue('mod(0:5, 3)', [0, 1, 2, 0, 1, 2])
        self.assertError('mod([1 2 3], [1 2])', 'Arrays have incompatible sizes')

    def test_hyperbolic(self):
        x = np.array([-1.5, 0, 0.5, 2])
        for name, func in (('sinh', np.sinh), ('cosh', np.cosh), ('tanh', np.tanh), ('asinh', np.arcsinh)):
            self.assertValue(f'{name}([-1.5 0 0.5 2])', func(x))
        self.assertValue('acosh([1 2 10])', np.arccosh([1, 2, 10]))
        self.assertValue('atanh([-0.5 0 0.5])', np.arctanh([-0.5, 0, 0.5]))
        self.assertValue('asinh(sinh(0.7))', 0.7)
        self.assertTrue(np.isnan(self.ans('acosh(0.5)')).all())

    def test_random_numbers(self):
        r = self.ans('rng(1); r = rand(3, 4);', var='r')
        self.assertEqual(r.shape, (3, 4))
        self.assertTrue(np.all((r >= 0) & (r < 1)))
        self.assertValue('rng(1); rand(3, 4)', r)                                # same seed, same numbers
        self.assertEqual(self.ans('rand').shape, (1, 1))
        self.assertEqual(self.ans('randn(2)').shape, (2, 2))
        x = self.ans('rng(5); randn(1, 20000)')
        self.assertLess(abs(x.mean()), 0.05)
        self.assertLess(abs(x.std() - 1), 0.05)
        self.assertNotEqual(self.ans('rng(1); rand').item(), self.ans('rng(2); rand').item())
        self.eng.execute("rng('default')")
        self.assertIsNone(self.eng.last_error)
        self.assertError("rng('fast')", 'rng(seed)')

    def test_size_and_length(self):
        self.out('A = zeros(3, 7);')
        self.assertValue('size(A)', [3, 7])
        self.assertValue('size(A, 2)', 7)
        self.assertValue('size(A, 3)', 1)
        self.assertValue('[m, n, p] = size(A);', 1, var='p')
        self.assertValue('length(A)', 7)
        self.assertValue('length([])', 0)
        self.assertValue("length('hello')", 5)

    def test_linspace(self):
        self.assertValue('linspace(0, 1, 5)', [0, 0.25, 0.5, 0.75, 1])
        self.assertEqual(self.ans('linspace(1, 2)').shape, (1, 100))
        self.assertValue('linspace(3, 7, 1)', 7)
        self.assertEqual(self.ans('linspace(0, 1, 0)').shape, (1, 0))

    def test_resize_and_reshape(self):
        self.assertValue('resize([1 2 3], 5)', [1, 2, 3, 0, 0])
        self.assertValue('resize([1 2 3], 2)', [1, 2])
        self.assertValue('resize([1; 2; 3], 4)', [[1], [2], [3], [0]])
        self.assertValue('resize([1 2; 3 4], [3 3])', [[1, 2, 0], [3, 4, 0], [0, 0, 0]])
        self.assertValue("resize([1 2 3], 5, 'FillValue', 9)", [1, 2, 3, 9, 9])
        self.assertValue("resize(1:5, 3, 'Side', 'leading')", [3, 4, 5])
        self.assertValue("resize([1 2], 4, 'Side', 'both')", [0, 1, 2, 0])
        self.assertEqual(text_of(self.ans("resize('ab', 4)")), 'ab  ')
        self.assertValue('reshape(1:6, 2, 3)', [[1, 3, 5], [2, 4, 6]])
        self.assertValue('reshape(1:6, [3 2])', [[1, 4], [2, 5], [3, 6]])
        self.assertValue('reshape(1:6, [], 2)', [[1, 4], [2, 5], [3, 6]])
        self.assertError('reshape(1:6, 4, 2)', 'Number of elements must not change')


class TestStatistics(EngineCase):
    A = np.array([[3.0, 1, 4], [1, 5, 9], [2, 6, 5]])

    def setUp(self):
        super().setUp()
        self.out('A = [3 1 4; 1 5 9; 2 6 5];')

    def test_max_and_min(self):
        self.assertValue('max(A)', self.A.max(axis=0))
        self.assertValue('min(A)', self.A.min(axis=0))
        self.assertValue('[m, i] = max(A);', [1, 3, 2], var='i')
        self.assertValue('max(A, [], 2)', self.A.max(axis=1).reshape(-1, 1))
        self.assertValue("max(A, [], 'all')", 9)
        self.assertValue("[m, k] = min(A, [], 'all');", 2, var='k')            # A(2) is the smallest
        self.assertValue('max([1 5 3])', 5)
        self.assertValue('max([1; 5; 3])', 5)
        self.assertValue('max(A, 4)', np.maximum(self.A, 4))
        self.assertValue('min([4 NaN 2])', 2)                                    # NaN is skipped
        self.assertTrue(np.isnan(self.ans('max([NaN NaN])')).all())
        self.assertValue('[m, i] = max([2 7 7]);', 2, var='i')                   # the first of equal values
        self.assertError('[m, i] = max(A, 4)', 'one output')

    def test_mean_median_std(self):
        self.assertValue('mean(A)', self.A.mean(axis=0))
        self.assertValue('mean(A, 2)', self.A.mean(axis=1).reshape(-1, 1))
        self.assertValue("mean(A, 'all')", self.A.mean())
        self.assertValue('mean([1 2 3 4])', 2.5)
        self.assertValue('median([3 1 4 1 5])', 3)
        self.assertValue('median([3 1 4 1])', 2)
        self.assertValue('median(A)', np.median(self.A, axis=0))
        self.assertValue('std([2 4 4 4 5 5 7 9])', np.std([2, 4, 4, 4, 5, 5, 7, 9], ddof=1))
        self.assertValue('std([2 4 4 4 5 5 7 9], 1)', 2)
        self.assertValue('std(A)', np.std(self.A, axis=0, ddof=1))
        self.assertValue('std(A, 0, 2)', np.std(self.A, axis=1, ddof=1).reshape(-1, 1))
        self.assertValue('std(5)', 0)
        self.assertTrue(np.isnan(self.ans('mean([])')).all())
        self.assertTrue(np.isnan(self.ans('mean([1 NaN 3])')).all())
        self.assertError('std(A, 2)', 'weight must be 0')


class TestTextOutputAndFiles(EngineCase):
    def setUp(self):
        super().setUp()
        self.folder = os.getcwd()
        self.tmp = tempfile.mkdtemp()
        os.chdir(self.tmp)

    def tearDown(self):
        self.eng.execute("fclose('all');")
        os.chdir(self.folder)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_disp(self):
        self.assertEqual(self.out("disp('Hello')"), 'Hello\n')
        self.assertEqual(self.out('disp(pi)'), '    3.1416\n')
        self.assertEqual(self.out('disp([1 2; 3 4])'), '    1   2\n    3   4\n')
        self.assertEqual(self.out("disp(['ab'; 'cd'])"), 'ab\ncd\n')
        self.assertEqual(self.out('disp([])'), '')
        self.assertEqual(self.out("disp(['x = ' num2str(2.5)])"), 'x = 2.5\n')
        self.assertTrue(all(isinstance(c, mc.RawText) for c in self.eng.execute("disp('a')")))

    def test_fprintf_to_the_screen(self):
        self.assertEqual(self.out("fprintf('%d squared is %d\\n', [1 2 3; 1 4 9])"),
                         '1 squared is 1\n2 squared is 4\n3 squared is 9\n')
        self.assertEqual(self.out("fprintf('%d, %d\\n', 1, 2, 3)"), '1, 2\n3, ')     # stops without data
        self.assertEqual(self.out("fprintf('%s = %.3f\\n', 'pi', pi)"), 'pi = 3.142\n')
        self.assertEqual(self.out("fprintf('100%% done\\t|\\n')"), '100% done\t|\n')
        self.assertEqual(self.out("fprintf('%5.1f|%-4d|%e\\n', 2.5, 7, 12345)"), '  2.5|7   |1.234500e+04\n')
        self.assertEqual(self.out("fprintf(1, 'x%c\\n', 'yz')"), 'xy\nxz\n')
        self.assertEqual(self.out("fprintf('%d\\n', [NaN Inf])"), 'NaN\nInf\n')
        self.assertEqual(self.out("fprintf('no newline')"), 'no newline')
        self.assertValue("n = fprintf('abc\\n');", 4, var='n')

    def test_sprintf_and_num2str(self):
        self.assertEqual(text_of(self.ans("sprintf('%.2f', pi)")), '3.14')
        self.assertEqual(text_of(self.ans("sprintf('%d, ', 1:3)")), '1, 2, 3, ')
        self.assertEqual(text_of(self.ans("sprintf('%d', 3.5)")), '3.500000e+00')    # as in MATLAB
        self.assertEqual(text_of(self.ans("sprintf('%s', 65)")), 'A')
        for expr, text in (('num2str(pi)', '3.1416'), ('num2str(42)', '42'), ('num2str([1 2 3])', '1  2  3'),
                           ('num2str(pi, 8)', '3.1415927'), ('num2str(-0.5)', '-0.5'),
                           ("num2str(pi, '%10.5f')", '   3.14159'), ('num2str(NaN)', 'NaN'),
                           ("num2str('abc')", 'abc'), ('num2str(123456.789)', '123456.789')):
            self.assertEqual(text_of(self.ans(expr)), text, expr)

    def test_write_and_read_a_file(self):
        self.assertValue("fid = fopen('data.txt', 'w');", 3, var='fid')
        self.assertEqual(self.out("fprintf(fid, '%f %f\\n', [1 2 3; 10 20 30]);"), '')
        self.assertValue('fclose(fid)', 0)
        with open(os.path.join(self.tmp, 'data.txt')) as f:
            self.assertEqual(f.read(), '1.000000 10.000000\n2.000000 20.000000\n3.000000 30.000000\n')
        self.assertValue("fid = fopen('data.txt'); B = fscanf(fid, '%f %f', [2 Inf])'; fclose(fid);",
                         [[1, 10], [2, 20], [3, 30]], var='B')
        self.out("fid = fopen('data.txt', 'r');")
        self.assertValue('[v, count] = fscanf(fid, "%f", 4);', [[1], [10], [2], [20]], var='v')
        self.assertValue('count', 4, var='count')
        self.assertValue('rest = fscanf(fid, "%f");', [[3], [30]], var='rest')   # continues where it stopped
        self.out('fclose(fid);')
        self.out("fid = fopen('log.txt', 'a'); fprintf(fid, 'one\\n'); fclose(fid);")
        self.out("fid = fopen('log.txt', 'a'); fprintf(fid, 'two\\n'); fclose(fid);")
        self.assertEqual(text_of(self.ans("fid = fopen('log.txt'); w = fscanf(fid, '%s'); fclose(fid);", var='w')),
                         'onetwo')
        self.assertValue("fid = fopen('csv.txt', 'w'); fprintf(fid, '1,2,3\\n4,5,6\\n'); fclose(fid); "
                         "fid = fopen('csv.txt'); M = fscanf(fid, '%d,%d,%d\\n', [3 Inf])'; fclose(fid);",
                         [[1, 2, 3], [4, 5, 6]], var='M')

    def test_file_identifiers(self):
        self.assertValue("[f, msg] = fopen('missing.txt');", -1, var='f')
        self.assertIn('No such file', text_of(self.eng.variables['msg']))
        self.assertValue("a = fopen('a.txt', 'w'); b = fopen('b.txt', 'w'); [a b]", [3, 4])
        self.assertValue("fopen('all')", [3, 4])
        self.assertValue("fclose(a); c = fopen('c.txt', 'w')", 3, var='c')       # free identifiers are reused
        self.assertError("fprintf(9, 'x')", 'Invalid file identifier')
        self.assertError('fclose(9)', 'Invalid file identifier')
        self.assertError("fscanf(b, '%f')", 'not open for reading')
        self.assertError("r = fopen('a.txt'); fprintf(r, 'x')", 'not open for writing')
        self.assertError("fopen('x.txt', 'q')", 'Invalid permission')
        self.assertValue("fclose('all'); fopen('all')", np.zeros((1, 0)))

    def test_current_folder(self):
        self.assertIn(os.path.realpath(self.tmp), os.path.realpath(self.out('pwd').strip()))
        os.mkdir(os.path.join(self.tmp, 'My Data'))
        self.out("cd 'My Data'")
        self.assertEqual(os.path.realpath(os.getcwd()), os.path.realpath(os.path.join(self.tmp, 'My Data')))
        self.out("fid = fopen('x.txt', 'w'); fclose(fid);")
        self.assertTrue(os.path.exists(os.path.join(self.tmp, 'My Data', 'x.txt')))
        self.out('cd ..')
        self.assertEqual(os.path.realpath(os.getcwd()), os.path.realpath(self.tmp))
        self.assertError('cd nowhere', 'Unable to change current folder')


class TestFilesAndFolders(EngineCase):
    """ls, mkdir, rmdir, isfile, isfolder, copyfile, movefile and delete, in a temporary folder."""

    def setUp(self):
        super().setUp()
        self.folder = os.getcwd()
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self.folder)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make(self, *names):
        for name in names:
            path = os.path.join(self.tmp, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write(name)

    def here(self, name=''):
        return os.path.join(self.tmp, name)

    def test_mkdir_and_isfolder(self):
        self.assertEqual(self.out('mkdir results'), '')
        self.assertTrue(os.path.isdir(self.here('results')))
        self.out("mkdir 'My Data'")                                  # quotes keep the spaces
        self.assertTrue(os.path.isdir(self.here('My Data')))
        self.out("mkdir('results', 'run1'); mkdir a/b/c")            # parent folder; missing folders are made
        self.assertTrue(os.path.isdir(self.here('results/run1')) and os.path.isdir(self.here('a/b/c')))
        self.assertIn('Warning: Directory already exists.', self.out('mkdir results'))
        self.assertIsNone(self.eng.last_error)
        self.assertValue("[s, msg] = mkdir('results');", True, var='s')
        self.assertEqual(text_of(self.eng.variables['msg']), 'Directory already exists.')
        self.make('f.txt')
        self.assertError('mkdir f.txt', 'a file of that name exists')
        self.assertValue("s = mkdir('f.txt');", False, var='s')
        self.assertEqual(self.eng.variables['s'].dtype, bool)
        for line, value in (("isfolder('results')", True), ('isfolder results', True), ("isfolder('f.txt')", False),
                            ("isfolder('none')", False), ("isfolder('..')", True)):
            self.assertValue(line, value)
        self.assertError('isfolder(3)', 'must be text')

    def test_isfile(self):
        self.make('data.txt', 'sub/x.m')
        for line, value in (("isfile('data.txt')", True), ('isfile data.txt', True), ("isfile('sub/x.m')", True),
                            ("isfile('sub')", False), ("isfile('none.txt')", False), ("isfile('')", False)):
            self.assertValue(line, value)
        self.assertEqual(self.ans("isfile('data.txt')").dtype, bool)

    def test_ls(self):
        self.make('b.txt', 'a.txt', 'Notes.m', '.hidden', 'sub/inner.txt')
        self.assertEqual(self.out('ls'), '  a.txt    b.txt    Notes.m  sub\n')   # sorted, hidden names left out
        self.assertEqual(self.out('ls sub'), '  inner.txt\n')
        self.assertEqual(self.out('ls *.txt'), '  a.txt  b.txt\n')
        self.assertEqual(self.out("ls('sub/*.txt')"), '  sub/inner.txt\n')
        self.assertEqual(text_of(self.ans("x = ls('*.txt');", var='x')), 'a.txtb.txt')
        self.assertEqual(self.eng.variables['x'].shape, (2, 5))           # one name per row
        self.assertEqual(self.ans("y = ls('*.txt');", var='y').dtype, mc.CHAR)
        self.assertError('ls missing*', 'not found')
        os.mkdir(self.here('empty'))
        self.assertEqual(self.out('ls empty'), '')
        self.assertEqual(self.ans("z = ls('empty');", var='z').size, 0)
        names = [f'file{k:02d}.txt' for k in range(12)]
        self.make(*[os.path.join('many', n) for n in names])
        rows = self.out('ls many').splitlines()                            # columns that fit in LS_WIDTH
        self.assertGreater(len(rows), 1)
        self.assertTrue(all(len(r) <= mc.LS_WIDTH + 2 for r in rows))
        self.assertEqual(rows[0].split()[:2], ['file00.txt', f'file{len(rows):02d}.txt'])   # down the columns

    def test_ls_is_a_variable_first(self):
        self.out('ls = 3')
        self.assertEqual(self.out('ls'), 'ls =\n    3\n')                    # the variable, not the function
        self.assertValue('ls + 1', 4)

    def test_rmdir(self):
        self.make('full/a.txt', 'full/sub/b.txt')
        os.mkdir(self.here('empty'))
        self.out('rmdir empty')
        self.assertFalse(os.path.exists(self.here('empty')))
        self.assertError('rmdir full', 'is not empty')
        self.assertTrue(os.path.isdir(self.here('full')))
        self.assertValue("[s, m] = rmdir('full');", False, var='s')
        self.assertIn('not empty', text_of(self.eng.variables['m']))
        self.assertError("rmdir('full', 'x')", "must be 's'")
        self.out('rmdir full s')                                          # with everything in it
        self.assertFalse(os.path.exists(self.here('full')))
        self.assertError('rmdir nowhere', 'not found')
        self.make('f.txt')
        self.assertError('rmdir f.txt', 'is not a folder')
        os.mkdir(self.here('cur'))
        self.out('cd cur')
        self.assertError("rmdir('../cur')", 'current folder')
        self.out('cd ..')
        self.assertValue("s = rmdir('cur')", True, var='s')

    def test_copyfile(self):
        self.make('a.txt', 'b.txt', 'c.m', 'src/x.txt', 'src/deep/y.txt')
        self.out('copyfile a.txt new.txt')
        with open(self.here('new.txt')) as f:
            self.assertEqual(f.read(), 'a.txt')
        self.assertTrue(os.path.isfile(self.here('a.txt')))
        os.mkdir(self.here('dest'))
        self.out("copyfile('a.txt', 'dest')")                             # into an existing folder
        self.assertTrue(os.path.isfile(self.here('dest/a.txt')))
        self.out('copyfile *.txt backup')                                 # several files: the folder is made
        self.assertEqual(sorted(os.listdir(self.here('backup'))), ['a.txt', 'b.txt', 'new.txt'])
        self.out('copyfile src copy')                                     # a folder: its contents, with subfolders
        self.assertTrue(os.path.isfile(self.here('copy/x.txt')) and os.path.isfile(self.here('copy/deep/y.txt')))
        self.out('copyfile b.txt a.txt')                                  # an existing file is replaced
        with open(self.here('a.txt')) as f:
            self.assertEqual(f.read(), 'b.txt')
        os.chdir(self.here('dest'))
        self.out("copyfile('../c.m')")                                    # into the current folder
        os.chdir(self.tmp)
        self.assertTrue(os.path.isfile(self.here('dest/c.m')))
        self.assertError('copyfile none.txt x.txt', 'No matching files were found')
        self.assertValue("[s, m] = copyfile('none.txt', 'x.txt');", False, var='s')
        self.assertEqual(text_of(self.eng.variables['m']), 'No matching files were found.')
        self.assertError('copyfile a.txt a.txt', 'onto itself')
        self.assertError('copyfile src src/inside', 'into itself')
        self.assertError('copyfile *.m a.txt', 'must be a folder')
        self.assertValue("s = copyfile('b.txt', 'd.txt')", True, var='s')

    def test_copyfile_f_replaces_read_only_files(self):
        self.make('a.txt', 'ro.txt')
        os.chmod(self.here('ro.txt'), 0o444)
        if os.access(self.here('ro.txt'), os.W_OK):
            self.skipTest('read-only files can be written (running as root?)')
        self.assertError('copyfile a.txt ro.txt', 'Permission denied')
        self.out("copyfile('a.txt', 'ro.txt', 'f')")
        with open(self.here('ro.txt')) as f:
            self.assertEqual(f.read(), 'a.txt')
        self.assertError("copyfile('a.txt', 'ro.txt', 'q')", "must be 'f'")

    def test_movefile(self):
        self.make('a.txt', 'b.txt', 'c.txt', 'old/x.txt')
        self.out('movefile a.txt renamed.txt')
        self.assertFalse(os.path.exists(self.here('a.txt')))
        self.assertTrue(os.path.isfile(self.here('renamed.txt')))
        self.out('movefile old new')                                      # a folder is renamed
        self.assertTrue(os.path.isfile(self.here('new/x.txt')) and not os.path.exists(self.here('old')))
        os.mkdir(self.here('box'))
        self.out("movefile('new', 'box')")                                # into an existing folder
        self.assertTrue(os.path.isfile(self.here('box/new/x.txt')))
        self.out("movefile b.txt 'My Files'")                             # a new name, with a space
        self.assertTrue(os.path.isfile(self.here('My Files')))
        self.out('movefile c.txt renamed.txt')                            # an existing file is replaced
        with open(self.here('renamed.txt')) as f:
            self.assertEqual(f.read(), 'c.txt')
        self.make('p.txt', 'q.txt')
        self.out('movefile *.txt moved')
        self.assertEqual(sorted(os.listdir(self.here('moved'))), ['p.txt', 'q.txt', 'renamed.txt'])
        self.assertError('movefile none.txt x', 'No matching files were found')
        self.assertError('movefile box box/inside', 'into itself')
        os.mkdir(self.here('other'))
        os.mkdir(self.here('box/other'))
        self.assertError('movefile other box', 'exists already')
        self.assertValue("[s, m] = movefile('none', 'x');", False, var='s')

    def test_delete(self):
        self.make('a.txt', 'b.txt', 'c.dat', 'd.dat', 'keep.m', 'sub/e.txt')
        self.out('delete a.txt')
        self.assertFalse(os.path.exists(self.here('a.txt')))
        self.out("delete('c.dat', 'd.dat')")
        self.assertFalse(os.path.exists(self.here('c.dat')) or os.path.exists(self.here('d.dat')))
        self.out('delete *.txt')                                          # the folder sub does not match *.txt
        self.assertFalse(os.path.exists(self.here('b.txt')))
        self.assertTrue(os.path.isfile(self.here('sub/e.txt')) and os.path.isfile(self.here('keep.m')))
        self.assertEqual(self.out("delete('none.txt')"), "Warning: File 'none.txt' not found.\n")
        self.assertIsNone(self.eng.last_error)
        self.assertIn('rmdir', self.out('delete sub'))                     # folders are not deleted
        self.assertTrue(os.path.isdir(self.here('sub')))
        self.out('delete *')                                              # files only
        self.assertEqual(os.listdir(self.here()), ['sub'])
        self.assertError('delete', 'Not enough input arguments')

    def test_command_syntax_with_file_names(self):
        self.make('data.txt')
        self.out("copyfile data.txt 'it''s here.txt'")                    # '' is a quote inside quotes
        self.assertTrue(os.path.isfile(self.here("it's here.txt")))
        self.out('copyfile "data.txt" "two words.txt"')
        self.assertTrue(os.path.isfile(self.here('two words.txt')))
        self.out('mkdir tilde_test; isfolder tilde_test')                 # ; ends the command, as usual
        self.assertValue('isfolder tilde_test', True)
        self.assertError('isfile + 1', 'Not enough input arguments')       # an expression, not command syntax
        self.assertError("mkdir 'unclosed", '')
        self.assertFalse(os.path.exists(self.here("'unclosed")) or os.path.exists(self.here('unclosed')))
        home = os.path.expanduser('~')
        self.assertValue('isfolder ~', os.path.isdir(home))


HAVE_MATPLOTLIB = importlib.util.find_spec('matplotlib') is not None


@unittest.skipUnless(HAVE_MATPLOTLIB, 'matplotlib is not installed')
class GraphicsCase(EngineCase):
    def setUp(self):
        super().setUp()
        self.eng.execute('close all')

    def axes(self):
        return self.eng.graphics.current.axes

    def lines(self):
        return self.axes().lines

    def render(self):
        """Draw the current figure off-screen: math text and legends must be drawable."""
        import io
        self.eng.graphics.current.figure.savefig(io.BytesIO(), format='png')


class TestPlot(GraphicsCase):
    def test_forms_of_plot(self):
        self.out('plot([4 5 6])')
        self.assertEqual(self.lines()[0].artist.get_xdata().tolist(), [1, 2, 3])
        self.out('plot([1 2; 3 4; 5 6])')                    # a line per column
        self.assertEqual([l.artist.get_ydata().tolist() for l in self.lines()], [[1, 3, 5], [2, 4, 6]])
        self.out('plot(1:2, [1 2; 3 4; 5 6])')               # x matches the rows' length
        self.assertEqual(len(self.lines()), 3)
        self.out('plot([1 2 3], [4 5 6], [1 2], [7 8])')
        self.assertEqual(len(self.lines()), 2)
        self.out('plot([1+2i 3+4i])')                          # complex: imaginary against real part
        self.assertEqual(self.lines()[0].artist.get_ydata().tolist(), [2, 4])
        self.assertIn('Imaginary parts', self.out('plot([1 2], [1i 2])'))
        self.out('plot(2, [1 2 3], \'o\')')                     # separate points
        self.assertEqual(len(self.lines()), 3)
        self.assertError('plot(1:3, 1:4)', 'Vectors must be the same length')
        self.assertError('plot([1 2], [3 4], [5 6])', 'single matrix Y or a list of pairs')
        self.assertError("plot(1:3, 'q-')", "Invalid LineSpec 'q-'")
        self.assertError("plot(1:3, 'LineColor', 'r')", 'Unrecognized property LineColor for class Line')
        self.assertError("plot('r')", 'Not enough input arguments')
        self.assertError("plot(1:3, 'LineWidth')", 'must come in pairs')

    def test_returned_handles_and_output(self):
        self.assertEqual(self.out('plot(1:3)'), '')            # nothing shown, ans unchanged
        self.assertNotIn('ans', self.eng.variables)
        h = self.ans('h = plot([1 2; 3 4])', var='h')
        self.assertEqual(h.shape, (2, 1))
        self.assertTrue(all(v % 1 == 0.5 for v in h.ravel()))  # never a figure number
        self.assertEqual(self.ans('get(gca, \'Children\')').ravel().tolist(), h.ravel()[::-1].tolist())

    def test_linespec_and_colors(self):
        self.assertEqual(mc.parse_linespec('r--o'), {'color': 'r', 'linestyle': '--', 'marker': 'o'})
        self.assertEqual(mc.parse_linespec('.-'), {'linestyle': '-', 'marker': '.'})
        self.assertEqual(mc.parse_linespec('k-.'), {'linestyle': '-.', 'color': 'k'})
        for bad in ('q', 'rr', '---', 'LineWidth', ''):
            self.assertIsNone(mc.parse_linespec(bad), bad)
        self.out("plot(1:3, 1:3, 'r--o', 1:3, 2:4, 'kd', 1:3, 3:5, ':')")
        red, black, third = self.lines()
        self.assertEqual((red.artist.get_linestyle(), red.artist.get_marker()), ('--', 'o'))
        self.assertEqual(self.ans(f"get({red.handle}, 'Color')").ravel().tolist(), [1, 0, 0])
        self.assertEqual((black.artist.get_linestyle(), black.artist.get_marker()), ('None', 'D'))
        import matplotlib
        import matplotlib.colors as mcolors
        cycle = [c['color'] for c in matplotlib.rcParams['axes.prop_cycle']]
        self.assertEqual(mcolors.to_hex(third.artist.get_color()), mcolors.to_hex(cycle[0]))  # matplotlib's order
        self.out('hold on; plot(1:3)')
        self.assertEqual(mcolors.to_hex(self.lines()[-1].artist.get_color()), mcolors.to_hex(cycle[1]))

    def test_hold(self):
        self.out("plot(1:3); title('First')")
        self.out('plot(3:-1:1)')                                 # replaces the plot and its title
        self.assertEqual(len(self.lines()), 1)
        self.assertEqual(text_of(self.ans("get(get(gca, 'Title'), 'String')")), '')
        self.out('hold on')
        self.assertTrue(self.ans('ishold').item())
        self.out('plot(1:3); plot(2:4)')
        self.assertEqual(len(self.lines()), 3)
        self.out('hold')                                         # toggles
        self.assertFalse(self.ans('ishold').item())
        self.assertEqual(text_of(self.ans("get(gca, 'NextPlot')")), 'replace')
        self.out('hold all')
        self.assertTrue(self.ans('ishold').item())
        self.out('hold off; plot(1)')
        self.assertEqual(len(self.lines()), 1)
        self.assertError('hold maybe', "Unknown option 'maybe'")


class TestLabelsAndLegend(GraphicsCase):
    def test_labels_and_tex(self):
        self.out("plot(1:3); h = xlabel('Time t_1 (s)');")
        label = self.axes().labels['xlabel']
        self.assertEqual(text_of(self.ans("get(h, 'String')")), 'Time t_1 (s)')
        self.assertIn('${}_{1}$', label.artist().get_text())
        self.out(r"ylabel('\alpha^{-2} and x_{max}'); title('Cost: $5', 'FontSize', 14)")
        self.assertIn(r'$\alpha$', self.axes().labels['ylabel'].artist().get_text())
        self.assertEqual(self.axes().labels['title'].artist().get_text(), 'Cost: $5')
        self.assertEqual(self.axes().labels['title'].artist().get_fontsize(), 14)
        self.out("xlabel('t_1', 'Interpreter', 'none')")
        self.assertEqual(label.artist().get_text(), 't_1')
        self.out(r"xlabel('$\sqrt{x}$', 'Interpreter', 'latex')")
        self.assertEqual(label.artist().get_text(), r'$\sqrt{x}$')
        self.assertEqual(mc.tex_to_mathtext(r'\notasymbol_1'), (r'\notasymbol${}_{1}$', True))
        self.assertEqual(mc.tex_to_mathtext('plain'), ('plain', False))
        self.render()
        self.assertError('xlabel(5)', 'The first input must be the text')
        self.assertError("title('x', 'Size', 3)", 'Unrecognized property Size for class Text')

    def test_legend(self):
        self.out("plot(1:3, 1:3, 1:3, 2:4, 1:3, 3:5)")
        self.out("legend('a', 'b')")
        legend = self.axes().legend
        self.assertEqual([label for _, label in legend.entries()], ['a', 'b'])     # the third is left out
        self.assertEqual([t.get_text() for t in legend.mpl_legend.get_texts()], ['a', 'b'])
        self.out('legend')                                                        # all lines
        self.assertEqual([label for _, label in legend.entries()], ['a', 'b', 'data3'])
        self.out("hold on; plot(1:3, 'DisplayName', 'new')")                       # added automatically
        self.assertEqual([label for _, label in self.axes().legend.entries()][-1], 'new')
        self.assertIn('Ignoring extra legend entries', self.out("legend('1', '2', '3', '4', '5')"))
        self.out("legend('x_1', 'Location', 'southwest', 'Orientation', 'horizontal')")
        self.assertEqual(text_of(self.ans("get(legend, 'Location')")), 'southwest')
        self.render()
        self.out('legend hide')
        self.assertFalse(self.axes().legend.mpl_legend.get_visible())
        self.out('legend boxoff; legend show')
        self.assertFalse(self.axes().legend.mpl_legend.get_frame_on())
        self.out('legend off')
        self.assertIsNone(self.axes().legend)
        self.assertIsNone(self.axes().ax.get_legend())
        h = self.ans('h = plot(1:3);', var='h')
        self.out(f"legend({h.item()}, 'only this')")
        self.assertEqual([label for _, label in self.axes().legend.entries()], ['only this'])
        self.assertError("legend('a', 'Location', 'up')", "Invalid value for Location property")


class TestSetAndGet(GraphicsCase):
    def test_set_and_get(self):
        h = self.ans('h = plot(1:3);', var='h').item()
        self.out(f"set({h}, 'color', [0 0.5 0], 'LINEW', 3, 'LineStyle', ':', 'Marker', 'square')")
        line = self.lines()[0]
        self.assertEqual(line.artist.get_linewidth(), 3)
        self.assertEqual(self.ans(f"get({h}, 'Color')").ravel().tolist(), [0, 0.5, 0])
        self.assertEqual(text_of(self.ans(f"get({h}, 'Marker')")), 's')
        self.assertEqual(text_of(self.ans(f"get({h}, 'LineStyle')")), ':')
        self.out(f"set({h}, 'Color', 'tab:orange', 'MarkerFaceColor', 'none', 'XData', [1 2 4])")
        self.assertEqual(text_of(self.ans(f"get({h}, 'MarkerFaceColor')")), 'none')
        self.assertEqual(self.ans(f"get({h}, 'XData')").tolist(), [[1, 2, 4]])
        self.assertError(f"set({h}, 'Line', 1)", 'Ambiguous property name Line for class Line')
        self.assertError(f"set({h}, 'LineColor', 'r')", 'Unrecognized property LineColor for class Line')
        self.assertError(f"set({h}, 'LineWidth', -1)", 'Invalid value for LineWidth property')
        self.assertError(f"set({h}, 'Color', 'notacolor')", 'Invalid value for Color property')
        self.assertError(f"set({h}, 'Type', 'x')", 'cannot be changed')
        self.assertError("set(99.5, 'Color', 'r')", 'Invalid or deleted graphics object')
        self.assertError("set('abc', 'Color', 'r')", 'must be graphics handles')
        self.assertError("x = set(gca, 'Box', 'off')", 'Too many output arguments')
        listing = self.out(f'get({h})')
        self.assertIn("LineStyle: ':'", listing)
        self.assertIn('MarkerSize', self.out(f'set({h})'))
        self.assertError('s = get(gca)', 'only lists the properties')

    def test_delete_graphics_objects(self):
        h = self.ans("h = plot([1 2; 3 4]); legend('a', 'b');", var='h').ravel().tolist()
        self.out('delete(h(1))')
        self.assertEqual([line.handle for line in self.lines()], [h[1]])
        self.assertEqual(self.ans("get(gca, 'Children')").ravel().tolist(), [h[1]])
        self.assertError('delete(h(1))', 'Invalid or deleted graphics object')
        self.render()                                                     # the legend follows the lines
        self.out('delete(legend)')
        self.assertIsNone(self.axes().legend)
        self.assertError("delete(get(gca, 'Title'))", 'cannot be deleted')
        self.out('delete(gca)')
        self.assertIsNone(self.eng.graphics.current.axes)
        self.out('plot(1:3); delete([gcf gca])')                          # the axes goes with its figure
        self.assertIsNone(self.eng.last_error)
        self.assertEqual(self.eng.graphics.figures, {})

    def test_axes_properties(self):
        self.out('plot(1:10, (1:10).^2)')
        self.out("set(gca, 'XLim', [0 5], 'YScale', 'log', 'XGrid', 'on', 'FontSize', 12, 'Box', 'off')")
        ax = self.axes().ax
        self.assertEqual(self.ans("get(gca, 'XLim')").tolist(), [[0, 5]])
        self.assertEqual(text_of(self.ans("get(gca, 'XLimMode')")), 'manual')
        self.assertEqual(ax.get_yscale(), 'log')
        self.assertFalse(ax.spines['top'].get_visible())
        self.assertEqual(text_of(self.ans("get(gca, 'XGrid')")), 'on')
        self.assertEqual(self.axes().labels['title'].artist().get_fontsize(), 12 * mc.LABEL_SIZE_FACTOR)
        self.out("set(gca, 'XDir', 'reverse', 'XLimMode', 'auto')")
        self.assertTrue(ax.xaxis.get_inverted())
        self.assertEqual(self.ans("get(gca, 'XLim')").tolist(), [[0, 10]])           # still [min max]
        self.out("set(gca, 'XTick', [0 2.5 10])")
        self.assertEqual(self.ans("get(gca, 'XTick')").tolist(), [[0, 2.5, 10]])
        self.assertEqual(self.ans("get(get(gca, 'XLabel'), 'Type')").tobytes(), mc.text_to_chars('text').tobytes())
        self.assertError("set(gca, 'XLim', [5 1])", 'min < max')
        self.assertError("set(gca, 'XScale', 'loglog')", "it must be 'linear' or 'log'")
        self.render()

    def test_figures(self):
        self.assertEqual(self.ans('f = figure', var='f').item(), 1)
        self.assertEqual(self.ans('g = figure(7);', var='g').item(), 7)
        self.assertEqual(self.ans('gcf').item(), 7)
        self.assertEqual(self.ans('n = figure;', var='n').item(), 2)  # the lowest free number
        self.out('figure(1)')
        self.assertEqual(self.ans('gcf').item(), 1)
        self.out("figure(3, 'Name', 'Results', 'NumberTitle', 'off')")
        figure = self.eng.graphics.figures[3]
        self.assertEqual(figure.title(), 'Results')
        self.out("set(gcf, 'NumberTitle', 'on')")
        self.assertEqual(figure.title(), 'Figure 3: Results')
        self.assertEqual(self.ans("get(gcf, 'Number')").item(), 3)
        self.assertEqual(self.ans("get(gcf, 'Position')").shape, (1, 4))
        self.assertEqual(self.ans("get(gca, 'Parent')").item(), 3)
        self.out('close')                                          # the current one: 3
        self.assertNotIn(3, self.eng.graphics.figures)
        self.assertEqual(self.ans('gcf').item(), 1)                # the one active before it
        self.out('close([1 2])')
        self.assertEqual(sorted(self.eng.graphics.figures), [7])
        self.assertError('figure(0)', 'whole number of 1 or more')
        self.assertError('close(1.5)', 'Invalid or deleted graphics object')
        self.out('plot(1:3); a = gca; clf')
        self.assertIsNone(self.eng.graphics.current.axes)
        self.assertError(f"get({self.eng.variables['a'].item()}, 'Type')", 'Invalid or deleted')
        self.out("plot(1:3); title('kept'); cla")
        self.assertEqual(self.lines(), [])
        self.assertEqual(text_of(self.ans("get(get(gca, 'Title'), 'String')")), 'kept')
        self.out('close all')
        self.assertEqual(self.eng.graphics.figures, {})

    def test_axis_and_limits(self):
        self.out('plot(0:0.1:2.05, sin(0:0.1:2.05))')
        self.assertEqual(self.ans('axis').shape, (1, 4))
        self.out('axis([0 3 -2 2])')
        self.assertEqual(self.ans('axis').tolist(), [[0, 3, -2, 2]])
        self.out('xlim([-Inf 1])')
        self.assertEqual(self.ans('xlim').tolist(), [[0, 1]])
        self.assertEqual(text_of(self.ans("xlim('mode')")), 'manual')
        self.out('xlim auto; axis tight')
        self.assertAlmostEqual(self.ans('xlim')[0, 1], 2.0)
        self.out('axis equal; axis ij; axis off')
        ax = self.axes().ax
        self.assertEqual(ax.get_aspect(), 1.0)
        self.assertTrue(ax.yaxis.get_inverted())
        self.assertFalse(ax.axison)
        self.out('axis on xy normal; axis square')
        self.assertFalse(ax.yaxis.get_inverted())
        self.assertError('axis([1 2 3])', '[xmin xmax ymin ymax]')
        self.assertError('axis sideways', "Unknown option 'sideways'")
        self.out('grid on')
        self.assertTrue(self.axes().grid['x'] and self.axes().grid['y'])
        self.out('grid minor')
        self.assertTrue(self.axes().grid['xminor'])
        self.out('grid off; box off')
        self.assertFalse(self.axes().grid['x'] or self.axes().grid['xminor'] or self.axes().box)
        self.render()

    def test_command_syntax(self):
        self.out('hold on')
        self.assertTrue(self.ans('ishold').item())
        self.out('axis equal tight')
        self.assertValue('hold = 3', 3, var='hold')          # assignment, not a command
        self.out('hold on')                                     # now the variable is shown, not a command
        self.assertNotIn('Error', self.out('close all'))
        self.assertEqual(self.eng.graphics.figures, {})
        self.assertIsNone(self.eng._command_syntax('disp hello'))  # only graphics commands take words

    def test_without_matplotlib(self):
        saved, saved_modules = dict(mc._MPL), {k: v for k, v in sys.modules.items() if k.startswith('matplotlib')}
        mc._MPL.clear()
        sys.modules['matplotlib'] = None   # makes "import matplotlib" fail
        try:
            eng = mc.MatlabEngine()
            self.assertIn('pip install matplotlib', ''.join(eng.execute('plot(1:3)')))
            self.assertIn('plot - Plot lines.', ''.join(eng.execute('help plot')))
        finally:
            del sys.modules['matplotlib']
            sys.modules.update(saved_modules)
            mc._MPL.update(saved)


@unittest.skipUnless(HAVE_MATPLOTLIB, 'matplotlib is not installed')
class TestFigureWindows(unittest.TestCase):
    """Figures shown in their own windows by the GUI."""

    def setUp(self):
        self.root = make_root(visible=True)
        self.app = mc.MatlabConsoleApp(self.root)
        self.root.update()

    def tearDown(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass
        gc.collect()

    def run_line(self, line):
        self.app.run_typed(line)
        self.root.update()
        return self.app.output_area.get('1.0', 'end-1c')

    def test_figure_windows(self):
        self.run_line('x = 0:0.1:6; plot(x, sin(x), x, cos(x)); legend; grid on')
        window = self.app.figure_windows[1]
        self.assertEqual(window.title(), 'Figure 1')
        self.assertTrue(window.winfo_ismapped())
        self.assertIsNotNone(window.toolbar)
        self.run_line('figure')
        self.assertEqual(sorted(self.app.figure_windows), [1, 2])
        window.event_generate('<FocusIn>')                       # clicking figure 1 makes it current
        self.assertEqual(self.app.engine.graphics.current.number, 1)
        self.run_line("set(gcf, 'Name', 'Waves')")
        self.assertEqual(window.title(), 'Figure 1: Waves')
        self.run_line("set(gcf, 'Position', [50 100 500 300])")
        position = self.app.engine.graphics.figures[1].get_position()
        self.assertEqual(position[2:], [500, 300])
        self.run_line('drawnow')
        window.close()                                            # the close button of the window
        self.assertNotIn(1, self.app.engine.graphics.figures)
        self.assertEqual(sorted(self.app.figure_windows), [2])
        self.assertEqual(self.app.engine.graphics.current.number, 2)
        self.assertFalse(window.winfo_exists())
        self.run_line('close all')
        self.assertEqual(self.app.figure_windows, {})
        content = self.run_line('plot(1:3)')
        self.assertNotIn('Error', content)
        self.assertEqual(sorted(self.app.figure_windows), [1])

    def test_exit_with_open_figures(self):
        self.run_line('plot(1:3); figure; plot(3:-1:1)')
        self.app.run_typed('exit')
        self.assertTrue(self.app.closed)


class TestDisplay(EngineCase):
    def test_readme_examples(self):
        self.out('A = [1 3 2.4 6 4.4 3.2 1];')
        self.assertEqual(self.out('A * 2'), 'ans =\n    2.0   6.0   4.8   12.0   8.8   6.4   2.0\n')
        self.assertEqual(self.out('B = [1 2; 3 4]'), 'B =\n    1   2\n    3   4\n')

    def test_adaptive_decimals(self):
        self.assertEqual(self.out('2.4 * 3'), 'ans =\n    7.2\n')
        self.assertEqual(self.out('1/3'), 'ans =\n    0.3333\n')
        self.assertEqual(self.out('[1 2.5]'), 'ans =\n    1.0   2.5\n')
        self.assertEqual(self.out('-1e-9 * 0'), 'ans =\n    0\n')

    def test_format_long(self):
        self.out('format long')
        self.assertEqual(self.out('pi'), 'ans =\n    3.141592653589793\n')
        self.assertEqual(self.out('0.1 + 0.2'), 'ans =\n    0.3\n')
        self.assertEqual(self.out('1e9 + 0.1'), 'ans =\n    1000000000.1\n')
        self.out('format short')
        self.assertEqual(self.out('pi'), 'ans =\n    3.1416\n')
        self.out('format long')
        self.out('format')                   # plain 'format' restores the default
        self.assertEqual(self.eng.format_mode, 'short')

    def test_small_large_and_special_values(self):
        self.assertEqual(self.out('0.00001'), 'ans =\n    1e-05\n')      # used to print 0
        self.assertEqual(self.out('sin(pi)'), 'ans =\n    1.2246e-16\n')
        self.assertEqual(self.out('123456789012'), 'ans =\n    1.2346e+11\n')
        self.assertEqual(self.out('[1 1e-6]'), 'ans =\n    1.0000   0.0000\n')
        self.assertEqual(self.out('[NaN -Inf 1.5]'), 'ans =\n    NaN   -Inf   1.5\n')

    def test_complex_and_empty(self):
        self.assertEqual(self.out('3 - 4i'), 'ans =\n    3 - 4i\n')
        self.assertEqual(self.out('(-8)^(1/3)'), 'ans =\n    1.0000 + 1.7321i\n')
        self.assertEqual(self.out('[]'), 'ans =\n    []\n')
        self.assertEqual(self.out('1:0'), 'ans =\n    1x0 empty double matrix\n')

    def test_large_results_are_summarised(self):
        self.assertIn('too large to display', self.out('x = 1:20000'))
        self.assertEqual(self.ans('x;', var='x').shape, (1, 20000))


class TestCommands(EngineCase):
    def test_clear_and_whos(self):
        self.out('a = 1; b = [1 2; 3 4]; c = 1 + 2i;')
        whos = self.out('whos')
        self.assertIn('b          2x2        double', whos)
        self.assertIn('c          1x1        double (complex)', whos)
        self.out('clear a b')
        self.assertEqual(sorted(self.eng.variables), ['c'])
        self.out('clear all')
        self.assertEqual(self.eng.variables, {})
        self.assertIn('Workspace is empty', self.out('whos'))

    def test_clc(self):
        self.assertIs(self.eng.execute('clc')[0], mc.CLEAR_SCREEN)

    def test_version(self):
        self.assertEqual(self.out('version'), f'{mc.APP_NAME} {mc.APP_VERSION}\n')
        self.assertEqual(self.out('version;'), '')
        self.assertValue('version = 2', 2, var='version')     # still usable as a variable name

    def test_exit(self):
        self.assertIs(self.eng.execute('exit')[-1], mc.EXIT_APP)
        self.assertIs(self.eng.execute('quit;')[-1], mc.EXIT_APP)
        out = self.eng.execute('a = 1, exit, b = 2')        # nothing runs after exit
        self.assertEqual(out[-1:], [mc.EXIT_APP])
        self.assertNotIn('b', self.eng.variables)
        self.assertValue('exit = 3', 3, var='exit')          # still usable as a variable name

    def test_showing_a_variable_keeps_ans(self):
        self.out('7; x = 4;')
        self.assertEqual(self.out('x'), 'x =\n    4\n')
        self.assertValue('ans', 7)

    def test_command_words_can_be_variables(self):
        self.assertValue('clc = 4', 4, var='clc')
        self.assertValue('format = 2', 2, var='format')
        self.assertError('format compact', "Unsupported format 'compact'")


class TestHelp(EngineCase):
    def all_documented(self):
        """Every command and function object, with the names it is registered under."""
        entries = list(mc.COMMANDS.items()) + list(mc.FUNCTIONS.items())
        return [(name, entry[0]) for name, entry in entries]

    def test_every_command_and_function_has_help(self):
        for name, func in self.all_documented():
            with self.subTest(name=name):
                text = mc.help_text(func)
                h1 = text.splitlines()[0]
                self.assertRegex(h1, r'^\S+ - [A-Z].*\.$', 'first line must be "name - Summary."')
                self.assertIn(h1.split(' - ')[0], func.matlab_names)
                self.assertGreater(len(text.splitlines()), 2, 'help needs more than the summary line')
                self.assertNotRegex(text, r'\{[A-Z_]+\}', 'unfilled {PARAMETER} field')
                self.assertIn(func.help_category, mc.HELP_CATEGORIES)

    def test_overview_lists_everything(self):
        text = self.out('help')
        self.assertTrue(text.startswith(f'  {mc.APP_NAME} {mc.APP_VERSION} - '), text)
        for name, func in self.all_documented():
            self.assertRegex(text, rf'\n    [^\n]*\b{re.escape(name)}\b[^\n]* - {re.escape(mc.help_summary(func))}\n')
        for ops, meaning in mc.OPERATOR_SUMMARY:
            self.assertRegex(text, re.escape(ops) + ' +' + re.escape(meaning))
        for category in mc.HELP_CATEGORIES:
            self.assertIn(f'\n  {category}\n', text)

    def test_help_fits_the_window(self):
        topics = ['', 'ops'] + [name for name, _ in self.all_documented()]
        for topic in topics:
            for line in self.out(f'help {topic}').splitlines():
                self.assertLessEqual(len(line), 76, f'help {topic}: {line!r}')

    def test_help_for_a_name(self):
        expected = '\n'.join(('  ' + line).rstrip() for line in mc.help_text(mc.fn_round).splitlines()) + '\n'
        self.assertEqual(self.out('help round'), expected)
        self.assertEqual(self.out('help ROUND'), expected)          # not case-sensitive
        self.assertIn('away from zero', expected)
        self.assertIn('the same as A.^B', self.out('help power'))
        self.assertIn('the same as A.*B', self.out('help times'))
        self.assertEqual(self.out('help quit'), self.out('help exit'))
        self.assertEqual(self.out('help inf'), self.out('help Inf'))
        self.assertIn(f'shows up to {mc.SHORT_DECIMALS} decimals', self.out('help format'))
        self.assertIn(f'{mc.APP_NAME} {mc.APP_VERSION}', self.out('help version'))
        self.assertIn('help NAME', self.out('help help'))
        two = self.out('help sin cos')
        self.assertIn('sin - Sine', two)
        self.assertIn('cos - Cosine', two)

    def test_help_for_operators(self):
        ops = self.out('help ops')
        self.assertIn('ops - Operators and special characters.', ops)
        for topic in ('.*', '+', '\\', "'", '==', ':', 'mtimes', 'operators'):
            self.assertEqual(self.out(f'help {topic}'), ops, topic)

    def test_help_for_unknown_names_and_variables(self):
        self.assertIn('foo not found', self.out('help foo'))
        self.out('x = 1;')
        self.assertIn('x is a variable', self.out('help x'))
        self.assertValue('help = 3', 3, var='help')                  # still usable as a variable name
        self.assertValue('format =2', 2, var='format')
        self.assertIn('ops - Operators', self.out('help ='))        # ...while 'help =' explains '='


def make_root(visible=False):
    try:
        root = tk.Tk(className=mc.APP_NAME)
    except tk.TclError as err:
        raise unittest.SkipTest(f'no display: {err}')
    if not visible:
        root.withdraw()
    return root


class TestGui(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = make_root(visible=True)  # the geometry is only reported for a mapped window
        cls.app = mc.MatlabConsoleApp(cls.root)
        cls.root.update()
        cls.text = cls.app.output_area
        cls.banner = cls.text.get('1.0', '1.end')  # captured before any test runs 'clc'

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def type_and_run(self, cmd):
        self.text.insert('input_start', cmd)
        self.app.process_command()
        return self.text.get('1.0', 'end-1c')

    def test_parameters_applied(self):
        self.assertEqual(self.root.title(), mc.APP_NAME)
        self.assertRegex(mc.APP_VERSION, r'^\d+\.\d+\.\d+$')
        self.assertEqual(mc.__version__, mc.APP_VERSION)
        self.assertTrue(self.banner.startswith(f'{mc.APP_NAME} {mc.APP_VERSION} '), self.banner)
        self.assertEqual(self.root.geometry().split('+')[0], f'{mc.WINDOW_WIDTH}x{mc.WINDOW_HEIGHT}')
        family = tk.font.Font(root=self.root, font=self.text.cget('font')).actual('family')
        self.assertTrue(tk.font.Font(root=self.root, family=family).metrics('fixed'), family)

    def test_read_only_history(self):
        before = self.text.get('1.0', 'input_start')
        self.text.insert('1.0', 'X')                         # e.g. typing with the cursor in old output
        self.text.delete('input_start - 1c')                 # Backspace / Ctrl+H at the prompt
        self.text.delete('1.0', 'end')                       # select-all + Delete / Cut
        self.text.replace('1.0', '1.5', 'Y')
        self.assertEqual(self.text.get('1.0', 'input_start'), before)
        self.text.insert('input_start', 'abc')               # the input line stays editable
        self.text.delete('input_start', 'end-1c')
        self.assertEqual(self.text.get('input_start', 'end-1c'), '')

    def test_execute_history_and_clc(self):
        content = self.type_and_run('q = 6 ./ 4')
        self.assertTrue(content.endswith('q =\n    1.5\n\n' + mc.PROMPT))
        self.app.history_up(None)
        self.assertEqual(self.text.get('input_start', 'end-1c'), 'q = 6 ./ 4')
        self.app.history_down(None)
        self.assertEqual(self.text.get('input_start', 'end-1c'), '')
        self.assertEqual(self.type_and_run('clc'), mc.PROMPT)

    def test_keypad_enter_is_bound(self):
        self.assertTrue(self.text.bind('<KP_Enter>'))


class TestEventLoop(unittest.TestCase):
    """Runs the real mainloop(), where tkinter re-raises errors from Tcl-called Python code.

    Regression test for 0.1.1, which crashed on the first key press.
    """

    def test_typing_while_mainloop_runs(self):
        root = make_root(visible=True)
        app = mc.MatlabConsoleApp(root)
        text = app.output_area

        class Key:  # what the <Key> binding receives for a plain key press
            state, keysym = 0, 'x'

        def user_session():
            app.on_key(Key())                                # our key handler (checks the selection)
            root.tk.call('tk::TextInsert', text._w, '1+1')  # Tk's own code for typed characters
            app.on_paste(None)
            text.tag_add('sel', '1.0', 'end-1c')             # select everything, then try to delete it
            root.tk.call('catch', f'{text._w} delete sel.first sel.last')
            text.tag_remove('sel', '1.0', 'end')
            root.tk.call('catch', f'{text._w} delete sel.first sel.last')  # no selection: Tk error, caught
            app.process_command()
            root.after(50, root.quit)

        root.after(50, user_session)
        root.after(10000, root.quit)  # safety net
        try:
            root.mainloop()  # would raise here if an error had been stored during the session
            content = text.get('1.0', 'end-1c')
            self.assertTrue(content.startswith(f'{mc.APP_NAME} {mc.APP_VERSION} '), content)
            self.assertIn('>> 1+1\nans =\n    2\n', content)
        finally:
            root.destroy()

    def test_exit_closes_the_window(self):
        root = make_root(visible=True)
        app = mc.MatlabConsoleApp(root)

        def type_exit():
            app.output_area.insert('input_start', 'exit')
            app.process_command()

        root.after(50, type_exit)
        timer = root.after(10000, root.quit)  # safety net: only fires if 'exit' did nothing
        root.mainloop()
        try:
            root.after_cancel(timer)
            root.destroy()
            self.fail("'exit' did not close the window")
        except tk.TclError:
            pass  # the application was destroyed, as expected


class InstallationCase(unittest.TestCase):
    """Base for the installation tests: every folder they write to is a temporary one."""

    PATCHED = ('CONFIG_FILE', 'MAC_SYSTEM_DIR', 'MAC_USER_DIR', 'LINUX_SYSTEM_DIRS', 'LINUX_USER_DIRS', '_can_write')

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.saved = {name: getattr(mc, name) for name in self.PATCHED}
        self.saved_xdg = os.environ.pop('XDG_DATA_HOME', None)
        t = self.tmp
        mc.CONFIG_FILE = os.path.join(t, 'home', '.matlab_calc.json')
        mc.MAC_SYSTEM_DIR, mc.MAC_USER_DIR = os.path.join(t, 'Applications'), os.path.join(t, 'home', 'Applications')
        mc.LINUX_SYSTEM_DIRS = (os.path.join(t, 'opt'), os.path.join(t, 'usr', 'applications'))
        mc.LINUX_USER_DIRS = (os.path.join(t, 'home', 'share'), os.path.join(t, 'home', 'share', 'applications'))
        os.makedirs(os.path.join(t, 'home'))

    def tearDown(self):
        for name, value in self.saved.items():
            setattr(mc, name, value)
        if self.saved_xdg is not None:
            os.environ['XDG_DATA_HOME'] = self.saved_xdg
        shutil.rmtree(self.tmp, ignore_errors=True)

    def config(self):
        with open(mc.CONFIG_FILE, encoding='utf-8') as f:
            return json.load(f)


class TestInstallation(InstallationCase):
    """Installing, updating from a local file and removing."""

    def test_linux_install_update_and_uninstall(self):
        ok, message = mc.install_app(False, 'linux')
        self.assertTrue(ok, message)
        target = mc.install_target(False, 'linux')
        self.assertTrue(os.path.isfile(target['script']))
        with open(target['launcher'], encoding='utf-8') as f:
            entry = f.read()
        self.assertIn(f'Name={mc.APP_NAME}\n', entry)
        self.assertIn(mc._desktop_exec([sys.executable, target['script']]), entry)
        if os.path.isfile(ICON):
            self.assertRegex(entry, r'\nIcon=/.+\.png\n')
        if shutil.which('desktop-file-validate'):
            result = subprocess.run(['desktop-file-validate', target['launcher']], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        installed = self.config()['installed']
        self.assertEqual((installed['version'], installed['systemwide']), (mc.APP_VERSION, False))
        self.assertTrue(self.config()['first_start_done'])
        self.assertTrue(mc.install_app(False, 'linux')[0])                     # installing again updates it
        ok, message = mc.uninstall_app('linux')
        self.assertTrue(ok, message)
        self.assertFalse(os.path.exists(target['launcher']) or os.path.exists(target['files']))
        self.assertNotIn('installed', self.config())
        self.assertIn('not installed', mc.uninstall_app('linux')[1])

    def test_systemwide_replaces_personal_copy(self):
        mc.install_app(False, 'linux')
        self.assertTrue(mc.install_app(True, 'linux')[0])
        self.assertTrue(os.path.exists(mc.install_target(True, 'linux')['launcher']))
        self.assertFalse(os.path.exists(mc.install_target(False, 'linux')['launcher']))  # not listed twice
        self.assertTrue(self.config()['installed']['systemwide'])

    def test_systemwide_without_rights(self):
        mc._can_write = lambda path: False
        ok, message = mc.install_app(True, 'linux')
        self.assertFalse(ok)
        self.assertIn('administrator rights', message)
        self.assertIn('sudo ', message)
        self.assertIn('--install-systemwide', message)

    def test_mac_app_bundle(self):
        ok, message = mc.install_app(True, 'darwin')
        self.assertTrue(ok, message)
        app = os.path.join(mc.MAC_SYSTEM_DIR, mc.APP_NAME + '.app')
        with open(os.path.join(app, 'Contents', 'Info.plist'), 'rb') as f:
            info = plistlib.load(f)
        self.assertEqual((info['CFBundleExecutable'], info['CFBundlePackageType']), (mc.APP_NAME, 'APPL'))
        self.assertEqual(info['CFBundleShortVersionString'], mc.APP_VERSION)
        launcher = os.path.join(app, 'Contents', 'MacOS', mc.APP_NAME)
        self.assertTrue(os.access(launcher, os.X_OK))
        with open(launcher, encoding='utf-8') as f:
            self.assertIn(f'"$(dirname "$0")/../Resources/{mc.APP_SLUG}.py"', f.read())
        self.assertTrue(os.path.isfile(os.path.join(app, 'Contents', 'Resources', mc.APP_SLUG + '.py')))
        if HAVE_PIL and os.path.isfile(ICON):
            self.assertEqual(info['CFBundleIconFile'], 'AppIcon')
            self.assertTrue(os.path.isfile(os.path.join(app, 'Contents', 'Resources', 'AppIcon.icns')))
        self.assertTrue(mc.uninstall_app('darwin')[0])
        self.assertFalse(os.path.exists(app))

    def test_unsupported_system(self):
        self.assertFalse(mc.install_app(False, 'win32')[0])
        self.assertIsNone(mc.startup_question({}, 'win32'))

    def test_desktop_exec_quoting(self):
        self.assertEqual(mc._desktop_exec(['/usr/bin/python3', '/opt/a/b.py']), '/usr/bin/python3 /opt/a/b.py')
        self.assertEqual(mc._desktop_exec(['/My Drive/x.py']), '"/My Drive/x.py"')
        self.assertEqual(mc._desktop_exec(['/a$b']), '"/a\\\\$b"')               # \$ then the file-level \\
        self.assertEqual(mc._desktop_exec(['/50%']), '/50%%')

    def test_startup_question(self):
        self.assertEqual(mc.startup_question({}, 'linux'), 'install')         # first start
        self.assertIsNone(mc.startup_question({'first_start_done': True}, 'linux'))
        mc.install_app(False, 'linux')
        config = self.config()
        self.assertIsNone(mc.startup_question(config, 'linux'))               # this version is installed
        config['installed']['version'] = '0.0.1'
        self.assertEqual(mc.startup_question(config, 'linux'), 'update')       # an older one is installed
        config['update_declined'] = mc.APP_VERSION
        self.assertIsNone(mc.startup_question(config, 'linux'))               # "Not now" was chosen
        config['update_declined'], config['installed']['version'] = None, '99.0.0'
        self.assertIsNone(mc.startup_question(config, 'linux'))               # never downgrade

    def test_install_commands(self):
        eng = mc.MatlabEngine()
        if mc.install_target(False) is None:
            self.skipTest('installing is not supported on this system')
        self.assertIn('is installed for you', ''.join(eng.execute('install')))
        self.assertIsNone(eng.last_error)
        self.assertIn('Removed', ''.join(eng.execute('uninstall')))
        mc._can_write = lambda path: False
        self.assertIn('administrator rights', ''.join(eng.execute('install systemwide')))
        self.assertIsNotNone(eng.last_error)
        self.assertIn('install systemwide', ''.join(eng.execute('help install')))

    def test_first_start_dialog(self):
        root = make_root(visible=True)
        try:
            app = mc.MatlabConsoleApp(root)
            asked = []
            original_show = mc.InstallDialog.show

            def answer(dialog, key):
                asked.append(dialog.title())
                dialog.after(100, dialog.buttons[key].invoke)
                return original_show(dialog)
            mc.InstallDialog.show = lambda dialog: answer(dialog, 'user')
            try:
                app.startup_check('linux')                                     # first start: asked, installs
                self.assertEqual(asked, [f'Install {mc.APP_NAME}?'])
                self.assertIn('is installed for you', app.output_area.get('1.0', 'end'))
                self.assertTrue(app.output_area.get('end-1l linestart', 'end-1c').startswith(mc.PROMPT))
                app.startup_check('linux')                                     # second start: not asked
                self.assertEqual(len(asked), 1)
            finally:
                mc.InstallDialog.show = original_show
            self.assertTrue(self.config()['first_start_done'])
        finally:
            root.destroy()


class TestScripts(unittest.TestCase):
    """Reading .m files (File > Open) and writing the session (File > Save), without a window."""

    def test_script_commands(self):
        text = ("% a comment line\n"
                "a = 1;   % set a\n"
                "\n"
                "B = [1 2\n"
                "     3 4]\n"
                "s = 'it''s ... [ not code';\n"
                "c = [1, 2, ...\n"
                "     3]\n"
                "  % indented comment\n"
                "x = '[', y = 2\r\n"
                "e = [1 2\n")
        self.assertEqual(mc.script_commands(text), [
            (2, 'a = 1;   % set a'), (4, 'B = [1 2\n     3 4]'), (6, "s = 'it''s ... [ not code';"),
            (7, 'c = [1, 2, ...\n     3]'), (10, "x = '[', y = 2"), (11, 'e = [1 2')])
        self.assertEqual(mc.script_commands(''), [])
        eng = mc.MatlabEngine()
        for _, command in mc.script_commands(text)[:5]:
            eng.execute(command)
            self.assertIsNone(eng.last_error, command)
        self.assertEqual(eng.variables['B'].tolist(), [[1, 2], [3, 4]])
        self.assertEqual(eng.variables['c'].tolist(), [[1, 2, 3]])

    def test_session_script(self):
        text = mc.session_script(['a = 1', 'B = [1 2\n3 4]', 'disp(a)  '])
        lines = text.splitlines()
        self.assertTrue(lines[0].startswith(f'% {mc.APP_NAME} {mc.APP_VERSION} session, saved '))
        self.assertEqual(lines[1:], ['a = 1', 'B = [1 2', '3 4]', 'disp(a)'])
        self.assertEqual([c for _, c in mc.script_commands(text)], ['a = 1', 'B = [1 2\n3 4]', 'disp(a)'])


class TestSettings(unittest.TestCase):
    """The settings of the Settings window, kept in the settings file."""

    def setUp(self):
        self.saved = mc.CONFIG_FILE, mc.UPDATE_CHECK_DAYS
        self.tmp = tempfile.mkdtemp()
        mc.CONFIG_FILE = os.path.join(self.tmp, '.matlab_calc.json')

    def tearDown(self):
        mc.CONFIG_FILE, mc.UPDATE_CHECK_DAYS = self.saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_defaults_and_storing(self):
        defaults = mc.user_settings()
        self.assertEqual(defaults, mc.default_settings())
        self.assertEqual(defaults['font_size'], mc.FONT_SIZE)
        mc.store_settings(dict(defaults, font_size=15, number_format='long', update_check_days=3))
        config = mc.load_config()
        self.assertEqual((config['font_size'], config['number_format'], config['update_check_days']), (15, 'long', 3))
        self.assertNotIn('check_updates', config)             # unchanged values are not stored
        self.assertEqual(mc.update_check_interval(config), 3)
        mc.store_settings(dict(mc.user_settings(), check_updates=False))
        self.assertEqual(mc.update_check_interval(mc.load_config()), 0)
        self.assertEqual(mc.user_settings()['update_check_days'], 3)  # remembered while turned off
        mc.store_settings(mc.default_settings())
        self.assertEqual(mc.load_config(), {})

    def test_invalid_values_are_ignored(self):
        mc.save_config({'font_size': 'big', 'number_format': 'huge', 'update_check_days': 'x', 'font_family': ' '})
        self.assertEqual(mc.user_settings(), mc.default_settings())
        mc.save_config({'font_size': 1000})
        self.assertEqual(mc.user_settings()['font_size'], mc.FONT_SIZE_LIMITS[1])

    def test_update_checks_turned_on_when_the_parameter_is_zero(self):
        mc.UPDATE_CHECK_DAYS = 0
        self.assertFalse(mc.user_settings()['check_updates'])
        mc.store_settings(dict(mc.user_settings(), check_updates=True, update_check_days=7))
        self.assertEqual(mc.update_check_interval(mc.load_config()), 7)
        self.assertTrue(mc.update_check_due(mc.load_config()))


class TestMenusAndWindows(InstallationCase):
    """Menu bar, About, Help and Settings windows, and File > Open / Save, in a real window."""

    def setUp(self):
        super().setUp()
        self.root = make_root(visible=True)
        self.app = mc.MatlabConsoleApp(self.root)
        self.root.update()
        self.text = self.app.output_area
        self.cwd = os.getcwd()
        self.folder = os.path.join(self.tmp, 'scripts')
        os.makedirs(self.folder)
        self.dialogs = []
        self.original_show = mc.InstallDialog.show
        mc.InstallDialog.show = lambda dialog: (self.dialogs.append((dialog.title(), dialog.message)),
                                                dialog.destroy())[0] and None

    def tearDown(self):
        mc.InstallDialog.show = self.original_show
        os.chdir(self.cwd)
        try:
            self.root.destroy()
        except tk.TclError:
            pass  # closed by the test
        gc.collect()
        super().tearDown()

    def labels(self, menu):
        return [menu.entrycget(i, 'label') for i in range(menu.index('end') + 1) if menu.type(i) == 'command']

    def invoke(self, menu, label):
        menu = self.app.menus[menu]
        menu.invoke(next(i for i in range(menu.index('end') + 1)
                         if menu.type(i) == 'command' and menu.entrycget(i, 'label') == label))

    def cascades(self):
        bar = self.app.menus['bar']
        return [bar.entrycget(i, 'label') for i in range(bar.index('end') + 1) if bar.type(i) == 'cascade']

    def console(self):
        return self.text.get('1.0', 'end-1c')

    def write_script(self, name, text):
        path = os.path.join(self.folder, name)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        return path

    def test_linux_menus(self):
        if self.app.mac_menus:                          # running on macOS: build the Linux layout
            self.app.mac_menus = False
            self.app.build_menus()
            self.app.bind_shortcuts()
        menus = self.app.menus
        self.assertEqual(self.labels(menus['file']), ['Open…', 'Save…', 'Settings…', 'Quit'])
        self.assertEqual(self.labels(menus['edit']), ['Cut', 'Copy', 'Paste', 'Select All'])
        self.assertEqual(self.labels(menus['help']),
                         [f'{mc.APP_NAME} Help', 'Check for Updates…', f'About {mc.APP_NAME}'])
        self.assertEqual(self.cascades(), ['File', 'Edit', 'Help'])
        self.assertEqual(menus['file'].entrycget(0, 'accelerator'), 'Ctrl+O')
        for key in ('<Control-o>', '<Control-s>', '<Control-q>', '<Control-plus>', '<Control-minus>', '<F1>'):
            self.assertTrue(self.text.bind(key), key)

    def test_mac_menus(self):
        self.app.mac_menus = True                       # the layout used with macOS (Aqua) Tk
        self.app.build_menus()
        menus = self.app.menus
        self.assertEqual(self.labels(menus['file']), ['Open…', 'Save…'])
        self.assertEqual(menus['file'].entrycget(0, 'accelerator'), 'Command-O')
        self.assertIn('Check for Updates…', self.labels(menus['help']))
        self.assertNotIn(f'About {mc.APP_NAME}', self.labels(menus['help']))
        self.assertEqual(self.cascades(), ['File', 'Edit', 'Window', 'Help'])
        # The application menu's About and Settings items, and the system's Help item, call these:
        self.root.tk.call('tkAboutDialog')
        self.root.tk.call('::tk::mac::ShowPreferences')
        self.root.tk.call('::tk::mac::ShowHelp')
        self.assertEqual(sorted(self.app.windows), ['about', 'help', 'settings'])

    def test_about_window(self):
        if self.app.mac_menus:
            self.root.tk.call('tkAboutDialog')            # Matlab_Calc > About Matlab_Calc
        else:
            self.invoke('help', f'About {mc.APP_NAME}')
        about = self.app.windows['about']
        self.root.update()
        shown = about.text()
        for part in (mc.APP_NAME, f'Version {mc.APP_VERSION}', 'github.com/varallyay/matlab_calc',
                     'Developed by Zoltan Varallyay, PhD', 'with the help of AI technology (Antigravity, Claude)'):
            self.assertIn(part, shown)
        self.assertEqual(' '.join(shown.split()).count(' '.join(mc.APP_CREDITS.split())), 1)
        self.assertEqual(about.title(), f'About {mc.APP_NAME}')
        if os.path.isfile(ICON):
            self.assertIsNotNone(about.icon_label)
        opened = []
        saved = mc.webbrowser.open
        mc.webbrowser.open = opened.append
        try:
            about.link.event_generate('<Button-1>')
        finally:
            mc.webbrowser.open = saved
        self.assertEqual(opened, [mc.APP_WEBSITE])
        self.assertIs(self.app.show_about(), about)    # a second click shows the same window

    def test_help_window(self):
        self.invoke('help', f'{mc.APP_NAME} Help')
        window = self.app.windows['help']
        content = window.text_area.get('1.0', 'end')
        mac = self.app.mac_menus
        for part in ('help round', 'help ops', 'File > Open', 'File > Save', 'Check for Updates', mc.APP_WEBSITE,
                     f'{mc.APP_NAME} > Settings' if mac else 'File > Settings', 'Cmd +, Cmd -' if mac else 'Ctrl +, Ctrl -'):
            self.assertIn(part, content)
        self.assertIn('Cmd +, Cmd -', mc.app_help_text(mac=True))
        self.assertIn(f'{mc.APP_NAME} > Settings', mc.app_help_text(mac=True))
        self.text.insert('input_start', 'half typed')
        window.list_button.invoke()                       # runs 'help' in the Command Window
        self.assertIn('>> help\n', self.console())
        self.assertIn('Console commands', self.console())
        self.assertEqual(self.text.get('input_start', 'end-1c'), 'half typed')
        kinds = {kind for kind, _ in mc.help_blocks(mc.app_help_text(False))}
        self.assertEqual(kinds, {'heading', 'text', 'item', 'code', 'gap'})

    def test_settings_window(self):
        if self.app.mac_menus:
            self.root.tk.call('::tk::mac::ShowPreferences')  # Matlab_Calc > Settings
        else:
            self.invoke('file', 'Settings…')
        dialog = self.app.windows['settings']
        self.assertIn(self.app.console_font.cget('family'), dialog.family_box.cget('values'))
        dialog.size.set('16')
        self.assertEqual(int(dialog.preview_font.cget('size')), 16)    # the preview follows
        dialog.number_format.set('long')
        dialog.check_updates.set(False)
        dialog.ok()
        self.assertEqual(self.app.font_size(), 16)
        self.assertEqual(int(self.app.prompt_font.cget('size')), 16)
        self.assertEqual(self.app.engine.format_mode, 'long')
        config = self.config()
        self.assertEqual((config['font_size'], config['number_format'], config['check_updates']), (16, 'long', False))
        self.assertFalse(mc.update_check_due(config))
        root2 = make_root()                               # the next start uses the settings
        try:
            app2 = mc.MatlabConsoleApp(root2)
            self.assertEqual((app2.font_size(), app2.engine.format_mode), (16, 'long'))
        finally:
            root2.destroy()
        dialog = self.app.show_settings()
        dialog.size.set('20')
        dialog.destroy()                                  # Cancel changes nothing
        self.assertEqual(self.app.font_size(), 16)
        dialog = self.app.show_settings()
        dialog.restore_defaults()
        dialog.ok()
        self.assertEqual(self.app.font_size(), mc.FONT_SIZE)
        self.assertEqual(self.app.console_font.cget('family'), mc.pick_font_family(self.root))
        self.assertFalse(any(key in self.config() for key in mc.SETTING_KEYS))

    def test_font_size_shortcuts(self):
        self.app.change_font_size(1)
        self.assertEqual(self.app.font_size(), mc.FONT_SIZE + 1)
        self.assertEqual(self.config()['font_size'], mc.FONT_SIZE + 1)
        self.app.change_font_size(-3)
        self.assertEqual(self.app.font_size(), mc.FONT_SIZE - 2)
        self.app.change_font_size(0)
        self.assertEqual(self.app.font_size(), mc.FONT_SIZE)
        self.assertNotIn('font_size', self.config())
        for _ in range(100):
            self.app.change_font_size(1)
        self.assertEqual(self.app.font_size(), mc.FONT_SIZE_LIMITS[1])

    def test_open_script(self):
        with open(os.path.join(self.folder, 'data.txt'), 'w', encoding='utf-8') as f:
            f.write('4 5 6\n')
        path = self.write_script('demo.m', "% Demo script\n"
                                           "a = 2 + 3\n"
                                           "B = [1 2\n"
                                           "     3 4];\n"
                                           "\n"
                                           "undefined_name + 1\n"
                                           "fid = fopen('data.txt'); d = fscanf(fid, '%d')', fclose(fid);\n"
                                           "fprintf('sum %d\\n', sum_of_d)   % an error: not defined\n"
                                           "disp(a * 2)\n")
        self.text.insert('input_start', 'typed')
        self.assertTrue(self.app.open_script(path))
        content = self.console()
        self.assertIn('>> a = 2 + 3\na =\n    5\n', content)
        self.assertIn('>> B = [1 2\n     3 4];\n>> undefined_name + 1\n', content)
        self.assertIn("Unrecognized function or variable 'undefined_name'", content)
        self.assertIn('d =\n    4   5   6\n', content)              # found next to the script
        self.assertIn('>> disp(a * 2)\n    10\n', content)         # runs on after an error
        self.assertNotIn('Demo script', content)                   # comment lines are not shown
        self.assertEqual(self.text.get('input_start', 'end-1c'), 'typed')
        self.assertEqual(os.getcwd(), self.cwd)                     # the current folder is back
        self.assertEqual(self.app.engine.variables['B'].tolist(), [[1, 2], [3, 4]])
        self.assertEqual(self.app.history[0], 'a = 2 + 3')
        self.assertEqual(len(self.app.history), 6)

    def test_script_that_changes_folder(self):
        path = self.write_script('go.m', 'cd ..\n')
        self.app.open_script(path)
        self.assertEqual(os.path.realpath(os.getcwd()), os.path.realpath(self.tmp))  # kept, as in MATLAB

    def test_open_missing_file(self):
        self.app.open_script(os.path.join(self.folder, 'missing.m'))
        self.assertIn('Error: Cannot open', self.console())
        self.assertTrue(self.console().endswith(mc.PROMPT))

    def test_save_session(self):
        self.app.save_session(os.path.join(self.folder, 'empty.m'))   # nothing typed yet
        self.assertEqual(self.dialogs[0][0], 'Save Session')
        self.assertFalse(os.path.exists(os.path.join(self.folder, 'empty.m')))
        for command in ('x = 3;', 'y = x ^ 2', 'z = [1 2', "fprintf('%d\\n', y)"):
            self.app.run_typed(command)
        path = os.path.join(self.folder, 'session.m')
        self.assertEqual(self.app.save_session(path), path)
        with open(path, encoding='utf-8') as f:
            lines = f.read().splitlines()
        self.assertTrue(lines[0].startswith(f'% {mc.APP_NAME} {mc.APP_VERSION} session'))
        self.assertEqual(lines[1:], ['x = 3;', 'y = x ^ 2', 'z = [1 2', "fprintf('%d\\n', y)"])
        self.assertIn(f'Saved 4 commands to {path}', self.console())
        root2 = make_root()                                         # File > Open runs it again
        try:
            app2 = mc.MatlabConsoleApp(root2)
            app2.open_script(path)
            self.assertEqual(app2.engine.variables['y'].item(), 9)
        finally:
            root2.destroy()

    def test_exit_in_a_script(self):
        path = self.write_script('bye.m', 'a = 1\nexit\nb = 2\n')
        self.assertFalse(self.app.open_script(path))
        self.assertTrue(self.app.closed)
        self.assertNotIn('b', self.app.engine.variables)

    def test_message_after_output_without_newline(self):
        self.app.run_typed("fprintf('no newline')")
        self.app.print_message('Hello')
        self.assertTrue(self.console().endswith('no newline\n  Hello\n\n' + mc.PROMPT), self.console())


class FakeGitHub:
    """A local web server answering like api.github.com and raw.githubusercontent.com.

    files maps a branch or tag name to {file name: bytes}; release is the tag of the
    newest release, or None for a project without releases.
    """

    def __init__(self):
        self.files, self.release, self.requests = {}, None, []
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                fake.requests.append(self.path)
                api = f'/api/repos/{mc.UPDATE_REPOSITORY}/releases/latest'
                raw = f'/raw/{mc.UPDATE_REPOSITORY}/'
                body = None
                if self.path == api and fake.release:
                    body = json.dumps({'tag_name': fake.release}).encode()
                elif self.path.startswith(raw):
                    ref, _, name = self.path[len(raw):].partition('/')
                    body = fake.files.get(ref, {}).get(name)
                self.send_response(200 if body is not None else 404)
                self.end_headers()
                self.wfile.write(body if body is not None else b'Not Found')

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}'

    def publish(self, version, ref='HEAD', icon=True):
        """Put a copy of this program with APP_VERSION = version on the fake GitHub."""
        with open(mc.THIS_SCRIPT, encoding='utf-8') as f:
            source = re.sub(r"^APP_VERSION = '[^']*'", f"APP_VERSION = '{version}'", f.read(), flags=re.M)
        self.files.setdefault(ref, {})[mc.APP_SLUG + '.py'] = source.encode()
        if icon and os.path.isfile(ICON):
            with open(ICON, 'rb') as f:
                self.files[ref][mc.APP_ICON_FILE] = f.read()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


class TestUpdates(InstallationCase):
    """Looking for, downloading and installing new versions - against a local fake GitHub."""

    PATCHED = InstallationCase.PATCHED + ('GITHUB_API', 'GITHUB_RAW')

    def setUp(self):
        gc.collect()  # free Tk objects of earlier tests here, not in a web server thread
        super().setUp()
        self.github = FakeGitHub()
        mc.GITHUB_API, mc.GITHUB_RAW = self.github.url + '/api', self.github.url + '/raw'

    def tearDown(self):
        self.github.stop()
        super().tearDown()

    def installed_version(self):
        return mc.file_version(mc.install_target(False, 'linux')['script'])

    def test_newest_version_from_main_branch(self):
        self.github.publish('9.9.9')
        latest = mc.find_latest_version()
        self.assertEqual(latest['version'], '9.9.9')
        self.assertTrue(latest['base'].endswith('/HEAD/'))

    def test_newest_release_is_preferred(self):
        self.github.publish('9.9.9')                       # newer, but not released yet
        self.github.publish('1.2.3', ref='v1.2.3')
        self.github.release = 'v1.2.3'
        self.assertEqual(mc.find_latest_version()['version'], '1.2.3')

    def test_invalid_download_is_rejected(self):
        self.github.files['HEAD'] = {mc.APP_SLUG + '.py': b'<html>Rate limit exceeded</html>'}
        with self.assertRaisesRegex(mc.UpdateError, 'not a valid'):
            mc.find_latest_version()
        self.github.files['HEAD'] = {}
        with self.assertRaisesRegex(mc.UpdateError, 'HTTP error 404'):
            mc.find_latest_version()

    def test_github_unreachable(self):
        mc.GITHUB_API = mc.GITHUB_RAW = 'http://127.0.0.1:9'   # nothing listens there
        with self.assertRaisesRegex(mc.UpdateError, 'Could not connect to GitHub'):
            mc.find_latest_version()
        mc.install_app(False, 'linux')
        ok, message = mc.update_installed_app('linux')
        self.assertFalse(ok)
        self.assertEqual(self.installed_version(), mc.APP_VERSION)    # nothing changed

    def test_update_installed_copy(self):
        self.assertIn('not installed', mc.update_installed_app('linux')[1])
        mc.install_app(False, 'linux')
        self.github.publish('9.9.9')
        ok, message = mc.update_installed_app('linux')
        self.assertTrue(ok, message)
        self.assertIn(f'updated from {mc.APP_VERSION} to 9.9.9', message)
        self.assertIn('Restart', message)
        self.assertEqual(self.installed_version(), '9.9.9')
        self.assertEqual(self.config()['installed']['version'], '9.9.9')
        self.assertGreater(self.config()['last_update_check'], time.time() - 60)
        self.assertIn('9.9.9 is up to date', mc.update_installed_app('linux')[1])  # nothing newer now

    def test_never_downgrades(self):
        mc.install_app(False, 'linux')
        self.github.publish('0.0.1')
        self.assertIn('is up to date', mc.update_installed_app('linux')[1])
        self.assertEqual(self.installed_version(), mc.APP_VERSION)

    def test_mac_bundle_update(self):
        mc.install_app(False, 'darwin')
        self.github.publish('9.9.9')
        self.assertTrue(mc.update_installed_app('darwin')[0])
        app = os.path.join(mc.MAC_USER_DIR, mc.APP_NAME + '.app')
        with open(os.path.join(app, 'Contents', 'Info.plist'), 'rb') as f:
            self.assertEqual(plistlib.load(f)['CFBundleShortVersionString'], '9.9.9')

    def test_weekly_schedule(self):
        week = mc.UPDATE_CHECK_DAYS * 86400
        now = time.time()
        self.assertTrue(mc.update_check_due({}, now))
        self.assertFalse(mc.update_check_due({'last_update_check': now - week + 3600}, now))
        self.assertTrue(mc.update_check_due({'last_update_check': now - week - 1}, now))
        self.assertFalse(mc.update_check_due({'last_update_check': 0, 'check_updates': False}, now))
        mc.install_app(False, 'linux')                     # installing starts the first week
        self.assertFalse(mc.update_check_due(self.config(), time.time()))

    def test_update_command(self):
        eng = mc.MatlabEngine()
        self.assertIn('off', ''.join(eng.execute('update off')))
        self.assertFalse(self.config()['check_updates'])
        self.assertIn('on', ''.join(eng.execute('update on')))
        self.assertTrue(self.config()['check_updates'])
        if mc.install_target(False) is None:
            self.skipTest('installing is not supported on this system')
        eng.execute('install')
        self.github.publish('9.9.9')
        self.assertIn('updated', ''.join(eng.execute('update')))
        self.assertIsNone(eng.last_error)
        self.assertIn('look every', ''.join(eng.execute('help update')))

    def answer_dialogs(self, answers, seen):
        """Make every InstallDialog answer by itself: answers[title] is the button to press."""
        original = mc.InstallDialog.show

        def show(dialog):
            seen.append((dialog.title(), dialog.icon_label is not None, dialog.checkbox is not None))
            dialog.after(100, dialog.buttons[answers[dialog.title()]].invoke)
            return original(dialog)
        mc.InstallDialog.show = show
        return original

    def test_weekly_check_in_the_window(self):
        root = make_root(visible=True)
        original = None
        try:
            app = mc.MatlabConsoleApp(root)
            mc.install_app(False, 'linux')
            config = self.config()
            config['last_update_check'] = 0                # the last check was long ago
            mc.save_config(config)
            self.github.publish('9.9.9')
            seen, restarted = [], []
            app.restart = restarted.append
            original = self.answer_dialogs({f'Update {mc.APP_NAME}?': 'update',
                                            f'Restart {mc.APP_NAME}?': 'restart'}, seen)
            app.startup_check('linux')                     # starts the look in the background
            deadline = time.time() + 20
            while len(seen) < 2 and time.time() < deadline:
                root.update()
                time.sleep(0.02)
            titles = [title for title, _, _ in seen]
            self.assertEqual(titles, [f'Update {mc.APP_NAME}?', f'Restart {mc.APP_NAME}?'])
            if os.path.isfile(ICON):
                self.assertTrue(all(has_icon for _, has_icon, _ in seen), 'both dialogs show the icon')
            self.assertTrue(seen[0][2], 'the update question has the weekly checkbox')
            self.assertEqual(self.installed_version(), '9.9.9')
            self.assertEqual(restarted, [mc.install_target(False, 'linux')['script']])
            self.assertIn('updated from', app.output_area.get('1.0', 'end'))
        finally:
            if original:
                mc.InstallDialog.show = original
            root.destroy()
            gc.collect()

    def test_skip_this_version(self):
        root = make_root(visible=True)
        original = None
        try:
            app = mc.MatlabConsoleApp(root)
            mc.install_app(False, 'linux')
            self.github.publish('9.9.9')
            seen = []
            original = self.answer_dialogs({f'Update {mc.APP_NAME}?': 'skip'}, seen)
            app.offer_update(mc.find_latest_version(), 'linux')
            app.offer_update(mc.find_latest_version(), 'linux')   # not asked again for 9.9.9
            self.assertEqual(len(seen), 1)
            self.assertEqual(self.config()['skipped_version'], '9.9.9')
            self.assertEqual(self.installed_version(), mc.APP_VERSION)
        finally:
            if original:
                mc.InstallDialog.show = original
            root.destroy()
            gc.collect()

    def test_check_for_updates_menu(self):
        root = make_root(visible=True)
        original = mc.InstallDialog.show
        seen = []

        def show(dialog):
            seen.append((dialog.title(), dialog.message))
            dialog.after(50, dialog.buttons['later' if 'later' in dialog.buttons else 'ok'].invoke)
            return original(dialog)
        mc.InstallDialog.show = show
        try:
            app = mc.MatlabConsoleApp(root)

            def check():
                thread = app.check_for_update_now('linux')
                self.assertIsNone(app.check_for_update_now('linux'))   # one look at a time
                deadline, count = time.time() + 20, len(seen)
                while len(seen) == count and time.time() < deadline:
                    root.update()
                    time.sleep(0.02)
                thread.join()
                self.assertFalse(app._looking_for_update)
                return seen[-1]

            self.github.publish(mc.APP_VERSION)
            title, message = check()
            self.assertEqual(title, 'Check for Updates')
            self.assertIn(f'{mc.APP_VERSION} is up to date', message)
            self.github.publish('9.9.9')
            self.assertIn('9.9.9 is available on GitHub', check()[1])  # not installed: no update offered
            mc.install_app(False, 'linux')
            config = self.config()
            config['skipped_version'] = '9.9.9'
            mc.save_config(config)
            self.assertEqual(check()[0], f'Update {mc.APP_NAME}?')   # offered although skipped before
            self.assertEqual(self.installed_version(), mc.APP_VERSION)
            mc.GITHUB_API = mc.GITHUB_RAW = 'http://127.0.0.1:9'
            self.assertIn('Could not look for a new version', check()[1])
            self.assertEqual(app.check_for_update_now.__doc__.split(':')[0], 'Help > Check for Updates')
        finally:
            mc.InstallDialog.show = original
            root.destroy()
            gc.collect()


class TestIcon(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(ICON):
            raise unittest.SkipTest(f'{ICON} not found')

    def check_alpha(self, alpha, size):
        """alpha(x, y) -> 0..255 for an icon processed with the default margin/rounding."""
        inner = round(size * (1 - 2 * mc.APP_ICON_MARGIN))
        off = (size - inner) // 2
        self.assertEqual(alpha(1, size // 2), 0, 'left margin should be transparent')
        self.assertEqual(alpha(size // 2, size - 2), 0, 'bottom margin should be transparent')
        self.assertEqual(alpha(off + 1, off + 1), 0, 'corner should be rounded off')
        self.assertGreater(alpha(off + inner // 2, off + 1), 200, 'top edge should be opaque')
        self.assertGreater(alpha(size // 2, size // 2), 200, 'centre should be opaque')

    @unittest.skipUnless(HAVE_PIL, 'Pillow not installed')
    def test_pillow_icon(self):
        img = mc.rounded_icon_image(ICON)
        from PIL import Image
        self.assertEqual(img.size[0], max(Image.open(ICON).size))
        self.check_alpha(lambda x, y: img.getpixel((x, y))[3], img.size[0])
        flat = mc.rounded_icon_image(ICON, margin=0, rounding=0)
        self.assertGreater(flat.getpixel((0, 0))[3], 200, 'margin=0, rounding=0 keeps the full square')

    def test_tk_fallback_icon(self):
        root = make_root()
        try:
            img = mc._rounded_icon_tk(root, ICON, mc.APP_ICON_MARGIN, mc.APP_ICON_ROUNDING)
            self.assertEqual(img.width(), img.height())
            self.check_alpha(lambda x, y: 0 if img.transparency_get(x, y) else 255, img.width())
            with tempfile.TemporaryDirectory() as tmp:  # Tk can write it back out as PNG
                img.write(os.path.join(tmp, 'icon.png'), format='png')
        finally:
            root.destroy()

    def test_missing_icon_is_ignored(self):
        root = make_root()
        try:
            self.assertIsNone(mc.load_app_icon(root, os.path.join(tempfile.gettempdir(), 'missing.png')))
        finally:
            root.destroy()


if __name__ == '__main__':
    unittest.main(verbosity=2)
