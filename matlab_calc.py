"""
Matlab_Calc - a MATLAB Command Window emulator built with Tkinter and NumPy.

Run with:  python3 matlab_calc.py
"""

import inspect
import math
import os
import re
import sys
import tkinter as tk
import tkinter.font as tkfont
from fractions import Fraction
from tkinter import scrolledtext

import numpy as np

# =============================================================================
#  Application parameters - edit these to customise the app
# =============================================================================
APP_NAME = 'Matlab_Calc'        # Window title, Tk class name and (macOS) menu-bar name
APP_VERSION = '0.4.0'           # Shown in the welcome banner; see the version history in README.md

WINDOW_WIDTH = 750              # Initial window width in pixels
WINDOW_HEIGHT = 520             # Initial window height in pixels

APP_ICON_FILE = 'icon.png'      # Icon image, looked up next to this script
APP_ICON_MARGIN = 0.1           # Transparent margin on each side, as a fraction of the icon size
APP_ICON_ROUNDING = 0.20        # Corner radius, as a fraction of the visible (shrunken) icon size

FONT_FAMILIES = ('Consolas', 'Menlo', 'Monaco', 'DejaVu Sans Mono', 'Courier New')  # first installed one is used
FONT_SIZE = 11
BACKGROUND_COLOR = '#1e1e1e'
TEXT_COLOR = '#d4d4d4'
PROMPT_COLOR = '#569cd6'
CURSOR_COLOR = 'white'
WINDOW_PADDING = 10             # Space around the console, in pixels
PROMPT = '>> '

SHORT_DECIMALS = 4              # Maximum decimals shown by 'format short' (the default)
LONG_DECIMALS = 15              # Maximum decimals shown by 'format long'
MAX_DISPLAY_ELEMENTS = 10000    # Bigger results are summarised instead of printed in full
# =============================================================================

__version__ = APP_VERSION


MAX_ELEMENTS = 10_000_000       # Guard against accidental huge allocations, e.g. 1:1e12
SCI_THRESHOLD = 1e10            # Magnitudes from here on are shown in scientific notation

MSG_INCOMPATIBLE = "Arrays have incompatible sizes for this operation."
MSG_UNBALANCED = "Error: Unbalanced or unexpected parenthesis or bracket."
MSG_MTIMES = (
    "Error using *\n"
    "Incorrect dimensions for matrix multiplication. Check that the number of\n"
    "columns in the first matrix matches the number of rows in the second\n"
    "matrix. To operate on each element of the matrix individually, use\n"
    "TIMES (.*) for elementwise multiplication."
)
MSG_MPOWER = (
    "Error using ^\n"
    "Incorrect dimensions for raising a matrix to a power. Check that\n"
    "the matrix is square and the power is a scalar. To operate on\n"
    "each element of the matrix individually, use POWER (.^) for\n"
    "elementwise power."
)


class MatlabError(Exception):
    """An error whose message is shown to the user as-is (MATLAB style)."""


# =============================================================================
#  Values and operators
#  Every value is a 2-D NumPy array (scalars are 1x1) of dtype float64 or
#  complex128, mirroring MATLAB's "everything is a matrix" model.
# =============================================================================
def to_matrix(x):
    """Convert any value to a 2-D array of float, complex, or bool (MATLAB's logical values)."""
    a = np.asarray(x)
    if a.dtype == bool:
        pass  # logical (true/false), e.g. the result of a comparison
    elif np.iscomplexobj(a):
        a = a.astype(complex)
        if not np.any(a.imag):  # MATLAB drops all-zero imaginary parts
            a = a.real.copy()
    else:
        a = a.astype(float)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(1, -1)
    if a.ndim > 2:
        raise MatlabError("Error: Only 2-D matrices are supported.")
    return a


def to_double(a):
    """Logical values take part in arithmetic and functions as the numbers 1 and 0, as in MATLAB."""
    return a.astype(float) if a.dtype == bool else a


def _is_scalar(a):
    return a.size == 1


def _is_square(a):
    return a.shape[0] == a.shape[1]


def _elementwise(func, a, b, symbol):
    """Apply func with MATLAB implicit expansion (same rules as NumPy broadcasting)."""
    try:
        return to_matrix(func(a, b))
    except ValueError:
        raise MatlabError(f"Error using {symbol}\n{MSG_INCOMPATIBLE}") from None


def _matrix_function(m, f):
    """f(M) for a diagonalisable square matrix, via its eigendecomposition."""
    vals, vecs = np.linalg.eig(m)
    res = vecs @ np.diag(f(vals)) @ np.linalg.inv(vecs)
    if not np.iscomplexobj(m) and np.allclose(res.imag, 0, atol=1e-12 * max(1.0, np.abs(res).max())):
        res = res.real
    return to_matrix(res)


def op_power(a, b):
    """.^ : element-wise power (complex result for negative base and fractional exponent)."""
    def pw(x, y):
        x, y = np.broadcast_arrays(x, y)
        if not np.iscomplexobj(x) and not np.iscomplexobj(y):
            if np.any((x < 0) & np.isfinite(y) & (y != np.round(y))):
                x = x.astype(complex)
        return np.power(x, y)
    return _elementwise(pw, a, b, '.^')


def op_mpower(a, b):
    """^ : matrix power (scalar^scalar, square^scalar or scalar^square)."""
    if _is_scalar(a) and _is_scalar(b):
        return op_power(a, b)
    if _is_scalar(b) and _is_square(a):
        p = complex(b.flat[0])
        if p.imag == 0 and float(p.real).is_integer():
            return to_matrix(np.linalg.matrix_power(a, int(p.real)))
        return _matrix_function(a, lambda lam: np.power(lam.astype(complex), p))
    if _is_scalar(a) and _is_square(b):
        base = complex(a.flat[0])
        return _matrix_function(b, lambda lam: np.power(base, lam))
    raise MatlabError(MSG_MPOWER)


def op_mtimes(a, b):
    """* : matrix product, or element-wise when either side is a scalar."""
    if _is_scalar(a) or _is_scalar(b):
        return to_matrix(a * b)
    if a.shape[1] != b.shape[0]:
        raise MatlabError(MSG_MTIMES)
    return to_matrix(a @ b)


def _solve(a, b):
    """X with A*X = B: exact for a square non-singular A, least squares (pseudo-inverse) otherwise."""
    if _is_square(a):
        try:
            return np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            pass
    return np.linalg.pinv(a) @ b


def op_mrdivide(a, b):
    """/ : division by a scalar, otherwise solves X*B = A."""
    if _is_scalar(b):
        return _elementwise(np.divide, a, b, '/')
    if a.shape[1] != b.shape[1]:
        raise MatlabError("Error using /\nMatrix dimensions must agree.")
    return to_matrix(_solve(b.T, a.T).T)


def op_mldivide(a, b):
    """\\ : division of a scalar, otherwise solves A*X = B."""
    if _is_scalar(a):
        return _elementwise(lambda x, y: y / x, a, b, '\\')
    if a.shape[0] != b.shape[0]:
        raise MatlabError("Error using \\\nMatrix dimensions must agree.")
    return to_matrix(_solve(a, b))


def _compare(func, symbol, real_parts=True):
    def op(a, b):
        if real_parts:  # MATLAB orders complex numbers by their real parts
            a, b = np.real(a), np.real(b)
        return _elementwise(func, a, b, symbol)
    return op


def _logical(func, symbol):
    return lambda a, b: _elementwise(lambda x, y: func(x != 0, y != 0), a, b, symbol)


def _logical_scalar(v, symbol):
    if v.size != 1:
        raise MatlabError(f"Operands to the {symbol} operator must be convertible to logical scalar values.")
    return bool(v.flat[0] != 0)


def op_colon(start, step, stop):
    """start:step:stop as a row vector (empty when the range is empty)."""
    if start.size == 0 or step.size == 0 or stop.size == 0:
        return np.zeros((1, 0))
    a, s, b = (float(np.real(v.flat[0])) for v in (start, step, stop))
    if math.isnan(a) or math.isnan(s) or math.isnan(b):
        return to_matrix(np.nan)
    if s == 0 or (s > 0 and a > b) or (s < 0 and a < b):
        return np.zeros((1, 0))
    n = (b - a) / s
    if not math.isfinite(n) or n + 1 > MAX_ELEMENTS:
        raise MatlabError("Error using :\nMaximum variable size allowed by the program is exceeded.")
    count = int(math.floor(n + 1e-10 * max(1.0, abs(n)))) + 1
    return to_matrix(a + s * np.arange(count))


BINARY_OPS = {
    '+': lambda a, b: _elementwise(np.add, a, b, '+'),
    '-': lambda a, b: _elementwise(np.subtract, a, b, '-'),
    '.*': lambda a, b: _elementwise(np.multiply, a, b, '.*'),
    './': lambda a, b: _elementwise(np.divide, a, b, './'),
    '.\\': lambda a, b: _elementwise(lambda x, y: y / x, a, b, '.\\'),
    '.^': op_power,
    '*': op_mtimes,
    '/': op_mrdivide,
    '\\': op_mldivide,
    '^': op_mpower,
    '==': _compare(np.equal, '==', real_parts=False),
    '~=': _compare(np.not_equal, '~=', real_parts=False),
    '<': _compare(np.less, '<'),
    '<=': _compare(np.less_equal, '<='),
    '>': _compare(np.greater, '>'),
    '>=': _compare(np.greater_equal, '>='),
    '&': _logical(np.logical_and, '&'),
    '|': _logical(np.logical_or, '|'),
}


def _with_double_operands(op):
    return lambda a, b: op(to_double(a), to_double(b))


for _symbol in ('+', '-', '.*', './', '.\\', '.^', '*', '/', '\\', '^'):  # true + true is 2, as in MATLAB
    BINARY_OPS[_symbol] = _with_double_operands(BINARY_OPS[_symbol])

UNARY_OPS = {
    '-': lambda a: to_matrix(-to_double(a)),
    '+': to_double,
    '~': lambda a: to_matrix(a == 0),
    "'": lambda a: to_matrix(a.conj().T),   # conjugate transpose
    ".'": lambda a: to_matrix(a.T),         # transpose
}


# =============================================================================
#  Built-in functions and constants
#
#  As in a MATLAB function file, every built-in starts with its help text: the
#  docstring. Its first line, "name - Summary.", is the H1 line listed by 'help';
#  'help name' shows the whole text. Fields such as {APP_NAME} or {SHORT_DECIMALS}
#  are filled in from the parameters when the help is displayed.
#
#  To add a function, write a Python function with such a docstring and decorate
#  it with @builtin. It receives MATLAB values (2-D NumPy arrays) and must return
#  one; the number of inputs it accepts comes from its signature, and parameters
#  with a default value (None) are optional.
# =============================================================================
CAT_COMMANDS = 'Console commands'
CAT_CONSTANTS = 'Constants'
CAT_MATH = 'Elementary math'
CAT_ELEMENTWISE = 'Element-wise arithmetic'
CAT_TRIG = 'Trigonometry'
CAT_ROUNDING = 'Rounding'
CAT_LOGICAL = 'Logical values and searching'
CAT_MATRIX = 'Matrices and linear algebra'
HELP_CATEGORIES = (CAT_COMMANDS, CAT_CONSTANTS, CAT_MATH, CAT_ELEMENTWISE, CAT_TRIG, CAT_ROUNDING, CAT_LOGICAL,
                   CAT_MATRIX)

FUNCTIONS = {}  # MATLAB name -> (Python function, minimum inputs, maximum inputs)


def builtin(*names, category):
    """Register the decorated Python function as the MATLAB function(s) `names`."""
    def register(func):
        params = list(inspect.signature(func).parameters.values())
        min_args = sum(p.default is inspect.Parameter.empty for p in params)
        for name in names:
            FUNCTIONS[name] = (func, min_args, len(params))
        func.matlab_names, func.help_category = names, category
        return func
    return register


def _given(*args):
    return [a for a in args if a is not None]


def _size_args(name, args):
    """Size arguments of eye/zeros/ones: (), (n), (r, c) or ([r c])."""
    if not args:
        return 1, 1
    vals = np.concatenate([np.real(a).ravel() for a in args])
    if len(args) == 1 and vals.size == 1:
        vals = np.array([vals[0], vals[0]])
    if vals.size != 2:
        raise MatlabError(f"Error using {name}\nOnly 2-D sizes are supported.")
    if not np.all(np.isfinite(vals)) or np.any(vals != np.round(vals)):
        raise MatlabError(f"Error using {name}\nSize inputs must be integers.")
    rows, cols = (max(int(v), 0) for v in vals)
    if rows * cols > MAX_ELEMENTS:
        raise MatlabError(f"Error using {name}\nRequested array exceeds the maximum allowed size.")
    return rows, cols


def _need_square(name, x):
    if not _is_square(x):
        raise MatlabError(f"Error using {name}\nMatrix must be square.")


def _componentwise(func, x):
    """Apply func to real and imaginary parts separately, as MATLAB's rounding functions do."""
    if np.iscomplexobj(x):
        return to_matrix(func(x.real) + 1j * func(x.imag))
    return to_matrix(func(x))


def _round_half_away(v):
    """Nearest integer with halves rounded away from zero (MATLAB), e.g. 2.5 -> 3 and -2.5 -> -3.

    NumPy's np.round rounds halves to the nearest even number instead (2.5 -> 2).
    """
    a = np.abs(v)
    whole = np.floor(a)
    return np.copysign(whole + (a - whole >= 0.5), v)


# ---- constants ---------------------------------------------------------------
@builtin('pi', category=CAT_CONSTANTS)
def fn_pi():
    """pi - Ratio of a circle's circumference to its diameter.

    pi is 3.141592653589793.

    See also e.
    """
    return to_matrix(np.pi)


@builtin('e', category=CAT_CONSTANTS)
def fn_e():
    """e - Euler's number, the base of the natural logarithm.

    e is 2.718281828459045, the same as exp(1).
    Note: MATLAB itself has no constant e; use exp(1) there.

    See also exp, log.
    """
    return to_matrix(np.e)


@builtin('Inf', 'inf', category=CAT_CONSTANTS)
def fn_inf():
    """Inf - Infinity.

    Inf is positive infinity, for example the result of 1/0; -Inf is negative
    infinity.

    See also NaN.
    """
    return to_matrix(np.inf)


@builtin('NaN', 'nan', category=CAT_CONSTANTS)
def fn_nan():
    """NaN - Not-a-Number.

    NaN is the result of undefined operations such as 0/0 or Inf - Inf.
    NaN is not equal to anything, not even to itself: NaN == NaN is 0.

    See also Inf.
    """
    return to_matrix(np.nan)


@builtin('i', 'j', category=CAT_CONSTANTS)
def fn_imaginary_unit():
    """i - Imaginary unit.

    i and j are the imaginary unit, sqrt(-1). Complex numbers can also be
    written directly, as in 3 + 4i or 3 + 4j.
    A variable named i or j hides the constant.

    See also sqrt, abs.
    """
    return to_matrix(1j)


# ---- elementary math ---------------------------------------------------------
@builtin('abs', category=CAT_MATH)
def fn_abs(x):
    """abs - Absolute value and complex magnitude.

    abs(X) is the absolute value of each element of X. For complex X it is
    the magnitude: abs(3 + 4i) is 5.

    See also sqrt.
    """
    return to_matrix(np.abs(x))


@builtin('sqrt', category=CAT_MATH)
def fn_sqrt(x):
    """sqrt - Square root.

    sqrt(X) is the square root of each element of X. Negative elements give
    complex results, as in MATLAB: sqrt(-4) is 0 + 2i.

    See also power, exp, log.
    """
    return to_matrix(np.emath.sqrt(x))


@builtin('exp', category=CAT_MATH)
def fn_exp(x):
    """exp - Exponential.

    exp(X) is e raised to the power of each element of X.

    See also log, power.
    """
    return to_matrix(np.exp(x))


@builtin('log', category=CAT_MATH)
def fn_log(x):
    """log - Natural logarithm.

    log(X) is the natural (base e) logarithm of each element of X. Negative
    elements give complex results: log(-1) is 0 + 3.1416i. log(0) is -Inf.

    See also log10, exp.
    """
    return to_matrix(np.emath.log(x))


@builtin('log10', category=CAT_MATH)
def fn_log10(x):
    """log10 - Common (base 10) logarithm.

    log10(X) is the base 10 logarithm of each element of X: log10(1000) is 3.
    Negative elements give complex results.

    See also log, exp.
    """
    return to_matrix(np.emath.log10(x))


# ---- element-wise arithmetic -------------------------------------------------
@builtin('power', category=CAT_ELEMENTWISE)
def fn_power(a, b):
    """power - Element-wise power, the same as A.^B.

    power(A, B) raises each element of A to the corresponding element of B.
    A and B must have compatible sizes: the same size, one of them a scalar,
    or a row and a column vector (which expand to a matrix).
    A negative base with a non-integer exponent gives a complex result.

    Examples:
        power([1 2 3], 2)     % 1  4  9
        power(2, [1 2 3])     % 2  4  8

    The ^ operator is the matrix power (see help ops).

    See also times, sqrt, exp.
    """
    return op_power(a, b)


@builtin('times', category=CAT_ELEMENTWISE)
def fn_times(a, b):
    """times - Element-wise multiplication, the same as A.*B.

    times(A, B) multiplies corresponding elements of A and B. A and B must
    have compatible sizes: the same size, one of them a scalar, or a row and
    a column vector (which expand to a matrix).

    Example:
        times([1 2 3], [4 5 6])     % 4  10  18

    The * operator is the matrix product (see help ops).

    See also power.
    """
    return BINARY_OPS['.*'](a, b)


# ---- trigonometry ------------------------------------------------------------
@builtin('sin', category=CAT_TRIG)
def fn_sin(x):
    """sin - Sine of an angle in radians.

    sin(X) is the sine of each element of X.

    See also asin, cos, tan.
    """
    return to_matrix(np.sin(x))


@builtin('cos', category=CAT_TRIG)
def fn_cos(x):
    """cos - Cosine of an angle in radians.

    cos(X) is the cosine of each element of X.

    See also acos, sin, tan.
    """
    return to_matrix(np.cos(x))


@builtin('tan', category=CAT_TRIG)
def fn_tan(x):
    """tan - Tangent of an angle in radians.

    tan(X) is the tangent of each element of X.

    See also atan, sin, cos.
    """
    return to_matrix(np.tan(x))


@builtin('asin', category=CAT_TRIG)
def fn_asin(x):
    """asin - Inverse sine, result in radians.

    asin(X) is the arcsine of each element of X, between -pi/2 and pi/2.
    Elements outside [-1, 1] give NaN (MATLAB returns a complex number).

    See also sin, acos, atan.
    """
    return to_matrix(np.arcsin(x))


@builtin('acos', category=CAT_TRIG)
def fn_acos(x):
    """acos - Inverse cosine, result in radians.

    acos(X) is the arccosine of each element of X, between 0 and pi.
    Elements outside [-1, 1] give NaN (MATLAB returns a complex number).

    See also cos, asin, atan.
    """
    return to_matrix(np.arccos(x))


@builtin('atan', category=CAT_TRIG)
def fn_atan(x):
    """atan - Inverse tangent, result in radians.

    atan(X) is the arctangent of each element of X, between -pi/2 and pi/2.

    See also tan, asin, acos.
    """
    return to_matrix(np.arctan(x))


# ---- rounding ----------------------------------------------------------------
@builtin('round', category=CAT_ROUNDING)
def fn_round(x, n=None):
    """round - Round to the nearest integer or to N decimals.

    round(X) rounds each element of X to the nearest integer. Halves are
    rounded away from zero, as in MATLAB: round(2.5) is 3, round(-2.5) is -3.

    round(X, N) rounds to N digits after the decimal point. A negative N
    rounds to the left of the decimal point: to tens, hundreds, ...

    Complex numbers are rounded in their real and imaginary parts separately.

    Examples:
        round(pi, 2)        % 3.14
        round(1234, -2)     % 1200

    See also floor, ceil.
    """
    if n is None:
        return _componentwise(_round_half_away, x)
    if n.size != 1 or np.iscomplexobj(n) or not float(n.flat[0]).is_integer():
        raise MatlabError("Error using round\nThe number of digits N must be an integer scalar.")
    digits = int(n.flat[0])

    def round_to(v):
        if digits > 308:   # finer than double precision: nothing to round
            return v
        if digits < -308:  # coarser than the largest double: every finite value rounds to 0
            return np.where(np.isfinite(v), 0.0 * v, v)
        if digits >= 0:
            scaled = v * 10.0 ** digits
            return np.where(np.isfinite(scaled), _round_half_away(scaled) / 10.0 ** digits, v)
        return _round_half_away(v / 10.0 ** -digits) * 10.0 ** -digits
    return _componentwise(round_to, x)


@builtin('floor', category=CAT_ROUNDING)
def fn_floor(x):
    """floor - Round toward minus infinity.

    floor(X) rounds each element of X to the nearest integer less than or
    equal to it: floor(2.7) is 2, floor(-2.5) is -3.
    Complex numbers are rounded in their real and imaginary parts separately.

    See also ceil, round.
    """
    return _componentwise(np.floor, x)


@builtin('ceil', category=CAT_ROUNDING)
def fn_ceil(x):
    """ceil - Round toward plus infinity.

    ceil(X) rounds each element of X to the nearest integer greater than or
    equal to it: ceil(2.1) is 3, ceil(-2.5) is -2.
    Complex numbers are rounded in their real and imaginary parts separately.

    See also floor, round.
    """
    return _componentwise(np.ceil, x)


# ---- logical values and searching --------------------------------------------
@builtin('find', category=CAT_LOGICAL)
def fn_find(x, n=None):
    """find - Positions of nonzero (true) elements.

    find(X) returns the positions of the nonzero elements of X, numbered
    down the columns as in A(k). With a condition it tells where the
    condition holds:
        A = [1 2 2.4 2.3];
        find(A >= 2)        % 2  3  4
    The result is a row if X is a row vector, otherwise a column, so after
    A = A' the same command gives [2; 3; 4].

    find(X, N) returns at most the first N positions.

    See also logical, true, false.
    """
    positions = np.flatnonzero(x.ravel(order='F')) + 1.0
    if n is not None:
        if n.size != 1 or np.iscomplexobj(n) or not (n.flat[0] >= 1 and float(n.flat[0]).is_integer()):
            raise MatlabError("Error using find\nSecond argument must be a positive scalar integer.")
        positions = positions[:int(n.flat[0])]
    if x.shape == (0, 0):
        return np.zeros((0, 0))
    return to_matrix(positions.reshape((1, -1) if x.shape[0] == 1 else (-1, 1)))


@builtin('logical', category=CAT_LOGICAL)
def fn_logical(x):
    """logical - Convert numbers to logical values (true or false).

    logical(X) is true (1) where X is nonzero and false (0) where X is zero.
    Logical values select elements: A(logical([1 0 1])) picks elements 1 and
    3, while A([1 0 1]) is an error because 0 is not a valid position.
    Comparisons such as A > 2 already give logical values.

    See also true, false, find.
    """
    if np.iscomplexobj(x) or np.any(np.isnan(x)):
        raise MatlabError("Error using logical\nOnly real numbers other than NaN can be converted to logical.")
    return to_matrix(x != 0)


@builtin('true', category=CAT_LOGICAL)
def fn_true(m=None, n=None):
    """true - Logical true (1).

    true is the logical value 1.
    true(N) is an N-by-N logical matrix of ones, true(M, N) or true([M N])
    an M-by-N one.

    See also false, logical.
    """
    return to_matrix(np.ones(_size_args('true', _given(m, n)), dtype=bool))


@builtin('false', category=CAT_LOGICAL)
def fn_false(m=None, n=None):
    """false - Logical false (0).

    false is the logical value 0.
    false(N) is an N-by-N logical matrix of zeros, false(M, N) or
    false([M N]) an M-by-N one.

    See also true, logical.
    """
    return to_matrix(np.zeros(_size_args('false', _given(m, n)), dtype=bool))


# ---- matrices and linear algebra ---------------------------------------------
@builtin('eye', category=CAT_MATRIX)
def fn_eye(m=None, n=None):
    """eye - Identity matrix.

    eye(N) is the N-by-N identity matrix.
    eye(M, N) or eye([M N]) is an M-by-N matrix with ones on the diagonal.

    See also zeros, ones.
    """
    return to_matrix(np.eye(*_size_args('eye', _given(m, n))))


@builtin('zeros', category=CAT_MATRIX)
def fn_zeros(m=None, n=None):
    """zeros - Matrix of zeros.

    zeros(N) is an N-by-N matrix of zeros.
    zeros(M, N) or zeros([M N]) is an M-by-N matrix of zeros.

    See also ones, eye.
    """
    return to_matrix(np.zeros(_size_args('zeros', _given(m, n))))


@builtin('ones', category=CAT_MATRIX)
def fn_ones(m=None, n=None):
    """ones - Matrix of ones.

    ones(N) is an N-by-N matrix of ones.
    ones(M, N) or ones([M N]) is an M-by-N matrix of ones.

    See also zeros, eye.
    """
    return to_matrix(np.ones(_size_args('ones', _given(m, n))))


@builtin('inv', category=CAT_MATRIX)
def fn_inv(x):
    """inv - Matrix inverse.

    inv(X) is the inverse of the square matrix X. A singular matrix gives
    an error. To solve A*x = b, A\\b is faster and more accurate than
    inv(A)*b.

    See also det.
    """
    _need_square('inv', x)
    try:
        return to_matrix(np.linalg.inv(x))
    except np.linalg.LinAlgError:
        raise MatlabError("Error using inv\nMatrix is singular to working precision.") from None


@builtin('det', category=CAT_MATRIX)
def fn_det(x):
    """det - Determinant.

    det(X) is the determinant of the square matrix X.

    See also inv.
    """
    _need_square('det', x)
    return to_matrix(np.linalg.det(x))


# ---- help texts --------------------------------------------------------------
OPERATORS_HELP = r"""ops - Operators and special characters.

Matrix arithmetic:
    +     Addition                          -     Subtraction, unary minus
    *     Matrix multiplication             ^     Matrix power
    /     Right division: A/B solves X*B = A
    \     Left division:  A\B solves A*X = B
    '     Complex conjugate transpose

Element-wise arithmetic:
    .*    Multiplication (also times(A, B))
    ./    Right division                    .\    Left division
    .^    Power (also power(A, B))
    .'    Transpose (without conjugation)

Comparison (the result is logical: 1 where true, 0 where false):
    ==  equal    ~=  not equal    <  <=  >  >=

Logical:
    &  and    |  or    ~  not    &&  ||  short-circuit and / or (scalars)

Special characters:
    =        Assignment, e.g. x = 2 or A(2,3) = 5
    [ ]      Build a matrix: [1 2; 3 4]
    ,        Separate elements, or statements
    ;        Separate rows, or suppress output at the end of a statement
    :        Range: a:b or a:step:b, e.g. 0:0.5:2
    ( )      Grouping, function arguments and indexing
    %        Comment: the rest of the line is ignored
    ...      Continue the statement on the next line

Indexing (elements are numbered from 1):
    A(i, j)       Element in row i, column j, e.g. A(2,3) + A(1,4)
    A(k)          Element k, counting down the columns: A(2) is A(2,1)
    A(i, :)       Row i;  A(:, j) is column j;  A(:) is a column of all
    A(1:2, [1 3]) Rows 1 to 2 of columns 1 and 3
    end           Last index in that position: A(end), A(end, 1), v(2:end)
    A(A > 2)      The elements for which a condition is true
Subscripts are positive whole numbers or conditions (logical values).

Changing elements:
    A(2,3) = 5    Set an element;  A(:, 1) = 0 sets a whole column
    A(A > 2) = 0  Set the elements for which a condition is true
    v(end+1) = 7  Positions beyond the end make the array grow
    A(2, :) = []  Delete row 2;  v([1 3]) = [] deletes elements 1 and 3

Precedence, from highest to lowest:
    ( )   ^ .^ ' .'   unary + - ~   * / \ .* ./ .\   + -   :
    comparisons   &   |   &&   ||
Operators of equal precedence are evaluated from left to right."""

OPERATOR_SUMMARY = (
    ("+  -  *  /  \\  ^  '", "matrix arithmetic"),
    (".*  ./  .\\  .^  .'", "element-wise arithmetic"),
    ("==  ~=  <  <=  >  >=", "comparison"),
    ("&  |  ~  &&  ||", "logical"),
    ("=  [ ]  ( )", "assignment, matrices, grouping"),
    (",  ;  :  %  ...", "separators, ranges, comments"),
    ("A(i, j)  A(k)  A(:, j)  end", "indexing"),
    ("A(A > 2)  A(2,3) = 5  A(2,:) = []", "conditions, changing, deleting"),
)

# 'help' followed by any of these shows OPERATORS_HELP
OPERATOR_TOPICS = {
    'ops', 'operators', '+', '-', '*', '/', '\\', '^', "'", '.*', './', '.\\', '.^', ".'",
    '==', '~=', '<', '<=', '>', '>=', '&', '|', '~', '&&', '||', '=', '[', ']', '(', ')', ':', '%', '...',
    'plus', 'minus', 'uplus', 'uminus', 'mtimes', 'mrdivide', 'mldivide', 'mpower', 'rdivide', 'ldivide',
    'transpose', 'ctranspose', 'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'and', 'or', 'not', 'colon',
    'horzcat', 'vertcat', 'end', 'paren', 'indexing', 'index', 'subsref', 'subsasgn', 'assignment',
}

_HELP_FIELDS = re.compile(r"\{(APP_NAME|APP_VERSION|SHORT_DECIMALS|LONG_DECIMALS)\}")


def help_text(func):
    """Help of a built-in or console command: its docstring, with parameter fields filled in."""
    doc = inspect.cleandoc(func.__doc__ or '')
    if not doc:  # docstrings are removed when Python runs with -OO
        return f"{func.matlab_names[0]} - No help available."
    return _HELP_FIELDS.sub(lambda m: str(globals()[m.group(1)]), doc)


def help_summary(func):
    """The one-line description from the help's first ("H1") line."""
    return help_text(func).splitlines()[0].partition(' - ')[2]


# =============================================================================
#  Parsing: statement splitting, tokenizer and recursive-descent parser
# =============================================================================
_IDENT_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_DIGITS = '0123456789'
# A trailing '.' belongs to the number only if it does not start an operator: 1./x is 1 ./ x
_NUMBER_RE = re.compile(r"(?:\d+\.\d+|\d+\.(?![*/\\^'])|\d+|\.\d+)(?:[eE][+-]?\d+)?", re.ASCII)
_OPERATORS = ('.^', '.*', './', '.\\', ".'", '==', '~=', '<=', '>=', '&&', '||',
              '+', '-', '*', '/', '\\', '^', '<', '>', '&', '|', '~', ':', '=')


def split_statements(text):
    """Split a command line into (statement, output_suppressed) pairs.

    Statements end at ',', ';' or a newline outside brackets/parentheses.
    '%' starts a comment and '...' continues the statement on the next line.
    """
    statements, buf, depth, i = [], [], 0, 0
    while i < len(text):
        ch = text[i]
        if ch == '%':
            end = text.find('\n', i)
            i = len(text) if end < 0 else end
            continue
        if text.startswith('...', i):
            end = text.find('\n', i)
            i = len(text) if end < 0 else end + 1
            buf.append(' ')
            continue
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth = max(depth - 1, 0)
        if depth == 0 and ch in ',;\n':
            statements.append((''.join(buf).strip(), ch == ';'))
            buf = []
        else:
            buf.append(ch)
        i += 1
    statements.append((''.join(buf).strip(), False))
    return [(s, suppressed) for s, suppressed in statements if s]


def _starts_new_element(text, j):
    """Inside [...], does whitespace followed by text[j:] begin a new element?  ([1 -2] vs [1 - 2])"""
    c = text[j]
    nxt = text[j + 1] if j + 1 < len(text) else ''
    if (c.isascii() and c.isalnum()) or c in '([':
        return True
    if c == '.':
        return nxt != '' and nxt in _DIGITS
    if c in '+-':
        return nxt not in ('', ' ', '\t', '\r', '\n')
    if c == '~':
        return nxt != '='
    return False


def tokenize(text):
    """Turn one statement into tokens (kind, text, value); kind is 'num', 'id', 'op', a bracket, ',', ';' or 'eof'."""
    tokens, stack, i, n = [], [], 0, len(text)

    def prev_ends_operand():
        if not tokens:
            return False
        kind, tok = tokens[-1][0], tokens[-1][1]
        return kind in ('num', 'id', ')', ']') or (kind == 'op' and tok in ("'", ".'"))

    while i < n:
        ch = text[i]
        in_brackets = bool(stack) and stack[-1] == '['
        if ch in ' \t\r\n':
            j, newline = i, False
            while j < n and text[j] in ' \t\r\n':
                newline = newline or text[j] == '\n'
                j += 1
            if in_brackets:
                if newline:
                    tokens.append((';', ';', None))       # a line break inside [...] starts a new row
                elif j < n and prev_ends_operand() and _starts_new_element(text, j):
                    tokens.append((',', ',', None))       # whitespace separates elements
            i = j
            continue
        if ch in _DIGITS or (ch == '.' and i + 1 < n and text[i + 1] in _DIGITS):
            m = _NUMBER_RE.match(text, i)
            tok, i = m.group(), m.end()
            value = float(tok)
            if i < n and text[i] in 'ij' and not (i + 1 < n and (text[i + 1].isalnum() or text[i + 1] == '_')):
                value, tok, i = complex(0, value), tok + text[i], i + 1   # imaginary literal, e.g. 4i
            tokens.append(('num', tok, value))
            continue
        if ch.isascii() and ch.isalpha():
            m = _IDENT_RE.match(text, i)
            tokens.append(('id', m.group(), None))
            i = m.end()
            continue
        if ch in '([':
            stack.append(ch)
            tokens.append((ch, ch, None))
            i += 1
            continue
        if ch in ')]':
            if not stack or stack[-1] != ('(' if ch == ')' else '['):
                raise MatlabError(MSG_UNBALANCED)
            stack.pop()
            tokens.append((ch, ch, None))
            i += 1
            continue
        if ch in ',;':
            tokens.append((ch, ch, None))
            i += 1
            continue
        if ch == "'":
            # Transpose when it directly follows an operand; otherwise it would open a character string
            if prev_ends_operand() and not (in_brackets and text[i - 1] in ' \t'):
                tokens.append(('op', "'", None))
                i += 1
                continue
            raise MatlabError("Error: Character strings are not supported by this calculator.")
        for op in _OPERATORS:
            if text.startswith(op, i):
                tokens.append(('op', op, None))
                i += len(op)
                break
        else:
            raise MatlabError(f"Error: Invalid text character '{ch}'. Check for unsupported symbols "
                              "or non-ASCII characters.")
    if stack:
        raise MatlabError(MSG_UNBALANCED)
    tokens.append(('eof', '', None))
    return tokens


class Parser:
    """Recursive-descent parser using MATLAB operator precedence (lowest first):

        ||   &&   |   &   < <= > >= == ~=   :   + -   * / \\ .* ./ .\\   unary + - ~   ^ .^ ' .'

    All binary operators are left-associative, so 2^3^2 = (2^3)^2 = 64 as in MATLAB.
    The result is a tree of tuples that evaluate() walks.
    """
    _LEVELS = (('||',), ('&&',), ('|',), ('&',), ('==', '~=', '<', '<=', '>', '>='),
               None,  # ':' ranges
               ('+', '-'), ('*', '/', '\\', '.*', './', '.\\'))

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self, offset=0):
        return self.tokens[min(self.pos + offset, len(self.tokens) - 1)]

    def advance(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def at(self, kind):
        return self.peek()[0] == kind

    def at_op(self, *ops):
        tok = self.peek()
        return tok[0] == 'op' and tok[1] in ops

    def expect(self, kind):
        if not self.at(kind):
            self.fail()
        return self.advance()

    def fail(self):
        tok = self.peek()
        if tok[0] == 'eof':
            raise MatlabError("Error: Invalid expression. The statement is incomplete.")
        raise MatlabError(f"Error: Invalid expression near '{tok[1]}'. Check for a missing operator "
                          "or a misplaced delimiter.")

    def parse_statement(self):
        """Returns (target, expression tree).

        target is None for a plain expression, a name for 'name = expression', or
        ('index', name, subscripts) for 'name(subscripts) = expression', e.g. A(2,3) = 5.
        """
        target = None
        if self.at('id') and self._assignment_follows():
            name = self.advance()[1]
            if name == 'end':
                raise MatlabError("Error: 'end' is a keyword and cannot be used as a variable name.")
            if self.at('('):
                self.advance()
                subscripts = [] if self.at(')') else [self._argument()]
                while self.at(','):
                    self.advance()
                    subscripts.append(self._argument())
                self.expect(')')
                target = ('index', name, subscripts)
            else:
                target = name
            self.advance()  # '='
        node = self.parse_expr()
        if self.at_op('='):
            raise MatlabError("Error: Invalid use of '='. Assignments have the form name = expression\n"
                              "or name(subscripts) = expression.")
        if not self.at('eof'):
            self.fail()
        return target, node

    def _assignment_follows(self):
        """Is the statement 'name = ...' or 'name(...) = ...'?"""
        nxt = self.peek(1)
        if nxt[0] == 'op' and nxt[1] == '=':
            return True
        if nxt[0] != '(':
            return False
        depth = 0
        for j in range(self.pos + 1, len(self.tokens)):
            kind = self.tokens[j][0]
            if kind in ('(', '['):
                depth += 1
            elif kind in (')', ']'):
                depth -= 1
                if depth == 0:
                    after = self.tokens[j + 1]
                    return after[0] == 'op' and after[1] == '='
        return False

    def parse_expr(self):
        return self._binary(0)

    def _binary(self, level):
        if level == len(self._LEVELS):
            return self._unary()
        ops = self._LEVELS[level]
        if ops is None:
            return self._range(level)
        node = self._binary(level + 1)
        while self.at_op(*ops):
            op = self.advance()[1]
            node = ('bin', op, node, self._binary(level + 1))
        return node

    def _range(self, level):
        first = self._binary(level + 1)
        if not self.at_op(':'):
            return first
        self.advance()
        second = self._binary(level + 1)
        if not self.at_op(':'):
            return ('range', first, None, second)
        self.advance()
        return ('range', first, second, self._binary(level + 1))

    def _unary(self):
        if self.at_op('-', '+', '~'):
            op = self.advance()[1]
            return ('un', op, self._unary())
        return self._power()

    def _power(self):
        node = self._primary()
        while True:
            if self.at_op('^', '.^'):
                op = self.advance()[1]
                node = ('bin', op, node, self._power_operand())
            elif self.at_op("'", ".'"):
                node = ('un', self.advance()[1], node)
            else:
                return node

    def _power_operand(self):  # allows 2^-1 and A.^-B
        if self.at_op('-', '+', '~'):
            op = self.advance()[1]
            return ('un', op, self._power_operand())
        return self._primary()

    def _primary(self):
        tok = self.peek()
        if tok[0] == 'num':
            self.advance()
            return ('num', tok[2])
        if tok[0] == 'id':
            self.advance()
            if tok[1] == 'end':  # last index, as in A(end); checked when evaluated
                return ('end',)
            if not self.at('('):
                return ('name', tok[1], None)
            self.advance()  # function arguments, or subscripts if the name is a variable
            args = []
            if not self.at(')'):
                args.append(self._argument())
                while self.at(','):
                    self.advance()
                    args.append(self._argument())
            self.expect(')')
            return ('name', tok[1], args)
        if tok[0] == '(':
            self.advance()
            node = self.parse_expr()
            self.expect(')')
            return node
        if tok[0] == '[':
            self.advance()
            return self._matrix()
        self.fail()

    def _argument(self):
        nxt = self.peek(1)
        if self.at_op(':') and nxt[0] in (',', ')'):  # a lone ':' selects a whole row/column, as in A(:, 1)
            self.advance()
            return ('colon_all',)
        return self.parse_expr()

    def _matrix(self):
        rows, row = [], []
        while True:
            if self.at(']'):
                self.advance()
                break
            if self.at(';'):
                self.advance()
                rows.append(row)
                row = []
                continue
            if self.at(','):
                self.advance()
                continue
            row.append(self.parse_expr())
            if not (self.at(',') or self.at(';') or self.at(']')):
                self.fail()
        rows.append(row)
        return ('matrix', [r for r in rows if r])


def _subscript(node, extent, variables, position, check_bounds=True):
    """Evaluate one subscript. Returns (0-based positions, kind, shape of the subscript).

    kind is 'colon' for a lone ':', 'logical' for a true/false mask such as A > 2, and
    'numeric' for positions. extent is the size of the indexed dimension (the number of
    elements for A(k)); it is also the value of 'end'. position is 1 or 2 for A(i, j)
    and None for A(k). Assignments pass check_bounds=False because they may grow A.
    """
    if node[0] == 'colon_all':
        return np.arange(extent), 'colon', None
    sub = evaluate(node, variables, end_value=extent)
    if sub.dtype == bool:  # logical indexing: the positions where the mask is true
        positions = np.flatnonzero(sub.ravel(order='F'))
        last = positions[-1] + 1 if positions.size else 0
        kind = 'logical'
    else:
        vals = sub.ravel(order='F')
        if (np.iscomplexobj(vals) or not np.all(np.isfinite(vals)) or np.any(vals < 1)
                or np.any(vals != np.floor(vals))):
            where = f"Index in position {position} is invalid. " if position else ""
            raise MatlabError(f"{where}Array indices must be positive integers or logical values.")
        last = vals.max() if vals.size else 0
        if last > MAX_ELEMENTS:
            raise MatlabError("Error: Requested array exceeds the maximum allowed size.")
        positions = vals.astype(int) - 1
        kind = 'numeric'
    if check_bounds and last > extent:
        if position:
            raise MatlabError(f"Index in position {position} exceeds array bounds. Index must not exceed {extent}.")
        raise MatlabError(f"Index exceeds the number of array elements. Index must not exceed {extent}.")
    return positions, kind, sub.shape


def index_matrix(value, arg_nodes, variables):
    """MATLAB indexing: A(i, j) selects rows i and columns j; A(k) counts elements down the columns.

    Subscripts are 1-based and may be scalars, vectors, ranges, ':', use 'end', or be
    conditions (logical masks) such as A(A > 2).
    """
    if len(arg_nodes) > 2:
        raise MatlabError("Error: Only one or two subscripts are supported, as in A(k) or A(i, j).")
    if len(arg_nodes) == 2:
        rows, _, _ = _subscript(arg_nodes[0], value.shape[0], variables, 1)
        cols, _, _ = _subscript(arg_nodes[1], value.shape[1], variables, 2)
        return to_matrix(value[np.ix_(rows, cols)])
    idx, kind, shape = _subscript(arg_nodes[0], value.size, variables, None)
    picked = value.ravel(order='F')[idx]
    if kind == 'colon':  # A(:) is a column of all elements
        return to_matrix(picked.reshape(-1, 1))
    if kind == 'logical':  # a mask acts like find(mask): a row for a row mask, otherwise a column
        shape = (1, idx.size) if shape[0] == 1 else (idx.size, 1)
    if 1 in shape and 1 in value.shape and value.size > 1:  # vector(vector) keeps the orientation of A
        return to_matrix(picked.reshape((1, -1) if value.shape[0] == 1 else (-1, 1)))
    return to_matrix(picked.reshape(shape, order='F'))  # otherwise the result has the subscript's shape


def assign_index(target, arg_nodes, value, variables):
    """A(subscripts) = B, following MATLAB.

    - Positions beyond the end make A grow; new elements are 0.
    - A scalar B is copied into every selected element; otherwise B must have one
      element per selected position (A(2, :) = [1 2 3 4]).
    - B = [] deletes the selected elements, rows or columns.
    - A logical A stays logical and a real A becomes complex if B is complex.
    target is the current value of A, or None if A does not exist yet.
    """
    if not arg_nodes:
        raise MatlabError("Error: An indexed assignment needs a subscript, as in A(2) = 5.")
    if len(arg_nodes) > 2:
        raise MatlabError("Error: Only one or two subscripts are supported, as in A(k) or A(i, j).")
    if target is None:  # the assignment creates the variable
        target = np.zeros((0, 0), dtype=bool if value.dtype == bool else float)
    if value.shape == (0, 0):
        return _delete_elements(target, arg_nodes, variables)
    if target.dtype == bool and value.dtype != bool:
        if np.iscomplexobj(value) or np.any(np.isnan(value)):
            raise MatlabError("Error: Only real numbers other than NaN can be stored in a logical array.")
        value = value != 0
    elif np.iscomplexobj(value) and not np.iscomplexobj(target):
        target = target.astype(complex)
    if len(arg_nodes) == 1:
        return _assign_linear(target, arg_nodes[0], value, variables)
    return _assign_rows_cols(target, arg_nodes, value, variables)


def _assign_linear(target, node, value, variables):
    """A(k) = B, with k counting elements down the columns."""
    idx, _, _ = _subscript(node, target.size, variables, None, check_bounds=False)
    if value.size not in (1, idx.size):
        raise MatlabError("Unable to perform assignment because the left and right sides have a "
                          "different number of elements.")
    shape = target.shape
    needed = int(idx.max()) + 1 if idx.size else 0
    if needed > target.size:  # grow: rows (and scalars) get longer, columns taller
        if shape == (0, 0) or shape[0] == 1:
            shape = (1, needed)
        elif shape[1] == 1:
            shape = (needed, 1)
        else:
            raise MatlabError("Attempt to grow array along ambiguous dimension.")
    flat = np.zeros(shape[0] * shape[1], dtype=target.dtype)
    flat[:target.size] = target.ravel(order='F')
    flat[idx] = value.flat[0] if value.size == 1 else value.ravel(order='F')
    return to_matrix(flat.reshape(shape, order='F'))


def _assign_rows_cols(target, arg_nodes, value, variables):
    """A(i, j) = B."""
    r, c = target.shape
    rows, rkind, _ = _subscript(arg_nodes[0], r, variables, 1, check_bounds=False)
    cols, ckind, _ = _subscript(arg_nodes[1], c, variables, 2, check_bounds=False)
    # ':' on a dimension that is still empty, as in B(:, 1) = [1; 2; 3] for a new B, takes its size from B
    if rkind == 'colon' and r == 0:
        rows = np.arange(value.shape[0] if ckind == 'colon' and c == 0 else _fitting_count(value, cols.size))
    if ckind == 'colon' and c == 0:
        cols = np.arange(value.shape[1] if rkind == 'colon' and r == 0 else _fitting_count(value, rows.size))
    m, k = rows.size, cols.size
    if value.size == 1:
        block = value.flat[0]
    elif value.size == m * k and [d for d in value.shape if d != 1] == [d for d in (m, k) if d != 1]:
        block = value.reshape(m, k)  # same size, or a row stored into a column (and vice versa)
    else:
        raise MatlabError(f"Unable to perform assignment because the size of the left side is {m}-by-{k} "
                          f"and the size of the right side is {value.shape[0]}-by-{value.shape[1]}.")
    new_r = max(r, int(rows.max()) + 1 if m else 0)
    new_c = max(c, int(cols.max()) + 1 if k else 0)
    if new_r * new_c > MAX_ELEMENTS:
        raise MatlabError("Error: Requested array exceeds the maximum allowed size.")
    result = np.zeros((new_r, new_c), dtype=target.dtype)
    result[:r, :c] = target
    if m and k:
        result[np.ix_(rows, cols)] = block
    return to_matrix(result)


def _fitting_count(value, other_count):
    """Length a ':' takes on an empty dimension so that value fits next to other_count positions."""
    if other_count and value.size % other_count == 0:
        return value.size // other_count
    return max(value.shape)


def _delete_elements(target, arg_nodes, variables):
    """A(...) = []: delete elements, or whole rows or columns."""
    if len(arg_nodes) == 1:
        idx, _, _ = _subscript(arg_nodes[0], target.size, variables, None)
        if idx.size == 0:
            return target
        keep = np.ones(target.size, dtype=bool)
        keep[idx] = False
        remaining = target.ravel(order='F')[keep]
        if target.shape[1] == 1 and target.shape[0] > 1:  # a column stays a column
            return to_matrix(remaining.reshape(-1, 1))
        return to_matrix(remaining.reshape(1, -1))  # a row; a matrix becomes a row, as in MATLAB
    r, c = target.shape
    rows, _, _ = _subscript(arg_nodes[0], r, variables, 1)
    cols, _, _ = _subscript(arg_nodes[1], c, variables, 2)
    if set(cols.tolist()) == set(range(c)):    # A(i, :) = [] deletes rows
        return to_matrix(np.delete(target, rows, axis=0))
    if set(rows.tolist()) == set(range(r)):    # A(:, j) = [] deletes columns
        return to_matrix(np.delete(target, cols, axis=1))
    raise MatlabError("A null assignment can have only one non-colon index.\n"
                      "(Delete whole rows, as in A(2, :) = [], or whole columns, as in A(:, 2) = [].)")


def evaluate(node, variables, end_value=None):
    """Evaluate a parse tree. Variables shadow built-in functions and constants, as in MATLAB.

    end_value is what 'end' stands for while a subscript is evaluated, e.g. in A(end - 1).
    """
    kind = node[0]
    if kind == 'num':
        return to_matrix(node[1])
    if kind == 'end':
        if end_value is None:
            raise MatlabError("Error: 'end' can only be used inside the subscript of a variable, as in A(end).")
        return to_matrix(end_value)
    if kind == 'colon_all':
        raise MatlabError("Error: A colon on its own can only be used as a subscript, as in A(:, 1).")
    if kind == 'name':
        name, args = node[1], node[2]
        if name in variables:
            if not args:  # A and A() are the whole variable
                return variables[name]
            return index_matrix(variables[name], args, variables)
        if name in FUNCTIONS:
            func, lo, hi = FUNCTIONS[name]
            args = args or []
            if not lo <= len(args) <= hi:
                problem = 'Not enough' if len(args) < lo else 'Too many'
                raise MatlabError(f"Error using {name}\n{problem} input arguments.")
            return func(*[to_double(evaluate(a, variables, end_value)) for a in args])
        raise MatlabError(f"Unrecognized function or variable '{name}'.")
    if kind == 'un':
        return UNARY_OPS[node[1]](evaluate(node[2], variables, end_value))
    if kind == 'bin':
        op = node[1]
        left = evaluate(node[2], variables, end_value)
        if op in ('&&', '||'):  # short-circuit
            lv = _logical_scalar(left, op)
            if lv == (op == '||'):
                return to_matrix(lv)
            return to_matrix(_logical_scalar(evaluate(node[3], variables, end_value), op))
        return BINARY_OPS[op](left, evaluate(node[3], variables, end_value))
    if kind == 'range':
        start = evaluate(node[1], variables, end_value)
        step = to_matrix(1.0) if node[2] is None else evaluate(node[2], variables, end_value)
        return op_colon(start, step, evaluate(node[3], variables, end_value))
    if kind == 'matrix':
        rows = []
        for row in node[1]:
            parts = [m for m in (evaluate(e, variables, end_value) for e in row) if m.size]
            if parts:
                try:
                    rows.append(np.hstack(parts))
                except ValueError:
                    raise MatlabError("Error using horzcat\nDimensions of arrays being concatenated "
                                      "are not consistent.") from None
        if not rows:
            return np.zeros((0, 0))
        try:
            return to_matrix(np.vstack(rows))
        except ValueError:
            raise MatlabError("Error using vertcat\nDimensions of arrays being concatenated "
                              "are not consistent.") from None
    raise MatlabError(f"Error: Unsupported expression ({kind}).")


# =============================================================================
#  Engine: runs command lines and formats results (independent of the GUI)
# =============================================================================
CLEAR_SCREEN = object()  # returned by MatlabEngine.execute() for 'clc'
EXIT_APP = object()      # returned by MatlabEngine.execute() for 'exit' / 'quit'

COMMANDS = {}  # command name -> (unbound MatlabEngine method, maximum number of arguments or None, argument regex)


def console_command(*names, max_args=0, arg_pattern=r"\S+"):
    """Register a MatlabEngine method as the console command(s) `names`."""
    def register(method):
        for name in names:
            COMMANDS[name] = (method, max_args, re.compile(arg_pattern))
        method.matlab_names, method.help_category = names, CAT_COMMANDS
        return method
    return register


def help_overview():
    """Text of 'help' without arguments: every command, operator and function with its summary."""
    unique = {id(entry[0]): entry[0] for entry in list(COMMANDS.values()) + list(FUNCTIONS.values())}
    by_category = {}
    for func in unique.values():
        by_category.setdefault(func.help_category, []).append((', '.join(func.matlab_names), help_summary(func)))
    width = max(len(label) for entries in by_category.values() for label, _ in entries)
    op_width = max(len(ops) for ops, _ in OPERATOR_SUMMARY)
    lines = [f"{APP_NAME} {APP_VERSION} - commands, operators and functions", ""]
    for category in list(HELP_CATEGORIES) + [c for c in by_category if c not in HELP_CATEGORIES]:
        entries = sorted(by_category.get(category, []), key=lambda entry: entry[0].lower())
        if entries:
            lines.append(category)
            lines += [f"  {label:<{width}}  - {summary}" for label, summary in entries]
            lines.append("")
        if category == CAT_COMMANDS:
            lines.append("Operators and special characters (help ops)")
            lines += [f"  {ops:<{op_width}}  {meaning}" for ops, meaning in OPERATOR_SUMMARY]
            lines.append("")
    lines.append('Type "help <name>" for details, e.g. help round.')
    return "\n".join(lines)


def _indent(text):
    return "\n".join(("  " + line).rstrip() for line in text.splitlines()) + "\n"


class MatlabEngine:
    def __init__(self):
        self.format_mode = 'short'  # 'short' or 'long'
        self.variables = {}
        self.last_error = None      # message of the error that stopped the last command line, if any

    # ---- execution ---------------------------------------------------------
    def execute(self, line):
        """Run one command line. Returns output chunks (str), CLEAR_SCREEN and/or EXIT_APP.

        Like MATLAB, the remaining statements on the line are skipped after an error or 'exit'.
        """
        output, error = [], None
        try:
            for text, suppressed in split_statements(line):
                output.extend(self._run_statement(text, suppressed))
                if output and output[-1] is EXIT_APP:
                    break
        except MatlabError as err:
            error = str(err)
        except np.linalg.LinAlgError as err:
            error = f"Error: {err}."
        except MemoryError:
            error = "Error: Out of memory."
        except RecursionError:
            error = "Error: The expression is nested too deeply."
        except Exception as err:  # unexpected NumPy/Python failure: report it, keep the app alive
            error = f"Error: {err}"
        self.last_error = error
        if error is not None:
            output.append(error + "\n")
        return output

    def _run_statement(self, text, suppressed):
        result = self._run_command(text, suppressed)
        if result is not None:
            return result
        target, node = Parser(tokenize(text)).parse_statement()
        with np.errstate(all='ignore'):  # 1/0 -> Inf and 0/0 -> NaN, as in MATLAB
            value = evaluate(node, self.variables)
            if isinstance(target, tuple):  # A(subscripts) = value changes elements of A
                _, target, subscripts = target
                value = assign_index(self.variables.get(target), subscripts, value, self.variables)
                self.variables[target] = value
            elif target is None and node[0] == 'name' and node[2] is None and node[1] in self.variables:
                target = node[1]  # typing a variable's name shows it without changing 'ans'
            else:
                target = target or 'ans'
                self.variables[target] = value
        return [] if suppressed else [f"{target} =\n{self.format_value(value)}\n"]

    # ---- console commands ----------------------------------------------------
    # Command syntax, e.g. 'format long'. As for the built-in functions, each
    # command's docstring is its help text, shown by 'help <name>'.
    def _run_command(self, text, suppressed=False):
        """Run text as a console command. Returns None if it is not one."""
        words = text.split()
        entry = COMMANDS.get(words[0])
        if entry is None:
            return None
        method, max_args, arg_re = entry
        args = words[1:]
        if args and args[0].startswith('=') and not args[0].startswith('==') and (len(args) > 1 or args[0] != '='):
            return None  # 'help = 3' or 'clear =5' assign a variable; 'help =' still asks about '='
        if (max_args is not None and len(args) > max_args) or not all(arg_re.fullmatch(a) for a in args):
            return None  # not command syntax, so the text is evaluated as an expression
        return method(self, args, suppressed)

    @console_command('clc')
    def cmd_clc(self, args, suppressed):
        """clc - Clear the Command Window.

        clc removes all text from the Command Window. The variables in the
        workspace are kept.

        See also clear.
        """
        return [CLEAR_SCREEN]

    @console_command('clear', max_args=None, arg_pattern=r"[A-Za-z][A-Za-z0-9_]*")
    def cmd_clear(self, args, suppressed):
        """clear - Remove variables from the workspace.

        clear or clear all removes all variables.
        clear X Y removes only the variables X and Y.

        See also whos, clc.
        """
        if not args or 'all' in args:
            self.variables.clear()
        else:
            for var in args:
                self.variables.pop(var, None)
        return []

    @console_command('exit', 'quit')
    def cmd_exit(self, args, suppressed):
        """exit - Close {APP_NAME}.

        exit (or quit) closes the window and ends the program. Statements after
        exit on the same line are not run.
        """
        return [EXIT_APP]

    @console_command('format', max_args=1, arg_pattern=r"[A-Za-z]+")
    def cmd_format(self, args, suppressed):
        """format - Set how numbers are displayed.

        format short    shows up to {SHORT_DECIMALS} decimals (the default).
        format long     shows up to {LONG_DECIMALS} decimals.
        format          restores the default, format short.

        Only the decimals a result needs are shown, the same number for all
        elements of a matrix. Very small or very large numbers are shown in
        scientific notation, e.g. 1e-05.
        """
        mode = args[0].lower() if args else 'short'
        if mode not in ('short', 'long'):
            raise MatlabError(f"Error using format\nUnsupported format '{args[0]}'. "
                              "Use 'format short' or 'format long'.")
        self.format_mode = mode
        return []

    @console_command('help', max_args=None)
    def cmd_help(self, args, suppressed):
        """help - Show help for commands, functions and operators.

        help          lists the available commands, operators and functions.
        help NAME     shows the help of NAME, e.g. help round.
        help ops      describes the operators and special characters;
                      help .* or help + show the same.

        As in MATLAB, the help of each command and function is the comment
        block at the start of its code: the docstring of the Python function
        that implements it in matlab_calc.py.
        """
        if not args:
            return [_indent(help_overview())]
        return [_indent(self._help_topic(topic)) for topic in args]

    def _help_topic(self, topic):
        for key in (topic, topic.lower()):  # help ROUND works too
            if key in COMMANDS:
                return help_text(COMMANDS[key][0])
            if key in FUNCTIONS:
                return help_text(FUNCTIONS[key][0])
            if key in OPERATOR_TOPICS:
                return OPERATORS_HELP
        if topic in self.variables:
            return f"{topic} is a variable in the workspace. Type {topic} to see its value."
        return f"{topic} not found.\n\nType help to list the available commands, operators and functions."

    @console_command('version')
    def cmd_version(self, args, suppressed):
        """version - Show the version number.

        version displays the name and version of the program, for example
        {APP_NAME} {APP_VERSION}. The version history is in README.md.
        """
        return [] if suppressed else [f"{APP_NAME} {APP_VERSION}\n"]

    @console_command('whos')
    def cmd_whos(self, args, suppressed):
        """whos - List the variables in the workspace.

        whos shows the name, size and class of every variable.

        See also clear.
        """
        if not self.variables:
            return ["  Workspace is empty.\n"]
        lines = [f"  {'Name':<10} {'Size':<10} {'Class':<10}", "  " + "-" * 32]
        for name in sorted(self.variables, key=str.lower):
            v = self.variables[name]
            cls = "logical" if v.dtype == bool else "double (complex)" if np.iscomplexobj(v) else "double"
            lines.append(f"  {name:<10} {f'{v.shape[0]}x{v.shape[1]}':<10} {cls:<10}")
        return ["\n".join(lines) + "\n"]

    # ---- number formatting -------------------------------------------------
    @property
    def max_decimals(self):
        return LONG_DECIMALS if self.format_mode == 'long' else SHORT_DECIMALS

    @staticmethod
    def _fixed_decimals(v, max_decimals):
        """Decimals needed to show v in fixed-point notation, capped at max_decimals."""
        if not math.isfinite(v) or v == 0:
            return 0
        if round(v, 12) == round(v, 14):  # hide IEEE-754 noise, e.g. 2.4*3 -> 7.199999999999999
            v = round(v, 12)
        if v == 0:
            return 0
        significant = max(0, 15 - int(math.floor(math.log10(abs(v)))))  # no digits beyond double precision
        frac = f"{v:.{min(max_decimals, significant)}f}".partition('.')[2]
        return len(frac.rstrip('0'))

    @staticmethod
    def _sci_decimals(v, max_decimals):
        if not math.isfinite(v) or v == 0:
            return 0
        mantissa = f"{v:.{max_decimals}e}".partition('e')[0]
        return len(mantissa.partition('.')[2].rstrip('0'))

    @staticmethod
    def _format_real(v, decimals, sci):
        if math.isnan(v):
            return 'NaN'
        if math.isinf(v):
            return 'Inf' if v > 0 else '-Inf'
        if sci:
            return '0' if v == 0 else f"{v:.{decimals}e}"
        if round(v, decimals) == 0:
            v = 0.0  # avoid '-0'
        return f"{v:.{decimals}f}"

    def format_number(self, z, decimals, sci):
        if isinstance(z, complex) or np.iscomplexobj(z):
            z = complex(z)
            sign = '-' if z.imag < 0 else '+'
            return (f"{self._format_real(z.real, decimals, sci)} {sign} "
                    f"{self._format_real(abs(z.imag), decimals, sci)}i")
        return self._format_real(float(z), decimals, sci)

    def format_value(self, value):
        a = to_matrix(value)
        rows, cols = a.shape
        kind = 'logical' if a.dtype == bool else 'double'
        if a.size == 0:
            return "    []" if (rows, cols) == (0, 0) and kind == 'double' else f"    {rows}x{cols} empty {kind} matrix"
        if a.size > MAX_DISPLAY_ELEMENTS:
            return (f"    {rows}x{cols} {kind} matrix (too large to display; "
                    "end the statement with ';' to suppress output)")
        if kind == 'logical':  # labelled as in MATLAB, e.g. "1x4 logical array"
            label = "logical" if a.size == 1 else f"{rows}x{cols} logical array"
            return f"  {label}\n" + self._format_numbers(a.astype(float))
        return self._format_numbers(a)

    def _format_numbers(self, a):
        rows, cols = a.shape
        max_dec = self.max_decimals
        parts = np.concatenate([a.real.ravel(), a.imag.ravel()]) if np.iscomplexobj(a) else a.ravel()
        parts = [float(p) for p in parts]
        nonzero = [abs(p) for p in parts if math.isfinite(p) and p != 0]
        hidden = [p for p in nonzero if round(p, max_dec) == 0]  # would print as 0
        # Scientific notation when everything is tiny or something is huge; mixed sizes keep fixed point
        sci = bool(nonzero) and (len(hidden) == len(nonzero) or max(nonzero) >= SCI_THRESHOLD)
        if sci:
            decimals = max(self._sci_decimals(p, max_dec) for p in parts)
        elif hidden:
            decimals = max_dec  # show 0.0000 rather than a misleading exact 0
        else:
            decimals = max(self._fixed_decimals(p, max_dec) for p in parts)
        grid = [[self.format_number(a[r, c], decimals, sci) for c in range(cols)] for r in range(rows)]
        widths = [max(len(grid[r][c]) for r in range(rows)) for c in range(cols)]
        return "\n".join("    " + "   ".join(grid[r][c].rjust(widths[c]) for c in range(cols))
                         for r in range(rows))


# =============================================================================
#  Application icon
# =============================================================================
def rounded_icon_image(path, margin=APP_ICON_MARGIN, rounding=APP_ICON_ROUNDING):
    """Pillow image of the icon shrunk by `margin` per side and with rounded corners.

    macOS draws the Dock icon edge to edge, so a full-bleed square picture looks bigger
    than native icons, which keep about 10 % transparent padding and rounded corners.
    """
    from PIL import Image, ImageChops, ImageDraw, ImageOps
    lanczos = getattr(Image, 'Resampling', Image).LANCZOS
    margin, rounding = _clamp_icon_params(margin, rounding)
    src = Image.open(path).convert('RGBA')
    size = max(src.size)
    inner = max(1, round(size * (1 - 2 * margin)))
    body = ImageOps.fit(src, (inner, inner), lanczos)
    ss = 4  # draw the corner mask 4x larger and scale it down for smooth edges
    mask = Image.new('L', (inner * ss, inner * ss), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, inner * ss - 1, inner * ss - 1),
                                           radius=round(rounding * inner * ss), fill=255)
    mask = mask.resize((inner, inner), lanczos)
    body.putalpha(ImageChops.multiply(body.getchannel('A'), mask))
    icon = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    offset = (size - inner) // 2
    icon.paste(body, (offset, offset))
    return icon


def _clamp_icon_params(margin, rounding):
    return min(max(float(margin), 0.0), 0.45), min(max(float(rounding), 0.0), 0.5)


def _rounded_icon_tk(root, path, margin, rounding):
    """Fallback without Pillow: nearest-neighbour scaling and hard-edged corners."""
    margin, rounding = _clamp_icon_params(margin, rounding)
    src = tk.PhotoImage(master=root, file=path)
    size = max(src.width(), src.height())
    scale = Fraction(1 - 2 * margin).limit_denominator(16)
    scaled = src.zoom(scale.numerator).subsample(scale.denominator) if scale != 1 else src
    inner = min(scaled.width(), scaled.height())
    x0, y0 = (scaled.width() - inner) // 2, (scaled.height() - inner) // 2
    off = (size - inner) // 2
    icon = tk.PhotoImage(master=root, width=size, height=size)  # starts fully transparent
    icon.tk.call(icon, 'copy', scaled, '-from', x0, y0, x0 + inner, y0 + inner,
                 '-to', off, off, '-compositingrule', 'set')
    radius = rounding * inner
    last = inner - 1
    for dy in range(int(math.ceil(radius))):
        for dx in range(int(math.ceil(radius))):
            if math.hypot(max(0.0, radius - dx - 0.5), max(0.0, radius - dy - 0.5)) > radius:
                for x, y in ((dx, dy), (last - dx, dy), (dx, last - dy), (last - dx, last - dy)):
                    icon.tk.call(icon, 'transparency', 'set', off + x, off + y, 1)
    return icon


def load_app_icon(root, path, margin=APP_ICON_MARGIN, rounding=APP_ICON_ROUNDING):
    """Tk image for the window/Dock icon, or None if the file is missing or unreadable."""
    if not os.path.isfile(path):
        return None
    try:
        from PIL import ImageTk
        return ImageTk.PhotoImage(rounded_icon_image(path, margin, rounding), master=root)
    except Exception:  # Pillow missing or unable to read the file: use plain Tk
        pass
    try:
        return _rounded_icon_tk(root, path, margin, rounding)
    except tk.TclError:
        return None


def set_macos_app_name(name):
    """Best effort: show `name` instead of 'Python' in the macOS menu bar.

    Needs PyObjC (pip install pyobjc-framework-Cocoa); does nothing without it or on
    other platforms. Must run before the Tk root window is created.
    """
    if sys.platform != 'darwin':
        return
    try:
        from Foundation import NSBundle
        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        if info is not None:
            info['CFBundleName'] = name
    except Exception:
        pass


def pick_font_family(root, candidates=FONT_FAMILIES):
    """First installed font from candidates, else Tk's default monospace font.

    A missing family (e.g. Consolas on a Mac without MS Office) can silently fall back to a
    proportional font, which breaks the column alignment of matrices.
    """
    installed = {f.lower(): f for f in tkfont.families(root)}
    for family in candidates:
        if family.lower() in installed:
            return installed[family.lower()]
    return tkfont.Font(root=root, font='TkFixedFont').actual('family')


# =============================================================================
#  GUI
# =============================================================================
# Runs in place of the text widget's own command. It is written in Tcl on purpose: an
# error raised inside a Python command called from Tcl (e.g. asking for "sel.first" when
# nothing is selected) is stored by tkinter and re-raised by mainloop(), even when the
# caller catches it. That crashed version 0.1.1 on the first key press.
_TEXT_GUARD_TCL = r'''
proc ::matlab_calc_text_guard {raw cmd args} {
    if {$cmd in {insert delete replace} && [llength $args] > 0
            && [$raw compare [lindex $args 0] < input_start]} {
        return
    }
    return [$raw $cmd {*}$args]
}
'''


class ConsoleText(scrolledtext.ScrolledText):
    """ScrolledText whose text before the 'input_start' mark is read-only for the user.

    Every insert/delete made by Tk's own bindings (typing, Backspace, Ctrl+H, Ctrl+T, cut,
    paste, middle-click, ...) passes through the Tcl guard above, which drops edits that
    would touch the read-only part. The app writes at the end of the buffer and uses
    clear() for 'clc'.
    """

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self._raw = self._w + '_raw'
        self.tk.eval(_TEXT_GUARD_TCL)
        self.tk.call('rename', self._w, self._raw)
        self.tk.call('interp', 'alias', '', self._w, '', '::matlab_calc_text_guard', self._raw)
        self.mark_set('input_start', '1.0')
        self.mark_gravity('input_start', tk.LEFT)

    def clear(self):
        self.tk.call(self._raw, 'delete', '1.0', 'end')

    def destroy(self):
        self.tk.call('rename', self._w, '')          # remove the guard
        self.tk.call('rename', self._raw, self._w)   # give the widget its own command back
        super().destroy()


class MatlabConsoleApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

        # Window / macOS Dock icon, shrunk and rounded so it matches native icons
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), APP_ICON_FILE)
        self.app_icon = load_app_icon(root, icon_path, APP_ICON_MARGIN, APP_ICON_ROUNDING)
        if self.app_icon is not None:
            try:
                self.root.iconphoto(True, self.app_icon)
            except tk.TclError:
                pass

        self.engine = MatlabEngine()
        self.history = []
        self.history_idx = 0

        family = pick_font_family(root)
        self.output_area = ConsoleText(
            root, wrap=tk.WORD, font=(family, FONT_SIZE), bg=BACKGROUND_COLOR, fg=TEXT_COLOR,
            insertbackground=CURSOR_COLOR, undo=False
        )
        self.output_area.pack(padx=WINDOW_PADDING, pady=WINDOW_PADDING, fill=tk.BOTH, expand=True)
        self.output_area.tag_configure("prompt", foreground=PROMPT_COLOR, font=(family, FONT_SIZE, "bold"))

        self.output_area.bind("<Return>", self.process_command)
        self.output_area.bind("<KP_Enter>", self.process_command)
        self.output_area.bind("<Up>", self.history_up)
        self.output_area.bind("<Down>", self.history_down)
        self.output_area.bind("<Left>", self.on_left)
        self.output_area.bind("<Home>", self.on_home)
        self.output_area.bind("<Key>", self.on_key)
        self.output_area.bind("<<Paste>>", self.on_paste)

        self.write_output(
            f"{APP_NAME} {APP_VERSION} - MATLAB Syntax Calculator\n"
            "Type help to list commands, operators and functions; help <name> for details.\n"
            + "-" * 72 + "\n"
        )
        self.show_prompt()
        self.output_area.focus_set()

    def show_prompt(self):
        self.output_area.insert(tk.END, PROMPT, "prompt")
        self.output_area.mark_set("input_start", "end-1c")
        self.output_area.mark_set(tk.INSERT, tk.END)
        self.output_area.see(tk.END)

    def write_output(self, text):
        self.output_area.insert(tk.END, text + "\n")
        self.output_area.see(tk.END)

    def _selection_in_readonly(self):
        try:
            return self.output_area.compare("sel.first", "<", "input_start")
        except tk.TclError:
            return False

    # ---- key handling (edits of the read-only part are blocked by ConsoleText) ----
    def on_left(self, event):
        if self.output_area.compare(tk.INSERT, "<=", "input_start"):
            return "break"

    def on_home(self, event):
        self.output_area.mark_set(tk.INSERT, "input_start")
        return "break"

    def on_paste(self, event):
        if self._selection_in_readonly():
            self.output_area.tag_remove(tk.SEL, "1.0", tk.END)
        if self.output_area.compare(tk.INSERT, "<", "input_start"):
            self.output_area.mark_set(tk.INSERT, tk.END)

    def on_key(self, event):
        # Allow copy/select-all shortcuts anywhere in the buffer (Ctrl on Windows/Linux, Cmd on macOS)
        if (event.state & 0x4 or event.state & 0x8) and event.keysym.lower() in ('c', 'a'):
            return
        if event.keysym in (
            'Shift_L', 'Shift_R', 'Control_L', 'Control_R',
            'Alt_L', 'Alt_R', 'Meta_L', 'Meta_R', 'Super_L', 'Super_R',
            'Left', 'Right', 'Up', 'Down', 'Home', 'End', 'Prior', 'Next', 'Escape'
        ):
            return
        # Typing while the cursor sits in old output continues at the prompt instead
        if self._selection_in_readonly():
            self.output_area.tag_remove(tk.SEL, "1.0", tk.END)
        if self.output_area.compare(tk.INSERT, "<", "input_start"):
            self.output_area.mark_set(tk.INSERT, tk.END)

    # ---- history -------------------------------------------------------------
    def _replace_current_input(self, text):
        self.output_area.delete("input_start", "end-1c")
        self.output_area.insert("input_start", text)
        self.output_area.mark_set(tk.INSERT, tk.END)
        self.output_area.see(tk.END)

    def history_up(self, event):
        if self.history and self.history_idx > 0:
            self.history_idx -= 1
            self._replace_current_input(self.history[self.history_idx])
        return "break"

    def history_down(self, event):
        if self.history_idx < len(self.history) - 1:
            self.history_idx += 1
            self._replace_current_input(self.history[self.history_idx])
        else:
            self.history_idx = len(self.history)
            self._replace_current_input("")
        return "break"

    # ---- execution -------------------------------------------------------------
    def process_command(self, event=None):
        raw_cmd = self.output_area.get("input_start", "end-1c").strip()
        self.output_area.mark_set(tk.INSERT, tk.END)
        self.output_area.insert(tk.END, "\n")

        if raw_cmd:
            self.history.append(raw_cmd)
            self.history_idx = len(self.history)
            if not self._execute_command(raw_cmd):
                return "break"  # the window has been closed by 'exit'
        self.show_prompt()
        return "break"

    def _execute_command(self, raw_cmd):
        """Show the command's output. Returns False if it closed the application."""
        for chunk in self.engine.execute(raw_cmd):
            if chunk is EXIT_APP:
                self.root.destroy()  # ends mainloop(), so the program exits
                return False
            if chunk is CLEAR_SCREEN:
                self.output_area.clear()
            else:
                self.write_output(chunk)
        return True


def main():
    set_macos_app_name(APP_NAME)
    root = tk.Tk(className=APP_NAME)
    MatlabConsoleApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
