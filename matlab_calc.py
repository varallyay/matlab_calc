"""
Matlab_Calc - a MATLAB Command Window emulator built with Tkinter and NumPy.

Run with:  python3 matlab_calc.py
"""

import gc
import inspect
import json
import math
import os
import plistlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import urllib.error
import urllib.parse
import urllib.request
import tkinter.font as tkfont
from fractions import Fraction
from tkinter import scrolledtext, ttk

import numpy as np

# =============================================================================
#  Application parameters - edit these to customise the app
# =============================================================================
APP_NAME = 'Matlab_Calc'        # Window title, Tk class name and (macOS) menu-bar name
APP_VERSION = '0.7.0'           # Shown in the welcome banner; see the version history in README.md

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

CONFIG_FILE = '~/.matlab_calc.json'  # Settings file in the home folder: first start, installation
UPDATE_REPOSITORY = 'varallyay/matlab_calc'  # GitHub project (owner/name) that publishes new versions
UPDATE_CHECK_DAYS = 7           # Look for a new version this often (days); 0 = only with the update command
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
CHAR = np.uint16  # MATLAB text (char arrays) is stored as UTF-16 character codes


class RawText(str):
    """Output shown exactly as it is (from disp or fprintf), without the usual blank line after it."""


def to_matrix(x):
    """Convert any value to a 2-D array of float, complex, bool (logical) or CHAR (text)."""
    a = np.asarray(x)
    if a.dtype == bool or a.dtype == CHAR:
        pass  # logical (true/false), e.g. the result of a comparison, or text
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
    """In arithmetic and most functions logical values count as 1 and 0, and text as its
    character codes ('a' + 1 is 98), as in MATLAB."""
    return a.astype(float) if a.dtype == bool or a.dtype == CHAR else a


def is_text(a):
    return a.dtype == CHAR


def text_to_chars(text):
    """A Python string as a MATLAB char row vector ('' is 0x0, as in MATLAB)."""
    if not text:
        return np.zeros((0, 0), dtype=CHAR)
    return np.frombuffer(text.encode('utf-16-le', 'surrogatepass'), dtype='<u2').astype(CHAR).reshape(1, -1)


def chars_to_text(a):
    """The text in a char array (or numbers used as character codes), row after row."""
    codes = np.clip(np.real(np.asarray(a)).astype(np.int64).ravel(), 0, 0xFFFF).astype('<u2')
    return codes.tobytes().decode('utf-16-le', 'replace')


def text_argument(value, function, what='The argument'):
    """The Python string in a text argument such as a file name; an error for anything else."""
    if not is_text(value) and value.size:
        raise MatlabError(f"Error using {function}\n{what} must be text, e.g. 'data.txt'.")
    return chars_to_text(value)


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
CAT_HYPERBOLIC = 'Hyperbolic functions'
CAT_ROUNDING = 'Rounding and remainders'
CAT_STATS = 'Data analysis'
CAT_LOGICAL = 'Logical values and searching'
CAT_RANDOM = 'Random numbers'
CAT_MATRIX = 'Matrices and arrays'
CAT_TEXT = 'Text and files'
HELP_CATEGORIES = (CAT_COMMANDS, CAT_CONSTANTS, CAT_MATH, CAT_ELEMENTWISE, CAT_TRIG, CAT_HYPERBOLIC, CAT_ROUNDING,
                   CAT_STATS, CAT_LOGICAL, CAT_RANDOM, CAT_MATRIX, CAT_TEXT)

FUNCTIONS = {}  # MATLAB name -> (Python function, minimum inputs, maximum inputs)


def builtin(*names, category, outputs=1, keep_types=False):
    """Register the decorated Python function as the MATLAB function(s) `names`.

    outputs is the largest number of results it returns ([a, b] = f(...) asks for 2; a
    function returning tuples must give at least as many as asked). Arguments arrive as
    numbers, with logical values and text converted, unless keep_types is set. If the
    function has the keyword-only parameter nargout, it learns how many results are wanted
    (0 for a statement such as disp(x), where it may return None for "no result"); a
    keyword-only parameter engine receives the MatlabEngine, e.g. to print text.
    """
    def register(func):
        params = list(inspect.signature(func).parameters.values())
        positional = [p for p in params if p.kind == p.POSITIONAL_OR_KEYWORD]
        keywords = {p.name for p in params if p.kind == p.KEYWORD_ONLY}
        variadic = any(p.kind == p.VAR_POSITIONAL for p in params)
        min_args = sum(p.default is inspect.Parameter.empty for p in positional)
        for name in names:
            FUNCTIONS[name] = (func, min_args, None if variadic else len(positional))
        func.matlab_names, func.help_category = names, category
        func.max_outputs, func.keep_types = outputs, keep_types
        func.wants_nargout, func.wants_engine = 'nargout' in keywords, 'engine' in keywords
        return func
    return register


_CURRENT_ENGINE = None  # the MatlabEngine running a command line (disp and fprintf print through it)


def call_function(name, arg_nodes, variables, end_value=None, nargout=1):
    """Call the built-in function `name`, checking the numbers of inputs and outputs as MATLAB does."""
    func, lo, hi = FUNCTIONS[name]
    arg_nodes = arg_nodes or []
    if len(arg_nodes) < lo or (hi is not None and len(arg_nodes) > hi):
        problem = 'Not enough' if len(arg_nodes) < lo else 'Too many'
        raise MatlabError(f"Error using {name}\n{problem} input arguments.")
    if nargout > func.max_outputs:
        raise MatlabError(f"Error using {name}\nToo many output arguments.")
    args = [evaluate(a, variables, end_value) for a in arg_nodes]
    if not func.keep_types:
        args = [to_double(a) for a in args]
    options = {}
    if func.wants_nargout:
        options['nargout'] = nargout
    if func.wants_engine:
        options['engine'] = _CURRENT_ENGINE or MatlabEngine()
    return func(*args, **options)


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
@builtin('find', category=CAT_LOGICAL, outputs=3)
def fn_find(x, n=None, *, nargout=1):
    """find - Positions of nonzero (true) elements.

    find(X) returns the positions of the nonzero elements of X, numbered
    down the columns as in A(k). With a condition it tells where the
    condition holds:
        A = [1 2 2.4 2.3];
        find(A >= 2)        % 2  3  4
    The result is a row if X is a row vector, otherwise a column, so after
    A = A' the same command gives [2; 3; 4].

    find(X, N) returns at most the first N positions.
    [row, col] = find(X) gives row and column numbers instead, and
    [row, col, v] = find(X) also the values.

    See also logical, max, min.
    """
    positions = np.flatnonzero(x.ravel(order='F'))
    if n is not None:
        if n.size != 1 or np.iscomplexobj(n) or not (n.flat[0] >= 1 and float(n.flat[0]).is_integer()):
            raise MatlabError("Error using find\nSecond argument must be a positive scalar integer.")
        positions = positions[:int(n.flat[0])]
    if x.shape == (0, 0):
        return np.zeros((0, 0)) if nargout <= 1 else (np.zeros((0, 0)),) * nargout
    shape = (1, -1) if x.shape[0] == 1 else (-1, 1)
    if nargout <= 1:
        return to_matrix((positions + 1.0).reshape(shape))
    results = (to_matrix((positions % x.shape[0] + 1.0).reshape(shape)),
               to_matrix((positions // x.shape[0] + 1.0).reshape(shape)),
               to_matrix(x.ravel(order='F')[positions].reshape(shape)))
    return results[:nargout]


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


# ---- helpers for the functions below ------------------------------------------
def _whole_number(value, function, what, minimum=0):
    """A non-negative (or >= minimum) whole number given as a scalar argument."""
    if value.size != 1 or np.iscomplexobj(value) or is_text(value):
        raise MatlabError(f"Error using {function}\n{what} must be a whole number.")
    number = float(value.flat[0])
    if not number.is_integer() or number < minimum:
        raise MatlabError(f"Error using {function}\n{what} must be a whole number of at least {minimum}.")
    return int(number)


def _reduce_axis(a, dim, function):
    """Direction for max, mean, ...: 0 down the columns, 1 along the rows, 'all', or None (dim > 2).

    Without dim MATLAB uses the first dimension longer than 1: columns of a matrix, the
    elements of a row vector.
    """
    if dim is None:
        return 1 if a.shape[0] == 1 and a.shape[1] != 1 else 0
    if is_text(dim):
        if chars_to_text(dim).lower() == 'all':
            return 'all'
        raise MatlabError(f"Error using {function}\nThe dimension must be 1, 2 or 'all'.")
    return {1: 0, 2: 1}.get(_whole_number(dim, function, 'The dimension', 1))


def _reduced_shape(a, axis):
    return (1, 1) if axis == 'all' else (1, a.shape[1]) if axis == 0 else (a.shape[0], 1)


# ---- hyperbolic functions ------------------------------------------------------
@builtin('sinh', category=CAT_HYPERBOLIC)
def fn_sinh(x):
    """sinh - Hyperbolic sine.

    sinh(X) is (exp(X) - exp(-X)) / 2 for each element of X.

    See also asinh, cosh, tanh.
    """
    return to_matrix(np.sinh(x))


@builtin('cosh', category=CAT_HYPERBOLIC)
def fn_cosh(x):
    """cosh - Hyperbolic cosine.

    cosh(X) is (exp(X) + exp(-X)) / 2 for each element of X.

    See also acosh, sinh, tanh.
    """
    return to_matrix(np.cosh(x))


@builtin('tanh', category=CAT_HYPERBOLIC)
def fn_tanh(x):
    """tanh - Hyperbolic tangent.

    tanh(X) is sinh(X) ./ cosh(X) for each element of X; it lies between
    -1 and 1.

    See also atanh, sinh, cosh.
    """
    return to_matrix(np.tanh(x))


@builtin('asinh', category=CAT_HYPERBOLIC)
def fn_asinh(x):
    """asinh - Inverse hyperbolic sine.

    asinh(X) is the number whose sinh is X, for each element of X:
    log(X + sqrt(X.^2 + 1)).

    See also sinh, acosh, atanh.
    """
    return to_matrix(np.arcsinh(x))


@builtin('acosh', category=CAT_HYPERBOLIC)
def fn_acosh(x):
    """acosh - Inverse hyperbolic cosine.

    acosh(X) is the number (>= 0) whose cosh is X, for each element of X.
    Elements below 1 give NaN (MATLAB returns a complex number).

    See also cosh, asinh, atanh.
    """
    return to_matrix(np.arccosh(x))


@builtin('atanh', category=CAT_HYPERBOLIC)
def fn_atanh(x):
    """atanh - Inverse hyperbolic tangent.

    atanh(X) is the number whose tanh is X, for each element of X;
    atanh(1) is Inf. Elements outside [-1, 1] give NaN (MATLAB returns a
    complex number).

    See also tanh, asinh, acosh.
    """
    return to_matrix(np.arctanh(x))


# ---- remainders ----------------------------------------------------------------
def _remainder(function, a, b, rule):
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        raise MatlabError(f"Error using {function}\nComplex numbers are not supported.")
    return _elementwise(rule, a, b, function)


@builtin('rem', category=CAT_ROUNDING)
def fn_rem(a, b):
    """rem - Remainder after division.

    rem(A, B) is A - fix(A./B).*B: the remainder has the sign of A,
    e.g. rem(7, 3) is 1 and rem(-7, 3) is -1. rem(A, 0) is NaN.
    A and B can be matrices of compatible sizes.

    See also mod, floor, round.
    """
    return _remainder('rem', a, b, np.fmod)


@builtin('mod', category=CAT_ROUNDING)
def fn_mod(a, b):
    """mod - Remainder after division (modulo).

    mod(A, B) is A - floor(A./B).*B: the result has the sign of B,
    e.g. mod(7, 3) is 1 and mod(-7, 3) is 2. mod(A, 0) is A.
    A and B can be matrices of compatible sizes.

    See also rem, floor, round.
    """
    return _remainder('mod', a, b, lambda x, y: np.where(y == 0, x, np.mod(x, y)))


# ---- data analysis -------------------------------------------------------------
def _extreme(function, a, b, dim, nargout, largest):
    """max / min: of a vector, of each column, element-wise between two arrays, or along dim."""
    a = to_double(a)
    if b is not None and b.shape != (0, 0):  # max(A, B): element by element
        if dim is not None or nargout > 1:
            raise MatlabError(f"Error using {function}\nWith two arrays to compare, {function} has one "
                              "output and no dimension argument.")
        return _elementwise(np.fmax if largest else np.fmin, a, to_double(b), function)
    axis = _reduce_axis(a, dim, function)
    if axis is None:  # along a dimension of length 1: the values themselves
        return (a, to_matrix(np.ones(a.shape))) if nargout > 1 else a
    work = a.ravel(order='F').reshape(-1, 1) if axis == 'all' else a
    if work.size == 0:
        empty = np.zeros((0, 0))
        return (empty, empty) if nargout > 1 else empty
    slices = work.T if axis in (0, 'all') else work
    key = np.abs(slices) if np.iscomplexobj(slices) else slices  # complex: by magnitude
    pick = np.nanargmax if largest else np.nanargmin
    positions = [0 if np.all(np.isnan(k)) else int(pick(k)) for k in key]  # NaN is skipped, as in MATLAB
    values = np.array([s[p] for s, p in zip(slices, positions)])
    shape = _reduced_shape(a, axis)
    result = to_matrix(values.reshape(shape))
    return (result, to_matrix(np.array(positions, dtype=float).reshape(shape) + 1)) if nargout > 1 else result


@builtin('max', category=CAT_STATS, outputs=2, keep_types=True)
def fn_max(a, b=None, dim=None, *, nargout=1):
    """max - Largest element.

    max(V) is the largest element of a vector V; max(A) of a matrix is a
    row with the largest element of each column. NaN is ignored.
    [M, I] = max(A) also gives the positions I of the largest elements
    (the first one if several are equal).
    max(A, B) is the larger of A and B, element by element.
    max(A, [], dim) works along dimension dim (1: columns, 2: rows), and
    max(A, [], 'all') gives the largest element of all.

    See also min, find, mean, median.
    """
    return _extreme('max', a, b, dim, nargout, largest=True)


@builtin('min', category=CAT_STATS, outputs=2, keep_types=True)
def fn_min(a, b=None, dim=None, *, nargout=1):
    """min - Smallest element.

    min(V) is the smallest element of a vector V; min(A) of a matrix is a
    row with the smallest element of each column. NaN is ignored.
    [M, I] = min(A) also gives the positions I of the smallest elements
    (the first one if several are equal).
    min(A, B) is the smaller of A and B, element by element.
    min(A, [], dim) works along dimension dim (1: columns, 2: rows), and
    min(A, [], 'all') gives the smallest element of all.

    See also max, find, mean, median.
    """
    return _extreme('min', a, b, dim, nargout, largest=False)


def _statistic(function, a, dim, compute):
    a = to_double(a)
    axis = _reduce_axis(a, dim, function)
    if axis is None:
        return compute(a, None, True)
    if axis == 'all':
        a, axis = a.ravel(order='F').reshape(-1, 1), 0
    if a.size == 0:
        return to_matrix(np.nan) if a.shape == (0, 0) else to_matrix(np.full(_reduced_shape(a, axis), np.nan))
    return to_matrix(compute(a, axis, False))


@builtin('mean', category=CAT_STATS, keep_types=True)
def fn_mean(a, dim=None):
    """mean - Average value.

    mean(V) is the average of the elements of a vector V; mean(A) of a
    matrix is a row with the average of each column.
    mean(A, dim) works along dimension dim (1: columns, 2: rows), and
    mean(A, 'all') averages all elements. NaN values give NaN.

    See also median, std, max, min.
    """
    return _statistic('mean', a, dim, lambda x, axis, same: x if same else np.mean(x, axis=axis, keepdims=True))


@builtin('median', category=CAT_STATS, keep_types=True)
def fn_median(a, dim=None):
    """median - Middle value.

    median(V) is the middle value of a vector V when sorted (the average
    of the two middle values for an even count); median(A) of a matrix is
    a row with the median of each column.
    median(A, dim) works along dimension dim, median(A, 'all') uses all
    elements. NaN values give NaN.

    See also mean, std, max, min.
    """
    return _statistic('median', a, dim,
                      lambda x, axis, same: x if same else np.median(x, axis=axis, keepdims=True))


@builtin('std', category=CAT_STATS, keep_types=True)
def fn_std(a, w=None, dim=None):
    """std - Standard deviation.

    std(V) is the standard deviation of the elements of a vector V,
    normalized by N-1 (N values); std(A) of a matrix is a row with one
    value per column.
    std(A, 1) normalizes by N instead; std(A, 0) is the same as std(A).
    std(A, w, dim) works along dimension dim, std(A, w, 'all') uses all
    elements. NaN values give NaN.

    See also mean, median.
    """
    weight = 0 if w is None or w.shape == (0, 0) else _whole_number(w, 'std', 'The weight')
    if weight not in (0, 1):
        raise MatlabError("Error using std\nThe weight must be 0 (divide by N-1) or 1 (divide by N).")

    def compute(x, axis, same):
        if same:
            return np.zeros(x.shape)
        count = x.shape[axis]
        return np.std(x, axis=axis, ddof=1 if weight == 0 and count > 1 else 0, keepdims=True)
    return _statistic('std', a, dim, compute)


# ---- random numbers ------------------------------------------------------------
_RANDOM = np.random.default_rng()


@builtin('rand', category=CAT_RANDOM)
def fn_rand(m=None, n=None):
    """rand - Uniformly distributed random numbers.

    rand is a random number between 0 and 1. rand(N) is an N-by-N matrix
    of them, rand(M, N) or rand([M N]) an M-by-N one.
    For other ranges: a + (b - a) * rand(...).

    See also randn, rng.
    """
    return to_matrix(_RANDOM.random(_size_args('rand', _given(m, n))))


@builtin('randn', category=CAT_RANDOM)
def fn_randn(m=None, n=None):
    """randn - Normally distributed random numbers.

    randn is a random number from the standard normal distribution (mean
    0, standard deviation 1). randn(N) is an N-by-N matrix of them,
    randn(M, N) or randn([M N]) an M-by-N one.
    For mean mu and standard deviation s: mu + s * randn(...).

    See also rand, rng.
    """
    return to_matrix(_RANDOM.standard_normal(_size_args('randn', _given(m, n))))


@builtin('rng', category=CAT_RANDOM, outputs=0, keep_types=True)
def fn_rng(seed):
    """rng - Control the random number generator.

    rng(seed) restarts rand and randn from the whole number seed, so they
    give the same numbers again. rng('default') is the same as rng(0),
    and rng('shuffle') starts from an unpredictable state.

    See also rand, randn.
    """
    global _RANDOM
    if is_text(seed):
        choice = chars_to_text(seed).lower()
        if choice not in ('default', 'shuffle'):
            raise MatlabError("Error using rng\nUse rng(seed), rng('default') or rng('shuffle').")
        _RANDOM = np.random.default_rng(0 if choice == 'default' else None)
    else:
        _RANDOM = np.random.default_rng(_whole_number(seed, 'rng', 'The seed'))


# ---- sizes and shapes ----------------------------------------------------------
@builtin('size', category=CAT_MATRIX, outputs=3, keep_types=True)
def fn_size(a, dim=None, *, nargout=1):
    """size - Size of an array.

    size(A) is [rows columns]; [m, n] = size(A) gives them separately
    (a third output is 1). size(A, 1) is the number of rows, size(A, 2)
    the number of columns.

    See also length, reshape, resize.
    """
    rows, cols = a.shape
    if dim is not None:
        d = _whole_number(dim, 'size', 'The dimension', 1)
        return to_matrix(rows if d == 1 else cols if d == 2 else 1)
    if nargout <= 1:
        return to_matrix([rows, cols])
    return tuple(to_matrix(v) for v in [rows, cols] + [1] * (nargout - 2))


@builtin('length', category=CAT_MATRIX, keep_types=True)
def fn_length(a):
    """length - Length of the largest dimension.

    length(A) is the number of elements of a vector A, and the larger of
    the number of rows and columns of a matrix; 0 if A is empty.

    See also size.
    """
    return to_matrix(0 if a.size == 0 else max(a.shape))


@builtin('linspace', category=CAT_MATRIX)
def fn_linspace(a, b, n=None):
    """linspace - Evenly spaced numbers.

    linspace(a, b) is a row of 100 evenly spaced numbers from a to b, and
    linspace(a, b, n) has n of them: linspace(0, 1, 5) is
    0  0.25  0.5  0.75  1.

    See also the colon operator a:step:b, resize.
    """
    if a.size != 1 or b.size != 1:
        raise MatlabError("Error using linspace\nThe start and end values must be scalars.")
    count = 100 if n is None else n.flat[0]
    if np.iscomplexobj(n) or not np.isfinite(count):
        raise MatlabError("Error using linspace\nThe number of points must be a whole number.")
    count = int(math.floor(np.real(count)))
    if count > MAX_ELEMENTS:
        raise MatlabError("Error using linspace\nRequested array exceeds the maximum allowed size.")
    if count < 1:
        return np.zeros((1, 0))
    if count == 1:
        return b
    return to_matrix(np.linspace(a.flat[0], b.flat[0], count))


def _size_vector(value, function):
    """[m n] from a size argument; a scalar m means "m in the first dimension that is not 1"."""
    sizes = np.real(value).ravel()
    if is_text(value) or np.iscomplexobj(value) or not sizes.size or np.any(sizes < 0) or \
            np.any(sizes != np.floor(sizes)) or not np.all(np.isfinite(sizes)):
        raise MatlabError(f"Error using {function}\nThe size must be whole numbers, e.g. 4 or [2 3].")
    return [int(v) for v in sizes]


@builtin('resize', category=CAT_MATRIX, keep_types=True)
def fn_resize(a, size, *options):
    """resize - Make an array larger or smaller.

    resize(A, [m n]) makes A m-by-n: elements are added at the end (0, or
    false/space for logical/text values) or removed from the end.
    resize(A, m) changes only the first dimension that is not 1 (the length
    of a vector): resize([1 2 3], 5) is 1 2 3 0 0.
    Options: resize(A, sz, 'FillValue', v) adds v instead of 0, and
    resize(A, sz, 'Side', 'leading') or 'both' changes the start (or both
    ends) instead of the end.

    See also reshape, size, linspace.
    """
    fill, side = None, 'trailing'
    if len(options) % 2:
        raise MatlabError("Error using resize\nOptions come in pairs, e.g. 'FillValue', 5.")
    for name, value in zip(options[::2], options[1::2]):
        option = text_argument(name, 'resize', 'The option name').lower()
        if option == 'fillvalue':
            if value.size != 1:
                raise MatlabError("Error using resize\nThe FillValue must be a single value.")
            fill = value
        elif option == 'side':
            side = text_argument(value, 'resize', 'The side').lower()
            if side not in ('trailing', 'leading', 'both'):
                raise MatlabError("Error using resize\nThe side must be 'trailing', 'leading' or 'both'.")
        else:
            raise MatlabError(f"Error using resize\nUnknown option '{option}'. Use 'FillValue' or 'Side'.")
    sizes = _size_vector(size, 'resize')
    if len(sizes) == 1:
        target = list(a.shape)
        target[1 if a.shape[0] == 1 and a.shape[1] != 1 else 0] = sizes[0]
    elif len(sizes) == 2:
        target = sizes
    else:
        raise MatlabError("Error using resize\nOnly 2-D sizes are supported.")
    if target[0] * target[1] > MAX_ELEMENTS:
        raise MatlabError("Error using resize\nRequested array exceeds the maximum allowed size.")
    result = a
    for axis, new in enumerate(target):
        old = result.shape[axis]
        if new <= old:  # remove elements
            start = 0 if side == 'trailing' else old - new if side == 'leading' else (old - new) // 2
            result = result[start:start + new, :] if axis == 0 else result[:, start:start + new]
        else:          # add elements
            grow = new - old
            lead = grow if side == 'leading' else grow // 2 if side == 'both' else 0
            pad = [(0, 0), (0, 0)]
            pad[axis] = (lead, grow - lead)
            value = 0 if fill is None else (fill.flat[0] if fill.dtype == result.dtype else to_double(fill).flat[0])
            if is_text(result) and fill is None:
                value = 32  # space
            result = np.pad(result, pad, constant_values=value)
    return to_matrix(result)


@builtin('reshape', category=CAT_MATRIX, keep_types=True)
def fn_reshape(a, m, n=None):
    """reshape - Arrange the elements of an array in a new shape.

    reshape(A, m, n) or reshape(A, [m n]) is an m-by-n array with the
    elements of A, taken down the columns: reshape(1:6, 2, 3) is
    [1 3 5; 2 4 6]. The number of elements must stay the same; [] for one
    size calculates it: reshape(1:6, [], 2).

    See also resize, size.
    """
    if n is None:
        sizes = [None if m.shape == (0, 0) else v for v in _size_vector(m, 'reshape')]
    else:
        sizes = [None if s.shape == (0, 0) else _whole_number(s, 'reshape', 'A size') for s in (m, n)]
    if len(sizes) != 2:
        raise MatlabError("Error using reshape\nOnly 2-D sizes are supported.")
    if sizes.count(None) > 1:
        raise MatlabError("Error using reshape\nOnly one size can be [].")
    known = [s for s in sizes if s is not None]
    if None in sizes:
        if known[0] == 0 or a.size % known[0]:
            raise MatlabError(f"Error using reshape\nThe {a.size} elements cannot be split into parts of {known[0]}.")
        sizes = [a.size // known[0] if s is None else s for s in sizes]
    if sizes[0] * sizes[1] != a.size:
        raise MatlabError("Error using reshape\nNumber of elements must not change. Use [] as one of the size "
                          "inputs to automatically calculate the appropriate size for that dimension.")
    return to_matrix(a.ravel(order='F').reshape(sizes, order='F'))


# ---- text and files ------------------------------------------------------------
# File identifiers (fid), as in MATLAB: 1 is the command window (standard output),
# 2 the command window too (standard error); fopen gives 3, 4, ... for files.
_FORMAT_SPEC = r"%(?P<flags>[-+ 0#]*)(?P<width>\d+)?(?:\.(?P<precision>\d+))?(?P<conversion>[diouxXfeEgGcs])"
_ESCAPES = {'n': '\n', 't': '\t', '\\': '\\', 'a': '\a', 'b': '\b', 'f': '\f', 'r': '\r', 'v': '\v'}


def _unescape(fmt):
    """Escape sequences in a format: \\n \\t \\\\ \\a \\b \\f \\r \\v, \\xN (hexadecimal), \\N (octal)."""
    def replace(match):
        code = match.group(1)
        if code[0] == 'x':
            return chr(min(int(code[1:], 16), 0x10FFFF))
        if code[0] in '01234567':
            return chr(int(code, 8))
        return _ESCAPES.get(code, '\\' + code)
    return re.sub(r"\\(x[0-9a-fA-F]+|[0-7]{1,3}|.)", replace, fmt, flags=re.S)


def _format_one(spec, value):
    """One conversion such as %6.2f. value is a number, or a str for %s / %c of text."""
    flags, width, precision, conv = spec.group('flags'), spec.group('width') or '', spec.group('precision'), \
        spec.group('conversion')
    prec = '' if precision is None else '.' + precision
    if conv in 'cs':
        if isinstance(value, str):
            return ("%" + flags.replace('0', '') + width + (prec if conv == 's' else '') + "s") % value
        if float(value).is_integer() and 0 <= value <= 0x10FFFF:  # a number printed as text: its character
            return ("%" + flags.replace('0', '') + width + "s") % chr(int(value))
        conv = 'e'  # as in MATLAB: a number that is not a character code is shown with %e
    if isinstance(value, str):
        value = ord(value)
    if math.isnan(value) or math.isinf(value):
        text = 'NaN' if math.isnan(value) else 'Inf' if value > 0 else '-Inf'
        return ("%" + flags.replace('0', '') + width + "s") % text
    if conv in 'diouxX':
        if not float(value).is_integer():
            conv = 'e'  # as in MATLAB: %d of a fraction is shown with %e
        else:
            return ("%" + flags + width + prec + {'i': 'd', 'u': 'd'}.get(conv, conv)) % int(value)
    return ("%" + flags + width + prec + conv) % value


def format_text(fmt, args):
    """MATLAB's sprintf: the format is used again and again until all values are printed.

    The values of all arguments are taken one after the other (down the columns);
    %s takes a whole text argument at once, other conversions take one element.
    """
    fmt, pieces, pos = _unescape(fmt), [], 0
    for match in re.finditer("%%|" + _FORMAT_SPEC, fmt):
        if match.start() > pos:
            pieces.append(('text', fmt[pos:match.start()]))
        pieces.append(('text', '%') if match.group() == '%%' else ('spec', match))
        pos = match.end()
    if pos < len(fmt):
        pieces.append(('text', fmt[pos:]))
    items = []
    for a in args:
        if is_text(a):
            if a.size:
                items.append(chars_to_text(a.ravel(order='F')))
        else:
            items.extend(float(v) for v in np.real(to_double(a)).ravel(order='F'))
    if not items or not any(kind == 'spec' for kind, _ in pieces):
        return ''.join(piece for kind, piece in pieces if kind == 'text')
    out, i, offset = [], 0, 0  # next value: items[i], and inside a text item its character offset
    while i < len(items):
        for kind, piece in pieces:
            if kind == 'text':
                out.append(piece)
                continue
            if i >= len(items):  # values used up: stop at the first conversion without a value
                return ''.join(out)
            item, conv = items[i], piece.group('conversion')
            if isinstance(item, str) and conv == 's':
                value, i, offset = item[offset:], i + 1, 0
            elif isinstance(item, str):
                value, offset = item[offset], offset + 1
                if offset >= len(item):
                    i, offset = i + 1, 0
            else:
                value, i = item, i + 1
            out.append(_format_one(piece, value))
    return ''.join(out)


def _file_id(value, function):
    if is_text(value) or value.size != 1 or np.iscomplexobj(value) or not float(value.flat[0]).is_integer():
        raise MatlabError(f"Error using {function}\nInvalid file identifier. Use fopen to generate a valid "
                          "file identifier.")
    return int(value.flat[0])


@builtin('disp', category=CAT_TEXT, outputs=0, keep_types=True)
def fn_disp(x, *, engine):
    """disp - Show a value without its name.

    disp(X) shows the value of X - a number, a matrix or text - without
    the "X =" line:
        disp('Hello')
        disp(['The result is ' num2str(pi)])
    An empty value shows nothing.

    See also fprintf, num2str, sprintf.
    """
    if x.size == 0:
        return None
    if is_text(x):
        engine.write(''.join(chars_to_text(row) + "\n" for row in x))
    else:
        engine.write(engine.format_value(to_double(x) if x.dtype == bool else x) + "\n")
    return None


@builtin('fprintf', category=CAT_TEXT, keep_types=True)
def fn_fprintf(first, *rest, nargout=1, engine=None):
    """fprintf - Write formatted text to the screen or to a file.

    fprintf(format, A, ...) writes to the command window, and
    fprintf(fid, format, A, ...) to a file opened with fid = fopen(...).
    The conversions in the format are replaced by the values in turn:
        %d whole number   %f fixed point   %e exponent   %g shortest
        %s text           %c character     %% percent sign
    with optional width and precision, e.g. %8.3f; \\n is a new line and
    \\t a tab. The format is repeated until all values are written:
        fprintf('%d squared is %d\\n', [1 2 3; 1 4 9])
    File identifier 1 is the command window. n = fprintf(...) gives the
    number of bytes written.

    See also sprintf, disp, fopen, fclose, fscanf, num2str.
    """
    if is_text(first):
        fid, fmt, values = 1, first, rest
    elif rest:
        fid, fmt, values = _file_id(first, 'fprintf'), rest[0], rest[1:]
    else:
        raise MatlabError("Error using fprintf\nThe format must be text, e.g. fprintf('%d\\n', x).")
    text = format_text(text_argument(fmt, 'fprintf', 'The format'), values)
    engine.write_to(fid, text, 'fprintf')
    return to_matrix(len(text.encode('utf-8'))) if nargout else None


@builtin('sprintf', category=CAT_TEXT, keep_types=True)
def fn_sprintf(fmt, *values):
    """sprintf - Format values as text.

    sprintf(format, A, ...) works like fprintf but returns the text
    instead of printing it:
        s = sprintf('%.2f', pi)          % '3.14'
        s = sprintf('%d, ', 1:3)         % '1, 2, 3, '

    See also fprintf, num2str, disp.
    """
    return text_to_chars(format_text(text_argument(fmt, 'sprintf', 'The format'), values))


def _number_text(value, digits):
    """One number as num2str writes it."""
    if np.iscomplexobj(value):
        imag = _number_text(abs(value.imag), digits)
        return _number_text(value.real, digits) + ('-' if value.imag < 0 else '+') + imag + 'i'
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return 'NaN' if math.isnan(value) else 'Inf' if value > 0 else '-Inf'
    if digits is None and value.is_integer():
        return str(int(value))
    return f"{value:.{digits or 5}g}"


@builtin('num2str', category=CAT_TEXT, keep_types=True)
def fn_num2str(x, precision=None):
    """num2str - Convert numbers to text.

    num2str(X) writes the numbers in X as text: num2str(pi) is '3.1416',
    num2str(42) is '42', num2str([1 2 3]) is '1  2  3'.
    num2str(X, N) uses N significant digits, and num2str(X, format) a
    format as in sprintf, e.g. num2str(pi, '%10.5f').
    Typical use: disp(['x = ' num2str(x)])

    See also sprintf, disp, fprintf.
    """
    if is_text(x):
        return x
    x = to_double(x)
    if x.size == 0:
        return text_to_chars('')
    if precision is not None and is_text(precision):
        fmt = chars_to_text(precision)
        rows = [format_text(fmt, [row.reshape(1, -1)]) for row in x]
    else:
        digits = None if precision is None else _whole_number(precision, 'num2str', 'The precision', 1)
        finite = np.abs(x[np.isfinite(x)])
        if digits is None and not np.all(np.isfinite(x) & (np.real(x) == np.round(np.real(x)))
                                         & (np.imag(x) == np.round(np.imag(x)))):
            biggest = finite.max() if finite.size and finite.max() > 0 else 1
            digits = max(int(math.floor(math.log10(biggest))) + 5, 5)
        cells = [[_number_text(v, digits) for v in row] for row in x]
        widths = [max(len(cells[r][c]) for r in range(len(cells))) for c in range(x.shape[1])]
        rows = ['  '.join(cell.rjust(widths[c]) for c, cell in enumerate(row)) for row in cells]
    width = max(len(r) for r in rows)
    return to_matrix(np.vstack([text_to_chars(r.ljust(width)) if width else np.zeros((1, 0), CHAR) for r in rows]))


@builtin('fopen', category=CAT_TEXT, outputs=2, keep_types=True)
def fn_fopen(filename, permission=None, *, nargout=1, engine):
    """fopen - Open a file.

    fid = fopen(filename, permission) opens a file and returns its file
    identifier for fprintf, fscanf and fclose; fid is -1 if it cannot be
    opened, and [fid, message] = fopen(...) also says why. Permissions:
        'r'  read (the default)      'r+'  read and write
        'w'  write a new file        'w+'  write and read
        'a'  add to the end          'a+'  add and read
    A file name without a folder is in the current folder (see pwd, cd).
    fopen('all') lists the identifiers of the open files.

    See also fclose, fprintf, fscanf.
    """
    name = text_argument(filename, 'fopen', 'The file name')
    if name == 'all' and permission is None:
        return to_matrix(np.array(sorted(engine.files), dtype=float).reshape(1, -1))
    mode = 'r' if permission is None else text_argument(permission, 'fopen', 'The permission')
    modes = {'r': 'rb', 'w': 'wb', 'a': 'ab', 'r+': 'r+b', 'w+': 'w+b', 'a+': 'a+b'}
    key = mode.replace('t', '').replace('b', '')
    if key not in modes:
        raise MatlabError(f"Error using fopen\nInvalid permission '{mode}'. Use 'r', 'w', 'a', 'r+', 'w+' or 'a+'.")
    try:
        handle = open(os.path.expanduser(name), modes[key])
    except OSError as err:
        fid, message = -1, err.strerror or str(err)
    else:
        fid, message = 3, ''
        while fid in engine.files:
            fid += 1
        engine.files[fid] = {'file': handle, 'name': name, 'mode': key}
    return (to_matrix(fid), text_to_chars(message)) if nargout >= 2 else to_matrix(fid)


@builtin('fclose', category=CAT_TEXT, keep_types=True)
def fn_fclose(fid, *, engine):
    """fclose - Close a file.

    fclose(fid) closes the file opened with fid = fopen(...), so that all
    text written to it is saved; it gives 0. fclose('all') closes every
    open file.

    See also fopen, fprintf, fscanf.
    """
    if is_text(fid):
        if chars_to_text(fid) != 'all':
            raise MatlabError("Error using fclose\nUse fclose(fid) or fclose('all').")
        for entry in engine.files.values():
            entry['file'].close()
        engine.files.clear()
        return to_matrix(0)
    number = _file_id(fid, 'fclose')
    if number not in (0, 1, 2):
        engine.file(number, 'fclose')['file'].close()
        del engine.files[number]
    return to_matrix(0)


_SCAN_NUMBER = r"[+-]?(?:(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|[iI][nN][fF](?:inity)?|[nN][aA][nN])"
_SCAN_PATTERNS = {'d': r"[+-]?\d+", 'i': r"[+-]?\d+", 'u': r"[+-]?\d+", 'x': r"[+-]?(?:0[xX])?[0-9a-fA-F]+",
                  'o': r"[+-]?[0-7]+", 's': r"\S+"}


def _scan(text, fmt, limit):
    """Read values from text with a scanf format, cycling through it as MATLAB's fscanf does.

    Returns (values, characters read, text_only) - values are numbers, or character
    codes for %s and %c; text_only is True if the format reads only text.
    """
    parts = re.findall(r"(\s+)|%(\*?)(\d*)(?:\.\d+)?([diouxfeEgGsc])|(%%|[^%\s]+)", fmt)
    conversions = [p for p in parts if p[3]]
    if not conversions:
        return [], 0, False
    text_only = all(p[3] in 'sc' for p in conversions)
    values, pos = [], 0
    while len(values) < limit:
        start = pos
        for space, skip, width, conv, literal in parts:
            if space:
                pos = len(text) - len(text[pos:].lstrip())
                continue
            if literal:
                literal = literal.replace('%%', '%')
                if not text.startswith(literal, pos):
                    return values, pos, text_only
                pos += len(literal)
                continue
            if conv != 'c':
                pos = len(text) - len(text[pos:].lstrip())
            segment = text[pos:pos + int(width)] if width else text[pos:]
            if conv == 'c':
                match_text = segment[:int(width) if width else 1]
                if not match_text:
                    return values, pos, text_only
            else:
                match = re.match(_SCAN_PATTERNS.get(conv, _SCAN_NUMBER), segment)
                if not match:
                    return values, pos, text_only
                match_text = match.group()
            pos += len(match_text)
            if skip:
                continue
            if conv in 'sc':
                values.extend(ord(ch) for ch in match_text)
            elif conv == 'x':
                values.append(float(int(match_text, 16)))
            elif conv == 'o':
                values.append(float(int(match_text, 8)))
            else:
                values.append(float(match_text))
            if len(values) >= limit:
                return values[:limit], pos, text_only
        if pos == start or pos >= len(text):
            break
    return values, pos, text_only


@builtin('fscanf', category=CAT_TEXT, outputs=2, keep_types=True)
def fn_fscanf(fid, fmt, size=None, *, nargout=1, engine):
    """fscanf - Read formatted data from a file.

    A = fscanf(fid, format) reads the rest of a file opened with fopen,
    using the format again and again; the numbers read form a column:
        fid = fopen('data.txt');  A = fscanf(fid, '%f');  fclose(fid);
    %d reads whole numbers, %f, %e and %g any numbers, %s words and %c
    characters; a format of only %s and %c gives text.
    fscanf(fid, format, n) reads at most n values, and fscanf(fid, format,
    [m n]) fills an m-by-n matrix column by column (n can be Inf):
        A = fscanf(fid, '%f %f', [2 Inf])'    % two numbers per line
    [A, count] = fscanf(...) also gives the number of values read.

    See also fopen, fclose, fprintf.
    """
    number = _file_id(fid, 'fscanf')
    if number in (0, 1, 2):
        raise MatlabError("Error using fscanf\nReading from the command window is not supported; open a file "
                          "with fopen.")
    entry = engine.file(number, 'fscanf')
    shape = None
    if size is None:
        limit = math.inf
    else:
        sizes = np.real(to_double(size)).ravel()
        if size.size not in (1, 2) or np.any(np.isnan(sizes)) or np.any(sizes < 0):
            raise MatlabError("Error using fscanf\nThe size must be n, Inf or [m n].")
        limit = float(math.prod(sizes))
        limit = int(limit) if math.isfinite(limit) else math.inf
        shape = [int(s) if math.isfinite(s) else None for s in sizes] if size.size == 2 else None
    handle = entry['file']
    try:
        start = handle.tell()
        data = handle.read().decode('utf-8', errors='replace')
    except (OSError, ValueError):
        raise MatlabError(f"Error using fscanf\nThe file {entry['name']} is not open for reading "
                          f"(it was opened with '{entry['mode']}').") from None
    values, used, text_only = _scan(data, _unescape(text_argument(fmt, 'fscanf', 'The format')), limit)
    handle.seek(start + len(data[:used].encode('utf-8')))
    count = len(values)
    if text_only:
        result = text_to_chars(''.join(chr(v) for v in values))
    elif shape is not None:
        rows = shape[0]
        cols = shape[1] if shape[1] is not None else (math.ceil(count / rows) if rows else 0)
        padded = np.zeros(rows * cols)
        padded[:min(count, rows * cols)] = values[:rows * cols]
        result = to_matrix(padded.reshape((rows, cols), order='F'))
    else:
        result = to_matrix(np.array(values, dtype=float).reshape(-1, 1)) if values else np.zeros((0, 1))
    return (result, to_matrix(count)) if nargout >= 2 else result


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

Text:
    'text'  "text"   Text (a char array), e.g. disp('Hello'); a quote
                     inside is doubled: 'it''s'

Several outputs:
    [r, c] = find(A)   Some functions give several results: find, size,
    [~, i] = max(v)    max, min, fopen, fscanf; ~ skips one.

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
    ("'text'  \"text\"  [a, b] = f(x)", "text, several outputs"),
)

# 'help' followed by any of these shows OPERATORS_HELP
OPERATOR_TOPICS = {
    'ops', 'operators', '+', '-', '*', '/', '\\', '^', "'", '.*', './', '.\\', '.^', ".'",
    '==', '~=', '<', '<=', '>', '>=', '&', '|', '~', '&&', '||', '=', '[', ']', '(', ')', ':', '%', '...',
    'plus', 'minus', 'uplus', 'uminus', 'mtimes', 'mrdivide', 'mldivide', 'mpower', 'rdivide', 'ldivide',
    'transpose', 'ctranspose', 'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'and', 'or', 'not', 'colon',
    'horzcat', 'vertcat', 'end', 'paren', 'indexing', 'index', 'subsref', 'subsasgn', 'assignment',
    'text', 'char', 'string', 'strings', '"', 'outputs', 'nargout',
}

_HELP_FIELDS = re.compile(r"\{(APP_NAME|APP_VERSION|SHORT_DECIMALS|LONG_DECIMALS|CONFIG_FILE|"
                          r"UPDATE_REPOSITORY|UPDATE_CHECK_DAYS)\}")


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
    statements, buf, stack, i = [], [], [], 0
    while i < len(text):
        ch = text[i]
        if ch == '"' or (ch == "'" and _quote_starts_text(text, i, bool(stack) and stack[-1] == '[')):
            end = _text_end(text, i) or len(text)  # % ; , inside text are just characters
            buf.append(text[i:end])
            i = end
            continue
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
            stack.append(ch)
        elif ch in ')]' and stack:
            stack.pop()
        if not stack and ch in ',;\n':
            statements.append((''.join(buf).strip(), ch == ';'))
            buf = []
        else:
            buf.append(ch)
        i += 1
    statements.append((''.join(buf).strip(), False))
    return [(s, suppressed) for s, suppressed in statements if s]


def _quote_starts_text(text, i, in_brackets):
    """Does the ' at text[i] start text (rather than being the transpose operator)?

    As in MATLAB, a quote right after a value - a name, a number, a closing bracket or the
    end of other text - is a transpose; inside [...] a space before it makes it text again.
    """
    j = i - 1
    while j >= 0 and text[j] in ' \t':
        j -= 1
    after_value = j >= 0 and (text[j].isalnum() or text[j] in "_)]}.'\"")
    return not after_value or (in_brackets and j < i - 1)


def _text_end(text, i):
    """Index just after the text literal starting with the quote at text[i] (None if unterminated).
    Inside, a doubled quote stands for one quote, as in 'it''s'."""
    quote, j = text[i], i + 1
    while True:
        k = text.find(quote, j)
        if k < 0:
            return None
        if text.startswith(quote * 2, k):
            j = k + 2
            continue
        return k + 1


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
    return c in '\'"'  # text


def tokenize(text):
    """Turn one statement into tokens (kind, text, value); kind is 'num', 'id', 'op', a bracket, ',', ';' or 'eof'."""
    tokens, stack, i, n = [], [], 0, len(text)

    def prev_ends_operand():
        if not tokens:
            return False
        kind, tok = tokens[-1][0], tokens[-1][1]
        return kind in ('num', 'id', 'str', ')', ']') or (kind == 'op' and tok in ("'", ".'"))

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
        if ch in '\'"':
            # Transpose when ' directly follows an operand; otherwise text in quotes
            if ch == "'" and prev_ends_operand() and not (in_brackets and text[i - 1] in ' \t'):
                tokens.append(('op', "'", None))
                i += 1
                continue
            end = _text_end(text, i)
            if end is None:
                raise MatlabError("Error: Text is not terminated properly: the closing "
                                  + ("quote ' is missing." if ch == "'" else 'quote " is missing.'))
            tokens.append(('str', text[i + 1:end - 1].replace(ch * 2, ch), None))
            i = end
            continue
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
        if self.at('[') and self._closing_followed_by_assignment(self.pos):
            return self._several_outputs()
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
        return nxt[0] == '(' and self._closing_followed_by_assignment(self.pos + 1)

    def _several_outputs(self):
        """[a, b, ~] = f(...): returns (('outputs', names), call), with None for each ~."""
        self.advance()  # '['
        names = []
        while not self.at(']'):
            if self.at(','):
                self.advance()
            elif self.at_op('~'):
                self.advance()
                names.append(None)
            elif self.at('id') and self.peek()[1] != 'end':
                names.append(self.advance()[1])
            else:
                raise MatlabError("Error: On the left of [a, b] = f(...) only variable names and ~ can be used.")
        self.advance()  # ']'
        self.advance()  # '='
        node = self.parse_expr()
        if not self.at('eof'):
            self.fail()
        if not names:
            raise MatlabError("Error: [] = ... has no variables to assign.")
        return ('outputs', names), node

    def _closing_followed_by_assignment(self, start):
        """Is the bracket opened at tokens[start] closed by a token followed by '='?"""
        depth = 0
        for j in range(start, len(self.tokens)):
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
        if tok[0] == 'str':
            self.advance()
            return ('str', tok[1])
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


def _same_kind(parts):
    """Text joined with numbers gives text, as in MATLAB: ['abc', 65] is 'abcA'."""
    if any(is_text(p) for p in parts) and not all(is_text(p) for p in parts):
        return [p if is_text(p) else np.clip(np.round(np.real(to_double(p))), 0, 0xFFFF).astype(CHAR)
                for p in parts]
    return parts


def evaluate(node, variables, end_value=None):
    """Evaluate a parse tree. Variables shadow built-in functions and constants, as in MATLAB.

    end_value is what 'end' stands for while a subscript is evaluated, e.g. in A(end - 1).
    """
    kind = node[0]
    if kind == 'num':
        return to_matrix(node[1])
    if kind == 'str':
        return text_to_chars(node[1])
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
            return call_function(name, args, variables, end_value)
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
                    rows.append(np.hstack(_same_kind(parts)))
                except ValueError:
                    raise MatlabError("Error using horzcat\nDimensions of arrays being concatenated "
                                      "are not consistent.") from None
        if not rows:
            return np.zeros((0, 0))
        try:
            return to_matrix(np.vstack(_same_kind(rows)))
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
        self.files = {}             # open files: file identifier -> {'file', 'name', 'mode'}
        self._output = []

    # ---- execution ---------------------------------------------------------
    def execute(self, line):
        """Run one command line. Returns output chunks (str), CLEAR_SCREEN and/or EXIT_APP.

        Like MATLAB, the remaining statements on the line are skipped after an error or 'exit'.
        """
        global _CURRENT_ENGINE
        output, error = [], None
        previous, _CURRENT_ENGINE, self._output = _CURRENT_ENGINE, self, output
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
        finally:
            _CURRENT_ENGINE = previous
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
            if isinstance(target, tuple) and target[0] == 'outputs':
                return self._assign_outputs(target[1], node, suppressed)
            if target is None and self._is_function_call(node):
                value = call_function(node[1], node[2], self.variables, nargout=0)
                if value is None:  # e.g. disp(x) or fprintf(...): nothing to show or keep in ans
                    return []
            else:
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

    def _is_function_call(self, node):
        return node[0] == 'name' and node[1] not in self.variables and node[1] in FUNCTIONS

    def _assign_outputs(self, names, node, suppressed):
        """[a, b] = f(...): store and show each result (except those marked ~)."""
        if not self._is_function_call(node):
            raise MatlabError("Error: Too many output arguments. Only a function call, such as "
                              "[r, c] = find(A), can give several results.")
        values = call_function(node[1], node[2], self.variables, nargout=len(names))
        values = values if isinstance(values, tuple) else (values,)
        if len(values) < len(names):
            raise MatlabError(f"Error using {node[1]}\nToo many output arguments.")
        shown = []
        for name, value in zip(names, values):
            if name is not None:
                self.variables[name] = value
                if not suppressed:
                    shown.append(f"{name} =\n{self.format_value(value)}\n")
        return shown

    def write(self, text):
        """Print text exactly as given (disp, fprintf to the screen)."""
        self._output.append(RawText(text))

    def file(self, fid, function):
        """The open file with identifier fid; an error if there is none."""
        if fid not in self.files:
            raise MatlabError(f"Error using {function}\nInvalid file identifier. Use fopen to generate a "
                              "valid file identifier.")
        return self.files[fid]

    def write_to(self, fid, text, function):
        """Write text to file identifier fid: 1 and 2 are the command window, 3, 4, ... open files."""
        if fid in (1, 2):
            self.write(text)
            return
        entry = self.file(fid, function)
        try:
            entry['file'].write(text.encode('utf-8'))
            entry['file'].flush()
        except (OSError, ValueError):
            raise MatlabError(f"Error using {function}\nThe file {entry['name']} is not open for writing "
                              f"(it was opened with '{entry['mode']}').") from None

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

    @console_command('install', max_args=1, arg_pattern=r"systemwide")
    def cmd_install(self, args, suppressed):
        """install - Install {APP_NAME} as an application.

        install             installs {APP_NAME} for you: it is then listed in
                            Launchpad (macOS) or in the application menu
                            (Linux) and starts without the terminal.
        install systemwide  installs it for all users of the computer; this
                            may need administrator rights.

        Typing install again updates the installed copy to this version.
        The settings file {CONFIG_FILE} records the installation.

        See also uninstall.
        """
        ok, message = install_app(systemwide=bool(args))
        if not ok:
            raise MatlabError(message)
        return [_indent(message)]

    @console_command('uninstall')
    def cmd_uninstall(self, args, suppressed):
        """uninstall - Remove the installed application.

        uninstall removes the installed copies of {APP_NAME} (systemwide and
        personal) with their launchers. The program that is running keeps
        working until you close it, and the settings file stays.

        See also install.
        """
        ok, message = uninstall_app()
        if not ok:
            raise MatlabError(message)
        return [_indent(message)]

    @console_command('pwd')
    def cmd_pwd(self, args, suppressed):
        """pwd - Show the current folder.

        pwd shows the current folder: files opened with fopen without a
        folder in their name are read from and written to this folder.
        At the start it is the folder the program was started from (your
        home folder when started from Launchpad or the application menu).

        See also cd, fopen.
        """
        return [_indent(os.getcwd())]

    @console_command('cd', max_args=None)
    def cmd_cd(self, args, suppressed):
        """cd - Change the current folder.

        cd folder     makes folder the current folder, e.g. cd Documents,
                      cd .. (one level up) or cd ~ (your home folder).
                      Put a name with spaces in quotes: cd 'My Data'.
        cd            shows the current folder, like pwd.

        See also pwd, fopen.
        """
        if not args:
            return [_indent(os.getcwd())]
        folder = ' '.join(args).strip()
        if len(folder) > 1 and folder[0] == folder[-1] and folder[0] in '\'"':
            folder = folder[1:-1]
        try:
            os.chdir(os.path.expanduser(folder))
        except OSError:
            raise MatlabError(f"Error using cd\nUnable to change current folder to '{folder}' "
                              "(Name is nonexistent or not a folder).") from None
        return []

    @console_command('update', max_args=1, arg_pattern=r"on|off")
    def cmd_update(self, args, suppressed):
        """update - Look for a new version and install it.

        update      looks on GitHub ({UPDATE_REPOSITORY}) for a newer
                    version; if there is one, it is downloaded and the
                    installed copy of {APP_NAME} is updated. Restart to use it.
        update off  turns off the automatic look every {UPDATE_CHECK_DAYS} days.
        update on   turns it on again.

        Only an installed {APP_NAME} is updated; see install.

        See also install, version.
        """
        if args:
            config = load_config()
            config['check_updates'] = args[0] == 'on'
            save_config(config)
            if args[0] == 'off':
                return [_indent("Automatic update checks are off. Type update to look for a new version.")]
            return [_indent(f"Automatic update checks are on: every {UPDATE_CHECK_DAYS} days."
                            if UPDATE_CHECK_DAYS > 0 else
                            "Automatic update checks are disabled by UPDATE_CHECK_DAYS = 0 in the program.")]
        ok, message = update_installed_app()
        if not ok:
            raise MatlabError(message)
        return [_indent(message)]

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
            cls = ("logical" if v.dtype == bool else "char" if v.dtype == CHAR
                   else "double (complex)" if np.iscomplexobj(v) else "double")
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
        kind = 'logical' if a.dtype == bool else 'char' if a.dtype == CHAR else 'double'
        if a.size == 0:
            if (rows, cols) == (0, 0) and kind != 'logical':
                return "    ''" if kind == 'char' else "    []"
            return f"    {rows}x{cols} empty {kind} " + ("array" if kind == 'char' else "matrix")
        if a.size > MAX_DISPLAY_ELEMENTS:
            return (f"    {rows}x{cols} {kind} matrix (too large to display; "
                    "end the statement with ';' to suppress output)")
        if kind == 'char':  # text in quotes, as in MATLAB
            lines = [f"    '{chars_to_text(a[r])}'" for r in range(rows)]
            return lines[0] if rows == 1 else f"  {rows}x{cols} char array\n" + "\n".join(lines)
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
#  Installation on macOS and Linux
#
#  Installing copies this script and its icon to a fixed place and adds a launcher,
#  so the program starts like any other application instead of from the terminal:
#    macOS   Matlab_Calc.app in /Applications (systemwide) or ~/Applications (only
#            for you); it appears in Launchpad and Spotlight.
#    Linux   program files in /opt/matlab_calc and a menu entry in
#            /usr/local/share/applications (systemwide), or ~/.local/share/matlab_calc
#            and ~/.local/share/applications (only for you).
#  The launcher runs the copy with the Python interpreter that did the installation,
#  which has NumPy. CONFIG_FILE in the home folder records that the question at the
#  first start was answered and what was installed.
# =============================================================================
MAC_SYSTEM_DIR = '/Applications'
MAC_USER_DIR = '~/Applications'
LINUX_SYSTEM_DIRS = ('/opt', '/usr/local/share/applications')   # program files, menu entry
LINUX_USER_DIRS = ('~/.local/share', '~/.local/share/applications')

THIS_SCRIPT = os.path.realpath(__file__)
APP_SLUG = re.sub(r'[^a-z0-9_]+', '_', APP_NAME.lower())  # file and folder names, e.g. matlab_calc


def load_config():
    """Settings from CONFIG_FILE; empty if the file is missing or unreadable."""
    try:
        with open(os.path.expanduser(CONFIG_FILE), encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_config(config):
    """Write the settings (via a temporary file, so a crash cannot leave half a file)."""
    path = os.path.expanduser(CONFIG_FILE)
    try:
        with open(path + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, sort_keys=True)
        os.replace(path + '.tmp', path)
        return True
    except OSError:
        return False


def _platform_name(platform):
    return 'macOS' if platform == 'darwin' else 'Linux' if platform.startswith('linux') else platform


def install_target(systemwide, platform=None):
    """Where an installation goes, or None on unsupported systems.

    Returns a dict with 'files' (the folder that gets the script), 'launcher' (the .app
    bundle or the .desktop file), 'script' (the installed script) and 'parents' (the
    folders that must be writable).
    """
    platform = platform or sys.platform
    if platform == 'darwin':
        base = MAC_SYSTEM_DIR if systemwide else MAC_USER_DIR
        app = os.path.join(os.path.expanduser(base), APP_NAME + '.app')
        files = os.path.join(app, 'Contents', 'Resources')
        parents = [os.path.expanduser(base)]
    elif platform.startswith('linux'):
        if systemwide:
            files_root, menu_dir = LINUX_SYSTEM_DIRS
        elif os.environ.get('XDG_DATA_HOME'):  # personal data folder set by the desktop
            files_root = os.environ['XDG_DATA_HOME']
            menu_dir = os.path.join(files_root, 'applications')
        else:
            files_root, menu_dir = LINUX_USER_DIRS
        files_root, menu_dir = os.path.expanduser(files_root), os.path.expanduser(menu_dir)
        files = os.path.join(files_root, APP_SLUG)
        app = os.path.join(menu_dir, APP_SLUG + '.desktop')
        parents = [files_root, menu_dir]
    else:
        return None
    return {'files': files, 'launcher': app, 'script': os.path.join(files, APP_SLUG + '.py'), 'parents': parents}


def _can_write(path):
    """Can this user create `path` (or write into it if it exists)?"""
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            return False
        path = parent
    return os.access(path, os.W_OK)


def running_installed_copy(platform=None):
    """'systemwide' or 'user' if this script is an installed copy, otherwise None."""
    for systemwide in (True, False):
        target = install_target(systemwide, platform)
        if target and os.path.realpath(target['script']) == THIS_SCRIPT:
            return 'systemwide' if systemwide else 'user'
    return None


def _version_tuple(version):
    try:
        return tuple(int(part) for part in str(version).split('.'))
    except ValueError:
        return (0,)


def _admin_hint(option):
    return (f"    sudo {shlex.quote(sys.executable or 'python3')} {shlex.quote(THIS_SCRIPT)} {option}")


_VERSION_RE = re.compile(r"""^APP_VERSION\s*=\s*['"]([^'"]+)['"]""", re.M)


def file_version(path):
    """APP_VERSION written in a copy of this program, or None."""
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            match = _VERSION_RE.search(f.read())
        return match.group(1) if match else None
    except OSError:
        return None


def _icon_for(source):
    """The icon file to install with `source`: the one next to it, else the one next to this script."""
    for folder in (os.path.dirname(source), os.path.dirname(THIS_SCRIPT)):
        if os.path.isfile(os.path.join(folder, APP_ICON_FILE)):
            return os.path.join(folder, APP_ICON_FILE)
    return None


def install_app(systemwide, platform=None, source=None, admin_option='--install-systemwide'):
    """Install this program, or the script `source` (a downloaded update). Returns (success, message)."""
    platform = platform or sys.platform
    source = source or THIS_SCRIPT
    version = file_version(source) or APP_VERSION
    target = install_target(systemwide, platform)
    if target is None:
        return False, f"Installing is only supported on macOS and Linux, not on {_platform_name(platform)}."
    if not all(_can_write(p) for p in target['parents']):
        if systemwide:
            return False, ("Installing for all users needs administrator rights. In a terminal, run\n"
                           f"{_admin_hint(admin_option)}\n"
                           "or type install to install it only for you.")
        return False, f"Cannot write to {' or '.join(target['parents'])}; nothing was installed."
    previous = file_version(target['script'])
    try:
        if platform == 'darwin':
            _install_mac(target, source, version)
        else:
            _install_linux(target, source)
    except (OSError, subprocess.SubprocessError) as err:
        return False, f"The installation failed: {err}"
    # A systemwide installation replaces a personal one, so the program is not listed twice
    other = install_target(not systemwide, platform)
    if systemwide and os.path.lexists(other['launcher']):
        try:
            _remove_installation(other)
        except OSError:
            pass
    config = load_config()
    config.update(first_start_done=True, update_declined=None, installed={
        'version': version, 'systemwide': systemwide, 'launcher': target['launcher'],
        'python': sys.executable})
    config.setdefault('last_update_check', time.time())  # the first automatic check comes a week later
    save_config(config)
    scope = "for all users" if systemwide else "for you"
    if previous and previous != version:
        return True, f"{APP_NAME} is updated from {previous} to {version} ({scope}):\n    {target['launcher']}"
    where = ("from Launchpad or the Applications folder" if platform == 'darwin'
             else "from the application menu (search for " + APP_NAME + ")")
    return True, (f"{APP_NAME} {version} is installed {scope}:\n    {target['launcher']}\n"
                  f"From now on it can be started {where}.")


def uninstall_app(platform=None):
    """Remove the installed copies (systemwide and personal). Returns (success, message)."""
    platform = platform or sys.platform
    if install_target(False, platform) is None:
        return False, f"Installing is only supported on macOS and Linux, not on {_platform_name(platform)}."
    removed, failed = [], []
    for systemwide in (True, False):
        target = install_target(systemwide, platform)
        if not (os.path.lexists(target['launcher']) or os.path.exists(target['script'])):
            continue
        try:
            _remove_installation(target)
            removed.append(target['launcher'])
        except OSError:
            failed.append(target['launcher'])
    config = load_config()
    if not failed:
        config.pop('installed', None)
        save_config(config)
    if failed:
        return False, ("Removing " + ", ".join(failed) + " needs administrator rights. In a terminal, run\n"
                       + _admin_hint('--uninstall'))
    if not removed:
        return True, f"{APP_NAME} is not installed."
    return True, "Removed " + ", ".join(removed) + "."


def _remove_installation(target):
    """Delete an installation; only folders that really contain an installed copy are removed."""
    if os.path.isdir(target['launcher']):          # macOS .app bundle
        if not os.path.exists(target['script']):
            raise OSError(f"{target['launcher']} does not look like a {APP_NAME} installation")
        shutil.rmtree(target['launcher'])
        return
    if os.path.lexists(target['launcher']):        # Linux menu entry
        os.remove(target['launcher'])
        _refresh_menus(os.path.dirname(target['launcher']))
    if os.path.exists(target['script']):
        shutil.rmtree(target['files'])


def _copy(src, dst):
    if os.path.exists(dst) and os.path.samefile(src, dst):
        return  # running the installed copy: nothing to copy
    shutil.copyfile(src, dst)


def _make_readable(folder):
    """Let every user read (and enter) the installed files, whatever the umask was."""
    for path, dirs, files in os.walk(folder):
        os.chmod(path, 0o755)
        for name in files:
            full = os.path.join(path, name)
            os.chmod(full, 0o755 if os.access(full, os.X_OK) else 0o644)


def _save_rounded_icon(src, dst):
    """Save the shrunk, rounded icon as a PNG (Pillow, else Tk, else a plain copy)."""
    try:
        rounded_icon_image(src).save(dst)
        return
    except Exception:
        pass
    root = tk._default_root
    try:
        temporary = root is None
        if temporary:
            root = tk.Tk()
            root.withdraw()
        try:
            _rounded_icon_tk(root, src, APP_ICON_MARGIN, APP_ICON_ROUNDING).write(dst, format='png')
            return
        finally:
            if temporary:
                root.destroy()
    except tk.TclError:  # no display, e.g. when installing over ssh
        _copy(src, dst)


def _install_linux(target, source):
    files = target['files']
    os.makedirs(files, exist_ok=True)
    _copy(source, target['script'])
    lines = ["[Desktop Entry]", "Type=Application", f"Name={APP_NAME}", "GenericName=Calculator",
             "Comment=MATLAB-style matrix calculator",
             "Exec=" + _desktop_exec([sys.executable or 'python3', target['script']])]
    icon = _icon_for(source)
    if icon:
        _copy(icon, os.path.join(files, APP_ICON_FILE))
        menu_icon = os.path.join(files, APP_SLUG + '_menu_icon.png')
        _save_rounded_icon(os.path.join(files, APP_ICON_FILE), menu_icon)
        lines.append(f"Icon={menu_icon}")
    lines += ["Terminal=false", "Categories=Utility;Calculator;", f"StartupWMClass={APP_NAME}"]
    os.makedirs(os.path.dirname(target['launcher']), exist_ok=True)
    with open(target['launcher'], 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(target['launcher'], 0o755)
    _make_readable(files)
    _refresh_menus(os.path.dirname(target['launcher']))


def _desktop_exec(args):
    """Quote arguments for the Exec key of a .desktop file (freedesktop.org rules)."""
    quoted = []
    for arg in args:
        arg = arg.replace('%', '%%')
        if re.search(r"[\s\"'\\><~|&;$*?#()`]", arg):
            arg = '"' + re.sub(r'(["`$\\])', r'\\\1', arg) + '"'
        quoted.append(arg.replace('\\', '\\\\'))  # the file format escapes backslashes once more
    return ' '.join(quoted)


def _refresh_menus(menu_dir):
    if shutil.which('update-desktop-database'):
        subprocess.run(['update-desktop-database', '-q', menu_dir],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def _install_mac(target, source, version):
    app, resources = target['launcher'], target['files']
    macos = os.path.join(app, 'Contents', 'MacOS')
    os.makedirs(resources, exist_ok=True)
    os.makedirs(macos, exist_ok=True)
    _copy(source, target['script'])
    icon = _icon_for(source)
    has_icon = False
    if icon:
        _copy(icon, os.path.join(resources, APP_ICON_FILE))
        has_icon = _save_icns(os.path.join(resources, APP_ICON_FILE), os.path.join(resources, 'AppIcon.icns'))
    executable = os.path.join(macos, APP_NAME)
    with open(executable, 'w', encoding='utf-8') as f:
        f.write("#!/bin/sh\n"
                f"# Starts {APP_NAME} with the Python interpreter that installed it.\n"
                f'exec {shlex.quote(sys.executable or "python3")} "$(dirname "$0")/../Resources/{APP_SLUG}.py" "$@"\n')
    os.chmod(executable, 0o755)
    info = {
        'CFBundleName': APP_NAME, 'CFBundleDisplayName': APP_NAME, 'CFBundleExecutable': APP_NAME,
        'CFBundleIdentifier': 'local.' + re.sub(r'[^A-Za-z0-9.-]', '-', APP_NAME.lower()),
        'CFBundlePackageType': 'APPL', 'CFBundleInfoDictionaryVersion': '6.0',
        'CFBundleShortVersionString': version, 'CFBundleVersion': version,
        'LSMinimumSystemVersion': '10.13', 'NSHighResolutionCapable': True,
    }
    if has_icon:
        info['CFBundleIconFile'] = 'AppIcon'
    with open(os.path.join(app, 'Contents', 'Info.plist'), 'wb') as f:
        plistlib.dump(info, f)
    _make_readable(app)
    os.utime(app)  # lets Finder and Launchpad notice the change
    lsregister = ('/System/Library/Frameworks/CoreServices.framework/Frameworks/'
                  'LaunchServices.framework/Support/lsregister')
    if os.path.exists(lsregister):
        subprocess.run([lsregister, '-f', app], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def _save_icns(src, dst):
    """Create the macOS icon file (Pillow, else the macOS tools sips and iconutil). Returns success."""
    try:
        from PIL import Image
        icon = rounded_icon_image(src).resize((1024, 1024), getattr(Image, 'Resampling', Image).LANCZOS)
        icon.save(dst, format='ICNS')
        return True
    except Exception:
        pass
    if not (shutil.which('sips') and shutil.which('iconutil')):
        return False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            png = os.path.join(tmp, 'icon.png')
            _save_rounded_icon(src, png)
            iconset = os.path.join(tmp, 'AppIcon.iconset')
            os.mkdir(iconset)
            for size in (16, 32, 128, 256, 512):
                for scale, suffix in ((1, ''), (2, '@2x')):
                    out = os.path.join(iconset, f"icon_{size}x{size}{suffix}.png")
                    subprocess.run(['sips', '-z', str(size * scale), str(size * scale), png, '--out', out],
                                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['iconutil', '-c', 'icns', iconset, '-o', dst],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def startup_question(config, platform=None):
    """What to ask when the program starts: 'install' at the first start, 'update' when an
    older version is installed and this newer one runs from elsewhere, otherwise None."""
    platform = platform or sys.platform
    if install_target(False, platform) is None or running_installed_copy(platform):
        return None
    if not config.get('first_start_done'):
        return 'install'
    installed = config.get('installed') or {}
    if (installed and os.path.lexists(installed.get('launcher', ''))
            and _version_tuple(installed.get('version')) < _version_tuple(APP_VERSION)
            and config.get('update_declined') != APP_VERSION):
        return 'update'
    return None


# ---- updates from GitHub -------------------------------------------------------
# New versions are published in the GitHub project UPDATE_REPOSITORY: the program
# file matlab_calc.py (and icon.png) in its top folder. If the project has GitHub
# releases, the newest release is used, otherwise the newest version of the main
# branch. Its APP_VERSION decides whether it is newer than the installed copy.
GITHUB_API = 'https://api.github.com'
GITHUB_RAW = 'https://raw.githubusercontent.com'
UPDATE_TIMEOUT = 15          # seconds allowed for each download
MAX_DOWNLOAD = 5_000_000     # bytes; the program file is about 100 kB


class UpdateError(Exception):
    """No new version could be found or downloaded; the message says why."""


def _download(url):
    request = urllib.request.Request(url, headers={'User-Agent': f"{APP_NAME}/{APP_VERSION}"})
    with urllib.request.urlopen(request, timeout=UPDATE_TIMEOUT) as response:
        data = response.read(MAX_DOWNLOAD + 1)
    if len(data) > MAX_DOWNLOAD:
        raise UpdateError(f"{url} is unexpectedly large; it was not used.")
    return data


def _connection_problem(err):
    reason = getattr(err, 'reason', err)
    text = f"Could not connect to GitHub to look for a new version ({reason})."
    if 'CERTIFICATE_VERIFY_FAILED' in str(reason):
        text += ("\nPython cannot check GitHub's security certificate. With Python from python.org on macOS,\n"
                 "run 'Install Certificates.command' in its folder in Applications.")
    return text


def find_latest_version():
    """The newest version on GitHub: {'version', 'source' (the program text), 'base' (its folder URL)}.

    Raises UpdateError when GitHub cannot be reached or the file is not a valid program.
    """
    ref = 'HEAD'  # the main branch, used when the project has no releases
    try:
        release = json.loads(_download(f"{GITHUB_API}/repos/{UPDATE_REPOSITORY}/releases/latest"))
        if isinstance(release, dict) and release.get('tag_name'):
            ref = release['tag_name']
    except (OSError, ValueError, UpdateError):  # no releases (404), API limit reached, offline, ...
        pass
    base = f"{GITHUB_RAW}/{UPDATE_REPOSITORY}/{urllib.parse.quote(ref)}/"
    try:
        data = _download(base + APP_SLUG + '.py')
    except urllib.error.HTTPError as err:
        raise UpdateError(f"{base}{APP_SLUG}.py could not be downloaded (HTTP error {err.code}).") from None
    except OSError as err:
        raise UpdateError(_connection_problem(err)) from None
    source = data.decode('utf-8', errors='replace')
    match = _VERSION_RE.search(source)
    try:
        compile(source, APP_SLUG + '.py', 'exec')  # a valid Python program, not e.g. an error page
    except (SyntaxError, ValueError):
        match = None
    if not match:
        raise UpdateError(f"The file downloaded from {base} is not a valid {APP_NAME} program.")
    return {'version': match.group(1), 'source': source, 'base': base}


def installed_copy(platform=None):
    """The installation to update: {'version', 'systemwide', 'script'}, or None if not installed."""
    recorded = (load_config().get('installed') or {}).get('systemwide')
    for systemwide in ((True, False) if recorded else (False, True)):
        target = install_target(systemwide, platform)
        if target and os.path.isfile(target['script']):
            return {'version': file_version(target['script']), 'systemwide': systemwide,
                    'script': target['script']}
    return None


def is_newer(version, than):
    return _version_tuple(version) > _version_tuple(than)


def update_check_due(config, now=None):
    """Is it time for the automatic look for a new version (every UPDATE_CHECK_DAYS days)?"""
    if UPDATE_CHECK_DAYS <= 0 or not config.get('check_updates', True):
        return False
    now = time.time() if now is None else now
    return now - float(config.get('last_update_check') or 0) >= UPDATE_CHECK_DAYS * 86400


def remember_update_check():
    config = load_config()
    config['last_update_check'] = time.time()
    save_config(config)


def install_update(latest, systemwide, platform=None):
    """Install a version found by find_latest_version (with its icon, if GitHub has one)."""
    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, APP_SLUG + '.py')
        with open(script, 'w', encoding='utf-8') as f:
            f.write(latest['source'])
        try:
            icon = _download(latest['base'] + APP_ICON_FILE)
            if icon.startswith(b'\x89PNG'):
                with open(os.path.join(tmp, APP_ICON_FILE), 'wb') as f:
                    f.write(icon)
        except (OSError, UpdateError):
            pass  # keep the current icon
        return install_app(systemwide, platform, source=script, admin_option='--update')


def update_installed_app(platform=None):
    """Look for a newer version on GitHub and install it over the installed copy. Returns (success, message)."""
    installed = installed_copy(platform)
    if installed is None:
        return False, f"{APP_NAME} is not installed, so there is nothing to update. Type install to install it."
    try:
        latest = find_latest_version()
    except UpdateError as err:
        return False, str(err)
    remember_update_check()
    if not is_newer(latest['version'], installed['version']):
        return True, (f"{APP_NAME} {installed['version']} is up to date "
                      f"(newest version on GitHub: {latest['version']}).")
    ok, message = install_update(latest, installed['systemwide'], platform)
    if ok:
        message += f"\nRestart {APP_NAME} to use version {latest['version']}."
    return ok, message


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

    def delete_raw(self, index1, index2):
        """Delete text without the read-only check, for the app's own output."""
        self.tk.call(self._raw, 'delete', index1, index2)

    def destroy(self):
        self.tk.call('rename', self._w, '')          # remove the guard
        self.tk.call('rename', self._raw, self._w)   # give the widget its own command back
        super().destroy()


def make_dialog_icon(root, size=64):
    """The rounded app icon, size x size pixels, for dialog windows (None without an icon file)."""
    path = os.path.join(os.path.dirname(THIS_SCRIPT), APP_ICON_FILE)
    if not os.path.isfile(path):
        return None
    try:
        from PIL import Image, ImageTk
        lanczos = getattr(Image, 'Resampling', Image).LANCZOS
        return ImageTk.PhotoImage(rounded_icon_image(path, margin=0).resize((size, size), lanczos), master=root)
    except Exception:  # no Pillow: plain Tk, scaled down by a whole factor
        pass
    try:
        image = _rounded_icon_tk(root, path, 0, APP_ICON_ROUNDING)
        factor = max(1, round(image.width() / size))
        return image.subsample(factor) if factor > 1 else image
    except tk.TclError:
        return None


class InstallDialog(tk.Toplevel):
    """Modal question with the app icon and a row of buttons; show() returns the chosen button's key.

    checkbox is an optional (label, tkinter.BooleanVar) pair shown above the buttons.
    """

    def __init__(self, root, title, message, buttons, note='', icon=None, checkbox=None):
        super().__init__(root)
        self.root, self.choice = root, None
        self.title(title)
        self.resizable(False, False)
        self.transient(root)
        frame = ttk.Frame(self, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)
        top = ttk.Frame(frame)
        top.pack(anchor=tk.W, fill=tk.X)
        self.icon_label = None
        if icon is not None:
            self.icon_label = ttk.Label(top, image=icon)
            self.icon_label.image = icon  # keep a reference, or Tk shows an empty square
            self.icon_label.pack(side=tk.LEFT, anchor=tk.N, padx=(0, 16))
        ttk.Label(top, text=message, justify=tk.LEFT, wraplength=420).pack(side=tk.LEFT, anchor=tk.W)
        self.checkbox = None
        if checkbox:
            self.checkbox = ttk.Checkbutton(frame, text=checkbox[0], variable=checkbox[1])
            self.checkbox.pack(anchor=tk.W, pady=(14, 0))
        row = ttk.Frame(frame)
        row.pack(anchor=tk.E, pady=(16, 0))
        self.buttons = {}
        for key, label in buttons:
            self.buttons[key] = ttk.Button(row, text=label, command=lambda k=key: self._choose(k))
            self.buttons[key].pack(side=tk.LEFT, padx=(8, 0))
        if note:
            ttk.Label(frame, text=note, justify=tk.LEFT, wraplength=460, foreground='gray').pack(
                anchor=tk.W, pady=(14, 0))
        default = self.buttons[buttons[0][0]]
        default.focus_set()
        self.bind('<Return>', lambda event: default.invoke())
        self.bind('<Escape>', lambda event: self._choose(None))
        self.protocol('WM_DELETE_WINDOW', lambda: self._choose(None))

    def _choose(self, key):
        self.choice = key
        self.destroy()

    def show(self):
        self.update_idletasks()  # centre over the main window
        x = self.root.winfo_rootx() + max(0, (self.root.winfo_width() - self.winfo_reqwidth()) // 2)
        y = self.root.winfo_rooty() + max(0, (self.root.winfo_height() - self.winfo_reqheight()) // 3)
        self.geometry(f"+{x}+{y}")
        try:
            self.wait_visibility()
            self.grab_set()
        except tk.TclError:
            pass
        self.wait_window()
        return self.choice


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
            elif isinstance(chunk, RawText):  # disp / fprintf output, exactly as printed
                self.output_area.insert(tk.END, chunk)
                self.output_area.see(tk.END)
            else:
                self.write_output(chunk)
        return True

    # ---- installation ------------------------------------------------------------
    def startup_check(self, platform=None):
        """At the first start ask whether to install the program; later offer to update it."""
        platform = platform or sys.platform
        config = load_config()
        question = startup_question(config, platform)
        if question is None:
            if not config.get('first_start_done'):
                config['first_start_done'] = True
                save_config(config)
            if update_check_due(config) and installed_copy(platform):
                self.check_for_update(platform)
            return
        mac = platform == 'darwin'
        message = None
        icon = self.dialog_icon()
        if question == 'install':
            home = os.path.expanduser('~')

            def place(systemwide):  # folder of the launcher, with the home folder shown as ~
                folder = os.path.dirname(install_target(systemwide, platform)['launcher'])
                return '~' + folder[len(home):] if folder.startswith(home + os.sep) else folder
            choice = InstallDialog(
                self.root, f"Install {APP_NAME}?",
                f"Welcome to {APP_NAME} {APP_VERSION}!\n\n"
                f"Install {APP_NAME} systemwide? It is then listed in "
                + ("Launchpad and the Applications folder" if mac else "the application menu")
                + " and starts like any other application, without the terminal.\n\n"
                f"Install systemwide: for all users of this computer\n    ({place(True)}"
                + ("" if mac else ", needs administrator rights") + ")\n"
                f"Only for me: just for your user account\n    ({place(False)})",
                [('systemwide', "Install systemwide"), ('user', "Only for me"), ('no', "Don't install")],
                note="This question is asked only once. The commands install and uninstall "
                     "can do the same later.", icon=icon, checkbox=self._weekly_checkbox(True)).show()
            if choice in ('systemwide', 'user'):
                message = install_app(choice == 'systemwide', platform)[1]
            config = load_config()
            config['first_start_done'] = True
            if self.weekly is not None:
                config['check_updates'] = self.weekly.get()
        else:
            installed = config['installed']
            choice = InstallDialog(
                self.root, f"Update {APP_NAME}?",
                f"{APP_NAME} {installed.get('version')} is installed on this computer.\n\n"
                f"Update it to version {APP_VERSION}, the version you are running now?",
                [('update', "Update"), ('later', "Not now")],
                note="If you choose Not now, you are not asked again for this version; "
                     "the command install updates it later.", icon=icon).show()
            if choice == 'update':
                message = install_app(bool(installed.get('systemwide')), platform)[1]
            config = load_config()
            if choice != 'update':
                config['update_declined'] = APP_VERSION
        save_config(config)
        if message:
            self.print_message(message)

    def dialog_icon(self):
        if getattr(self, '_dialog_icon', None) is None:
            self._dialog_icon = make_dialog_icon(self.root)
        return self._dialog_icon

    def _weekly_checkbox(self, checked):
        """(label, variable) for the 'look for new versions' checkbox, or None if checks are off."""
        self.weekly = None
        if UPDATE_CHECK_DAYS <= 0:
            return None
        self.weekly = tk.BooleanVar(self.root, value=checked)
        period = "every week" if UPDATE_CHECK_DAYS == 7 else f"every {UPDATE_CHECK_DAYS} days"
        return (f"Look for new versions on GitHub {period}", self.weekly)

    def check_for_update(self, platform=None):
        """Look for a new version on GitHub without blocking the window; offer it if there is one."""
        result = {}

        def look():  # runs in a background thread: no Tk calls here
            try:
                result['latest'] = find_latest_version()
            except Exception as err:  # offline, GitHub unreachable, ...: try again at the next start
                result['error'] = err

        # Free unused Tk objects (e.g. of closed dialogs) here in the main thread: if Python's
        # garbage collector ran into them in the background thread, Tcl would abort the program.
        gc.collect()
        thread = threading.Thread(target=look, daemon=True)
        thread.start()

        def wait():
            if thread.is_alive():
                self.root.after(200, wait)
            elif 'latest' in result:
                self.offer_update(result['latest'], platform)
        self.root.after(200, wait)
        return thread

    def offer_update(self, latest, platform=None):
        """Ask whether to install the newer version found on GitHub; install it and offer a restart."""
        remember_update_check()
        config = load_config()
        installed = installed_copy(platform)
        if (installed is None or not is_newer(latest['version'], installed['version'])
                or config.get('skipped_version') == latest['version']):
            return
        icon = self.dialog_icon()
        choice = InstallDialog(
            self.root, f"Update {APP_NAME}?",
            f"A new version of {APP_NAME} is available.\n\n"
            f"    Installed version:  {installed['version']}\n"
            f"    New version:  {latest['version']}\n\n"
            "Download and install it now?",
            [('update', "Update now"), ('later', "Not now"), ('skip', "Skip this version")],
            note=f"From github.com/{UPDATE_REPOSITORY}. The command update looks for a new version at any time.",
            icon=icon, checkbox=self._weekly_checkbox(config.get('check_updates', True))).show()
        config = load_config()
        if self.weekly is not None:
            config['check_updates'] = self.weekly.get()
        if choice == 'skip':
            config['skipped_version'] = latest['version']
        save_config(config)
        if choice != 'update':
            return
        ok, message = install_update(latest, installed['systemwide'], platform)
        self.print_message(message)
        if ok and InstallDialog(
                self.root, f"Restart {APP_NAME}?",
                f"{APP_NAME} {latest['version']} is installed.\n\n"
                "Restart now to use it? The variables in the workspace are not kept.",
                [('restart', "Restart now"), ('later', "Later")], icon=icon).show() == 'restart':
            self.restart(installed['script'])

    def restart(self, script):
        """Close the window and run `script` (the updated program) in place of this process."""
        self.root.destroy()
        os.execv(sys.executable, [sys.executable, script])

    def print_message(self, text):
        """Show text above the prompt, as if a command had printed it; typed input is kept."""
        typed = self.output_area.get("input_start", "end-1c")
        self.output_area.delete_raw("input_start linestart", "end")
        self.write_output(_indent(text))
        self.show_prompt()
        self.output_area.insert("input_start", typed)
        self.output_area.mark_set(tk.INSERT, tk.END)



def main(argv=None):
    """Start the program. --install, --install-systemwide, --uninstall and --update work without a window."""
    import argparse
    parser = argparse.ArgumentParser(prog=os.path.basename(THIS_SCRIPT),
                                     description=f"{APP_NAME} {APP_VERSION} - a MATLAB Command Window emulator.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--install', action='store_true',
                        help="install the application for the current user, then exit")
    action.add_argument('--install-systemwide', action='store_true',
                        help="install the application for all users (e.g. with sudo), then exit")
    action.add_argument('--uninstall', action='store_true', help="remove the installed application, then exit")
    action.add_argument('--update', action='store_true',
                        help="install a newer version from GitHub over the installed one, then exit")
    parser.add_argument('--version', action='version', version=f"{APP_NAME} {APP_VERSION}")
    args, _ = parser.parse_known_args(argv)  # macOS may add an argument such as -psn_0_123 from Finder
    if args.install or args.install_systemwide or args.uninstall or args.update:
        if args.update:
            ok, message = update_installed_app()
        elif args.uninstall:
            ok, message = uninstall_app()
        else:
            ok, message = install_app(args.install_systemwide)
        print(message)
        return 0 if ok else 1
    try:  # started from Launchpad or Finder the current folder is '/': files would go there
        if os.getcwd() == os.path.abspath(os.sep) or not os.access(os.getcwd(), os.W_OK):
            os.chdir(os.path.expanduser('~'))
    except OSError:
        pass
    set_macos_app_name(APP_NAME)
    root = tk.Tk(className=APP_NAME)
    app = MatlabConsoleApp(root)
    root.lift()
    root.after(300, app.startup_check)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
