"""
Matlab_Calc - a MATLAB Command Window emulator built with Tkinter and NumPy.

Run with:  python3 matlab_calc.py
"""

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
APP_VERSION = '0.1.3'           # Shown in the welcome banner; see the version history in README.md

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
    """Convert any numeric value to a 2-D float or complex array."""
    a = np.asarray(x)
    if np.iscomplexobj(a):
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

UNARY_OPS = {
    '-': lambda a: to_matrix(-a),
    '+': lambda a: a,
    '~': lambda a: to_matrix(a == 0),
    "'": lambda a: to_matrix(a.conj().T),   # conjugate transpose
    ".'": lambda a: to_matrix(a.T),         # transpose
}


# =============================================================================
#  Built-in constants and functions
# =============================================================================
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


def fn_inv(x):
    _need_square('inv', x)
    try:
        return to_matrix(np.linalg.inv(x))
    except np.linalg.LinAlgError:
        raise MatlabError("Error using inv\nMatrix is singular to working precision.") from None


def fn_det(x):
    _need_square('det', x)
    return to_matrix(np.linalg.det(x))


def _unary_fn(np_func):
    return lambda x: to_matrix(np_func(x))


CONSTANTS = {
    'pi': np.pi, 'e': np.e,
    'Inf': np.inf, 'inf': np.inf, 'NaN': np.nan, 'nan': np.nan,
    'i': 1j, 'j': 1j,
}

FUNCTIONS = {  # name: (callable, minimum number of arguments, maximum number of arguments)
    'sin': (_unary_fn(np.sin), 1, 1),
    'cos': (_unary_fn(np.cos), 1, 1),
    'tan': (_unary_fn(np.tan), 1, 1),
    'asin': (_unary_fn(np.arcsin), 1, 1),
    'acos': (_unary_fn(np.arccos), 1, 1),
    'atan': (_unary_fn(np.arctan), 1, 1),
    'sqrt': (_unary_fn(np.emath.sqrt), 1, 1),     # sqrt(-4) = 2i, as in MATLAB
    'exp': (_unary_fn(np.exp), 1, 1),
    'log': (_unary_fn(np.emath.log), 1, 1),       # log(-1) = pi*i
    'log10': (_unary_fn(np.emath.log10), 1, 1),
    'abs': (_unary_fn(np.abs), 1, 1),
    'eye': (lambda *a: to_matrix(np.eye(*_size_args('eye', a))), 0, 2),
    'zeros': (lambda *a: to_matrix(np.zeros(_size_args('zeros', a))), 0, 2),
    'ones': (lambda *a: to_matrix(np.ones(_size_args('ones', a))), 0, 2),
    'inv': (fn_inv, 1, 1),
    'det': (fn_det, 1, 1),
}


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
        """Returns (assigned variable name or None, expression tree)."""
        target = None
        nxt = self.peek(1)
        if self.at('id') and nxt[0] == 'op' and nxt[1] == '=':
            target = self.advance()[1]
            self.advance()
        node = self.parse_expr()
        if self.at_op('='):
            raise MatlabError("Error: Only assignments of the form 'name = expression' are supported.")
        if not self.at('eof'):
            self.fail()
        return target, node

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
            if not self.at('('):
                return ('name', tok[1], None)
            self.advance()
            args = []
            if not self.at(')'):
                args.append(self.parse_expr())
                while self.at(','):
                    self.advance()
                    args.append(self.parse_expr())
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


def evaluate(node, variables):
    """Evaluate a parse tree. Variables shadow constants and functions, as in MATLAB."""
    kind = node[0]
    if kind == 'num':
        return to_matrix(node[1])
    if kind == 'name':
        name, args = node[1], node[2]
        if name in variables:
            if args:  # x() just returns x
                raise MatlabError(f"Error: Indexing into variables, as in {name}(...), is not supported.")
            return variables[name]
        if name in CONSTANTS:
            if args:
                raise MatlabError(f"Error using {name}\nToo many input arguments.")
            return to_matrix(CONSTANTS[name])
        if name in FUNCTIONS:
            func, lo, hi = FUNCTIONS[name]
            args = args or []
            if not lo <= len(args) <= hi:
                problem = 'Not enough' if len(args) < lo else 'Too many'
                raise MatlabError(f"Error using {name}\n{problem} input arguments.")
            return func(*[evaluate(a, variables) for a in args])
        raise MatlabError(f"Unrecognized function or variable '{name}'.")
    if kind == 'un':
        return UNARY_OPS[node[1]](evaluate(node[2], variables))
    if kind == 'bin':
        op = node[1]
        left = evaluate(node[2], variables)
        if op in ('&&', '||'):  # short-circuit
            lv = _logical_scalar(left, op)
            if lv == (op == '||'):
                return to_matrix(lv)
            return to_matrix(_logical_scalar(evaluate(node[3], variables), op))
        return BINARY_OPS[op](left, evaluate(node[3], variables))
    if kind == 'range':
        start = evaluate(node[1], variables)
        step = to_matrix(1.0) if node[2] is None else evaluate(node[2], variables)
        return op_colon(start, step, evaluate(node[3], variables))
    if kind == 'matrix':
        rows = []
        for row in node[1]:
            parts = [m for m in (evaluate(e, variables) for e in row) if m.size]
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


class MatlabEngine:
    def __init__(self):
        self.format_mode = 'short'  # 'short' or 'long'
        self.variables = {}

    # ---- execution ---------------------------------------------------------
    def execute(self, line):
        """Run one command line. Returns output chunks (str), CLEAR_SCREEN and/or EXIT_APP.

        Like MATLAB, the remaining statements on the line are skipped after an error or 'exit'.
        """
        output = []
        try:
            for text, suppressed in split_statements(line):
                output.extend(self._run_statement(text, suppressed))
                if output and output[-1] is EXIT_APP:
                    break
        except MatlabError as err:
            output.append(f"{err}\n")
        except np.linalg.LinAlgError as err:
            output.append(f"Error: {err}.\n")
        except MemoryError:
            output.append("Error: Out of memory.\n")
        except RecursionError:
            output.append("Error: The expression is nested too deeply.\n")
        except Exception as err:  # unexpected NumPy/Python failure: report it, keep the app alive
            output.append(f"Error: {err}\n")
        return output

    def _run_statement(self, text, suppressed):
        result = self._run_command(text)
        if result is not None:
            return result
        target, node = Parser(tokenize(text)).parse_statement()
        with np.errstate(all='ignore'):  # 1/0 -> Inf and 0/0 -> NaN, as in MATLAB
            value = evaluate(node, self.variables)
        if target is None and node[0] == 'name' and node[2] is None and node[1] in self.variables:
            target = node[1]  # typing a variable's name shows it without changing 'ans'
        else:
            target = target or 'ans'
            self.variables[target] = value
        return [] if suppressed else [f"{target} =\n{self.format_value(value)}\n"]

    def _run_command(self, text):
        """Console commands (command syntax). Returns None if text is not a command."""
        words = text.split()
        name, args = words[0], words[1:]
        if name == 'clc' and not args:
            return [CLEAR_SCREEN]
        if name in ('exit', 'quit') and not args:
            return [EXIT_APP]
        if name == 'whos' and not args:
            return [self._whos()]
        if name == 'format' and len(args) <= 1 and all(w.isalpha() for w in args):
            mode = args[0].lower() if args else 'short'
            if mode not in ('short', 'long'):
                raise MatlabError(f"Error using format\nUnsupported format '{args[0]}'. "
                                  "Use 'format short' or 'format long'.")
            self.format_mode = mode
            return []
        if name == 'clear' and all(_IDENT_RE.fullmatch(w) for w in args):
            if not args or 'all' in args:
                self.variables.clear()
            else:
                for var in args:
                    self.variables.pop(var, None)
            return []
        return None

    def _whos(self):
        if not self.variables:
            return "  Workspace is empty.\n"
        lines = [f"  {'Name':<10} {'Size':<10} {'Class':<10}", "  " + "-" * 32]
        for name in sorted(self.variables, key=str.lower):
            v = self.variables[name]
            cls = "double (complex)" if np.iscomplexobj(v) else "double"
            lines.append(f"  {name:<10} {f'{v.shape[0]}x{v.shape[1]}':<10} {cls:<10}")
        return "\n".join(lines) + "\n"

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
        if a.size == 0:
            return "    []" if (rows, cols) == (0, 0) else f"    {rows}x{cols} empty double matrix"
        if a.size > MAX_DISPLAY_ELEMENTS:
            return (f"    {rows}x{cols} double matrix (too large to display; "
                    "end the statement with ';' to suppress output)")
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
            "Commands: 'format short', 'format long', 'clear <var>', 'clear all', 'whos', 'clc', 'exit'\n"
            + "-" * 80 + "\n"
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
