# Matlab_Calc — MATLAB Command Window Emulator (`matlab_calc.py`)

**Version 0.3.0** — see [Version History](#version-history).

A lightweight, interactive desktop calculator built with Python, Tkinter, and NumPy that emulates the look, feel, and matrix-first syntax of the **MATLAB Command Window**.

---

## Features

- **Inline Interactive Command Window**: Type expressions directly after the `>> ` prompt in a single scrollable console window—just like native MATLAB. Previous outputs and prompts are protected from every kind of edit (typing, Backspace, Delete, cut, paste, Emacs-style shortcuts such as Ctrl+H) while remaining selectable for copying.
- **Native MATLAB Matrix & Vector Syntax**: Define row vectors (`[1 2 3]` or `[1, 2, 3]`), column vectors (`[1; 2; 3]`), 2D matrices (`[1 2; 3 4]`), block matrices (`[A A; A A]`) and ranges (`1:5`, `0:0.1:1`). MATLAB's whitespace rules apply inside brackets: `[1 -2]` has two elements, `[1 - 2]` has one.
- **Indexing**: Use single elements, rows, columns and sub-matrices in calculations, as in `A(2,3) + A(1,4)`, `A(end, :)` or `v(2:end)`.
- **MATLAB Semantics**: Every value is a matrix (scalars are 1×1), operators follow MATLAB precedence and left-to-right associativity (`2*A.^2`, `-A.^2`, `2^3^2 = 64`), and element-wise operators work on scalars too (`2 ./ 4 = 0.5`).
- **Distinction Between Matrix & Element-Wise Operators**:
  - Matrix operations: `*`, `/`, `\`, `^`, `'` (conjugate transpose), `.'` (transpose).
  - Element-wise operations: `.*`, `./`, `.\`, `.^`.
  - Raises MATLAB-style errors when operands have incompatible dimensions.
- **Multiple Statements and Comments**: `a = 1; b = 2, c = a + b` runs three statements (a trailing `;` suppresses output); `%` starts a comment and `...` continues a line. Multi-line pastes run line by line.
- **Smart Adaptive Number Formatting**:
  - Automatically aligns matrix columns and matches the decimal precision across all elements of a vector/matrix to the maximum significant decimals needed by the calculation (capped at 4 digits in `format short` and 15 digits in `format long`).
  - Very small or very large numbers switch to scientific notation (`1e-05`, `1.2346e+11`); `Inf`, `NaN` and complex results (`1 + 2i`) display as in MATLAB.
- **Workspace Management**: Persistent variables across calculations, automatic `ans` assignment, output suppression with trailing `;`, and built-in workspace commands (`whos`, `clear`, `clc`, `format short`, `format long`).
- **Built-in Help**: `help` lists every command, operator and function; `help <name>` shows the full help of one of them, as in MATLAB.

---

## Requirements

- **Python** 3.8+
- **NumPy** (`pip install numpy`)
- **Tkinter** (included with standard Python installations)
- *Optional:* **Pillow** (`pip install pillow`) — smooth, anti-aliased edges for the rounded app icon. Without it the icon is still shrunk and rounded, but its edges are slightly jagged.
- *Optional, macOS:* **PyObjC** (`pip install pyobjc-framework-Cocoa`) — shows `APP_NAME` instead of “Python” in the menu bar.

---

## Running the Application

```bash
python3 matlab_calc.py
```

Place `icon.png` next to `matlab_calc.py` to use it as the window / Dock icon.

---

## Configuration

All settings are constants at the top of `matlab_calc.py`:

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `APP_NAME` | `'Matlab_Calc'` | Window title, Tk class name and macOS menu-bar name |
| `APP_VERSION` | `'0.3.0'` | Version number, shown in the welcome banner and by the `version` command (also available as `matlab_calc.__version__`) |
| `WINDOW_WIDTH`, `WINDOW_HEIGHT` | `750`, `520` | Initial window size in pixels |
| `APP_ICON_FILE` | `'icon.png'` | Icon image, looked up next to the script |
| `APP_ICON_MARGIN` | `0.1` | Transparent margin on each side, as a fraction of the icon size (keeps the Dock icon the same size as native macOS icons) |
| `APP_ICON_ROUNDING` | `0.20` | Corner radius, as a fraction of the visible icon size (`0` = square corners, `0.5` = circle) |
| `FONT_FAMILIES`, `FONT_SIZE` | `('Consolas', 'Menlo', …)`, `11` | The first installed monospace font is used, so matrix columns stay aligned on every OS |
| `BACKGROUND_COLOR`, `TEXT_COLOR`, `PROMPT_COLOR`, `CURSOR_COLOR` | dark theme | Console colors |
| `WINDOW_PADDING`, `PROMPT` | `10`, `'>> '` | Space around the console; prompt text |
| `SHORT_DECIMALS`, `LONG_DECIMALS` | `4`, `15` | Maximum decimals for `format short` / `format long` |
| `MAX_DISPLAY_ELEMENTS` | `10000` | Larger results are summarised instead of printed (end the statement with `;`) |

---

## Supported Syntax & Operations

### 1. Vectors and Matrices
```matlab
>> A = [1 3 2.4 6 4.4 3.2 1];   % Semicolon suppresses output
>> A * 2
ans =
    2.0   6.0   4.8   12.0   8.8   6.4   2.0

>> B = [1 2; 3 4]
B =
    1   2
    3   4

>> x = B \ [5; 6]
x =
    -4.0
     4.5

>> a = 2; b = a ./ 4, c = 1:0.5:2
b =
    0.5

c =
    1.0   1.5   2.0
```

### 2. Operators
Listed from highest to lowest precedence group:

| Operator | Description | Example |
| :--- | :--- | :--- |
| `^` | Matrix power (square matrix and scalar exponent, or scalar base and square matrix) | `B ^ 2`, `2 ^ B` |
| `.^` | Element-wise power (also `power(A, B)`) | `A .^ 2` |
| `'` | Conjugate transpose | `A'` |
| `.'` | Non-conjugate transpose | `A.'` |
| `-` `+` `~` | Unary minus, plus, logical NOT | `-A`, `~A` |
| `*` | Matrix or scalar multiplication | `A * B`, `A * 2` |
| `.*` | Element-wise multiplication (also `times(A, B)`) | `A .* A` |
| `/` | Scalar division or matrix right-division (solves `X*B = A`) | `A / 3`, `A / B` |
| `\` | Matrix left-division (solves `A*X = B`) | `B \ [5; 6]` |
| `./` `.\` | Element-wise right / left division | `A ./ B` |
| `+` `-` | Addition, subtraction (with implicit expansion: `[1 2] + [10; 20]`) | `A + 1` |
| `:` | Range `start:stop` or `start:step:stop` | `0:0.1:1` |
| `==` `~=` `<` `<=` `>` `>=` | Comparisons (result is 1 or 0) | `A > 2` |
| `&` `\|` `&&` `\|\|` | Logical AND / OR | `a > 0 & a < 5` |

`/` and `\` use an exact solve for square non-singular matrices and least squares (pseudo-inverse) otherwise.

### 3. Indexing
Elements of a vector or matrix are numbered from 1, and the result can be used in any calculation:

```matlab
>> A = [1 2 3 4; 5 6 7 8; 9 10 11 12];
>> A(2,3) + A(1,4)
ans =
    11

>> A(end, :)
ans =
    9   10   11   12

>> A(1:2, [1 3])
ans =
    1   3
    5   7

>> v = [10 20 30 40];
>> v(2:end) - v(1)
ans =
    10   20   30
```

| Form | Meaning |
| :--- | :--- |
| `A(i, j)` | Element in row `i`, column `j` |
| `A(k)` | Element `k`, counting down the columns as in MATLAB (`A(2)` is `A(2,1)`); for a vector simply its `k`-th element |
| `A(i, :)`, `A(:, j)` | Row `i`, column `j` |
| `A(:)` | All elements as one column |
| `A(1:2, [1 3])` | Several rows and columns at once: any vector or range of positions |
| `end` | The last position in that place: `A(end)`, `A(end, 1)`, `v(2:end)`, `v(end-1)` |

Subscripts can be any expression that gives positive whole numbers, including other indexed values (`A(v(1)/10, end)` is `A(1, 4)`). Positions outside the matrix give MATLAB's error messages, e.g. `Index in position 2 exceeds array bounds. Index must not exceed 4.` A variable hides a function of the same name, so after `sin = [5 6 7]`, `sin(2)` is 6.

### 4. Built-in Constants & Mathematical Functions
- **Constants**: `pi`, `e`, `Inf`, `NaN`, `i`, `j` (imaginary unit; literals such as `3 + 4i` also work)
- **Trigonometric**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
- **Exponential & Logarithmic**: `sqrt`, `exp`, `log` (natural log), `log10` — `sqrt`, `log` and `log10` return complex results for negative inputs, as in MATLAB
- **Element-wise Arithmetic**:
  - `power(A, B)` — element-wise power, the same as `A.^B` (`power(2, [1 2 3])` is `2 4 8`)
  - `times(A, B)` — element-wise multiplication, the same as `A.*B` (`times([1 2 3], [4 5 6])` is `4 10 18`)
- **Matrix Construction & Linear Algebra**:
  - `eye(n)` — $n \times n$ identity matrix
  - `zeros(n)`, `zeros(r, c)` or `zeros([r c])` — matrix of zeros
  - `ones(n)`, `ones(r, c)` or `ones([r c])` — matrix of ones
  - `inv(M)` — matrix inverse
  - `det(M)` — matrix determinant
  - `abs(x)` — absolute value / magnitude
- **Rounding** (element-wise; complex numbers are rounded in their real and imaginary parts separately):
  - `round(x)` — nearest integer, halves rounded away from zero as in MATLAB (`round(2.5) = 3`, `round(-2.5) = -3`)
  - `round(x, n)` — round to `n` decimals (`round(pi, 2) = 3.14`); a negative `n` rounds to tens, hundreds, … (`round(1234, -2) = 1200`)
  - `floor(x)` — round toward minus infinity (`floor(-2.5) = -3`)
  - `ceil(x)` — round toward plus infinity (`ceil(-2.5) = -2`)

### 5. Console & Workspace Commands
| Command | Description |
| :--- | :--- |
| `format short` / `format` | Cap display precision at up to 4 decimal places (default) |
| `format long` | Cap display precision at up to 15 decimal places |
| `whos` | List all variables currently in the workspace with their size and class |
| `clear <var> ...` | Remove specific variable(s) from the workspace |
| `clear` / `clear all` | Clear all variables from the workspace |
| `clc` | Clear the command window screen |
| `help` | List all commands, operators and functions |
| `help <name>` | Show the help of a command or function, e.g. `help round`; `help ops` (or `help .*`, `help +`, …) explains the operators |
| `version` | Show the application name and version number |
| `exit` / `quit` | Close the application |

Typing a variable's name shows its value without changing `ans`.

### 6. Getting Help
`help` on its own lists the console commands, the operators and every function grouped by topic, each with a one-line description. `help <name>` shows the full help text:

```matlab
>> help floor
  floor - Round toward minus infinity.

  floor(X) rounds each element of X to the nearest integer less than or
  equal to it: floor(2.7) is 2, floor(-2.5) is -3.
  Complex numbers are rounded in their real and imaginary parts separately.

  See also ceil, round.
```

As in MATLAB, where a function's help is the comment block at the top of its `.m` file, the help of every command and function here is the comment block at the start of its code: the docstring of the Python function that implements it. The first line (`floor - Round toward minus infinity.`) is the summary shown in the `help` list.

### 7. Adding Your Own Functions
A new built-in function needs only a Python function with a help docstring and the `@builtin` decorator, placed next to the others in `matlab_calc.py`. It then works in calculations and appears in `help` automatically:

```python
@builtin('cube', category=CAT_MATH)
def fn_cube(x):
    """cube - Cube of each element.

    cube(X) is X.^3, the cube of each element of X.

    See also power, sqrt.
    """
    return op_power(x, to_matrix(3))
```

Arguments arrive as 2-D NumPy arrays (scalars are 1×1) and the function must return one (`to_matrix` converts any result). The number of inputs comes from the Python signature; parameters with a default value (`n=None`) are optional. Console commands work the same way with the `@console_command` decorator on a `MatlabEngine` method. The tests check that every command and function has a help text whose first line has the form `name - Summary.`

### 8. Not Supported
Changing elements by indexed assignment (`A(2,3) = 5`), logical indexing (`A(A > 2)`), more than two subscripts, character strings, user functions/scripts, and control flow (`if`, `for`, …). `asin`/`acos` return `NaN` outside [-1, 1]. Non-integer matrix powers use an eigendecomposition, so they require a diagonalisable matrix.

---

## Keyboard Shortcuts

- **`Enter`** (or keypad **`Enter`**): Execute the command at the current `>> ` prompt.
- **`Up` / `Down` Arrows**: Cycle through previous commands in history.
- **`Home`**: Move the cursor to the start of the current input (right after `>> `).
- **`Ctrl+C` / `Cmd+C`**: Copy selected text from anywhere in the console history.
- **`Ctrl+V` / `Cmd+V`**: Paste into the active command line.

---

## Testing

```bash
python3 test_matlab_calc.py
```

Runs 55 tests covering operators, functions, indexing, the help texts, precedence, parsing, number display, workspace commands, the read-only console (including a session inside the real Tk event loop) and the rounded icon. The GUI and icon tests are skipped when no display is available.

---

## Version History

### 0.3.0 — 2026-10-31
- **Indexing:** elements, rows, columns and sub-matrices can be used in calculations, e.g. `A(2,3) + A(1,4)`, `A(k)`, `A(i, :)`, `A(:, j)`, `A(:)`, `A(1:2, [1 3])`, with `end` for the last position (`v(2:end)`). Results have MATLAB's shapes, and invalid positions give MATLAB's error messages.
- **Help:** `help ops` (also `help end`) has a new section on indexing.

### 0.2.0 — 2026-10-31
- **New command:** `help` lists all commands, operators and functions; `help <name>` shows the help of one of them (`help round`), and `help ops` explains the operators.
- **Help texts:** as in MATLAB, every command, function and constant carries its help as the comment block (docstring) at the start of its code; the first line is the summary shown by `help`.
- **New functions:** `power(A, B)`, the same as `A.^B`, and `times(A, B)`, the same as `A.*B`.
- **Code:** built-in functions, constants and console commands are registered with the `@builtin` and `@console_command` decorators, so a new function only needs one decorated Python function with its help text.
- **Fix:** a command word followed by `=` is now an assignment, so `help = 3` creates a variable named `help`.

### 0.1.4 — 2026-10-30
- **New command:** `version` shows the application name and version number.
- **New functions:** `round(x)`, `round(x, n)`, `floor(x)` and `ceil(x)`. `round` follows MATLAB and rounds halves away from zero.

### 0.1.3 — 2026-10-30
- **New command:** `exit` (or `quit`) closes the application. Statements after it on the same line are not run.

### 0.1.2 — 2026-10-30
- **Fixed crash on the first key press:** the read-only protection of the console was implemented as a Python callback; Tk errors raised inside it (for example when checking for a text selection while nothing is selected) were re-raised by Tkinter's event loop and closed the app. The protection now runs entirely in Tcl.
- **Tests:** added a test that types inside the running Tk event loop.

### 0.1.1 — 2026-10-30
- **Correct results:** replaced the Python-operator translation with a MATLAB expression parser. Element-wise operators now work on scalars (`2 ./ 4` returned `0`, `2 .^ 3` returned `2`), and precedence and associativity match MATLAB (`2*A.^2`, `-A.^2`, `2^3^2 = 64`).
- **Input fixes:** `==` comparisons, several statements per line, `%` comments, `[]`, sizes computed from expressions (`zeros(n, n)`), and variable names starting with `clear` now work.
- **New syntax:** ranges (`1:5`), left division (`A\b`, `.\`), comparison and logical operators, and the constants `Inf`, `NaN`, `i` and `j`.
- **Display:** very small and very large numbers use scientific notation instead of showing `0`; `Inf`, `NaN` and complex results look as they do in MATLAB; very large results are summarised.
- **Console:** earlier output is fully read-only (Ctrl+H could delete the prompt), keypad Enter works, and a monospace font is picked automatically so matrix columns line up on macOS.
- **Configuration:** settings are parameters at the top of the file (`APP_NAME`, `APP_VERSION`, `WINDOW_WIDTH`, `WINDOW_HEIGHT`, fonts, colours, …).
- **App icon:** shrunk with a transparent margin (`APP_ICON_MARGIN`) and rounded corners (`APP_ICON_ROUNDING`) to match native macOS Dock icons.
- **Tests:** added `test_matlab_calc.py`.

### 0.1.0
- Initial version: Tkinter command window with a `>>` prompt and command history, MATLAB matrix and vector syntax, matrix and element-wise operators, adaptive number formatting, and the workspace commands `whos`, `clear`, `clc` and `format`.
