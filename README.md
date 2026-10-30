# Matlab_Calc — MATLAB Command Window Emulator (`matlab_calc.py`)

**Version 0.1.3** — see [Version History](#version-history).

A lightweight, interactive desktop calculator built with Python, Tkinter, and NumPy that emulates the look, feel, and matrix-first syntax of the **MATLAB Command Window**.

---

## Features

- **Inline Interactive Command Window**: Type expressions directly after the `>> ` prompt in a single scrollable console window—just like native MATLAB. Previous outputs and prompts are protected from every kind of edit (typing, Backspace, Delete, cut, paste, Emacs-style shortcuts such as Ctrl+H) while remaining selectable for copying.
- **Native MATLAB Matrix & Vector Syntax**: Define row vectors (`[1 2 3]` or `[1, 2, 3]`), column vectors (`[1; 2; 3]`), 2D matrices (`[1 2; 3 4]`), block matrices (`[A A; A A]`) and ranges (`1:5`, `0:0.1:1`). MATLAB's whitespace rules apply inside brackets: `[1 -2]` has two elements, `[1 - 2]` has one.
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
| `APP_VERSION` | `'0.1.3'` | Version number, shown in the welcome banner (also available as `matlab_calc.__version__`) |
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
| `.^` | Element-wise power | `A .^ 2` |
| `'` | Conjugate transpose | `A'` |
| `.'` | Non-conjugate transpose | `A.'` |
| `-` `+` `~` | Unary minus, plus, logical NOT | `-A`, `~A` |
| `*` | Matrix or scalar multiplication | `A * B`, `A * 2` |
| `.*` | Element-wise multiplication | `A .* A` |
| `/` | Scalar division or matrix right-division (solves `X*B = A`) | `A / 3`, `A / B` |
| `\` | Matrix left-division (solves `A*X = B`) | `B \ [5; 6]` |
| `./` `.\` | Element-wise right / left division | `A ./ B` |
| `+` `-` | Addition, subtraction (with implicit expansion: `[1 2] + [10; 20]`) | `A + 1` |
| `:` | Range `start:stop` or `start:step:stop` | `0:0.1:1` |
| `==` `~=` `<` `<=` `>` `>=` | Comparisons (result is 1 or 0) | `A > 2` |
| `&` `\|` `&&` `\|\|` | Logical AND / OR | `a > 0 & a < 5` |

`/` and `\` use an exact solve for square non-singular matrices and least squares (pseudo-inverse) otherwise.

### 3. Built-in Constants & Mathematical Functions
- **Constants**: `pi`, `e`, `Inf`, `NaN`, `i`, `j` (imaginary unit; literals such as `3 + 4i` also work)
- **Trigonometric**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
- **Exponential & Logarithmic**: `sqrt`, `exp`, `log` (natural log), `log10` — `sqrt`, `log` and `log10` return complex results for negative inputs, as in MATLAB
- **Matrix Construction & Linear Algebra**:
  - `eye(n)` — $n \times n$ identity matrix
  - `zeros(n)`, `zeros(r, c)` or `zeros([r c])` — matrix of zeros
  - `ones(n)`, `ones(r, c)` or `ones([r c])` — matrix of ones
  - `inv(M)` — matrix inverse
  - `det(M)` — matrix determinant
  - `abs(x)` — absolute value / magnitude

### 4. Console & Workspace Commands
| Command | Description |
| :--- | :--- |
| `format short` / `format` | Cap display precision at up to 4 decimal places (default) |
| `format long` | Cap display precision at up to 15 decimal places |
| `whos` | List all variables currently in the workspace with their size and class |
| `clear <var> ...` | Remove specific variable(s) from the workspace |
| `clear` / `clear all` | Clear all variables from the workspace |
| `clc` | Clear the command window screen |
| `exit` / `quit` | Close the application |

Typing a variable's name shows its value without changing `ans`.

### 5. Not Supported
Array indexing (`A(2,3)`) and indexed assignment, character strings, user functions/scripts, and control flow (`if`, `for`, …). `asin`/`acos` return `NaN` outside [-1, 1]. Non-integer matrix powers use an eigendecomposition, so they require a diagonalisable matrix.

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

Runs 38 tests covering operators, precedence, parsing, number display, workspace commands, the read-only console (including a session inside the real Tk event loop) and the rounded icon. The GUI and icon tests are skipped when no display is available.

---

## Version History

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
