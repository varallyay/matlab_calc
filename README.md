# MATLAB Command Window Emulator (`matlab_calc.py`)

A lightweight, interactive desktop calculator built with Python, Tkinter, and NumPy that emulates the look, feel, and matrix-first syntax of the **MATLAB Command Window**.

---

## Features

- **Inline Interactive Command Window**: Type expressions directly after the `>> ` prompt in a single scrollable console window—just like native MATLAB. Previous outputs and prompts are protected from accidental deletion while remaining selectable for copying.
- **Native MATLAB Matrix & Vector Syntax**: Define row vectors (`[1 2 3]` or `[1, 2, 3]`), column vectors (`[1; 2; 3]`), 2D matrices (`[1 2; 3 4]`), and block matrices.
- **Distinction Between Matrix & Element-Wise Operators**:
  - Matrix operations: `*` (matrix/scalar multiplication), `/` (scalar or matrix right-division via pseudo-inverse), `^` (matrix power for square matrices), `'` (conjugate transpose), `.'` (transpose).
  - Element-wise operations: `.*`, `./`, `.^`.
  - Raises MATLAB-style dimension errors when matrix operators (such as `^` or `*`) are used on incompatible vector/matrix dimensions.
- **Smart Adaptive Number Formatting**:
  - Automatically aligns matrix columns and matches the decimal precision across all elements of a vector/matrix to the maximum significant decimals needed by the calculation (capped at 4 digits in `format short` and 15 digits in `format long`).
- **Workspace Management**: Persistent variables across calculations, automatic `ans` assignment, output suppression with trailing `;`, and built-in workspace commands (`whos`, `clear`, `clc`, `format short`, `format long`).

---

## Requirements

- **Python** 3.8+
- **NumPy** (`pip install numpy`)
- **Tkinter** (included with standard Python installations)

---

## Running the Application

```bash
python3 matlab_calc.py
```

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
```

### 2. Matrix vs. Element-Wise Operators
| Operator | Description | Example |
| :--- | :--- | :--- |
| `*` | Matrix or scalar multiplication | `A * B`, `A * 2` |
| `.*` | Element-wise multiplication | `A .* A` |
| `/` | Scalar division or matrix right-division (`A * pinv(B)`) | `A / 3`, `B / A` |
| `./` | Element-wise right-division | `A ./ B` |
| `^` | Matrix power (requires square matrix and scalar exponent) | `B ^ 2` |
| `.^` | Element-wise power | `A .^ 2` |
| `'` | Conjugate transpose | `A'` |
| `.'` | Non-conjugate transpose | `A.'` |

### 3. Built-in Constants & Mathematical Functions
- **Constants**: `pi`, `e`
- **Trigonometric**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
- **Exponential & Logarithmic**: `sqrt`, `exp`, `log` (natural log), `log10`
- **Matrix Construction & Linear Algebra**:
  - `eye(n)` — $n \times n$ identity matrix
  - `zeros(n)` or `zeros(r, c)` — matrix of zeros
  - `ones(n)` or `ones(r, c)` — matrix of ones
  - `inv(M)` — matrix inverse
  - `det(M)` — matrix determinant
  - `abs(x)` — absolute value / magnitude

### 4. Console & Workspace Commands
| Command | Description |
| :--- | :--- |
| `format short` | Cap display precision at up to 4 decimal places (default) |
| `format long` | Cap display precision at up to 15 decimal places |
| `whos` | List all variables currently in the workspace with their size and class |
| `clear <var>` | Remove specific variable(s) from the workspace |
| `clear` / `clear all` | Clear all variables from the workspace |
| `clc` | Clear the command window screen |

---

## Keyboard Shortcuts

- **`Enter`**: Execute the command at the current `>> ` prompt.
- **`Up` / `Down` Arrows**: Cycle through previous commands in history.
- **`Home`**: Move the cursor to the start of the current input (right after `>> `).
- **`Ctrl+C` / `Cmd+C`**: Copy selected text from anywhere in the console history.
- **`Ctrl+V` / `Cmd+V`**: Paste into the active command line.
