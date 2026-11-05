# Matlab_Calc — MATLAB Command Window Emulator (`matlab_calc.py`)

**Version 0.9.0** — see [Version History](#version-history).

A lightweight, interactive desktop calculator built with Python, Tkinter, and NumPy that emulates the look, feel, and matrix-first syntax of the **MATLAB Command Window**.

---

## Features

- **Inline Interactive Command Window**: Type expressions directly after the `>> ` prompt in a single scrollable console window—just like native MATLAB. Previous outputs and prompts are protected from every kind of edit (typing, Backspace, Delete, cut, paste, Emacs-style shortcuts such as Ctrl+H) while remaining selectable for copying.
- **Native MATLAB Matrix & Vector Syntax**: Define row vectors (`[1 2 3]` or `[1, 2, 3]`), column vectors (`[1; 2; 3]`), 2D matrices (`[1 2; 3 4]`), block matrices (`[A A; A A]`) and ranges (`1:5`, `0:0.1:1`). MATLAB's whitespace rules apply inside brackets: `[1 -2]` has two elements, `[1 - 2]` has one.
- **Indexing**: Use single elements, rows, columns and sub-matrices in calculations, as in `A(2,3) + A(1,4)`, `A(end, :)` or `v(2:end)`; change them (`A(2,3) = 5`), delete them (`A(2, :) = []`) or select them by a condition (`A(A > 2)`, `A(A > 2) = 0`). `find` gives the positions where a condition holds.
- **MATLAB Semantics**: Every value is a matrix (scalars are 1×1), operators follow MATLAB precedence and left-to-right associativity (`2*A.^2`, `-A.^2`, `2^3^2 = 64`), and element-wise operators work on scalars too (`2 ./ 4 = 0.5`).
- **Distinction Between Matrix & Element-Wise Operators**:
  - Matrix operations: `*`, `/`, `\`, `^`, `'` (conjugate transpose), `.'` (transpose).
  - Element-wise operations: `.*`, `./`, `.\`, `.^`.
  - Raises MATLAB-style errors when operands have incompatible dimensions.
- **Several Outputs**: functions such as `max`, `min`, `size`, `find`, `fopen` and `fscanf` return more than one result: `[m, i] = max(A)`, `[rows, cols] = size(A)`; `~` skips an output (`[~, i] = min(v)`).
- **Text and Files**: text in quotes (`'Hello'`), `disp`, `fprintf`, `sprintf` and `num2str` for formatted output, and `fopen` / `fprintf` / `fscanf` / `fclose` with file identifiers to write and read text files.
- **Plots**: `plot`, `figure`, `hold`, `xlabel`, `ylabel`, `title`, `legend`, `grid`, `axis`, `set` and `get` draw line plots in figure windows that look like MATLAB's, with a toolbar to zoom, pan and save (made with matplotlib).
- **Data Analysis and Random Numbers**: `max`, `min`, `mean`, `median`, `std`, `rand`, `randn` and more, working down the columns of a matrix as in MATLAB.
- **Multiple Statements and Comments**: `a = 1; b = 2, c = a + b` runs three statements (a trailing `;` suppresses output); `%` starts a comment and `...` continues a line. Multi-line pastes run line by line.
- **Smart Adaptive Number Formatting**:
  - Automatically aligns matrix columns and matches the decimal precision across all elements of a vector/matrix to the maximum significant decimals needed by the calculation (capped at 4 digits in `format short` and 15 digits in `format long`).
  - Very small or very large numbers switch to scientific notation (`1e-05`, `1.2346e+11`); `Inf`, `NaN` and complex results (`1 + 2i`) display as in MATLAB.
- **Workspace Management**: Persistent variables across calculations, automatic `ans` assignment, output suppression with trailing `;`, and built-in workspace commands (`whos`, `clear`, `clc`, `format short`, `format long`).
- **Installable**: on macOS and Linux it installs itself as a normal application (asked at the first start), so it starts from Launchpad or the application menu.
- **Automatic Updates**: once a week the installed program looks for a new version on [GitHub](https://github.com/varallyay/matlab_calc) and, if you agree, downloads and installs it.
- **Built-in Help**: `help` lists every command, operator and function; `help <name>` shows the full help of one of them, as in MATLAB. *Help > Matlab_Calc Help* describes the program.
- **Menus**: *File > Open* runs a script (`.m` file) line by line, *File > Save* saves the commands of the session as a script, *Settings* chooses the font, the number format and the update checks, and *About* shows the version and the credits.

---

## Requirements

- **Python** 3.8+
- **NumPy** (`pip install numpy`)
- **Tkinter** (included with standard Python installations)
- *Optional, for plots:* **matplotlib** (`pip install matplotlib`; included in Anaconda). Without it everything else works, and the plot commands say how to install it.
- *Optional:* **Pillow** (`pip install pillow`) — smooth, anti-aliased edges for the rounded app icon. Without it the icon is still shrunk and rounded, but its edges are slightly jagged. On macOS the icon of the installed app is made with Pillow, or with the built-in tools `sips` and `iconutil`.
- *Optional, macOS:* **PyObjC** (`pip install pyobjc-framework-Cocoa`) — shows `APP_NAME` instead of “Python” in the menu bar.

---

## Installing and Running

Put `icon.png` next to `matlab_calc.py` (it becomes the window, Dock and menu icon) and start the program once from a terminal:

```bash
python3 matlab_calc.py
```

At this first start a window asks whether to install it:

| Button | What happens |
| :--- | :--- |
| **Install systemwide** | For all users of the computer. macOS: `/Applications/Matlab_Calc.app`. Linux: program files in `/opt/matlab_calc`, menu entry in `/usr/local/share/applications` (needs administrator rights, see below). |
| **Only for me** | For your user account only. macOS: `~/Applications/Matlab_Calc.app`. Linux: `~/.local/share/matlab_calc` and `~/.local/share/applications`. |
| **Don't install** | Nothing is installed; the program keeps working from the terminal. |

The question is asked only once. After installing, start Matlab_Calc like any other application: from **Launchpad**, **Spotlight** or the **Applications** folder on macOS, or from the **application menu** on Linux (search for *Matlab_Calc*); the terminal is no longer needed.

- **How it works:** installing copies `matlab_calc.py` and `icon.png` to the place above and creates a launcher (a `Matlab_Calc.app` bundle on macOS, a `.desktop` menu entry on Linux). The launcher starts the copy with the same Python that did the installation, so NumPy is always found, and it keeps working if you move or delete the folder you installed from.
- **Updating from a downloaded file:** when you start a newer version from your download folder (`python3 matlab_calc.py`), it offers to update the installed copy (**Update** / **Not now**; *Not now* is remembered for that version). Typing `install` updates it at any time.
- **Later:** the commands `install`, `install systemwide` and `uninstall` do the same from the command window (see `help install`).
- **From a terminal**, without opening the window:

  ```bash
  python3 matlab_calc.py --install              # for you
  python3 matlab_calc.py --install-systemwide   # for all users
  python3 matlab_calc.py --uninstall            # remove installed copies
  python3 matlab_calc.py --update               # install a newer version from GitHub
  python3 matlab_calc.py --version
  ```

  On Linux, installing for all users needs administrator rights: run `sudo python3 matlab_calc.py --install-systemwide` (with the full path of the Python that has NumPy, e.g. `sudo /opt/anaconda3/bin/python3 ...`). If you try it without the rights, the program prints the exact command. On macOS, administrator accounts can install into `/Applications` directly.
- **Settings file:** `~/.matlab_calc.json` in your home folder records that the question was answered, what was installed and when updates were last looked for. Delete it to be asked again the next time you start the program from your download folder.

### Updates from GitHub
Once a week (`UPDATE_CHECK_DAYS`), when the installed program starts, it looks in the background for a newer version in the GitHub project [varallyay/matlab_calc](https://github.com/varallyay/matlab_calc) (`UPDATE_REPOSITORY`). The window stays usable meanwhile, and without an internet connection nothing happens until the next start. If a newer version exists, a window with the program icon asks:

| Button | What happens |
| :--- | :--- |
| **Update now** | Downloads the new version (and its icon) and installs it over the installed copy. A second window offers **Restart now** (the variables in the workspace are not kept) or **Later**. |
| **Not now** | Nothing changes; you are asked again at the next weekly check. |
| **Skip this version** | You are not asked about this version again; a later version is offered as usual. |

The checkbox *Look for new versions on GitHub every week* in this window (and in the installation question) turns the weekly check off or on; the Settings window also sets how many days to wait between checks. **Help > Check for Updates…** looks right away and says what it found: it offers a newer version (even one you skipped before), or tells you that the program is up to date or that GitHub could not be reached. In the command window, `update` looks for a new version right away and installs it, and `update off` / `update on` switch the weekly check. Only an installed program is updated; a program started from your download folder is never changed (Check for Updates then offers to open the GitHub page).

**Publishing a new version** (for the maintainer): put `matlab_calc.py`, and `icon.png` if it changed, in the top folder of the repository and increase `APP_VERSION`. If the project has GitHub *releases*, the newest release is used, so a version is offered only once it is released; without releases, the newest version on the main branch is used as soon as it is pushed. An update is offered only when its `APP_VERSION` is higher than the installed one, so an older version is never installed by accident. The downloaded file is checked to be a valid program before it replaces the installed copy.

If Python from the python.org installer on macOS cannot verify GitHub's security certificate, the `update` command says so; running *Install Certificates.command* in that Python's folder in Applications fixes it (Anaconda's Python does not need this).

---

## Menus and Settings

| macOS | Linux | What it does |
| :--- | :--- | :--- |
| **Matlab_Calc > About Matlab_Calc** | **Help > About Matlab_Calc** | The program icon, name and version, the GitHub page and the credits |
| **Matlab_Calc > Settings…** (⌘,) | **File > Settings…** | Font, number format and update checks, see below |
| **File > Open…** (⌘O) | **File > Open…** (Ctrl+O) | Runs the commands of a script (`.m` file) |
| **File > Save…** (⌘S) | **File > Save…** (Ctrl+S) | Saves the commands of this session as a script |
| **Edit** | **Edit** | Cut, Copy, Paste, Select All |
| **Help > Matlab_Calc Help** | **Help > Matlab_Calc Help** (F1) | What the program can do, and how to get more help with the `help` command |
| **Help > Check for Updates…** | **Help > Check for Updates…** | Looks for a new version on GitHub right away |
| **Matlab_Calc > Quit Matlab_Calc** (⌘Q) | **File > Quit** (Ctrl+Q) | Closes the program, like `exit` |

On macOS the *Matlab_Calc* menu (with About, Settings and Quit) and the *Window* menu are the standard ones of the system; the program name appears there when PyObjC is installed (see Requirements). Older Tk versions call the Settings item *Preferences…*.

### Running and saving scripts
**File > Open…** runs a MATLAB script (`.m` file) one line at a time, exactly as if the lines were typed after the prompt: each command and its result appear in the command window, and the commands are added to the command history (↑/↓).

- Blank lines and comment lines (`% ...`) are skipped. A command continues on the next lines while a `[` is open (`A = [1 2` / `3 4]`) or after `...`.
- An error is shown and the script goes on with the next line, as when typing.
- While the script runs, the current folder is the folder of the script (as with MATLAB's `run`), so `fopen('data.txt')` finds the data files next to it. Afterwards the previous folder is restored, unless the script changed it with `cd`.

**File > Save…** writes every command of the session (typed or run from a script) to a `.m` file, one per line, below a comment line with the date. *File > Open* runs it again later and gives the same variables. The same file name is offered the next time.

### Settings
The Settings window keeps its choices in the settings file `~/.matlab_calc.json`, so they apply at every start; **Restore Defaults** goes back to the parameters at the top of `matlab_calc.py`.

| Setting | Default | Meaning |
| :--- | :--- | :--- |
| Font | first installed font of `FONT_FAMILIES` | Font of the command window; only monospace fonts are listed, so matrix columns stay aligned. A preview shows the choice. |
| Size | `FONT_SIZE` (11) | Font size, from 6 to 40 (`FONT_SIZE_LIMITS`). **Ctrl/⌘ +** and **Ctrl/⌘ −** change it in the command window directly, **Ctrl/⌘ 0** restores the default. |
| Format at the start | `short` | Number format when the program starts: `format short` (up to 4 decimals) or `format long` (up to 15). The `format` command still changes it during a session. |
| Look for new versions on GitHub every … days | on, 7 days (`UPDATE_CHECK_DAYS`) | The automatic look for a new version; the same as `update on` / `update off`. |

---

## Configuration

All settings are constants at the top of `matlab_calc.py`:

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `APP_NAME` | `'Matlab_Calc'` | Window title, Tk class name and macOS menu-bar name |
| `APP_CREDITS` | `'Developed by Zoltan Varallyay, PhD with the help of AI technology (Antigravity, Claude)'` | Shown in the About window |
| `APP_VERSION` | `'0.9.0'` | Version number, shown in the welcome banner and by the `version` command (also available as `matlab_calc.__version__`) |
| `WINDOW_WIDTH`, `WINDOW_HEIGHT` | `750`, `520` | Initial window size in pixels |
| `APP_ICON_FILE` | `'icon.png'` | Icon image, looked up next to the script |
| `APP_ICON_MARGIN` | `0.1` | Transparent margin on each side, as a fraction of the icon size (keeps the Dock icon the same size as native macOS icons) |
| `APP_ICON_ROUNDING` | `0.20` | Corner radius, as a fraction of the visible icon size (`0` = square corners, `0.5` = circle) |
| `FONT_FAMILIES`, `FONT_SIZE` | `('Consolas', 'Menlo', …)`, `11` | Default font: the first installed monospace font is used, so matrix columns stay aligned on every OS. The Settings window can choose another font and size. |
| `FONT_SIZE_LIMITS` | `(6, 40)` | Smallest and largest font size in the Settings window and with Ctrl/⌘ + and − |
| `BACKGROUND_COLOR`, `TEXT_COLOR`, `PROMPT_COLOR`, `CURSOR_COLOR` | dark theme | Console colors |
| `WINDOW_PADDING`, `PROMPT` | `10`, `'>> '` | Space around the console; prompt text |
| `SHORT_DECIMALS`, `LONG_DECIMALS` | `4`, `15` | Maximum decimals for `format short` / `format long` |
| `MAX_DISPLAY_ELEMENTS` | `10000` | Larger results are summarised instead of printed (end the statement with `;`) |
| `FIGURE_WIDTH`, `FIGURE_HEIGHT` | `560`, `420` | Size of the plot area of a new figure window in pixels (MATLAB's default) |
| `FIGURE_COLOR` | `'#f0f0f0'` | Background of the figure windows (MATLAB's light gray; the plot area is white) |
| `PLOT_LINE_WIDTH` | `1.5` | Default width of plotted lines in points (matplotlib's default; MATLAB uses 0.5) |
| `CONFIG_FILE` | `'~/.matlab_calc.json'` | Settings file in the home folder: remembers the answer to the installation question, what was installed, the update checks and the choices of the Settings window |
| `UPDATE_REPOSITORY` | `'varallyay/matlab_calc'` | GitHub project (owner/name) where new versions are published |
| `APP_WEBSITE` | `'https://github.com/' + UPDATE_REPOSITORY` | Web page shown in the About window and the Help window |
| `UPDATE_CHECK_DAYS` | `7` | Look for a new version this often, in days; `0` = only with the `update` command (the Settings window can still turn the checks on) |
| `SCRIPT_EXTENSION` | `'.m'` | File type of *File > Open* and *File > Save* |

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
| `==` `~=` `<` `<=` `>` `>=` | Comparisons; the result is logical (1 where true, 0 where false) | `A > 2` |
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

#### Changing and deleting elements
The same subscripts on the left of `=` change elements; the whole variable is shown afterwards (unless the line ends with `;`):

```matlab
>> A = [1 2 3; 4 5 6];
>> A(2,3) = 50
A =
    1   2    3
    4   5   50

>> A(:, 1) = 0
A =
    0   2    3
    0   5   50

>> A(3, 4) = 1
A =
    0   2    3   0
    0   5   50   0
    0   0    0   1

>> A(2, :) = []
A =
    0   2   3   0
    0   0   0   1
```

- A scalar is copied into every selected element (`A(:, 1) = 0`); otherwise the right side must have one value per selected element (`A(1, :) = [7 8 9 10]`).
- Assigning beyond the end makes the matrix grow and fills the new elements with 0 (`v(end+1) = 7`, `A(3, 4) = 1`). Assigning to a name that does not exist creates it: `w(3) = 7` gives `w = [0 0 7]`.
- `= []` deletes: whole rows (`A(2, :) = []`), whole columns (`A(:, [1 3]) = []`) or elements of a vector (`v([1 3]) = []`). Deleting with a single subscript from a matrix leaves a row of the remaining elements, as in MATLAB.

#### Conditions
Comparisons give *logical* values (shown as `1x4 logical array`): 1 where the condition holds, 0 where it does not. Used as a subscript, they select the elements for which the condition is true; `find` returns their positions:

```matlab
>> A = [1 2 2.4 2.3];
>> A >= 2
ans =
  1x4 logical array
    0   1   1   1

>> A(A >= 2)
ans =
    2.0   2.4   2.3

>> find(A >= 2)
ans =
    2   3   4

>> A = A';
>> find(A >= 2)
ans =
    2
    3
    4

>> A(A > 2.35) = 0
A =
    1.0
    2.0
    0.0
    2.3
```

- Conditions can be combined with `&` (and), `|` (or) and `~` (not): `v(v > 15 & v < 35)`.
- A condition on one row or column selects columns or rows: `A(A(:, 1) > 4, :)` keeps the rows whose first element is greater than 4.
- `find(X)` numbers the positions down the columns, like `A(k)`. The result is a row if `X` is a row vector and a column otherwise; `find(X, N)` returns at most the first `N` positions.
- Only logical values select elements: `A(logical([1 0 1]))` picks elements 1 and 3, but `A([1 0 1])` is an error because 0 is not a position.
- In arithmetic, logical values count as 1 and 0 (`true + true` is 2), so `(v > 15) * 1` turns a condition into numbers.

### 4. Several Outputs
Some functions give more than one result. Put the names of the variables that receive them in square brackets on the left of `=`; `~` skips an output you do not need:

```matlab
>> A = [8 1 6; 3 5 7; 4 9 2];
>> [m, i] = max(A)
m =
    8   9   7

i =
    1   3   2

>> [rows, cols] = size(A)
rows =
    3

cols =
    3

>> [r, c] = find(A > 6)
r =
    1
    3
    2

c =
    1
    2
    3

>> [~, k] = min([4 2 8 2])
k =
    2
```

- Functions with several outputs: `max`, `min` (value and position), `size` (one output per dimension), `find` (rows, columns and values), `fopen` (file identifier and error message) and `fscanf` (data and count).
- Without brackets only the first output is returned, as in MATLAB: `max(A)` gives just the maximum values.
- Asking for more outputs than a function has gives MATLAB's error `Too many output arguments.`

### 5. Text
Text is written in single quotes; a doubled quote inside stands for one quote (`'it''s'`). Double quotes (`"text"`) are accepted too and give the same kind of text. Text is a row of characters (a `char` array), so it can be joined with brackets, indexed and measured like any vector:

```matlab
>> s = 'Hello'
s =
    'Hello'

>> size(s)
ans =
    1   5

>> disp(['Result: ' num2str(pi, 4)])
Result: 3.142
```

| Function | What it does |
| :--- | :--- |
| `disp(X)` | Shows text or a value without the `name =` line |
| `fprintf(format, A, ...)` | Prints formatted text in the command window; the format is repeated until all values are used |
| `sprintf(format, A, ...)` | The same formatting, returned as text |
| `num2str(x)`, `num2str(x, N)`, `num2str(x, format)` | A number as text, with about 5 significant digits, `N` digits or a format |

Formats use the conversions `%d`, `%i`, `%u`, `%f`, `%e`, `%g`, `%x`, `%o`, `%c` and `%s` with optional flags, width and precision (`%8.3f`, `%-5d`, `%+e`), and the escapes `\n`, `\t`, `\\` and `%%`. Values are taken down the columns, so each column of a matrix fills one round of the format:

```matlab
>> fprintf('%d squared is %d\n', [1 2 3; 1 4 9])
1 squared is 1
2 squared is 4
3 squared is 9
```

As in MATLAB, output without `\n` at the end leaves the next `>>` prompt on the same line.

### 6. Writing and Reading Files
A file is opened with `fopen`, which returns a *file identifier*: a number (3, 4, …) that `fprintf`, `fscanf` and `fclose` use to name the file. Identifier `1` is the command window, so `fprintf(1, ...)` is the same as `fprintf(...)`.

```matlab
>> fid = fopen('data.txt', 'w');
>> fprintf(fid, '%d %.1f\n', [1 2 3; 0.5 1.5 2.5]);
>> fclose(fid);
>> fid = fopen('data.txt');
>> M = fscanf(fid, '%d %f', [2 Inf])'
M =
    1.0   0.5
    2.0   1.5
    3.0   2.5

>> fclose(fid);
```

| Function | What it does |
| :--- | :--- |
| `fid = fopen(name)`, `fopen(name, mode)` | Opens a file; `mode` is `'r'` (read, the default), `'w'` (write a new file), `'a'` (append) or `'r+'`, `'w+'`, `'a+'` (read and write). Returns `-1` if the file cannot be opened; `[fid, msg] = fopen(...)` also gives the reason |
| `fopen('all')` | The identifiers of all open files |
| `fprintf(fid, format, A, ...)` | Writes formatted text to the file; `n = fprintf(...)` gives the number of bytes |
| `A = fscanf(fid, format)` | Reads the data that match the format, repeating it, as a column. `fscanf(fid, format, N)` reads at most `N` values, `fscanf(fid, format, [m n])` or `[m Inf]` fills an `m`-row matrix down the columns. With only `%c` / `%s` conversions the result is text. `[A, count] = fscanf(...)` also gives the number of values read |
| `fclose(fid)`, `fclose('all')` | Closes one or all open files |

File names are relative to the current folder, shown by `pwd` and changed with `cd`. When the program is started from Launchpad or the application menu, the current folder is your home folder.

### 7. Plots
`plot` draws lines in a **figure window**, which opens beside the command window. As in MATLAB, the window has a light gray background, a white plot area with a box and inward tick marks, a bold title, and a toolbar at the top to zoom, pan, go back to the original view and save the plot as an image (PNG, PDF, SVG, …). The plots are drawn with matplotlib, and the line colors and color order are matplotlib's.

```
x = linspace(0, 4*pi, 200);
plot(x, sin(x), x, cos(x), '--')
hold on
plot(x(1:10:end), 0.5*sin(x(1:10:end)), 'o')
xlabel('Time t_1 (s)'), ylabel('Amplitude A^2'), title('Signals with \alpha = 0.5')
legend('sin', 'cos', 'samples')
grid on
```

| Function | What it does |
| :--- | :--- |
| `plot(y)`, `plot(x, y)`, `plot(x1, y1, x2, y2, …)` | Lines; each column of a matrix is a line. `plot(x, y, 'r--o')` sets color, line style and marker (a LineSpec), `plot(…, 'LineWidth', 2)` any line property. `h = plot(…)` gives the line handles. |
| `figure`, `figure(n)` | Opens a new figure window, or makes figure `n` the current one (the next plot goes there). `figure('Name', 'Results')` sets properties. |
| `hold on`, `hold off`, `hold` | Keep the plot and add the next ones to it (the color order continues), or let the next plot replace it. `ishold` tells which. |
| `xlabel`, `ylabel`, `title` | Axis labels and the title. As in MATLAB, `x_1`, `x^2`, `x_{max}` and `\alpha`, `\pi`, `\circ`, … are shown as subscript, superscript and symbols; `'Interpreter', 'none'` shows the text as typed. |
| `legend('a', 'b', …)`, `legend` | A legend of the lines (unnamed lines are `data1`, `data2`, …); `'Location', 'southwest'` places it. `legend off`, `legend hide`/`show`, `legend boxoff`. Lines added with `hold on` join the legend. |
| `grid on`/`off`/`minor`, `box on`/`off` | Grid lines; the box around the plot. |
| `axis([xmin xmax ymin ymax])`, `xlim([a b])`, `ylim([a b])` | Axis limits; `axis tight`, `equal`, `image`, `square`, `ij`, `off`, `auto` change the look. Without inputs they return the limits. |
| `gcf`, `gca` | Handles of the current figure and its axes. |
| `set(h, 'Name', value, …)`, `get(h, 'Name')`, `get(h)` | Change or read properties, e.g. `set(gca, 'FontSize', 12)`, `set(h, 'LineWidth', 2)`, `get(gca, 'XLim')`. Names can be in any case and shortened (`'linew'`). `get(h)` lists all properties. |
| `close`, `close(n)`, `close all`, `clf`, `cla`, `drawnow` | Close figures, clear the figure or the axes, draw right away. |

- **Handles are numbers**, as in MATLAB before R2014b: figures are numbered 1, 2, …, and axes, lines, labels and legends get numbers such as 1.5, 2.5, … that are never a figure number. Store them in variables: `h = plot(x, y); set(h, 'Color', 'k')`.
- **Properties** (see `help set`): figure `Name`, `Color`, `Position`, `NumberTitle`; axes `XLim`, `YLim`, `XScale`/`YScale` (`'linear'`, `'log'`), `XGrid`, `YGrid`, `Box`, `FontSize`, `LineWidth`, `Color`, `XDir`/`YDir`, `XTick`/`YTick`, `TickDir`, `NextPlot`, `Title`, `XLabel`, `YLabel`, `Children`; line `Color`, `LineStyle`, `LineWidth`, `Marker`, `MarkerSize`, `MarkerFaceColor`, `MarkerEdgeColor`, `XData`, `YData`, `DisplayName`, `Visible`; text `String`, `FontSize`, `FontWeight`, `FontAngle`, `Color`, `Interpreter`; legend `Location`, `Orientation`, `FontSize`, `Box`, `Visible`.
- **Colors** are matplotlib colors: the letters `r g b c m y k w` (matplotlib's shades), names such as `'red'` or `'tab:orange'`, `'#ff8000'`, or an RGB triplet `[1 0.5 0]`. `get` returns colors as RGB triplets.
- **Command syntax**: `hold on`, `grid on`, `axis equal tight`, `legend off`, `close all` are the same as `hold('on')`, … — as long as no variable has that name.
- **Figures are drawn after every command line** (`drawnow` draws within a line). Without hold on, `plot` resets the axes, including the title and labels, so give those after `plot`. Clicking a figure window makes it the current figure; **Ctrl+W** / **⌘W** in it closes it.

### 8. Built-in Constants & Functions
- **Constants**: `pi`, `e`, `Inf`, `NaN`, `i`, `j` (imaginary unit; literals such as `3 + 4i` also work)
- **Trigonometric**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
- **Hyperbolic**: `sinh`, `cosh`, `tanh` and their inverses `asinh`, `acosh`, `atanh` (`acosh` and `atanh` return `NaN` outside their domain, like `asin` and `acos`)
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
  - `rem(x, y)` — remainder after division, with the sign of `x` (`rem(-7, 3) = -1`)
  - `mod(x, y)` — modulus, with the sign of `y` (`mod(-7, 3) = 2`); `mod(x, 0)` is `x`
- **Data Analysis** (on a vector: all its elements; on a matrix: each column, or along dimension `dim`; `'all'` uses every element):
  - `max(A)`, `min(A)` — largest / smallest elements; `[m, i] = max(A)` also gives their positions. `max(A, B)` compares element by element (`max([1 5 3], 4)` is `4 5 4`), `max(A, [], dim)` works along a dimension. `NaN` values are ignored
  - `mean(A)`, `median(A)` — average and middle value (`mean(A, 2)` per row, `mean(A, 'all')`)
  - `std(A)` — standard deviation (normalised by `N-1`; `std(A, 1)` by `N`; `std(A, w, dim)`)
- **Random Numbers**:
  - `rand(n)`, `rand(r, c)` — uniformly distributed numbers between 0 and 1
  - `randn(n)`, `randn(r, c)` — normally distributed numbers (mean 0, standard deviation 1)
  - `rng(seed)`, `rng('default')`, `rng('shuffle')` — repeatable or new random sequences
- **Size and Shape**:
  - `size(A)`, `size(A, dim)`, `[r, c] = size(A)` — number of rows and columns
  - `length(A)` — the largest dimension (0 for an empty matrix)
  - `linspace(a, b, n)` — `n` evenly spaced numbers from `a` to `b` (100 if `n` is not given): `linspace(0, 1, 5)` is `0 0.25 0.5 0.75 1`
  - `resize(A, sz)` — pad with zeros or cut to size `sz` (`resize([1 2 3], 5)` is `1 2 3 0 0`); options `'FillValue', v` and `'Side', 'trailing' | 'leading' | 'both'`
  - `reshape(A, r, c)` — the same elements arranged in `r` rows and `c` columns (one size may be `[]`)
- **Text and Files**: `disp`, `fprintf`, `sprintf`, `num2str`, `fopen`, `fclose`, `fscanf` (see sections 5 and 6)
- **Graphics**: `plot`, `figure`, `hold`, `ishold`, `xlabel`, `ylabel`, `title`, `legend`, `grid`, `box`, `axis`, `xlim`, `ylim`, `gca`, `gcf`, `set`, `get`, `close`, `clf`, `cla`, `drawnow` (see section 7)
- **Logical Values & Searching**:
  - `find(X)`, `find(X, N)` — positions of the nonzero (true) elements, e.g. `find(A >= 2)`; `[row, col] = find(X)` or `[row, col, v] = find(X)` give rows, columns and values
  - `logical(X)` — convert numbers to logical values (nonzero becomes true)
  - `true`, `false` — the logical values 1 and 0; `true(n)`, `false(r, c)` make logical matrices

### 9. Console & Workspace Commands
| Command | Description |
| :--- | :--- |
| `format short` / `format` | Cap display precision at up to 4 decimal places (default) |
| `format long` | Cap display precision at up to 15 decimal places |
| `whos` | List all variables currently in the workspace with their size and class |
| `clear <var> ...` | Remove specific variable(s) from the workspace |
| `clear` / `clear all` | Clear all variables from the workspace |
| `clc` | Clear the command window screen |
| `pwd` | Show the current folder, where `fopen` looks for files |
| `cd <folder>` | Change the current folder (`cd ..`, `cd ~/Documents`); `cd` alone shows it |
| `help` | List all commands, operators and functions |
| `help <name>` | Show the help of a command or function, e.g. `help round`; `help ops` (or `help .*`, `help +`, …) explains the operators |
| `version` | Show the application name and version number |
| `install` | Install the program for you (or update the installed copy), so it can be started from Launchpad / the application menu |
| `install systemwide` | Install it for all users of the computer |
| `uninstall` | Remove the installed copies |
| `update` | Look for a new version on GitHub and install it over the installed copy |
| `update off` / `update on` | Turn the weekly look for new versions off or on |
| `exit` / `quit` | Close the application |

Typing a variable's name shows its value without changing `ans`.

### 10. Getting Help
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

### 11. Adding Your Own Functions
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

Arguments arrive as 2-D NumPy arrays (scalars are 1×1) and the function must return one (`to_matrix` converts any result). The number of inputs comes from the Python signature; parameters with a default value (`n=None`) are optional. For several outputs, give `outputs=2` (or more) to `@builtin`, add a keyword-only parameter `nargout` and return a tuple with that many values; `keep_types=True` passes logical and text arguments unchanged instead of converting them to numbers, and `command_syntax=True` allows calls such as `hold on`. Console commands work the same way with the `@console_command` decorator on a `MatlabEngine` method. The tests check that every command and function has a help text whose first line has the form `name - Summary.`

### 12. Not Supported
More than two subscripts and N-dimensional arrays, cell arrays, structures, the separate `string` class (`"text"` gives ordinary `char` text), user functions/scripts, and control flow (`if`, `for`, …). `asin`/`acos` return `NaN` outside [-1, 1], and `acosh` below 1 and `atanh` outside [-1, 1] likewise. Files are read and written as text (UTF-8); `fread`, `fwrite`, `fgetl` and `textscan` are not available. Graphics: one axes per figure (no `subplot` yet), no `pcolor`, `surf`, `bar` or other plot types, no dot syntax (`h.LineWidth = 2`; use `set`), and `get` takes one handle at a time. Non-integer matrix powers use an eigendecomposition, so they require a diagonalisable matrix.

---

## Keyboard Shortcuts

- **`Enter`** (or keypad **`Enter`**): Execute the command at the current `>> ` prompt.
- **`Up` / `Down` Arrows**: Cycle through previous commands in history.
- **`Home`**: Move the cursor to the start of the current input (right after `>> `).
- **`Ctrl+C` / `Cmd+C`**: Copy selected text from anywhere in the console history.
- **`Ctrl+V` / `Cmd+V`**: Paste into the active command line.
- **`Ctrl+A` / `Cmd+A`**: Select all text.
- **`Ctrl+O` / `Cmd+O`**, **`Ctrl+S` / `Cmd+S`**: Open (run) a script, save the session.
- **`Ctrl +` / `Cmd +`**, **`Ctrl −` / `Cmd −`**, **`Ctrl 0` / `Cmd 0`**: Larger, smaller or default font size (remembered).
- **`Ctrl+W` / `Cmd+W`** in a figure window: close the figure.
- **`F1`** (Linux) / **`Cmd+?`** (macOS): Matlab_Calc Help. **`Cmd+,`** (macOS): Settings. **`Ctrl+Q` / `Cmd+Q`**: Quit.

---

## Testing

```bash
python3 test_matlab_calc.py
```

Runs 139 tests covering installation (in temporary folders, so your system is not touched), updates (against a local stand-in for GitHub, without internet access), operators, functions, indexing, assignment, conditions and `find`, several outputs, text, formatted output and files (in a temporary folder), the statistics and random-number functions, the help texts, the menus, the About, Help and Settings windows, running and saving scripts, the update check of the Help menu, plotting (off-screen) and figure windows, precedence, parsing, number display, workspace commands, the read-only console (including a session inside the real Tk event loop) and the rounded icon. The GUI and icon tests are skipped when no display is available, and the plot tests when matplotlib is not installed.

---

## Credits

Developed by Zoltan Varallyay, PhD with the help of AI technology (Antigravity, Claude).

---

## Version History

### 0.9.0 — 2026-11-04
- **Plots:** `plot` with all its usual forms (vectors, matrices, several x/y pairs, LineSpecs such as `'r--o'`, line properties), drawn with matplotlib in figure windows that look like MATLAB's, with a toolbar to zoom, pan and save.
- **New functions:** `figure`, `hold`, `ishold`, `xlabel`, `ylabel`, `title` (with MATLAB's `x_1`, `x^2`, `\alpha` markup), `legend`, `grid`, `box`, `axis`, `xlim`, `ylim`, `gca`, `gcf`, `set`, `get`, `close`, `clf`, `cla` and `drawnow`.
- **Graphics handles** are numbers, as in MATLAB before R2014b; `set` and `get` change and read the properties of figures, axes, lines, labels and legends.
- **Command syntax** for the graphics commands: `hold on`, `grid on`, `axis equal`, `close all`, ….
- **Colors** and the color order are matplotlib's.
- **Parameters:** `FIGURE_WIDTH`, `FIGURE_HEIGHT`, `FIGURE_COLOR` and `PLOT_LINE_WIDTH`.

### 0.8.0 — 2026-11-04
- **Menus:** a File menu with **Open…** (runs a `.m` script line by line, as if typed) and **Save…** (saves the commands of the session as a `.m` script), an Edit menu, and a Help menu with **Matlab_Calc Help** and **Check for Updates…**. On Linux, File also has Settings and Quit, and Help has About.
- **About window:** the program icon, name, version, the GitHub page (clickable) and the credits, instead of the empty standard panel of macOS.
- **Settings window** (macOS: *Matlab_Calc > Settings…*, which was greyed out before; Linux: *File > Settings…*): font and font size of the command window with a preview, the number format at the start, and the automatic update check with its interval. The choices are kept in `~/.matlab_calc.json`.
- **Font size from the keyboard:** Ctrl/⌘ +, Ctrl/⌘ − and Ctrl/⌘ 0.
- **Help window:** describes what the program can do and how the `help` command gives more help; a button lists every command and function in the command window.
- **Check for Updates…** looks for a new version right away and reports the result.
- **Parameters:** `APP_CREDITS`, `APP_WEBSITE`, `FONT_SIZE_LIMITS` and `SCRIPT_EXTENSION`.

### 0.7.0 — 2026-11-02
- **Several outputs:** `[a, b] = f(...)` assigns each output of a function to its own variable, and `~` skips one. `max`, `min`, `size`, `find` (`[row, col, v]`), `fopen` (`[fid, msg]`) and `fscanf` (`[A, count]`) have several outputs.
- **Text:** `'text'` and `"text"` give character arrays that can be joined, indexed and displayed (`'Hello'`, `2x2 char array`); `whos` lists them as `char`.
- **Output and files:** `disp`, `fprintf`, `sprintf` and `num2str`; `fopen`, `fclose` and `fscanf` read and write text files through file identifiers (1 is the command window, opened files get 3, 4, …).
- **New functions:** `rem` and `mod`; the hyperbolic functions `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`; `max`, `min`, `mean`, `median` and `std`; `rand`, `randn` and `rng`; `size`, `length`, `linspace`, `resize` and `reshape`.
- **New commands:** `pwd` and `cd`. When started from Launchpad or the application menu, the current folder is the home folder, so files are written there.
- **Help:** new topics *Hyperbolic functions*, *Rounding and remainders*, *Data analysis*, *Random numbers*, *Matrices and arrays* and *Text and files*; `help ops` explains text and several outputs.

### 0.6.0 — 2026-11-02
- **Updates from GitHub:** the installed program looks for a new version in [varallyay/matlab_calc](https://github.com/varallyay/matlab_calc) once a week (in the background) and offers **Update now**, **Not now** or **Skip this version**; after updating it offers to restart. The newest GitHub release is used, or the main branch if there are no releases.
- **New command:** `update` looks for a new version right away (`update off` / `update on` switch the weekly check); `--update` does the same from a terminal.
- **Dialogs:** the installation and update questions show the program icon and a checkbox for the weekly check.
- **Parameters:** `UPDATE_REPOSITORY` and `UPDATE_CHECK_DAYS`.

### 0.5.0 — 2026-11-02
- **Installation on macOS and Linux:** at the first start a window asks whether to install the program systemwide, only for the current user, or not at all. Installed, it is listed in Launchpad / the Applications folder (macOS: a `Matlab_Calc.app` bundle with its own icon) or in the application menu (Linux: a `.desktop` entry) and starts without the terminal.
- **Settings file:** `~/.matlab_calc.json` remembers the answer and the installation, so the question is asked only once.
- **Updates:** starting a newer version from elsewhere offers to update the installed copy.
- **New commands:** `install`, `install systemwide` and `uninstall`; the same as the terminal options `--install`, `--install-systemwide` and `--uninstall` (useful with `sudo` on Linux). `--version` prints the version.

### 0.4.0 — 2026-11-02
- **Changing elements:** `A(2,3) = 5`, `A(:, 1) = 0`, `A(1:2, end) = [7; 8]` and so on. Matrices grow when assigning beyond their end (new elements are 0), and assigning to a new name creates the variable.
- **Deleting elements:** `A(2, :) = []`, `A(:, [1 3]) = []`, `v([1 3]) = []`.
- **Conditions:** comparisons now give logical values (shown as `logical array`, listed as `logical` by `whos`), which select elements: `A(A > 2)`, `A(A > 2) = 0`, `A(A(:, 1) > 4, :)`.
- **New functions:** `find` (positions of nonzero or true elements, e.g. `find(A >= 2)`), `logical`, `true` and `false`.
- **Help:** `help ops` has new sections on conditions and on changing and deleting elements.

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
