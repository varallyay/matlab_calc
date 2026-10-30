import os
import tkinter as tk
from tkinter import scrolledtext
import numpy as np
import re

class MatlabError(Exception):
    """Custom exception for MATLAB-specific operator and dimension errors."""
    pass


class MatlabMatrix(np.ndarray):
    """
    Custom NumPy array subclass to seamlessly map MATLAB operators:
    - '*'  -> Matrix / Scalar multiplication
    - '.*' -> Element-wise multiplication (mapped to '@' internally)
    - '/'  -> Scalar division or Matrix right-division
    - './' -> Element-wise division (mapped to '//' internally)
    - '^'  -> Matrix/Scalar power (mapped to '**' internally)
    - '.^' -> Element-wise power (mapped to '%' internally)
    """
    def __new__(cls, input_array):
        obj = np.asarray(input_array, dtype=float).view(cls)
        if obj.ndim == 0:
            return obj.item()
        if obj.ndim == 1:
            obj = obj.reshape(1, -1)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    @property
    def H(self):
        """Conjugate transpose (') in MATLAB."""
        return self.conj().T

    def _unwrap_if_scalar(self, res):
        if isinstance(res, np.ndarray) and (res.ndim == 0 or res.size == 1):
            return float(res.flat[0])
        return res.view(MatlabMatrix) if isinstance(res, np.ndarray) else res

    # '*' in MATLAB: Matrix multiplication (or scalar multiplication)
    def __mul__(self, other):
        if np.isscalar(other) or (isinstance(other, np.ndarray) and other.size == 1):
            return self._unwrap_if_scalar(super().__mul__(other))
        res = super().__matmul__(other)
        return self._unwrap_if_scalar(res)

    def __rmul__(self, other):
        if np.isscalar(other) or (isinstance(other, np.ndarray) and other.size == 1):
            return self._unwrap_if_scalar(super().__rmul__(other))
        res = super().__rmatmul__(other)
        return self._unwrap_if_scalar(res)

    # '.*' in MATLAB (mapped to '@' in translated Python string): Element-wise multiplication
    def __matmul__(self, other):
        res = super().__mul__(other)
        return self._unwrap_if_scalar(res)

    def __rmatmul__(self, other):
        res = super().__rmul__(other)
        return self._unwrap_if_scalar(res)

    # '/' in MATLAB: Scalar division or Matrix right division (A * inv(B))
    def __truediv__(self, other):
        if np.isscalar(other) or (isinstance(other, np.ndarray) and other.size == 1):
            return self._unwrap_if_scalar(super().__truediv__(other))
        res = super().__matmul__(np.linalg.pinv(other))
        return self._unwrap_if_scalar(res)

    # './' in MATLAB (mapped to '//' in translated Python string): Element-wise division
    def __floordiv__(self, other):
        res = super().__truediv__(other)
        return self._unwrap_if_scalar(res)

    def __rfloordiv__(self, other):
        res = super().__rtruediv__(other)
        return self._unwrap_if_scalar(res)

    # '.^' in MATLAB (mapped to '%' in translated Python string): Element-wise power
    def __mod__(self, other):
        res = super().__pow__(other)
        return self._unwrap_if_scalar(res)

    def __rmod__(self, other):
        res = super().__rpow__(other)
        return self._unwrap_if_scalar(res)

    # '^' in MATLAB (mapped to '**' in translated Python string): Matrix Power
    def __pow__(self, other):
        is_scalar_exp = np.isscalar(other) or (isinstance(other, np.ndarray) and other.size == 1)
        is_square_base = (self.ndim == 2 and self.shape[0] == self.shape[1]) or self.size == 1
        if not (is_square_base and is_scalar_exp):
            raise MatlabError(
                "Error using ^\n"
                "Incorrect dimensions for raising a matrix to a power. Check that\n"
                "the matrix is square and the power is a scalar. To operate on\n"
                "each element of the matrix individually, use POWER (.^) for\n"
                "elementwise power."
            )
        exp_val = float(other.flat[0]) if isinstance(other, np.ndarray) else float(other)
        if self.size == 1:
            return float(self.flat[0]) ** exp_val
        if exp_val == int(exp_val):
            res = np.linalg.matrix_power(self, int(exp_val))
            return self._unwrap_if_scalar(res)
        vals, vecs = np.linalg.eig(np.asarray(self))
        res = vecs @ np.diag(vals ** exp_val) @ np.linalg.inv(vecs)
        if np.isrealobj(self) and np.allclose(res.imag, 0):
            res = res.real
        return self._unwrap_if_scalar(res)

    def __rpow__(self, other):
        is_scalar_base = np.isscalar(other) or (isinstance(other, np.ndarray) and other.size == 1)
        is_square_exp = (self.ndim == 2 and self.shape[0] == self.shape[1]) or self.size == 1
        if not (is_scalar_base and is_square_exp):
            raise MatlabError(
                "Error using ^\n"
                "Incorrect dimensions for raising a matrix to a power. Check that\n"
                "the matrix is square and the power is a scalar. To operate on\n"
                "each element of the matrix individually, use POWER (.^) for\n"
                "elementwise power."
            )
        base_val = float(other.flat[0]) if isinstance(other, np.ndarray) else float(other)
        if self.size == 1:
            return base_val ** float(self.flat[0])
        vals, vecs = np.linalg.eig(np.asarray(self))
        res = vecs @ np.diag(base_val ** vals) @ np.linalg.inv(vecs)
        if np.isrealobj(self) and np.allclose(res.imag, 0):
            res = res.real
        return self._unwrap_if_scalar(res)


def create_matlab_matrix(rows):
    """Constructs a 2D MatlabMatrix from nested lists, supporting block concatenation."""
    processed_rows = []
    for r in rows:
        row_elems = []
        for item in r:
            if isinstance(item, np.ndarray):
                row_elems.append(np.atleast_2d(item))
            else:
                row_elems.append(np.array([[float(item)]]))
        processed_rows.append(np.hstack(row_elems))
    full_arr = np.vstack(processed_rows)
    return MatlabMatrix(full_arr)


class MatlabTranslator:
    """Translates MATLAB expression syntax into executable Python code."""

    @staticmethod
    def find_matching_bracket(s, start_idx):
        depth = 0
        for i in range(start_idx, len(s)):
            if s[i] == '[':
                depth += 1
            elif s[i] == ']':
                depth -= 1
                if depth == 0:
                    return i
        raise SyntaxError("Unmatched '[' in expression.")

    @classmethod
    def split_rows(cls, content):
        rows, current = [], []
        depth_p, depth_b = 0, 0
        for ch in content:
            if ch == '(': depth_p += 1
            elif ch == ')': depth_p -= 1
            elif ch == '[': depth_b += 1
            elif ch == ']': depth_b -= 1
            
            if ch == ';' and depth_p == 0 and depth_b == 0:
                rows.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)
        if current or content.endswith(';'):
            if ''.join(current).strip():
                rows.append(''.join(current).strip())
        return rows

    @classmethod
    def split_elements(cls, row_str):
        row_str = row_str.strip()
        if not row_str:
            return []
        
        tokens = []
        current = []
        depth_p, depth_b = 0, 0
        i, n = 0, len(row_str)
        
        while i < n:
            ch = row_str[i]
            if ch in '([':
                if ch == '(': depth_p += 1
                else: depth_b += 1
                current.append(ch)
            elif ch in ')]':
                if ch == ')': depth_p -= 1
                else: depth_b -= 1
                current.append(ch)
            elif depth_p == 0 and depth_b == 0 and (ch.isspace() or ch == ','):
                if current:
                    tokens.append(''.join(current))
                    current = []
                while i + 1 < n and (row_str[i+1].isspace() or row_str[i+1] == ','):
                    i += 1
            else:
                current.append(ch)
            i += 1
            
        if current:
            tokens.append(''.join(current))

        # Re-merge tokens split around binary operators (e.g., [1 + 2, 3] -> ['1+2', '3'])
        binary_ops = {'+', '-', '*', '/', '^', '.*', './', '.^'}
        merged = []
        for tok in tokens:
            if not merged:
                merged.append(tok)
            else:
                prev = merged[-1]
                if (prev in binary_ops or tok in binary_ops or 
                    prev.endswith(('+', '-', '*', '/', '^', '.*'))):
                    merged[-1] = prev + tok
                else:
                    merged.append(tok)
        return merged

    @classmethod
    def replace_brackets(cls, expr):
        while '[' in expr:
            start = expr.find('[')
            end = cls.find_matching_bracket(expr, start)
            inner = expr[start+1:end]
            rows = cls.split_rows(inner)
            row_strs = []
            for r in rows:
                elems = cls.split_elements(r)
                elem_strs = [cls.translate_expr(e) for e in elems]
                row_strs.append("(" + ", ".join(elem_strs) + ",)")
            replacement = "_mat((" + ", ".join(row_strs) + ",))"
            expr = expr[:start] + replacement + expr[end+1:]
        return expr

    @classmethod
    def translate_expr(cls, expr):
        expr = expr.strip()
        expr = cls.replace_brackets(expr)
        
        # Replace transpose operators: .' (non-conjugate) and ' (conjugate)
        expr = re.sub(r"\.'", ".T", expr)
        expr = re.sub(r"'", ".H", expr)
        
        # Replace element-wise and power operators with custom Python dunder tokens
        expr = expr.replace('.*', '@')
        expr = expr.replace('./', '//')
        expr = expr.replace('.^', '%')
        expr = expr.replace('^', '**')
        return expr


class MatlabConsoleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MATLAB Command Window Emulator")
        self.root.geometry("750x520")

        # Load and apply application icon (sets macOS Dock / window icon)
        self.app_icon = None
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
        if os.path.isfile(icon_path):
            try:
                self.app_icon = tk.PhotoImage(file=icon_path)
                self.root.iconphoto(True, self.app_icon)
            except tk.TclError:
                pass

        self.format_mode = 'short'  # 'short' (4 digits) or 'long' (15 digits)
        self.base_env = {
            '_mat': create_matlab_matrix,
            'pi': np.pi,
            'e': np.e,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
            'sqrt': np.sqrt, 'exp': np.exp, 'log': np.log, 'log10': np.log10,
            'abs': np.abs,
            'eye': lambda n: MatlabMatrix(np.eye(int(n))),
            'zeros': lambda *s: MatlabMatrix(np.zeros(s if len(s) > 1 else (int(s[0]), int(s[0])))),
            'ones': lambda *s: MatlabMatrix(np.ones(s if len(s) > 1 else (int(s[0]), int(s[0])))),
            'inv': lambda x: MatlabMatrix(np.linalg.inv(x)),
            'det': lambda x: float(np.linalg.det(x))
        }
        self.user_vars = {}
        self.history = []
        self.history_idx = 0

        # Output / Interactive Console Area
        self.output_area = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, font=("Consolas", 11), bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="white", undo=False
        )
        self.output_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.output_area.tag_configure("prompt", foreground="#569cd6", font=("Consolas", 11, "bold"))

        # Bindings for inline command editing
        self.output_area.bind("<Return>", self.process_command)
        self.output_area.bind("<Up>", self.history_up)
        self.output_area.bind("<Down>", self.history_down)
        self.output_area.bind("<BackSpace>", self.on_backspace)
        self.output_area.bind("<Left>", self.on_left)
        self.output_area.bind("<Home>", self.on_home)
        self.output_area.bind("<Key>", self.on_key)
        self.output_area.bind("<<Paste>>", self.on_paste)
        self.output_area.bind("<<Cut>>", self.on_cut)

        self.write_output(
            "MATLAB Syntax Calculator Initialized.\n"
            "Commands: 'format short', 'format long', 'clear <var>', 'clear all', 'whos', 'clc'\n"
            "--------------------------------------------------------------------------------\n"
        )
        self.show_prompt()
        self.output_area.focus_set()

    def show_prompt(self):
        self.output_area.insert(tk.END, ">> ", "prompt")
        self.output_area.mark_set("input_start", "end-1c")
        self.output_area.mark_gravity("input_start", tk.LEFT)
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

    def on_backspace(self, event):
        if self._selection_in_readonly():
            return "break"
        if self.output_area.compare(tk.INSERT, "<=", "input_start"):
            return "break"

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

    def on_cut(self, event):
        if self._selection_in_readonly():
            return "break"

    def on_key(self, event):
        # Allow copy/select-all shortcuts anywhere in the buffer
        if (event.state & 0x4 or event.state & 0x8) and event.keysym.lower() in ('c', 'a'):
            return
        # Ignore pure modifier keys or navigation keys handled elsewhere
        if event.keysym in (
            'Shift_L', 'Shift_R', 'Control_L', 'Control_R',
            'Alt_L', 'Alt_R', 'Meta_L', 'Meta_R', 'Super_L', 'Super_R',
            'Left', 'Right', 'Up', 'Down', 'Home', 'End', 'Prior', 'Next', 'Escape'
        ):
            return
        if self._selection_in_readonly():
            self.output_area.tag_remove(tk.SEL, "1.0", tk.END)
        if self.output_area.compare(tk.INSERT, "<", "input_start"):
            self.output_area.mark_set(tk.INSERT, tk.END)

    def _needed_decimals(self, val, max_decimals):
        val = float(val)
        if np.isnan(val) or np.isinf(val):
            return 0
        # Suppress IEEE-754 floating-point representation noise (e.g., 2.4*2 -> 4.800000000000001)
        if round(val, 12) == round(val, 14):
            val = round(val, 12)
        s = f"{val:.{max_decimals}f}"
        if '.' not in s:
            return 0
        frac = s.split('.')[1].rstrip('0')
        return len(frac)

    def format_number(self, val, decimals=None):
        val = float(val)
        if np.isnan(val) or np.isinf(val):
            return str(val)
        max_decimals = 15 if self.format_mode == 'long' else 4
        if decimals is None:
            decimals = self._needed_decimals(val, max_decimals)
        if round(val, decimals) == 0:
            val = 0.0
        return f"{val:.{decimals}f}"

    def format_result(self, res):
        max_decimals = 15 if self.format_mode == 'long' else 4
        if isinstance(res, (int, float, np.number)):
            decimals = self._needed_decimals(res, max_decimals)
            return f"    {self.format_number(res, decimals)}"
        elif isinstance(res, np.ndarray):
            arr = np.atleast_2d(res)
            rows, cols = arr.shape
            decimals = max(
                (self._needed_decimals(arr[r, c], max_decimals) for r in range(rows) for c in range(cols)),
                default=0
            )
            str_grid = [[self.format_number(arr[r, c], decimals) for c in range(cols)] for r in range(rows)]
            col_widths = [max(len(str_grid[r][c]) for r in range(rows)) for c in range(cols)]
            lines = []
            for r in range(rows):
                row_str = "   ".join(str_grid[r][c].rjust(col_widths[c]) for c in range(cols))
                lines.append("    " + row_str)
            return "\n".join(lines)
        return str(res)

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

    def process_command(self, event=None):
        raw_cmd = self.output_area.get("input_start", "end-1c").strip()
        self.output_area.mark_set(tk.INSERT, tk.END)
        self.output_area.insert(tk.END, "\n")

        if not raw_cmd:
            self.show_prompt()
            return "break"

        self.history.append(raw_cmd)
        self.history_idx = len(self.history)

        self._execute_command(raw_cmd)
        self.show_prompt()
        return "break"

    def _execute_command(self, raw_cmd):
        # Check if output is suppressed by trailing semicolon
        suppress = raw_cmd.endswith(';')
        cmd = raw_cmd[:-1].strip() if suppress else raw_cmd

        # 1. Handle Built-in Console Commands
        if cmd == "clc":
            self.output_area.delete("1.0", tk.END)
            return

        if cmd == "format long":
            self.format_mode = 'long'
            return
        elif cmd == "format short":
            self.format_mode = 'short'
            return

        if cmd == "whos":
            if not self.user_vars:
                self.write_output("  Workspace is empty.\n")
            else:
                header = f"  {'Name':<10} {'Size':<10} {'Class':<10}"
                self.write_output(header + "\n  " + "-" * 32)
                for k, v in self.user_vars.items():
                    sz = f"{v.shape[0]}x{v.shape[1]}" if isinstance(v, np.ndarray) else "1x1"
                    cls_name = "double"
                    self.write_output(f"  {k:<10} {sz:<10} {cls_name:<10}")
                self.write_output("")
            return

        if cmd.startswith("clear"):
            parts = cmd.split()
            if len(parts) == 1 or parts[1] == "all":
                self.user_vars.clear()
            else:
                for var_name in parts[1:]:
                    self.user_vars.pop(var_name, None)
            return

        # 2. Handle Mathematical Expressions & Assignments
        try:
            # Check for variable assignment: e.g., A = [1 2 3]*2
            assign_match = re.match(r'^([a-zA-Z_]\w*)\s*=\s*(.+)$', cmd)
            if assign_match:
                var_name = assign_match.group(1)
                expr_str = assign_match.group(2)
            else:
                var_name = 'ans'
                expr_str = cmd

            py_expr = MatlabTranslator.translate_expr(expr_str)
            
            eval_env = {**self.base_env, **self.user_vars}
            result = eval(py_expr, {"__builtins__": {}}, eval_env)

            # Store in workspace
            self.user_vars[var_name] = result

            if not suppress:
                self.write_output(f"{var_name} =\n{self.format_result(result)}\n")

        except MatlabError as me:
            self.write_output(f"{me}\n")
        except ZeroDivisionError:
            self.write_output("Error: Division by zero.\n")
        except ValueError as ve:
            self.write_output(f"Error using operator: Matrix dimensions must agree. ({ve})\n")
        except NameError as ne:
            self.write_output(f"Undefined function or variable: {ne}\n")
        except Exception as e:
            self.write_output(f"Syntax or Calculation Error: {e}\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = MatlabConsoleApp(root)
    root.mainloop()