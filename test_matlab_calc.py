"""Tests for matlab_calc.py.   Run:  python3 test_matlab_calc.py

The engine tests need only NumPy. The GUI and icon tests need a display (they are
skipped otherwise); the anti-aliased icon test also needs Pillow.
"""
import importlib.util
import os
import sys
import tempfile
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
        self.assertNotIn('Error', text, f"{line!r} -> {text}")
        self.assertNotIn('Unrecognized', text, f"{line!r} -> {text}")
        return self.eng.variables[var]

    def assertValue(self, line, expected, var='ans'):
        got = self.ans(line, var)
        expected = mc.to_matrix(expected)
        self.assertEqual(got.shape, expected.shape, line)
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12, err_msg=line)

    def assertError(self, line, fragment):
        text = self.out(line)
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
        self.assertError("'abc'", 'strings are not supported')
        self.assertError('x = 1; x(1)', 'Indexing into variables')
        self.assertError('x(1) = 2', 'Only assignments')
        self.assertError('sin(1, 2)', 'Too many input arguments')
        self.assertError('sin()', 'Not enough input arguments')
        self.assertError('2 € 3', 'Invalid text character')

    def test_error_stops_rest_of_line(self):
        text = self.out('a = 1, foo, b = 2')
        self.assertIn('a =', text)
        self.assertNotIn('b =', text)
        self.assertNotIn('b', self.eng.variables)


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
