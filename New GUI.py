import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import numpy as np
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks
import math
import re
import os
import csv

def inv_gaussian_func(x, A, mu, lambda_):
    """Standard Inverse Gaussian PDF scaled by Amplitude A (Optimized)"""
    x_safe = np.maximum(x, 1e-9)
    term = np.sqrt(lambda_ / (2 * np.pi * x_safe * x_safe * x_safe))
    exp_term = np.exp(-lambda_ * (x_safe - mu) ** 2 / (2 * mu * mu * x_safe))
    return A * term * exp_term

# --- Scalar helpers (plain floats: much cheaper than 1-element numpy arrays inside the fit loops) ---
def _ig_unit_at(x, mu, lambda_):
    """Unit-amplitude Inverse Gaussian PDF evaluated at a single point."""
    x = max(float(x), 1e-9)
    return math.sqrt(lambda_ / (2 * math.pi * x ** 3)) * math.exp(-lambda_ * (x - mu) ** 2 / (2 * mu ** 2 * x))

def _safe_mu(m, l):
    """Mean of the IG distribution whose mode is m, for shape l."""
    return m * math.sqrt(l / max(l - 3 * m, 1e-9))

def _amp_for_height(H, m, l):
    """Amplitude A that gives a peak of height H at mode m."""
    return H / max(_ig_unit_at(m, _safe_mu(m, l), l), 1e-12)

def _peak_mode(params):
    A, mu, lambda_ = params
    ratio = mu / lambda_
    return mu * (np.sqrt(1 + 2.25 * ratio**2) - 1.5 * ratio)

def _smoothed_peak(y, window=5):
    """Index and height of the maximum of y after a short moving average (edges padded)."""
    w = min(window, len(y))
    if w < 2:
        k = int(np.argmax(y))
        return k, y[k]
    pad = w // 2
    padded = np.pad(y, (pad, w - 1 - pad), mode='edge')
    smooth = np.convolve(padded, np.ones(w) / w, mode='valid')
    k = int(np.argmax(smooth))
    return k, smooth[k]

def _parabola_peak(x, y, k, half=2):
    """Vertex (x, height) of a least-squares parabola through the points k-half..k+half.
    Returns None if the parabola doesn't open downward or its vertex falls outside those points."""
    lo, hi = max(0, k - half), min(len(y), k + half + 1)
    if hi - lo < 3: return None
    xs = x[lo:hi] - x[k]  # centred on the top point for a well-conditioned fit
    a, b, c = np.polyfit(xs, y[lo:hi], 2)
    if a >= 0: return None
    xv = -b / (2 * a)
    if not (xs[0] <= xv <= xs[-1]): return None
    return x[k] + xv, c - b * b / (4 * a)

def _value_at_index(arr, idx):
    """arr at a possibly fractional index: 100.3 -> 70% of arr[100] + 30% of arr[101]."""
    i0 = int(math.floor(idx))
    f = idx - i0
    if f == 0 or i0 + 1 >= len(arr):
        return arr[i0]
    return arr[i0] * (1 - f) + arr[i0 + 1] * f

def _read_numeric_file(path):
    """Reads whitespace/comma separated numeric rows, skipping headers and junk lines."""
    parsed_data = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                # Replace commas with spaces to handle CSV formatted .dat files
                parts = [float(x) for x in line.replace(',', ' ').split()]
                if len(parts) >= 2:
                    parsed_data.append(parts)
            except ValueError:
                continue
    return np.array(parsed_data)

# --- Speed-fit helpers ---
def _natural_key(name):
    """Sort key so 'run2' comes before 'run10'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]

def _is_not_good(folder_name):
    return folder_name.strip().lower() == "not good"

def _scan_dat_folders(root_dir):
    """Returns [(folder, [dat paths])] for every folder under root_dir holding .dat files,
    never entering folders named 'Not Good' (or anything inside them)."""
    groups = []
    if _is_not_good(os.path.basename(os.path.normpath(root_dir))):
        return groups
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = sorted((d for d in dirnames if not _is_not_good(d)), key=_natural_key)
        dats = sorted((f for f in filenames if f.lower().endswith('.dat')), key=_natural_key)
        if dats:
            groups.append((dirpath, [os.path.join(dirpath, f) for f in dats]))
    return groups

def _peaks_of(ds):
    """{'Peak 1': (q, d), 'Peak 2': (q, d)} for the fitted peaks of a dataset."""
    peaks = {}
    for label, key in (("Peak 1", 'g1_params'), ("Peak 2", 'g2_params')):
        params = ds.get(key)
        if params is None: continue
        A, mu, lambda_ = params
        if lambda_ > 0 and mu > 0:
            q_peak = float(_peak_mode(params))
            if q_peak > 0 and np.isfinite(q_peak):
                peaks[label] = (q_peak, 2 * np.pi / q_peak)
    return peaks

def _weighted_batch_stats(items):
    """items: [(d_mean, d_std, n_indices)] for the files of one sample, one peak.
    Inverse-variance weighted average and weighted std. Files fitted at a single index are left
    out; files with several indices but zero spread get the smallest std among the others.
    Returns (avg, std, [note per item])."""
    # Rounding leaves identical peaks with a std of ~1e-15 rather than 0; treat that as no spread
    items = [(d, s if s > 1e-9 * max(abs(d), 1.0) else 0.0, n) for d, s, n in items]
    multi =[i for i, (_, _, n) in enumerate(items) if n > 1]
    positive = [items[i][1] for i in multi if items[i][1] > 0]
    notes = []
    for d, s, n in items:
        if n <= 1: notes.append("excluded: single fit")
        elif s > 0: notes.append("weighted")
        elif positive: notes.append("std 0 -> smallest other std")
        else: notes.append("unweighted (no spread in any file)")

    if not multi:
        # Nothing can be weighted: plain statistics over all files
        d_all = [d for d, _, _ in items]
        notes = ["unweighted (only single fits)"] * len(items)
        return float(np.mean(d_all)), (float(np.std(d_all, ddof=1)) if len(d_all) > 1 else 0.0), notes

    d = np.array([items[i][0] for i in multi])
    if not positive:
        return float(np.mean(d)), (float(np.std(d, ddof=1)) if len(d) > 1 else 0.0), notes

    s_min = min(positive)
    w = np.array([1.0 / (items[i][1] if items[i][1] > 0 else s_min) ** 2 for i in multi])
    avg = float(np.sum(w * d) / np.sum(w))
    std = float(np.sqrt(np.sum(w * (d - avg) ** 2) / np.sum(w)))
    return avg, std, notes

class ScatteringFitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Scattering Analysis - Optimized Edition")
        self.root.geometry("1400x950")

        self.datasets = []
        self.selected_index = None

        # --- NEW: Background Tracking ---
        self.bg_dataset = None

        # Pending after() jobs (used to coalesce bursts of redraws / refits)
        self._redraw_job = None
        self._param_job = None
        self._scroll_pending = {}
        self._scroll_job = {}

        # UI Control Variables
        self.is_combined = tk.BooleanVar(value=False)
        self.normalize_res = tk.BooleanVar(value=False)
        self.show_errors = tk.BooleanVar(value=True)
        self.res_as_scatter = tk.BooleanVar(value=True)
        self.show_dashed = tk.BooleanVar(value=True)
        self.res_fit_mode = tk.StringVar(value="None")
        self.plot_scale = tk.StringVar(value="log log")
        self.res_scale = tk.StringVar(value="lin lin")

        # Peak Constraint Variables
        self.fix_p1 = tk.BooleanVar(value=False)
        self.fix_p2 = tk.BooleanVar(value=False)
        # Doubles so a peak can be fixed between two data points (e.g. 100.3)
        self.p1_idx_var = tk.DoubleVar(value=100)
        self.p2_idx_var = tk.DoubleVar(value=150)

        # Label Customization Variables
        self.main_title_var = tk.StringVar(value="Scattering Data")
        self.res_title_var = tk.StringVar(value="Residual Fitting (Peak Region)")
        self.x_axis_var = tk.StringVar(value="q (nm⁻¹)")
        self.y_axis_var = tk.StringVar(value="I(q) (a.u.)")
        self._last_labels = self._current_labels()

        # Background Parameter UI Variables
        self.bg_vars = {'a': tk.StringVar(), 'b': tk.StringVar(), 'c': tk.StringVar()}

        # 2-Point Linear Background Variables
        self.use_lin_bg = tk.BooleanVar(value=False)
        self.lin_idx1_var = tk.IntVar(value=15)
        self.lin_idx2_var = tk.IntVar(value=120)

        # UI string variables
        self.bg_file_var = tk.StringVar(value="Current BG: None")

        # Speed-fit run state (None when not running)
        self.sf = None
        self.sf_svg = tk.BooleanVar(value=False)
        self.sf_png = tk.BooleanVar(value=False)
        self.sf_sample_var = tk.StringVar()
        self.sf_file_var = tk.StringVar()
        self.sf_status_var = tk.StringVar()
        self.sf_idx1_var = tk.StringVar()
        self.sf_idx2_var = tk.StringVar()
        self.sf_idx2 = None  # Idx 2 chosen in the panel; carried over to the next files of the run

        self.setup_ui()

    # --- Background Subtraction Logic ---
    def load_background(self):
        """Opens a file dialog to load an independent background file."""
        file_path = filedialog.askopenfilename(title="Select Background .dat file", filetypes=[("Data files", "*.dat"), ("All files", "*.*")])
        if not file_path: return

        try:
            data = _read_numeric_file(file_path)
            if data.size == 0:
                raise ValueError("No valid numerical data found.")

            q, I = data[:, 0], data[:, 1]
            name = os.path.basename(file_path)

            self.bg_dataset = {'name': name, 'q': q, 'I_raw': I.copy()}
            self.bg_file_var.set(f"Current BG: {name}")

            self.apply_background()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load background file:\n{str(e)}")

    def clear_background(self):
        self.bg_dataset = None
        self.bg_file_var.set("Current BG: None")
        self.apply_background()

    def _subtract_bg(self, ds):
        if self.bg_dataset:
            # Use np.interp to robustly map the background intensity to the target's q-grid
            bg_I_mapped = np.interp(ds['q'], self.bg_dataset['q'], self.bg_dataset['I_raw'])
            ds['I'] = ds['I_raw'] - bg_I_mapped
        else:
            ds['I'] = ds['I_raw'].copy()

    def apply_background(self):
        for ds in self.datasets:
            self._subtract_bg(ds)

        # Trigger a full recalculation of the analytical background models and Gaussian fits
        self.process_fits(recalc_bg=True)

    # --- Smooth Scrolling ---
    def _on_mousewheel(self, event):
        """Routes the wheel to whichever scroll area is under the cursor."""
        try:
            w = self.root.winfo_containing(event.x_root, event.y_root)
        except (KeyError, tk.TclError):
            return
        if isinstance(w, tk.Listbox):
            return  # let the file list scroll itself
        while w is not None:
            if w is self.sidebar_canvas or w is self.canvas_scroll:
                self._smooth_scroll(w, -event.delta / 120 * 60)
                return
            w = w.master

    def _wheel_scrolls_panel(self, event):
        """Stops spinboxes/comboboxes from changing value when the wheel passes over them."""
        self._on_mousewheel(event)
        return "break"

    def _smooth_scroll(self, canvas, pixels):
        pending = self._scroll_pending.get(canvas, 0.0)
        if pending * pixels < 0:
            pending = 0.0  # direction reversed: drop the leftover momentum
        self._scroll_pending[canvas] = max(-600.0, min(600.0, pending + pixels))
        if self._scroll_job.get(canvas) is None:
            self._scroll_step(canvas)

    def _scroll_step(self, canvas):
        """Eases toward the target: each frame covers ~35% of the remaining distance."""
        remaining = self._scroll_pending.get(canvas, 0.0)
        region = canvas.cget('scrollregion').split()
        if abs(remaining) < 1 or len(region) != 4:
            self._scroll_pending[canvas] = 0.0
            self._scroll_job[canvas] = None
            return
        step = remaining * 0.35
        if abs(step) < 1: step = math.copysign(1, remaining)
        height = max(float(region[3]) - float(region[1]), 1.0)
        canvas.yview_moveto(canvas.yview()[0] + step / height)
        self._scroll_pending[canvas] = remaining - step
        self._scroll_job[canvas] = self.root.after(12, self._scroll_step, canvas)

    def setup_ui(self):
        # --- NEW: Scrollable Sidebar ---
        self.sidebar_outer = tk.Frame(self.root, width=340, bg="#f4f4f4")
        self.sidebar_outer.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar_outer.pack_propagate(False)

        self.sidebar_canvas = tk.Canvas(self.sidebar_outer, bg="#f4f4f4", highlightthickness=0)
        self.sidebar_scrollbar = ttk.Scrollbar(self.sidebar_outer, orient="vertical", command=self.sidebar_canvas.yview)
        self.sidebar = tk.Frame(self.sidebar_canvas, bg="#f4f4f4")

        self.sidebar_canvas.configure(yscrollcommand=self.sidebar_scrollbar.set)
        self.sidebar_scrollbar.pack(side="right", fill="y")
        self.sidebar_canvas.pack(side="left", fill="both", expand=True)
        self.sidebar_canvas.create_window((0, 0), window=self.sidebar, anchor="nw", width=310)
        self.sidebar.bind("<Configure>", lambda e: self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all")))

        # Action Buttons
        self.btn_speed = tk.Button(self.sidebar, text="Speed-fit", command=self.start_speed_fit,
                                   bg="#8e44ad", fg="white", font=('Arial', 10, 'bold'))
        self.btn_speed.pack(fill=tk.X, pady=2)
        self.btn_upload = tk.Button(self.sidebar, text="Upload .dat Files", command=self.load_files,
                                    bg="#2c3e50", fg="white", font=('Arial', 10, 'bold'))
        self.btn_upload.pack(fill=tk.X, pady=2)
        self.btn_remove_all = tk.Button(self.sidebar, text="Remove All Files", command=self.remove_all_files,
                                        bg="#c0392b", fg="white", font=('Arial', 10, 'bold'))
        self.btn_remove_all.pack(fill=tk.X, pady=2)
        self.btn_save = tk.Button(self.sidebar, text="Batch Save All (Plots & CSV)", command=self.save_plots,
                                  bg="#27ae60", fg="white", font=('Arial', 10, 'bold'))
        self.btn_save.pack(fill=tk.X, pady=2)

        self.file_listbox = tk.Listbox(self.sidebar, height=6, exportselection=False)
        self.file_listbox.pack(fill=tk.X, pady=5)
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)

        # --- NEW: Experimental Background Subtraction ---
        sub_group = tk.LabelFrame(self.sidebar, text="Experimental Background Subtraction", padx=5, pady=5)
        sub_group.pack(fill=tk.X, pady=5)

        tk.Label(sub_group, textvariable=self.bg_file_var, fg="#2980b9", font=('Arial', 9, 'bold')).pack(anchor="w")
        tk.Button(sub_group, text="Upload Background File", command=self.load_background).pack(fill=tk.X, pady=2)
        tk.Button(sub_group, text="Clear Background", command=self.clear_background).pack(fill=tk.X, pady=2)

        # Right-Click Context Menu
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="Rename File", command=self.rename_file)
        self.context_menu.add_command(label="Remove File", command=self.remove_file)
        self.file_listbox.bind("<Button-3>", self.show_context_menu)
        self.file_listbox.bind("<Button-2>", self.show_context_menu)

        # Labels & Titles
        cust_group = tk.LabelFrame(self.sidebar, text="Labels & Titles", padx=5, pady=5)
        cust_group.pack(fill=tk.X, pady=5)

        def add_entry(parent, label_text, var):
            f = tk.Frame(parent); f.pack(fill=tk.X, pady=1)
            tk.Label(f, text=label_text, width=8, anchor="w").pack(side=tk.LEFT)
            e = ttk.Entry(f, textvariable=var)
            e.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            e.bind("<Return>", lambda e: self.on_label_change())
            e.bind("<FocusOut>", lambda e: self.on_label_change())

        add_entry(cust_group, "Main Title", self.main_title_var)
        add_entry(cust_group, "Res Title", self.res_title_var)
        add_entry(cust_group, "X-Axis", self.x_axis_var)
        add_entry(cust_group, "Y-Axis", self.y_axis_var)

        # Region Controls
        ctrl_group = tk.LabelFrame(self.sidebar, text="Indices (Enter to update)", padx=5, pady=5)
        ctrl_group.pack(fill=tk.X, pady=5)
        self.spin_ctrls = {}
        for label, key in [("H-Start", "hs"), ("H-End", "he"), ("G-Start", "gs"), ("G-End", "ge"), ("T-Start", "ts"), ("T-End", "te")]:
            f = tk.Frame(ctrl_group); f.pack(fill=tk.X, pady=1)
            tk.Label(f, text=label, width=8).pack(side=tk.LEFT)
            sb = ttk.Spinbox(f, from_=0, to=99999, width=10, command=self._schedule_param_change)
            sb.pack(side=tk.RIGHT)
            sb.bind("<Return>", lambda e: self.on_param_change())
            sb.bind("<FocusOut>", lambda e: self.on_param_change())
            sb.bind("<MouseWheel>", self._wheel_scrolls_panel)
            self.spin_ctrls[key] = sb

        # Background Parameter Overrides
        bg_param_group = tk.LabelFrame(self.sidebar, text="Background Params (a*q^-b + c)", padx=5, pady=5)
        bg_param_group.pack(fill=tk.X, pady=5)
        for key in ['a', 'b', 'c']:
            f = tk.Frame(bg_param_group); f.pack(fill=tk.X, pady=1)
            tk.Label(f, text=f"Param {key}:", width=8, anchor="w").pack(side=tk.LEFT)
            e = ttk.Entry(f, textvariable=self.bg_vars[key])
            e.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            e.bind("<Return>", lambda e: self.on_manual_bg_change())
            e.bind("<FocusOut>", lambda e: self.on_manual_bg_change())

        # 2-Point Log-Log Linear Background
        lin_bg_group = tk.LabelFrame(self.sidebar, text="2-Point Log-Log Linear BG", padx=5, pady=5)
        lin_bg_group.pack(fill=tk.X, pady=5)

        tk.Checkbutton(lin_bg_group, text="Enable 2-Point Linear", variable=self.use_lin_bg, command=self.on_lin_bg_toggle).pack(anchor="w")

        f_idx = tk.Frame(lin_bg_group); f_idx.pack(fill=tk.X)
        tk.Label(f_idx, text="Idx 1:").pack(side=tk.LEFT)
        e_idx1 = ttk.Entry(f_idx, textvariable=self.lin_idx1_var, width=6)
        e_idx1.pack(side=tk.LEFT, padx=2)
        e_idx1.bind("<Return>", lambda e: self.on_lin_bg_change())
        e_idx1.bind("<FocusOut>", lambda e: self.on_lin_bg_change())

        tk.Label(f_idx, text="Idx 2:").pack(side=tk.LEFT, padx=(5,0))
        e_idx2 = ttk.Entry(f_idx, textvariable=self.lin_idx2_var, width=6)
        e_idx2.pack(side=tk.LEFT, padx=2)
        e_idx2.bind("<Return>", lambda e: self.on_lin_bg_change())
        e_idx2.bind("<FocusOut>", lambda e: self.on_lin_bg_change())

        # Scaling Sections
        scale_group = tk.LabelFrame(self.sidebar, text="Plot Scaling", padx=5, pady=5)
        scale_group.pack(fill=tk.X, pady=5)

        f_main_scale = tk.Frame(scale_group); f_main_scale.pack(fill=tk.X)
        tk.Label(f_main_scale, text="Main:").pack(side=tk.LEFT)
        self.main_scale_cb = ttk.Combobox(f_main_scale, textvariable=self.plot_scale,
                                          values=["lin lin", "log lin", "log log", "guinier"], state="readonly", width=10)
        self.main_scale_cb.pack(side=tk.RIGHT); self.main_scale_cb.bind("<<ComboboxSelected>>", lambda e: self.request_redraw())
        self.main_scale_cb.bind("<MouseWheel>", self._wheel_scrolls_panel)

        f_res_scale = tk.Frame(scale_group); f_res_scale.pack(fill=tk.X, pady=2)
        tk.Label(f_res_scale, text="Resid:").pack(side=tk.LEFT)
        self.res_scale_cb = ttk.Combobox(f_res_scale, textvariable=self.res_scale,
                                         values=["lin lin", "log lin", "log log", "guinier"], state="readonly", width=10)
        self.res_scale_cb.pack(side=tk.RIGHT); self.res_scale_cb.bind("<<ComboboxSelected>>", lambda e: self.request_redraw())
        self.res_scale_cb.bind("<MouseWheel>", self._wheel_scrolls_panel)

        # Fit Residuals Group
        res_group = tk.LabelFrame(self.sidebar, text="Residual Peak Fit", padx=5, pady=5)
        res_group.pack(fill=tk.X, pady=5)
        for mode in ["None", "Single", "Double"]:
            tk.Radiobutton(res_group, text=f"{mode} Peak Fit", variable=self.res_fit_mode,
                           value=mode, command=self.on_fit_mode_change).pack(anchor="w")

        # Peak Constraints Group
        constraint_group = tk.LabelFrame(self.sidebar, text="Peak Constraints (by Index)", padx=5, pady=5)
        constraint_group.pack(fill=tk.X, pady=5)

        f_p1 = tk.Frame(constraint_group); f_p1.pack(fill=tk.X, pady=2)
        tk.Checkbutton(f_p1, text="Fix P1 Idx:", variable=self.fix_p1, command=self.on_fit_mode_change).pack(side=tk.LEFT)
        e_p1 = ttk.Entry(f_p1, textvariable=self.p1_idx_var, width=8)
        e_p1.pack(side=tk.RIGHT)
        e_p1.bind("<Return>", lambda e: self.on_fit_mode_change())
        e_p1.bind("<FocusOut>", lambda e: self.on_fit_mode_change())

        f_p2 = tk.Frame(constraint_group); f_p2.pack(fill=tk.X, pady=2)
        tk.Checkbutton(f_p2, text="Fix P2 Idx:", variable=self.fix_p2, command=self.on_fit_mode_change).pack(side=tk.LEFT)
        e_p2 = ttk.Entry(f_p2, textvariable=self.p2_idx_var, width=8)
        e_p2.pack(side=tk.RIGHT)
        e_p2.bind("<Return>", lambda e: self.on_fit_mode_change())
        e_p2.bind("<FocusOut>", lambda e: self.on_fit_mode_change())

        # View Toggles
        opt_group = tk.LabelFrame(self.sidebar, text="View Options", padx=5, pady=5)
        opt_group.pack(fill=tk.X, pady=5)
        tk.Checkbutton(opt_group, text="Combined View (All Files)", variable=self.is_combined, command=self.request_redraw).pack(anchor="w")
        tk.Checkbutton(opt_group, text="Normalize Residuals", variable=self.normalize_res, command=self.request_redraw).pack(anchor="w")
        tk.Checkbutton(opt_group, text="Scatter Residuals", variable=self.res_as_scatter, command=self.request_redraw).pack(anchor="w")
        tk.Checkbutton(opt_group, text="Show Dashed Peaks", variable=self.show_dashed, command=self.request_redraw).pack(anchor="w")

        # --- Scrollable Container (Main Plot) ---
        self.plot_container = tk.Frame(self.root)
        self.plot_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas_scroll = tk.Canvas(self.plot_container, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.plot_container, orient="vertical", command=self.canvas_scroll.yview)
        self.scrollable_frame = tk.Frame(self.canvas_scroll)
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas_scroll.configure(scrollregion=self.canvas_scroll.bbox("all")))
        self._plot_window = self.canvas_scroll.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas_scroll.bind("<Configure>", lambda e: self.canvas_scroll.itemconfig(self._plot_window, width=e.width))
        self.canvas_scroll.configure(yscrollcommand=self.scrollbar.set)

        # One persistent figure + canvas, redrawn in place (rebuilding the Tk widget each time was the main slowdown)
        self.fig = Figure(figsize=(15, 6))
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=self.scrollable_frame)
        self.canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Pan / zoom / save toolbar for the live plot
        self.toolbar = NavigationToolbar2Tk(self.canvas_widget, self.plot_container, pack_toolbar=False)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        # Speed-fit panel (shown above the plot only while a run is active)
        self.sf_panel = tk.Frame(self.plot_container, bg="#efe6f5", padx=8, pady=6)
        tk.Label(self.sf_panel, textvariable=self.sf_sample_var, bg="#efe6f5", anchor="w",
                 font=('Arial', 10, 'bold')).pack(fill=tk.X)
        tk.Label(self.sf_panel, textvariable=self.sf_file_var, bg="#efe6f5", anchor="w").pack(fill=tk.X)
        sf_idx_row = tk.Frame(self.sf_panel, bg="#efe6f5"); sf_idx_row.pack(fill=tk.X, pady=(2, 4))
        for text, var in [("Idx 1 =", self.sf_idx1_var), ("Idx 2 =", self.sf_idx2_var)]:
            tk.Label(sf_idx_row, text=text, bg="#efe6f5", fg="#5b2c6f",
                     font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 2))
            e = ttk.Entry(sf_idx_row, textvariable=var, width=6, font=('Arial', 10, 'bold'))
            e.pack(side=tk.LEFT, padx=(0, 8))
            e.bind("<Return>", lambda ev: self.sf_jump_to_idx())
        tk.Button(sf_idx_row, text="Go", command=self.sf_jump_to_idx).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(sf_idx_row, textvariable=self.sf_status_var, bg="#efe6f5", anchor="w", fg="#5b2c6f",
                 font=('Arial', 10, 'bold')).pack(side=tk.LEFT, fill=tk.X)
        sf_btns = tk.Frame(self.sf_panel, bg="#efe6f5"); sf_btns.pack(fill=tk.X)
        for text, cmd, color in [("Approve", self.sf_approve, "#27ae60"), ("Skip", self.sf_skip, "#7f8c8d"),
                                 ("Next file", self.sf_finish_file, "#e67e22")]:
            tk.Button(sf_btns, text=text, command=cmd, bg=color, fg="white", width=10,
                      font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=2)
        tk.Frame(sf_btns, width=24, bg="#efe6f5").pack(side=tk.LEFT)
        for text, cmd in [("Skip file", self.sf_skip_file), ("Finish file now", self.sf_finish_file),
                          ("Stop Speed-fit", self.sf_stop)]:
            tk.Button(sf_btns, text=text, command=cmd).pack(side=tk.LEFT, padx=2)
        sf_save_opts = tk.Frame(sf_btns, bg="#efe6f5"); sf_save_opts.pack(side=tk.RIGHT)
        tk.Checkbutton(sf_save_opts, text="Save SVG", variable=self.sf_svg, bg="#efe6f5").pack(anchor="w")
        tk.Checkbutton(sf_save_opts, text="Save PNG", variable=self.sf_png, bg="#efe6f5").pack(anchor="w")

        self.scrollbar.pack(side="right", fill="y")
        self.canvas_scroll.pack(side="left", fill="both", expand=True)

        # Bind the global scroll handler
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)

    # --- UI Helpers ---
    def _sync_bg_entries(self, ds):
        self.bg_vars['a'].set(f"{ds.get('bg_a', 1.0):.5e}")
        self.bg_vars['b'].set(f"{ds.get('bg_b', 4.0):.5f}")
        self.bg_vars['c'].set(f"{ds.get('bg_c', 0.0):.5e}")

    def _update_ui_from_dataset(self, ds):
        """Refreshes the sidebar inputs to match the actively selected dataset."""
        if not ds: return
        for k in ['hs', 'he', 'gs', 'ge', 'ts', 'te']:
            self.spin_ctrls[k].set(ds[k])
        self._sync_bg_entries(ds)

        # Sync UI to this dataset's specific fit settings
        self.res_fit_mode.set(ds.get('fit_mode', "None"))
        self.fix_p1.set(ds.get('fix_p1', False))
        self.p1_idx_var.set(ds.get('p1_idx', 30))
        self.fix_p2.set(ds.get('fix_p2', False))
        self.p2_idx_var.set(ds.get('p2_idx', 60))

        self.use_lin_bg.set(ds.get('use_lin_bg', False))
        self.lin_idx1_var.set(ds.get('lin_idx1', 15))
        self.lin_idx2_var.set(ds.get('lin_idx2', 120))

    def _selected_ds(self):
        if self.selected_index is None or self.selected_index >= len(self.datasets): return None
        return self.datasets[self.selected_index]

    # --- Robust File Loading (Text Safe) ---
    def _make_dataset(self, path):
        """Reads one .dat file into a dataset dict (raises on unreadable files)."""
        data = _read_numeric_file(path)

        if data.size == 0:
            raise ValueError("No valid numerical scattering data found in the file.")

        if data.shape[1] >= 3:
            q, I, sig = data[:, 0], data[:, 1], data[:, 2]
        else:
            q, I, sig = data[:, 0], data[:, 1], np.sqrt(np.clip(data[:, 1], 1e-9, None))

        name = os.path.basename(path)

        q_len = len(q)
        hs = min(4, q_len)
        he = min(15, q_len)
        gs = min(15, q_len)
        ge = min(200, q_len)
        ts = min(200, q_len)
        te = min(400, q_len)

        ds = {
            'name': name, 'path': path, 'q': q, 'I': I.copy(), 'I_raw': I.copy(), 'sig': sig,
            'hs': hs, 'he': he, 'gs': gs, 'ge': ge, 'ts': ts, 'te': te,
            'bg_a': 1.0, 'bg_b': 4.0, 'bg_c': 0.0, 'slope': 4.0,
            'fit_y': None, 'g1_params': None, 'g2_params': None, 'res_total': None,
            'fit_mode': "None",
            'fix_p1': False, 'p1_idx': 30,
            'fix_p2': False, 'p2_idx': 60,
            'manual_bg': False,
            'use_lin_bg': False, 'lin_idx1': min(15, q_len-1), 'lin_idx2': min(120, q_len-1)
        }
        # Files added after a background was loaded get the same subtraction
        self._subtract_bg(ds)
        return ds

    def load_files(self):
        file_paths = filedialog.askopenfilenames(title="Select .dat files", filetypes=[("Data files", "*.dat"), ("All files", "*.*")])
        if not file_paths: return

        new_datasets = []
        for path in file_paths:
            try:
                ds = self._make_dataset(path)
                name = ds['name']
                self.datasets.append(ds)
                new_datasets.append(ds)
                self.file_listbox.insert(tk.END, name)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load {os.path.basename(path)}\n{str(e)}")

        if not new_datasets: return
        if self.selected_index is None:
            self.selected_index = 0
            self.file_listbox.selection_set(0)
        # Fit only the newly added files (previously files added later were never fitted)
        self.process_fits(recalc_bg=True, datasets_to_process=new_datasets)
        self._update_ui_from_dataset(self._selected_ds())

    def remove_all_files(self):
        if not self.datasets:
            return

        # Prompt the user to confirm the mass deletion
        if not messagebox.askyesno("Confirm Deletion", "Are you sure you want to remove all loaded files? This cannot be undone."):
            return

        # Clear the backend data
        self.datasets.clear()

        # --- NEW: Clear BG ---
        self.bg_dataset = None
        self.bg_file_var.set("Current BG: None")

        # Clear the UI listbox
        self.file_listbox.delete(0, tk.END)
        self.selected_index = None

        # Reset the sidebar control variables to zero
        for k in ['hs', 'he', 'gs', 'ge', 'ts', 'te']:
            self.spin_ctrls[k].set(0)
        for k in ['a', 'b', 'c']:
            self.bg_vars[k].set("0")

        # Refresh the canvas to show an empty state
        self.update_plots()

    def on_file_select(self, event):
        selection = self.file_listbox.curselection()
        if selection:
            self.selected_index = selection[0]
            self._update_ui_from_dataset(self.datasets[self.selected_index])
            self.request_redraw()

    def _schedule_param_change(self):
        """Spinbox arrows fire on every click; wait for a short pause before refitting."""
        if self._param_job is not None:
            self.root.after_cancel(self._param_job)
        self._param_job = self.root.after(250, self._run_param_change)

    def _run_param_change(self):
        self._param_job = None
        self.on_param_change()

    def on_param_change(self):
        ds = self._selected_ds()
        if ds is None: return
        try:
            new_vals = {k: int(sb.get()) for k, sb in self.spin_ctrls.items()}
        except ValueError:
            return

        bg_changed = any(ds[k] != new_vals[k] for k in ('hs', 'he', 'ts', 'te'))
        g_changed = any(ds[k] != new_vals[k] for k in ('gs', 'ge'))
        if not (bg_changed or g_changed):
            return  # e.g. focus simply left the box: nothing to refit
        ds.update(new_vals)

        if bg_changed:
            ds['manual_bg'] = False # Unlock if user edits the background region
            # Process only the active dataset to save computation and prevent overwrites
            self.process_fits(recalc_bg=True, datasets_to_process=[ds])
        else:
            # Only the peak window moved: the background is unchanged, just refit the peaks
            self._update_gaussian_fits(ds)
            self.request_redraw()

    def _current_labels(self):
        return (self.main_title_var.get(), self.res_title_var.get(), self.x_axis_var.get(), self.y_axis_var.get())

    def on_label_change(self):
        labels = self._current_labels()
        if labels != self._last_labels:
            self._last_labels = labels
            self.request_redraw()

    def request_redraw(self, delay=15):
        """Coalesces bursts of redraw requests (fast clicking, key repeat) into one draw."""
        if self._redraw_job is not None:
            self.root.after_cancel(self._redraw_job)
        self._redraw_job = self.root.after(delay, self.update_plots)

    def update_plots(self):
        if self._redraw_job is not None:
            self.root.after_cancel(self._redraw_job)
            self._redraw_job = None

        if self.datasets and self.is_combined.get():
            self._build_figure(self.datasets, True, self.normalize_res.get(), fig=self.fig)
        elif self._selected_ds() is not None:
            self._build_figure([self._selected_ds()], False, False, fig=self.fig)
        else:
            self.fig.clear()

        self.canvas_widget.draw_idle()
        self.toolbar.update()  # reset the zoom/pan history to the new plot

    # --- Plot Building Engine ---
    @staticmethod
    def _reset_axes(ax):
        """Removes all plotted data from an axes but keeps its (expensive) tick machinery."""
        for artist in list(ax.lines) + list(ax.collections) + list(ax.patches):
            artist.remove()
        ax.containers.clear()  # errorbar groupings; their artists were removed above
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.relim()
        ax.set_autoscale_on(True)  # a toolbar zoom switches autoscaling off
        # Start from linear like a fresh axes, so limits come out exactly as before
        ax.set_xscale('linear'); ax.set_yscale('linear')

    def _build_figure(self, datasets, is_combined, normalize_res, fig=None):
        if fig is not None and len(fig.axes) == 2:
            # Reuse the live axes: rebuilding them recreates every tick object, which dominates draw time
            ax_data, ax_res = fig.axes
            self._reset_axes(ax_data)
            self._reset_axes(ax_res)
        else:
            if fig is None:
                fig = Figure(figsize=(15, 6))
            else:
                fig.clear()
            ax_data = fig.add_subplot(121)
            ax_res = fig.add_subplot(122)
        fig.subplots_adjust(left=0.05, bottom=0.15, right=0.78, top=0.90, wspace=0.55)

        main_m, res_m = self.plot_scale.get(), self.res_scale.get()
        x_label = self.x_axis_var.get()
        y_label = self.y_axis_var.get()
        main_title = self.main_title_var.get()
        res_title = self.res_title_var.get()
        show_errors = self.show_errors.get()
        res_style = '.' if self.res_as_scatter.get() else '-'
        show_dashed = self.show_dashed.get()

        for i, ds in enumerate(datasets):
            color = cm.tab10(i % 10)
            main_slc = slice(ds['hs'], ds['te'])
            q_r, I_r = ds['q'][main_slc], ds['I'][main_slc]

            # --- MAIN PLOT ---
            if main_m == "guinier": q_p, I_p = q_r**2, np.log(np.clip(I_r, 1e-9, None))
            else: q_p, I_p = q_r, I_r

            lbl_data = f"{ds['name']} (b={ds['slope']:.2f})"
            if show_errors and main_m != "guinier":
                ax_data.errorbar(q_p, I_p, yerr=ds['sig'][main_slc], fmt='.', ms=3, alpha=0.3, color=color, label=lbl_data)
            else:
                ax_data.plot(q_p, I_p, '.', ms=3, alpha=0.3, color=color, label=lbl_data)

            fit_c = 'red' if not is_combined else color
            if ds['fit_y'] is not None:
                fy = ds['fit_y'][main_slc]
                fy_p = np.log(np.clip(fy, 1e-9, None)) if main_m == "guinier" else fy
                ax_data.plot(q_p, fy_p, '--', color=fit_c, lw=1.2)

            # --- RESIDUAL PLOT ---
            if ds['fit_y'] is not None:
                res_slc = slice(ds['he'], ds['ts'])
                q_res_region = ds['q'][res_slc]
                I_res_region = ds['I'][res_slc]
                fit_y_region = ds['fit_y'][res_slc]

                res_v = I_res_region - fit_y_region
                max_abs = np.max(np.abs(res_v)) if len(res_v) else 0
                norm = max_abs if normalize_res and max_abs != 0 else 1.0
                y_res = res_v / norm

                if res_m == "guinier": q_rs, I_rs = q_res_region**2, np.log(np.clip(y_res, 1e-9, None))
                else: q_rs, I_rs = q_res_region, y_res

                ax_res.plot(q_rs, I_rs, res_style, ms=2 if res_style=='.' else 1, alpha=0.3, color=color)

                if ds['res_total'] is not None:
                    peak_strs = []
                    if ds['g1_params'] is not None:
                        q1 = _peak_mode(ds['g1_params'])
                        peak_strs.append(f"q1={q1:.3f} nm⁻¹ (d={2*np.pi/q1:.1f} nm)")
                    if ds['g2_params'] is not None:
                        q2 = _peak_mode(ds['g2_params'])
                        peak_strs.append(f"q2={q2:.3f} nm⁻¹ (d={2*np.pi/q2:.1f} nm)")

                    if peak_strs:
                        leg_res = f"{ds['name']}:\n  " + "   ".join(peak_strs)
                    else:
                        leg_res = f"{ds['name']}"

                    if show_dashed:
                        c1, c2 = (color, color) if is_combined else ('gray', 'blue')
                        if ds['g1_params'] is not None:
                            g1_c = inv_gaussian_func(q_res_region, *ds['g1_params']) / norm
                            if res_m == "guinier": g1_c = np.log(np.clip(g1_c, 1e-9, None))
                            ax_res.plot(q_rs, g1_c, '--', color=c1, lw=0.8)

                        if ds['g2_params'] is not None:
                            g2_c = inv_gaussian_func(q_res_region, *ds['g2_params']) / norm
                            if res_m == "guinier": g2_c = np.log(np.clip(g2_c, 1e-9, None))
                            ax_res.plot(q_rs, g2_c, '--', color=c2, lw=0.8)

                    rt = ds['res_total'][res_slc] / norm
                    rt_p = np.log(np.clip(rt, 1e-9, None)) if res_m == "guinier" else rt
                    ax_res.plot(q_rs, rt_p, '-', color=fit_c, lw=2, label=leg_res)

            if i == len(datasets) - 1:
                last = len(ds['q']) - 1
                h_q, t_q = ds['q'][min(ds['he'], last)], ds['q'][min(ds['ts'], last)]
                shade_s = h_q**2 if main_m == "guinier" else h_q
                shade_e = t_q**2 if main_m == "guinier" else t_q
                ax_data.axvspan(shade_s, shade_e, color='red', alpha=0.05)

        if len(datasets) == 1 and not is_combined:
            ax_data.set_title(f"{datasets[0]['name']} - {main_title}")
        else:
            ax_data.set_title(main_title + (" (Combined View)" if is_combined else ""))

        ax_res.set_title(res_title + (" (Normalized)" if normalize_res and is_combined else ""))

        ax_data.set_xlabel(x_label)
        ax_data.set_ylabel(y_label)
        ax_res.set_xlabel(x_label)
        ax_res.set_ylabel("Residuals")

        ax_data.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

        if main_m == "log log": ax_data.set_xscale('log'); ax_data.set_yscale('log')
        elif main_m == "log lin": ax_data.set_xscale('linear'); ax_data.set_yscale('log')
        else: ax_data.set_xscale('linear'); ax_data.set_yscale('linear')

        if res_m == "log log":
            ax_res.set_xscale('log'); ax_res.set_yscale('log')
        elif res_m == "log lin":
            ax_res.set_xscale('linear'); ax_res.set_yscale('log')
        else:
            ax_res.set_xscale('linear'); ax_res.set_yscale('linear')
            ax_res.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2), useMathText=True)

        ax_res.axhline(0, color='black', lw=0.8, ls=':');
        ax_res.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

        return fig

    # --- Speed-fit: semi-automatic run over a folder tree ---
    # Each folder of .dat files is one sample. For every file, the 2-point background's Idx 1 is swept
    # up from SF_START_IDX (4) until "Next file" is clicked or the upper limit is reached.
    SF_START_IDX = 4
    SF_MIN_IDX = 4      # the 2-point fit uses Idx1-4 .. Idx1, so this keeps all five points
    SF_IDX2 = 120

    def start_speed_fit(self):
        if self.sf is not None: return
        root_dir = filedialog.askdirectory(title="Select the folder to Speed-fit")
        if not root_dir: return

        groups = _scan_dat_folders(root_dir)
        n_files = sum(len(files) for _, files in groups)
        if not n_files:
            messagebox.showinfo("Speed-fit", "No .dat files found (folders named 'Not Good' are skipped).")
            return
        msg = (f"Found {n_files} .dat file(s) in {len(groups)} sample folder(s), skipping 'Not Good' folders.\n\n"
               "Currently loaded files will be cleared. Start Speed-fit?")
        if not messagebox.askyesno("Speed-fit", msg): return

        self.datasets.clear()
        self.file_listbox.delete(0, tk.END)
        self.selected_index = None
        self.sf = {'groups': groups, 'g': 0, 'f': -1, 'sample_rows': [], 'n_files': n_files, 'done_files': 0}
        self.sf_idx2 = self.SF_IDX2
        for b in (self.btn_speed, self.btn_upload, self.btn_remove_all, self.btn_save):
            b.config(state=tk.DISABLED)
        self.sf_panel.pack(side=tk.TOP, fill=tk.X, before=self.toolbar)
        self._sf_next_file()

    def _sf_end(self, completed):
        self.sf = None
        self.sf_panel.pack_forget()
        for b in (self.btn_speed, self.btn_upload, self.btn_remove_all, self.btn_save):
            b.config(state=tk.NORMAL)
        if completed:
            messagebox.showinfo("Speed-fit", "Speed-fit complete. Results were saved next to the data in each sample folder.")

    def _sf_next_file(self):
        sf = self.sf
        sf['f'] += 1
        # Finished the last file of a sample folder: save the sample, then move to the next folder
        while sf['g'] < len(sf['groups']) and sf['f'] >= len(sf['groups'][sf['g']][1]):
            self._sf_save_sample()
            sf['g'] += 1
            sf['f'] = 0
            sf['sample_rows'] = []
        if sf['g'] >= len(sf['groups']):
            self._sf_end(completed=True)
            return

        path = sf['groups'][sf['g']][1][sf['f']]
        try:
            ds = self._make_dataset(path)
        except Exception as e:
            messagebox.showerror("Speed-fit", f"Could not read {path}\n{e}\nSkipping it.")
            sf['done_files'] += 1
            self._sf_next_file()
            return

        n = len(ds['q'])
        ds['use_lin_bg'] = True
        ds['fit_mode'] = "Single"
        ds['lin_idx2'] = min(self.sf_idx2, n - 1)
        sf.update(ds=ds, path=path, dir=1, idx=self.SF_START_IDX, approved=[])

        self.datasets[:] = [ds]
        self.file_listbox.delete(0, tk.END)
        self.file_listbox.insert(tk.END, ds['name'])
        self.file_listbox.selection_set(0)
        self.selected_index = 0

        if not self._sf_idx_in_range(sf['idx']):
            messagebox.showwarning("Speed-fit", f"{ds['name']} has too few points for the index sweep. Skipping it.")
            sf['done_files'] += 1
            self._sf_next_file()
            return
        self._sf_show()

    def _sf_idx_in_range(self, idx):
        return self.SF_MIN_IDX <= idx <= self.sf['ds']['lin_idx2'] - 5

    def _sf_show(self):
        """Fits the current file with the current Idx 1 and refreshes the panel."""
        sf = self.sf
        ds = sf['ds']
        ds['lin_idx1'] = sf['idx']
        ds['manual_bg'] = False
        self.process_fits(recalc_bg=True, datasets_to_process=[ds])
        self._place_regions_from_lin_bg(ds)

        sample_dir, files = sf['groups'][sf['g']]
        self.sf_sample_var.set(f"Sample {sf['g'] + 1}/{len(sf['groups'])}:  {sample_dir}")
        self.sf_file_var.set(f"File {sf['f'] + 1}/{len(files)}:  {sf['path']}")
        n_ok = len(sf['approved'])
        self.sf_idx1_var.set(str(sf['idx']))
        self.sf_idx2_var.set(str(ds['lin_idx2']))
        self.sf_status_var.set(f"(Idx 1 going up)   ·   "
                               f"{n_ok} approved for this file   ·   {sf['done_files']}/{sf['n_files']} files done")

    def sf_jump_to_idx(self):
        """Applies the Idx 1 / Idx 2 typed in the panel; the sweep then continues from there (Idx 1 + 1, same Idx 2)."""
        if self.sf is None: return
        sf = self.sf
        ds = sf['ds']
        n = len(ds['q'])
        try:
            i1 = int(float(self.sf_idx1_var.get()))
            i2 = int(float(self.sf_idx2_var.get()))
        except ValueError:
            messagebox.showwarning("Speed-fit", "Idx 1 and Idx 2 must be whole numbers.")
            self._sf_show()
            return
        if not (0 < i2 <= n - 1):
            messagebox.showwarning("Speed-fit", f"Idx 2 must be between 1 and {n - 1} for this file.")
            self._sf_show()
            return
        if not (self.SF_MIN_IDX <= i1 <= i2 - 5):
            messagebox.showwarning("Speed-fit", f"Idx 1 must be between {self.SF_MIN_IDX} and Idx 2 - 5 ({i2 - 5}).")
            self._sf_show()
            return
        ds['lin_idx2'] = i2
        self.sf_idx2 = i2
        sf['idx'] = i1
        self._sf_show()

    def sf_approve(self):
        if self.sf is None: return
        sf = self.sf
        ds = sf['ds']
        snap = dict(ds)  # fits are re-assigned (not mutated) on the next step, so a shallow copy is a stable snapshot
        snap['name'] = f"{ds['name']} (Idx 1 = {sf['idx']}, Idx 2 = {ds['lin_idx2']})"
        sf['approved'].append({'idx': sf['idx'], 'idx2': ds['lin_idx2'], 'he': ds['he'], 'mode': ds.get('fit_mode', "None"),
                               'peaks': _peaks_of(ds), 'snap': snap})
        self._sf_step()

    def sf_skip(self):
        if self.sf is None: return
        self._sf_step()

    def _sf_step(self):
        sf = self.sf
        sf['idx'] += 1
        if self._sf_idx_in_range(sf['idx']):
            self._sf_show()
        else:
            self.sf_finish_file()  # ran out of usable indices

    def sf_finish_file(self):
        if self.sf is None: return
        self._sf_save_file()
        self.sf['done_files'] += 1
        self._sf_next_file()

    def sf_skip_file(self):
        if self.sf is None: return
        self.sf['done_files'] += 1
        self._sf_next_file()

    def sf_stop(self):
        if self.sf is None: return
        if messagebox.askyesno("Stop Speed-fit", "Stop Speed-fit?\n\nSample folders that were already completed stay saved. "
                                                  "The current sample folder will not get its summary CSV."):
            self._sf_end(completed=False)

    def _sf_save_file(self):
        """Saves the approved fits of the current file into <sample folder>/<file name>/."""
        sf = self.sf
        recs = sf['approved']
        if not recs: return
        ds = sf['ds']
        clean_name = ds['name'].rsplit('.', 1)[0]
        folder = os.path.join(sf['groups'][sf['g']][0], clean_name)

        self.root.config(cursor="watch"); self.root.update_idletasks()
        try:
            os.makedirs(folder, exist_ok=True)
            save_svg, save_png = self.sf_svg.get(), self.sf_png.get()
            for r in recs:
                if not (save_svg or save_png): break
                base = os.path.join(folder, f"{clean_name}_Idx1_{r['idx']:03d}_Idx2_{r['idx2']:03d}")
                fig = self._build_figure([r['snap']], is_combined=False, normalize_res=False)
                if save_png:
                    fig.savefig(f"{base}.png", format="png", dpi=300, bbox_inches="tight")
                if save_svg:
                    fig.savefig(f"{base}.svg", format="svg", bbox_inches="tight")
                fig.clf()

            stats = {}
            for label in ("Peak 1", "Peak 2"):
                pts = [r['peaks'][label] for r in recs if label in r['peaks']]
                if not pts: continue
                q_vals = [p[0] for p in pts]
                d_vals = [p[1] for p in pts]
                stats[label] = {'n': len(d_vals), 'q': float(np.mean(q_vals)), 'd': float(np.mean(d_vals)),
                                'std': float(np.std(d_vals, ddof=1)) if len(d_vals) > 1 else 0.0}

            with open(os.path.join(folder, f"{clean_name}_Index_Sweep.csv"), mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Idx 1", "Idx 2", "H-End", "Fit Mode", "Peak Name", "q_peak (A^-1)", "d_peak (A)"])
                for r in sorted(recs, key=lambda r: (r['idx2'], r['idx'])):
                    for label, (q_peak, d_peak) in sorted(r['peaks'].items()):
                        writer.writerow([r['idx'], r['idx2'], r['he'], r['mode'], label, f"{q_peak:.5f}", f"{d_peak:.5f}"])
                writer.writerow([])
                writer.writerow(["Peak Name", "N indices", "mean q_peak (A^-1)", "mean d_peak (A)", "std d_peak (A)"])
                for label, s in stats.items():
                    writer.writerow([label, s['n'], f"{s['q']:.5f}", f"{s['d']:.5f}", f"{s['std']:.5f}"])

            sf['sample_rows'].append({'name': ds['name'], 'stats': stats})
        except Exception as e:
            messagebox.showerror("Speed-fit", f"Could not save results for {ds['name']}:\n{e}")
        finally:
            self.root.config(cursor="")

    def _sf_save_sample(self):
        """Writes the sample-level CSV: one row per file and peak, with inverse-variance weighted batch stats."""
        sf = self.sf
        rows = sf['sample_rows']
        if not rows: return
        sample_dir = sf['groups'][sf['g']][0]
        try:
            with open(os.path.join(sample_dir, "Speed_Fit_Results.csv"), mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Peak Name", "N indices", "q_peak (A^-1)", "d_peak (A)", "File std_d (A)",
                                 "Weighting", "Batch avg_d (A)", "Batch std_d (A)"])
                for label in ("Peak 1", "Peak 2"):
                    files = [r for r in rows if label in r['stats']]
                    if not files: continue
                    items = [(r['stats'][label]['d'], r['stats'][label]['std'], r['stats'][label]['n']) for r in files]
                    avg, std, notes = _weighted_batch_stats(items)
                    for r, note in zip(files, notes):
                        s = r['stats'][label]
                        writer.writerow([r['name'], label, s['n'], f"{s['q']:.5f}", f"{s['d']:.5f}", f"{s['std']:.5f}",
                                         note, f"{avg:.5f}", f"{std:.5f}"])
        except Exception as e:
            messagebox.showerror("Speed-fit", f"Could not save the summary for {sample_dir}:\n{e}")

    # --- Batch Save Functionality ---
    def save_plots(self):
        if not self.datasets:
            messagebox.showwarning("Empty", "There is no data currently loaded to save.")
            return

        folder = filedialog.askdirectory(title="Select Folder to Batch Save Data")
        if not folder: return

        self.root.config(cursor="watch")
        self.root.update_idletasks()

        try:
            for ds in self.datasets:
                clean_name = ds['name'].rsplit('.', 1)[0]
                base_name = f"{clean_name}_Scattering_and_Residuals"
                full_path = os.path.join(folder, base_name)

                # 1. Save Main Plot (Data + Fits)
                fig = self._build_figure([ds], is_combined=False, normalize_res=self.normalize_res.get())
                fig.savefig(f"{full_path}.png", format="png", dpi=300, bbox_inches="tight")
                fig.savefig(f"{full_path}.svg", format="svg", bbox_inches="tight")
                fig.clf()

                # 2. NEW: Save Standalone Raw Residuals Plot (PNG & SVG)
                if ds['fit_y'] is not None:
                    res_slc = slice(ds['he'], ds['ts'])
                    q_res = ds['q'][res_slc]

                    # If using a loaded background file, ds['I'] is already subtracted.
                    # If using standard fit, calculate the raw subtraction here.
                    if ds.get('bg_file_data') is not None:
                        raw_residuals = ds['I'][res_slc]
                    else:
                        raw_residuals = ds['I'][res_slc] - ds['fit_y'][res_slc]

                    fig_res = Figure(figsize=(8, 5))
                    ax_res_only = fig_res.add_subplot(111)

                    res_style = '.' if self.res_as_scatter.get() else '-'
                    ax_res_only.plot(q_res, raw_residuals, res_style, ms=4 if res_style=='.' else 1.5, color='#2c3e50')
                    ax_res_only.axhline(0, color='black', lw=1, ls=':')

                    ax_res_only.set_title(f"{ds['name']} - Raw Residuals (No Fit)")
                    ax_res_only.set_xlabel(self.x_axis_var.get())
                    ax_res_only.set_ylabel("Raw Residuals")

                    # Apply X-axis scaling from user preferences
                    res_m = self.res_scale.get()
                    if res_m in ["log log", "log lin"]:
                        ax_res_only.set_xscale('log')
                    else:
                        ax_res_only.set_xscale('linear')

                    # Y-axis forced to linear since residuals cross zero
                    ax_res_only.set_yscale('linear')
                    ax_res_only.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2), useMathText=True)

                    res_img_path = os.path.join(folder, f"{clean_name}_Raw_Residuals")
                    fig_res.savefig(f"{res_img_path}.png", format="png", dpi=300, bbox_inches="tight")
                    fig_res.savefig(f"{res_img_path}.svg", format="svg", bbox_inches="tight")
                    fig_res.clf()

            # Save Combined Views if multiple datasets are loaded
            if len(self.datasets) > 1:
                fig_unnorm = self._build_figure(self.datasets, is_combined=True, normalize_res=False)
                unnorm_path = os.path.join(folder, "Combined_Scattering_and_Residuals_Unnormalized")
                fig_unnorm.savefig(f"{unnorm_path}.png", format="png", dpi=300, bbox_inches="tight")
                fig_unnorm.savefig(f"{unnorm_path}.svg", format="svg", bbox_inches="tight")
                fig_unnorm.clf()

                fig_norm = self._build_figure(self.datasets, is_combined=True, normalize_res=True)
                norm_path = os.path.join(folder, "Combined_Scattering_and_Residuals_Normalized")
                fig_norm.savefig(f"{norm_path}.png", format="png", dpi=300, bbox_inches="tight")
                fig_norm.savefig(f"{norm_path}.svg", format="svg", bbox_inches="tight")
                fig_norm.clf()

            # Generate Peak Stats CSV
            csv_path = os.path.join(folder, "Scattering_Fit_Results.csv")
            peak1_d_values = []
            peak2_d_values = []
            rows_to_write = []

            def get_peak_stats(params):
                if params is not None:
                    A, mu, lambda_ = params
                    if lambda_ > 0 and mu > 0:
                        q_peak = _peak_mode(params)
                        d_peak = 2 * np.pi / q_peak
                        return q_peak, d_peak
                return None, None

            for ds in self.datasets:
                q1, d1 = get_peak_stats(ds['g1_params'])
                if d1 is not None:
                    peak1_d_values.append(d1)
                    rows_to_write.append((ds['name'], "Peak 1", q1, d1))

                q2, d2 = get_peak_stats(ds['g2_params'])
                if d2 is not None:
                    peak2_d_values.append(d2)
                    rows_to_write.append((ds['name'], "Peak 2", q2, d2))

            avg_d1 = np.mean(peak1_d_values) if peak1_d_values else 0
            std_d1 = np.std(peak1_d_values, ddof=1) if len(peak1_d_values) > 1 else 0

            avg_d2 = np.mean(peak2_d_values) if peak2_d_values else 0
            std_d2 = np.std(peak2_d_values, ddof=1) if len(peak2_d_values) > 1 else 0

            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Peak Name", "q_peak (A^-1)", "d_peak (A)", "Batch avg_d (A)", "Batch std_d (A)"])
                for row in rows_to_write:
                    name, peak_label, q_peak, d_peak = row
                    if peak_label == "Peak 1":
                        writer.writerow([name, peak_label, f"{q_peak:.5f}", f"{d_peak:.5f}", f"{avg_d1:.5f}", f"{std_d1:.5f}"])
                    elif peak_label == "Peak 2":
                        writer.writerow([name, peak_label, f"{q_peak:.5f}", f"{d_peak:.5f}", f"{avg_d2:.5f}", f"{std_d2:.5f}"])

            messagebox.showinfo("Success", f"Successfully batch saved all plots and CSV Data to:\n{folder}")
        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred while saving:\n{str(e)}")
        finally:
            self.root.config(cursor="")

    # --- Context Menu & Logic ---
    def show_context_menu(self, event):
        if not self.datasets or self.sf is not None: return
        try:
            idx = self.file_listbox.nearest(event.y)
            if idx >= 0:
                self.file_listbox.selection_clear(0, tk.END)
                self.file_listbox.selection_set(idx)
                self.selected_index = idx
                ds = self.datasets[idx]
                self._update_ui_from_dataset(ds)
                self.update_plots()
                self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def rename_file(self):
        if self.selected_index is None: return
        current_name = self.datasets[self.selected_index]['name']
        new_name = simpledialog.askstring("Rename File", "Enter a new name:", initialvalue=current_name, parent=self.root)
        if new_name and new_name.strip():
            self.datasets[self.selected_index]['name'] = new_name.strip()
            self.file_listbox.delete(self.selected_index)
            self.file_listbox.insert(self.selected_index, new_name.strip())
            self.file_listbox.selection_set(self.selected_index)
            self.request_redraw()

    def remove_file(self):
        if self.selected_index is None: return
        del self.datasets[self.selected_index]
        self.file_listbox.delete(self.selected_index)

        if len(self.datasets) > 0:
            new_idx = max(0, self.selected_index - 1)
            self.selected_index = new_idx
            self.file_listbox.selection_set(new_idx)
            self._update_ui_from_dataset(self.datasets[new_idx])
        else:
            self.selected_index = None
            for k in ['hs', 'he', 'gs', 'ge', 'ts', 'te']: self.spin_ctrls[k].set(0)
            for k in ['a', 'b', 'c']: self.bg_vars[k].set("0")
        self.update_plots()

    def on_manual_bg_change(self):
        ds = self._selected_ds()
        if ds is None: return

        try:
            a = float(self.bg_vars['a'].get())
            b = float(self.bg_vars['b'].get())
            c = float(self.bg_vars['c'].get())
        except ValueError:
            self._update_ui_from_dataset(ds)
            return

        # Compare against what the entries display (rounded), so merely tabbing
        # through the boxes does not lock the background with rounded values.
        shown = (float(f"{ds.get('bg_a', 1.0):.5e}"), float(f"{ds.get('bg_b', 4.0):.5f}"), float(f"{ds.get('bg_c', 0.0):.5e}"))
        if (a, b, c) == shown:
            return

        ds['bg_a'], ds['bg_b'], ds['bg_c'] = a, b, c
        ds['slope'] = b
        ds['manual_bg'] = True  # Lock the background from auto-recalculating

        q_safe_all = np.clip(ds['q'], 1e-9, None)
        ds['fit_y'] = a * (q_safe_all ** -b) + c

        self._update_gaussian_fits(ds)
        self.request_redraw()


    def on_lin_bg_toggle(self):
        ds = self._selected_ds()
        if ds is None: return
        ds['use_lin_bg'] = self.use_lin_bg.get()
        ds['manual_bg'] = False  # Unlock manual bg so it recalculates
        self.process_fits(recalc_bg=True, datasets_to_process=[ds])
        if ds['use_lin_bg']:
            self._place_regions_from_lin_bg(ds)

    def on_lin_bg_change(self):
        ds = self._selected_ds()
        if ds is None: return
        try:
            i1, i2 = self.lin_idx1_var.get(), self.lin_idx2_var.get()
        except (ValueError, tk.TclError):
            return
        if ds['lin_idx1'] == i1 and ds['lin_idx2'] == i2:
            return
        ds['lin_idx1'], ds['lin_idx2'] = i1, i2
        if ds.get('use_lin_bg', False):
            ds['manual_bg'] = False
            self.process_fits(recalc_bg=True, datasets_to_process=[ds])
            self._place_regions_from_lin_bg(ds)

    def _place_regions_from_lin_bg(self, ds):
        """After a 2-point fit, sets H-End where the early residuals turn positive,
        G-Start = H-End, and G-End = T-Start = Idx 2. The spinboxes stay editable afterwards."""
        if ds.get('fit_y') is None: return
        n = len(ds['q'])
        i1 = max(0, min(ds['lin_idx1'], n - 1))
        i2 = max(i1, min(ds['lin_idx2'], n - 1))
        res = ds['I'] - ds['fit_y']

        # First index from Idx 1 on where residuals are positive for a few points in a row,
        # so a single noisy positive point among the negatives doesn't count
        # Only search the early half of the Idx 1..Idx 2 range; if residuals never turn positive there, keep Idx 1
        run = 3
        he = i1
        for k in range(i1, i1 + (i2 - i1) // 2):
            if np.all(res[k:k + run] > 0):
                he = k
                break

        ds['he'] = ds['gs'] = he
        ds['ge'] = ds['ts'] = i2
        ds['hs'] = min(ds['hs'], he)
        ds['te'] = max(ds['te'], i2)

        self._update_gaussian_fits(ds)
        self._update_ui_from_dataset(ds)
        self.request_redraw()

    def on_fit_mode_change(self):
        """Saves constraint UI to the active dataset and triggers a local refit"""
        ds = self._selected_ds()
        if ds is None: return

        # Read UI state, keeping the stored index if an entry holds non-integer text
        new = {'fit_mode': self.res_fit_mode.get(), 'fix_p1': self.fix_p1.get(), 'fix_p2': self.fix_p2.get(),
               'p1_idx': ds.get('p1_idx'), 'p2_idx': ds.get('p2_idx')}
        try: new['p1_idx'] = self.p1_idx_var.get()
        except (ValueError, tk.TclError): pass
        try: new['p2_idx'] = self.p2_idx_var.get()
        except (ValueError, tk.TclError): pass

        if all(ds.get(k) == v for k, v in new.items()):
            return  # nothing changed (e.g. focus left an entry)
        ds.update(new)

        # Refit ONLY the active dataset's peaks based on new constraints
        self._update_gaussian_fits(ds)
        self.request_redraw()

    # --- Fitting Logic ---
    def fit_ig_core(self, q, I, q_min, q_max, force_mu=None, maxfev=2000, fixed_mode=None, fixed_amp=None):
        if len(q) < 4: return [0, max(1e-5, (q_min+q_max)/2), 1.0]

        safe_q_min = max(1e-5, q_min)
        safe_q_max = max(safe_q_min + 1e-3, q_max)

        max_idx = np.argmax(I)
        peak_amp = max(1e-5, I[max_idx])

        if fixed_mode is not None and fixed_amp is not None:
            m = max(1e-5, fixed_mode)
            I_m = max(1e-5, fixed_amp)

            def constrained_func(x, lambda_):
                return inv_gaussian_func(x, _amp_for_height(I_m, m, lambda_), _safe_mu(m, lambda_), lambda_)

            min_lambda = 3.001 * m
            p0 = [max(10.0 * m, min_lambda + 1e-4)]
            bounds = ([min_lambda], [np.inf])

            try:
                popt, _ = curve_fit(constrained_func, q, I, p0=p0, bounds=bounds, maxfev=maxfev)
                lambda_ = popt[0]
                return [_amp_for_height(I_m, m, lambda_), _safe_mu(m, lambda_), lambda_]
            except Exception:
                lambda_fb = max(10.0 * m, min_lambda + 1e-4)
                return [_amp_for_height(I_m, m, lambda_fb), _safe_mu(m, lambda_fb), lambda_fb]
        else:
            # Unconstrained Fit: Optimize Height (H), Mode (m), and Ratio (R=l/m)
            def func(x, H, m, R):
                l = R * m
                return inv_gaussian_func(x, _amp_for_height(H, m, l), _safe_mu(m, l), l)

            m_guess = q[max_idx] if force_mu is None else force_mu
            m_guess = np.clip(m_guess, safe_q_min * 0.501, safe_q_max * 1.499)

            # HARD LIMIT: Amplitude can never exceed 3x the max data point
            max_H = max(np.max(I) * 3.0, 1e-3)

            p0 = [peak_amp, m_guess, 10.0]
            bounds = ([0, safe_q_min*0.5, 3.001], [max_H, safe_q_max*1.5, 1000.0])

            try:
                popt, _ = curve_fit(func, q, I, p0=p0, bounds=bounds, maxfev=maxfev)
                H_fit, m_fit, R_fit = popt
                l_fit = R_fit * m_fit
                return [_amp_for_height(H_fit, m_fit, l_fit), _safe_mu(m_fit, l_fit), l_fit]
            except Exception:
                l_fb = 10.0 * m_guess
                return [_amp_for_height(peak_amp, m_guess, l_fb), _safe_mu(m_guess, l_fb), l_fb]

    def fit_double_simultaneous(self, q, I, q_min, q_max, fixed_m1=None, fixed_m2=None, fixed_h1=None, fixed_h2=None):
        safe_q_min = max(1e-5, q_min)
        safe_q_max = max(safe_q_min + 1e-3, q_max)

        # Initial peak detection
        peaks, _ = find_peaks(I, distance=max(3, len(q)//20))
        if len(peaks) >= 2:
            top_peaks = sorted(peaks, key=lambda p: I[p], reverse=True)[:2]
            idx1, idx2 = top_peaks[0], top_peaks[1]
        else:
            idx1 = np.argmax(I)
            window = max(3, len(q) // 10)
            I_masked = np.copy(I)
            start_idx = max(0, idx1 - window)
            end_idx = min(len(q), idx1 + window)
            I_masked[start_idx:end_idx] = np.min(I)
            idx2 = np.argmax(I_masked)

        m1_guess = fixed_m1 if fixed_m1 else q[idx1]
        m1_guess = np.clip(m1_guess, safe_q_min * 0.501, safe_q_max * 1.499)

        m2_guess = fixed_m2 if fixed_m2 else q[idx2]
        m2_guess = np.clip(m2_guess, safe_q_min * 0.501, safe_q_max * 1.499)
        if abs(m1_guess - m2_guess) < 1e-5: m2_guess += 1e-3

        max_res = max(np.max(I), 1e-5)
        amp1 = max(I[np.argmin(np.abs(q - m1_guess))], max_res * 0.05)
        amp2 = max(I[np.argmin(np.abs(q - m2_guess))], max_res * 0.05)

        get_A = _amp_for_height
        safe_mu = _safe_mu

        # ---------------------------------------------------------
        # THE DYNAMIC ANCHORING ENGINE
        # ---------------------------------------------------------
        fm1 = fixed_m1 if fixed_m1 else m1_guess
        fm2 = fixed_m2 if fixed_m2 else m2_guess

        # Lock the exact y-values of the dataset at the chosen indices
        # (blended heights are passed in when a fixed index falls between data points)
        T1 = fixed_h1 if fixed_m1 and fixed_h1 is not None else I[np.argmin(np.abs(q - fm1))]
        T2 = fixed_h2 if fixed_m2 and fixed_h2 is not None else I[np.argmin(np.abs(q - fm2))]

        # Helper to calculate the normalized overlap of the tails
        def get_n(x, m, R):
            l = R * m
            mu = safe_mu(m, l)
            return _ig_unit_at(x, mu, l) / max(_ig_unit_at(m, mu, l), 1e-12)

        # Map optimization parameters depending on which peaks are locked
        if fixed_m1 and fixed_m2:
            def param_mapper(p):
                R1, R2 = p[0], p[1]
                n1_at_m2 = get_n(fm2, fm1, R1)
                n2_at_m1 = get_n(fm1, fm2, R2)
                D = max(1.0 - n1_at_m2 * n2_at_m1, 1e-5)
                # Solve the 2x2 algebraic intersection
                H1 = (T1 - T2 * n2_at_m1) / D
                H2 = (T2 - T1 * n1_at_m2) / D
                return [max(H1, 1e-5), fm1, R1, max(H2, 1e-5), fm2, R2]
            p0 = [10.0, 10.0]
            bounds = ([3.001, 3.001], [100.0, 100.0])

        elif fixed_m1:
            def param_mapper(p):
                R1, H2, m2, R2 = p
                # Dynamically calculate H1 so Peak1 + Peak2 perfectly equals Data at m1
                n2_at_m1 = get_n(fm1, m2, R2)
                H1 = T1 - H2 * n2_at_m1
                return [max(H1, 1e-5), fm1, R1, H2, m2, R2]
            p0 = [10.0, amp2, m2_guess, 10.0]
            bounds = ([3.001, 0, safe_q_min*0.5, 3.001], [100.0, max_res*3.0, safe_q_max*1.5, 100.0])

        elif fixed_m2:
            def param_mapper(p):
                H1, m1, R1, R2 = p
                # Dynamically calculate H2 so Peak1 + Peak2 perfectly equals Data at m2
                n1_at_m2 = get_n(fm2, m1, R1)
                H2 = T2 - H1 * n1_at_m2
                return [H1, m1, R1, max(H2, 1e-5), fm2, R2]
            p0 = [amp1, m1_guess, 10.0, 10.0]
            bounds = ([0, safe_q_min*0.5, 3.001, 3.001], [max_res*3.0, safe_q_max*1.5, 100.0, 100.0])

        else:
            def param_mapper(p): return [p[0], p[1], p[2], p[3], p[4], p[5]]
            p0 = [amp1, m1_guess, 10.0, amp2, m2_guess, 10.0]
            bounds = ([0, safe_q_min*0.5, 3.001, 0, safe_q_min*0.5, 3.001],
                      [max_res*3.0, safe_q_max*1.5, 100.0, max_res*3.0, safe_q_max*1.5, 100.0])

        def objective(p):
            H1, m1, R1, H2, m2, R2 = param_mapper(p)
            l1, l2 = R1 * m1, R2 * m2

            y1 = inv_gaussian_func(q, get_A(H1, m1, l1), safe_mu(m1, l1), l1)
            y2 = inv_gaussian_func(q, get_A(H2, m2, l2), safe_mu(m2, l2), l2)
            diff = (y1 + y2) - I

            # Apply a gentle 5x penalty if Peak 1 tries to exceed the data curve in its domain,
            # keeping it tidy without crushing it into a needle.
            diff[(diff > 0) & (y1 > y2)] *= 5.0

            return diff

        try:
            res = least_squares(objective, p0, bounds=bounds, max_nfev=5000)
            H1, m1, R1, H2, m2, R2 = param_mapper(res.x)
            l1, l2 = R1 * m1, R2 * m2

            p1 = [get_A(H1, m1, l1), safe_mu(m1, l1), l1]
            p2 = [get_A(H2, m2, l2), safe_mu(m2, l2), l2]

            if p1[1] > p2[1]: return p2, p1
            return p1, p2
        except Exception:
            try:
                if hasattr(self, 'fit_double_peeling'):
                    return self.fit_double_peeling(q, I, q_min, q_max)
            except Exception: pass

            l1_fb, l2_fb = 10.0 * m1_guess, 10.0 * m2_guess
            p1_fb = [get_A(amp1*0.8, m1_guess, l1_fb), safe_mu(m1_guess, l1_fb), l1_fb]
            p2_fb = [get_A(amp2*0.8, m2_guess, l2_fb), safe_mu(m2_guess, l2_fb), l2_fb]
            if p1_fb[1] > p2_fb[1]: return p2_fb, p1_fb
            return p1_fb, p2_fb

    def _update_gaussian_fits(self, ds):
        try:
            q, I = ds['q'], ds['I']

            # Isolate the Gaussian fitting domain
            gs, ge = ds['gs'], ds['ge']
            q_p = q[gs:ge]
            res_p = I[gs:ge] - ds['fit_y'][gs:ge]

            # Safe boundary values for the fit functions based on the new slice
            q_min_val = q_p[0] if len(q_p) > 0 else q[0]
            q_max_val = q_p[-1] if len(q_p) > 0 else q[-1]

            # Pull parameters directly from the dataset, NOT the active UI
            mode = ds.get('fit_mode', "None")
            ds['g1_params'], ds['g2_params'], ds['res_total'] = None, None, None

            # Fixed peaks may sit between data points: q and residual height are blended linearly
            res_all = I - ds['fit_y']
            fixed_mode1, fixed_amp1 = None, None
            if ds.get('fix_p1', False):
                idx1 = ds.get('p1_idx', 0)
                if 0 <= idx1 <= len(q) - 1:
                    fixed_mode1 = _value_at_index(q, idx1)
                    fixed_amp1 = _value_at_index(res_all, idx1)

            fixed_mode2, fixed_amp2 = None, None
            if ds.get('fix_p2', False):
                idx2 = ds.get('p2_idx', 0)
                if 0 <= idx2 <= len(q) - 1:
                    fixed_mode2 = _value_at_index(q, idx2)
                    fixed_amp2 = _value_at_index(res_all, idx2)

            if mode == "Single":
                if fixed_mode1 is None and len(res_p) > 0:
                    # Snap the peak top onto the residuals' peak; only the width is fitted. The top point is
                    # found on smoothed residuals (so one noisy spike doesn't win), then a parabola through the
                    # raw points around it gives a q between data points: 5 points first, widened to 7, 9 ... 15
                    # when noise hides the curvature. Ticking "Fix P1" overrides this.
                    k, fixed_amp1 = _smoothed_peak(res_p)
                    fixed_mode1 = q_p[k]
                    for half in range(2, 8):
                        vertex = _parabola_peak(q_p, res_p, k, half)
                        if vertex is not None:
                            fixed_mode1, fixed_amp1 = vertex
                            break
                ds['g1_params'] = self.fit_ig_core(
                    q_p, res_p, q_min_val, q_max_val,
                    fixed_mode=fixed_mode1, fixed_amp=fixed_amp1
                )
                ds['res_total'] = inv_gaussian_func(q, *ds['g1_params'])

            elif mode == "Double":
                ds['g1_params'], ds['g2_params'] = self.fit_double_simultaneous(
                    q_p, res_p, q_min_val, q_max_val,
                    fixed_m1=fixed_mode1, fixed_m2=fixed_mode2,
                    fixed_h1=fixed_amp1, fixed_h2=fixed_amp2
                )
                ds['res_total'] = inv_gaussian_func(q, *ds['g1_params']) + inv_gaussian_func(q, *ds['g2_params'])
        except Exception:
            pass

    def _fit_background(self, ds):
        """Returns [a, b, c] for the a*q^-b + c background of one dataset, or None if the regions are too small."""
        q, I = ds['q'], ds['I']

        idx_head = np.arange(len(q))[ds['hs']:ds['he']]
        idx_tail = np.arange(len(q))[ds['ts']:ds['te']]
        idx_bg = np.concatenate([idx_head, idx_tail])

        if len(idx_bg) < 3: return None

        if ds.get('use_lin_bg', False):
            i1 = ds['lin_idx1']
            i2 = ds['lin_idx2']

            # EXACTLY the indices requested: [i1-4, i1-3, i1-2, i1-1, i1, i2]
            target_indices = list(range(i1 - 4, i1 + 1)) + [i2]
            valid_indices = [idx for idx in target_indices if 0 <= idx < len(q) and q[idx] > 0 and I[idx] > 0]

            valid_q = q[valid_indices]
            valid_I = I[valid_indices]

            if len(valid_q) < 3:
                return [1.0, 4.0, 0.0]

            c = 0.0
            a = 1.0
            b = 4.0

            # Math Safeguard 1: Clamp C to prevent log(I - C) from blowing up to -infinity
            max_c = np.min(valid_I) * 0.95

            # Math Safeguard 2: Damping factor to prevent "too high values" divergence
            learning_rate = 0.25

            log_q = np.log(valid_q)
            # Target C is set so the curve reaches the tail value (the last point in the array)
            tail_q = valid_q[-1]
            tail_I = valid_I[-1]

            for _ in range(30): # Iterative loop
                # Fit the linear log-log line aq^-b across the chosen points, after subtracting current C
                m, log_A = np.polyfit(log_q, np.log(valid_I - c), 1)

                a = np.exp(log_A)
                b = -m

                c_target = tail_I - (a * (tail_q ** -b))

                # Safely update C
                c = c + learning_rate * (c_target - c)
                c = max(0.0, min(c, max_c))

            return [a, b, c]

        try:
            if len(idx_tail) > 3 and len(idx_head) > 3:
                tail_end_start = int(len(idx_tail) * 0.8)
                idx_tail_end = idx_tail[tail_end_start:]
                c_guess = np.median(I[idx_tail_end]) if len(idx_tail_end) > 0 else np.min(I)

                def tail_func(x, a, c): return a * (x ** -4.0) + c

                mid_tail = idx_tail[len(idx_tail) // 2]
                a_guess = max(1e-5, (I[mid_tail] - c_guess) * (q[mid_tail] ** 4.0))

                popt_tail, _ = curve_fit(
                    tail_func, q[idx_tail], I[idx_tail],
                    p0=[a_guess, c_guess],
                    bounds=([1e-5, -np.inf], [np.inf, np.inf]),
                    maxfev=2000
                )
                a_step2, c_step2 = popt_tail

                def full_bg_func(x, a, b, c): return a * (x ** -b) + c

                p_bg, _ = curve_fit(
                    full_bg_func, q[idx_bg], I[idx_bg],
                    p0=[a_step2, 4.0, c_step2],
                    bounds=([1e-5, 1e-5, -np.inf], [np.inf, 20.0, np.inf]),
                    maxfev=5000
                )
            else:
                p_bg, _ = curve_fit(
                    lambda x, a, b, c: a*(x**-b)+c, q[idx_bg], I[idx_bg],
                    p0=[1, 4, np.min(I)], bounds=([1e-5, 1e-5, -np.inf], [np.inf, 20.0, np.inf])
                )
            return p_bg
        except Exception:
            return [1.0, 4.0, np.min(I) if len(I) > 0 else 0]

    def process_fits(self, recalc_bg=True, datasets_to_process=None):
        self.root.config(cursor="watch"); self.root.update_idletasks()

        # Default to all datasets if a specific list isn't provided
        target_datasets = datasets_to_process if datasets_to_process is not None else self.datasets

        try:
            for ds in target_datasets:
                # Check the lock before recalculating
                if recalc_bg and not ds.get('manual_bg', False):
                    p_bg = self._fit_background(ds)
                    if p_bg is None: continue

                    # Only update dictionary if auto-fitting
                    ds['bg_a'], ds['bg_b'], ds['bg_c'] = p_bg[0], p_bg[1], p_bg[2]
                    ds['slope'] = p_bg[1]

                    q_safe_all = np.clip(ds['q'], 1e-9, None)
                    ds['fit_y'] = p_bg[0] * (q_safe_all ** -p_bg[1]) + p_bg[2]

                self._update_gaussian_fits(ds)
        finally:
            self.root.config(cursor="")

        # Keep the a/b/c boxes showing the freshly fitted background of the selected file
        sel = self._selected_ds()
        if sel is not None and any(ds is sel for ds in target_datasets):
            self._sync_bg_entries(sel)
        self.request_redraw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ScatteringFitApp(root)
    root.mainloop()
