import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk, ImageOps
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers as pylab_helpers
from matplotlib.widgets import SpanSelector
import datetime
import json
import os
MODEL_DIMENSIONS = {
    "4343": {"Probe": (3072, 3072)},
    "3643": {"Probe": (3072, 3072)},
    "2430": {"Probe": (3840, 3072)},
    "1824": {"Probe": (3840, 3072)},
    "1616": {"Probe": (3840, 3072)},
    "1624": {"Probe": (3840, 3072)},
}
ROI_PRESETS_XY = {
    "4343": ((1, 1), (3072, 3072)),
    "3643": ((1, 1), (3065, 2556)),
    "2430": ((1, 1), (3840, 3072)),
    "1616": ((0, 548), (1650, 2191)),
    "1824": ((1, 1), (2303, 3072)),
    "1624": ((0, 548), (2477, 2191)),
}
MODEL_OPTIONS = list(MODEL_DIMENSIONS.keys())
WHITE_DIM_KEY = "Probe"
CTF_DIM_KEY = "Probe"
ROTATE_FLIP_MODELS = {"1616", "1624"}


# ==============================
# 1) Bresenham으로 라인 프로파일 추출
# ==============================
def bresenham_line_profile(array, x1, y1, x2, y2):
    """
    (x1, y1)부터 (x2, y2)까지의 픽셀 intensity 값을 추출.
    array: 2D numpy array
    """
    points = []
    dx = abs(x2 - x1)
    sx = 1 if x1 < x2 else -1
    dy = -abs(y2 - y1)
    sy = 1 if y1 < y2 else -1
    err = dx + dy

    cur_x, cur_y = x1, y1
    while True:
        if 0 <= cur_y < array.shape[0] and 0 <= cur_x < array.shape[1]:
            points.append(array[cur_y, cur_x])
        if cur_x == x2 and cur_y == y2:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            cur_x += sx
        if e2 <= dx:
            err += dx
            cur_y += sy

    return np.array(points)

# ==============================
# 2) Matplotlib 그래프 Pan/Zoom 처리용 클래스
# ==============================
class PlotInteractor:
    """
    - 좌클릭 드래그(Pan)
    - 마우스 휠(Zoom)
    """
    def __init__(self, ax):
        self.ax = ax
        self.fig = ax.figure
        self.pressing = False
        self.xpress = None
        self.ypress = None

    def on_button_press(self, event):
        if event.button == 1:
            self.pressing = True
            self.xpress = event.xdata
            self.ypress = event.ydata

    def on_button_release(self, event):
        if event.button == 1:
            self.pressing = False
            self.xpress = None
            self.ypress = None

    def on_mouse_move(self, event):
        if not self.pressing:
            return
        if event.xdata is None or event.ydata is None:
            return

        dx = event.xdata - self.xpress
        dy = event.ydata - self.ypress
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        self.ax.set_xlim(cur_xlim[0] - dx, cur_xlim[1] - dx)
        self.ax.set_ylim(cur_ylim[0] - dy, cur_ylim[1] - dy)
        self.ax.figure.canvas.draw_idle()

        self.xpress = event.xdata
        self.ypress = event.ydata

    def on_scroll(self, event):
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            return

        zoom_factor = 1.1 if event.button == 'up' else 1/1.1
        new_width = (cur_xlim[1] - cur_xlim[0]) * zoom_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * zoom_factor

        relx = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        rely = (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([
            xdata - new_width * relx,
            xdata + new_width * (1 - relx)
        ])
        self.ax.set_ylim([
            ydata - new_height * rely,
            ydata + new_height * (1 - rely)
        ])
        self.ax.figure.canvas.draw_idle()

# ==============================
# 3) Tkinter 메인 클래스
# ==============================
class ImageViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IMG 파일 뷰어 (CTF 계산 포함 - 3lp/mm 확장)")
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "white_import_config.json")
        self.persisted_white_config = self.load_white_config()
        default_model = None
        if self.persisted_white_config:
            default_model = self.persisted_white_config.get("last_model")
        if default_model not in MODEL_DIMENSIONS:
            default_model = MODEL_OPTIONS[0]
        self.model_var = tk.StringVar(value=default_model)
        self.current_model = default_model
        self.use_3lp_var = tk.BooleanVar(value=True)
        self.selection_fig = None

        # 이미지 상태
        self.image_array_original = None
        self.image_array_8bit = None  # 8bit 변환된 배열 (밝기/대조 조정용)
        self.display_image = None
        self.tk_image = None
        self.zoom_level = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # 마우스 드래그 판별
        self.is_dragging = False
        self.is_right_dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.last_offset_x = 0
        self.last_offset_y = 0
        self.right_drag_start_x = 0
        self.right_drag_start_y = 0
        self.last_brightness = 0.5
        self.last_contrast = 1.0

        # 점/선 (라인 프로파일)
        self.point1 = None
        self.point2 = None
        self.line_id = None
        self.point_ids = []

        # 파일 경로
        self.selected_file_path = None  # CTF 파일
        self.white_file_path = None     # White 파일
        self.white_array = None         # White 파일의 원본 배열
        
        # White 파일 로드 설정 저장
        self.white_config = None  # (img_type, endian, width, height, offset)
        
        # White ROI 설정
        self.white_roi = None  # (x, y, w, h)
        self.use_roi_sensitivity = False  # ROI 감도 계산 여부
        
        # 공통 설정 저장 (CTF와 White가 같은 설정 사용)
        self.common_config = None  # (img_type, endian, offset)
        
        # 결과 창 추적
        self.result_windows = []  # 결과 창 리스트
        
        # 이미지 밝기/대조 조정용
        self.auto_balance_center = None  # (x, y) - auto balance 중심점
        self.brightness_level = 0.5  # L (Level) - 밝기, 0.0~1.0
        self.contrast_width = 1.0   # W (Width) - 대조, 0.0~2.0

        self.create_widgets()
        self.bind_events()

    def create_widgets(self):
        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        tk.Label(top_frame, text="Model:").pack(side=tk.LEFT, padx=5)
        self.model_combo = ttk.Combobox(top_frame, textvariable=self.model_var, state="readonly", width=6)
        self.model_combo['values'] = MODEL_OPTIONS
        self.model_combo.pack(side=tk.LEFT, padx=5)

        self.use_3lp_check = tk.Checkbutton(top_frame, text="3lp", variable=self.use_3lp_var)
        self.use_3lp_check.pack(side=tk.LEFT, padx=5)

        btn_open = tk.Button(top_frame, text="Open IMG", command=self.on_click_open_file)
        btn_open.pack(side=tk.LEFT, padx=5)

        self.btn_plot = tk.Button(top_frame, text="Line plot", command=self.show_line_plot)
        self.btn_plot.pack(side=tk.LEFT, padx=5)

        self.canvas = tk.Canvas(self, bg="black", width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def bind_events(self):
        self.canvas.bind("<ButtonPress-1>", self.on_left_button_press)
        self.canvas.bind("<B1-Motion>", self.on_left_button_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_button_release)
        self.canvas.bind("<Button-3>", self.on_right_button_press)
        self.canvas.bind("<B3-Motion>", self.on_right_button_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_button_release)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)  # Linux
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)  # Linux
    def load_white_config(self):
        if not os.path.exists(self.config_path):
            return {"by_model": {}, "last_model": None}
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {"by_model": {}, "last_model": None}

        if isinstance(data, dict) and "by_model" in data:
            data.setdefault("by_model", {})
            data.setdefault("last_model", None)
            return data

        if isinstance(data, dict) and "img_type" in data:
            return {"by_model": {"default": data}, "last_model": None}

        return {"by_model": {}, "last_model": None}

    def get_white_config_for_model(self, model):
        if not self.persisted_white_config:
            return None
        by_model = self.persisted_white_config.get("by_model", {})
        return by_model.get(model)

    def save_white_config(self, model, config):
        if not model or not config:
            return
        if not self.persisted_white_config:
            self.persisted_white_config = {"by_model": {}, "last_model": None}
        by_model = self.persisted_white_config.setdefault("by_model", {})
        by_model[model] = config
        self.persisted_white_config["last_model"] = model
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.persisted_white_config, f, indent=2, ensure_ascii=True)
        except Exception:
            return

    def model_requires_transform(self, model):
        return model in ROTATE_FLIP_MODELS

    def get_model_dimensions(self, model, role):
        return MODEL_DIMENSIONS.get(model, {}).get(role)

    def get_white_display_dimensions(self, model):
        dims = self.get_model_dimensions(model, WHITE_DIM_KEY)
        if not dims:
            return None, None
        w, h = dims
        if self.model_requires_transform(model):
            return h, w
        return w, h

    def transform_roi_for_rotflip(self, x, y, w, h, orig_w, orig_h):
        corners = [(x, y), (x + w - 1, y), (x, y + h - 1), (x + w - 1, y + h - 1)]
        transformed = []
        for cx, cy in corners:
            nx = orig_h - 1 - cy
            ny = orig_w - 1 - cx
            transformed.append((nx, ny))
        xs = [p[0] for p in transformed]
        ys = [p[1] for p in transformed]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return int(min_x), int(min_y), int(max_x - min_x + 1), int(max_y - min_y + 1)

    def get_roi_defaults_for_model(self, model):
        dims = self.get_model_dimensions(model, WHITE_DIM_KEY)
        if not dims:
            return {"x": "0", "y": "0", "w": "0", "h": "0"}
        base_w, base_h = dims
        if model in ROI_PRESETS_XY:
            (x1, y1), (x2, y2) = ROI_PRESETS_XY[model]
            x = int(x1)
            y = int(y1)
            w = int(x2 - x1 + 1)
            h = int(y2 - y1 + 1)
        else:
            x, y, w, h = 0, 0, base_w, base_h
        if self.model_requires_transform(model):
            x, y, w, h = self.transform_roi_for_rotflip(x, y, w, h, base_w, base_h)
        return {"x": str(x), "y": str(y), "w": str(w), "h": str(h)}

    def close_all_matplotlib_figures(self):
        managers = list(pylab_helpers.Gcf.get_all_fig_managers())
        for manager in managers:
            try:
                plt.close(manager.canvas.figure)
            except Exception:
                pass

    # ==============================
    # (1) 파일 열기 + 설정
    # ==============================
    def on_click_open_file(self):
        model = self.model_var.get()
        if model not in MODEL_DIMENSIONS:
            messagebox.showerror("Error", "Please select a model.")
            return
        messagebox.showinfo("File Select", "Select White IMG file.")
        white_path = filedialog.askopenfilename(
            title="Select White IMG file",
            filetypes=[("IMG files", "*.img"), ("All files", "*.*")]
        )
        if not white_path:
            return
        self.white_file_path = white_path

        messagebox.showinfo("File Select", "Select CTF IMG file.")
        ctf_path = filedialog.askopenfilename(
            title="Select CTF IMG file",
            filetypes=[("IMG files", "*.img"), ("All files", "*.*")]
        )
        if not ctf_path:
            return
        self.selected_file_path = ctf_path

        self.open_config_window(is_ctf_file=False, is_first=True)

    def open_config_window(self, is_ctf_file=True, is_first=True):
        model = self.model_var.get()
        if model not in MODEL_DIMENSIONS:
            messagebox.showerror("Error", "Please select a model.")
            return

        file_type = "CTF" if is_ctf_file else "White"
        config_win = tk.Toplevel(self)
        config_win.title(f"{file_type} File Import")

        default_img_type = "16-bit Unsigned"
        default_endian = "Little-endian"
        default_offset = "0"
        cfg = None
        if not is_first and self.common_config:
            img_type, endian, offset = self.common_config
            default_img_type = img_type
            default_endian = endian
            default_offset = str(offset)
        else:
            cfg = self.get_white_config_for_model(model)
            if is_first and cfg:
                default_img_type = cfg.get("img_type", default_img_type)
                default_endian = cfg.get("endian", default_endian)
                default_offset = str(cfg.get("offset", 0))

        ctf_dims = self.get_model_dimensions(model, CTF_DIM_KEY)
        white_dims = self.get_model_dimensions(model, WHITE_DIM_KEY)
        white_disp_w, white_disp_h = self.get_white_display_dimensions(model)
        if not ctf_dims or not white_dims:
            messagebox.showerror("Error", "Model dimensions not found.")
            config_win.destroy()
            return

        ctf_w, ctf_h = ctf_dims
        white_w, white_h = white_dims

        roi_defaults = self.get_roi_defaults_for_model(model)
        use_roi_default = False
        if cfg:
            roi_defaults = {
                "x": str(cfg.get("roi_x", roi_defaults["x"])),
                "y": str(cfg.get("roi_y", roi_defaults["y"])),
                "w": str(cfg.get("roi_w", roi_defaults["w"])),
                "h": str(cfg.get("roi_h", roi_defaults["h"])),
            }
            use_roi_default = bool(cfg.get("use_roi_sensitivity", False))

        tk.Label(config_win, text="Image Type:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        image_type_var = tk.StringVar(value=default_img_type)
        combo_type = ttk.Combobox(config_win, textvariable=image_type_var, state="readonly")
        combo_type['values'] = ["8-bit Unsigned", "16-bit Unsigned", "16-bit Signed"]
        combo_type.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        tk.Label(config_win, text="Endianness:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        endian_var = tk.StringVar(value=default_endian)
        combo_endian = ttk.Combobox(config_win, textvariable=endian_var, state="readonly")
        combo_endian['values'] = ["Little-endian", "Big-endian"]
        combo_endian.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        tk.Label(config_win, text="Offset to first image (bytes):").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        offset_var = tk.StringVar(value=default_offset)
        tk.Entry(config_win, textvariable=offset_var, width=10).grid(row=2, column=1, padx=5, pady=5, sticky="w")

        if is_ctf_file:
            size_text = "Model size ({}): {}x{}".format(CTF_DIM_KEY, ctf_w, ctf_h)
        else:
            size_text = "Model size ({}): {}x{}".format(WHITE_DIM_KEY, white_disp_w, white_disp_h)
        tk.Label(config_win, text=size_text).grid(row=3, column=0, columnspan=2, padx=5, pady=5)

        tk.Label(config_win, text="--- ROI Settings ---", font=("Arial", 9, "bold")).grid(row=4, column=0, columnspan=2, padx=5, pady=10)

        tk.Label(config_win, text="ROI X:").grid(row=5, column=0, padx=5, pady=5, sticky="e")
        x_var = tk.StringVar(value=roi_defaults["x"])
        tk.Entry(config_win, textvariable=x_var, width=10).grid(row=5, column=1, padx=5, pady=5, sticky="w")

        tk.Label(config_win, text="ROI Y:").grid(row=6, column=0, padx=5, pady=5, sticky="e")
        y_var = tk.StringVar(value=roi_defaults["y"])
        tk.Entry(config_win, textvariable=y_var, width=10).grid(row=6, column=1, padx=5, pady=5, sticky="w")

        tk.Label(config_win, text="ROI Width:").grid(row=7, column=0, padx=5, pady=5, sticky="e")
        w_var = tk.StringVar(value=roi_defaults["w"])
        tk.Entry(config_win, textvariable=w_var, width=10).grid(row=7, column=1, padx=5, pady=5, sticky="w")

        tk.Label(config_win, text="ROI Height:").grid(row=8, column=0, padx=5, pady=5, sticky="e")
        h_var = tk.StringVar(value=roi_defaults["h"])
        tk.Entry(config_win, textvariable=h_var, width=10).grid(row=8, column=1, padx=5, pady=5, sticky="w")

        use_roi_var = tk.BooleanVar(value=use_roi_default)
        tk.Checkbutton(config_win, text="Use ROI sensitivity", variable=use_roi_var).grid(row=9, column=0, columnspan=2, padx=5, pady=5)

        def on_ok():
            img_type = image_type_var.get()
            endian = endian_var.get()
            try:
                offset = int(offset_var.get())
                roi_x = int(x_var.get())
                roi_y = int(y_var.get())
                roi_w = int(w_var.get())
                roi_h = int(h_var.get())
            except ValueError:
                messagebox.showerror("Error", "All values must be integers.")
                return

            if roi_x < 0 or roi_y < 0 or roi_x + roi_w > white_disp_w or roi_y + roi_h > white_disp_h:
                messagebox.showerror("Error", "ROI is outside image.\nImage size: {}x{}".format(white_disp_w, white_disp_h))
                return

            config_win.destroy()
            self.common_config = (img_type, endian, offset)
            self.white_roi = (roi_x, roi_y, roi_w, roi_h)
            self.use_roi_sensitivity = use_roi_var.get()

            if not is_ctf_file:
                self.save_white_config(model, {
                    "img_type": img_type,
                    "endian": endian,
                    "offset": offset,
                    "roi_x": roi_x,
                    "roi_y": roi_y,
                    "roi_w": roi_w,
                    "roi_h": roi_h,
                    "use_roi_sensitivity": self.use_roi_sensitivity,
                })

            self.current_model = model
            if is_ctf_file:
                self.load_img_with_config(img_type, endian, ctf_w, ctf_h, offset, model=model)
            else:
                self.load_white_file_with_config(img_type, endian, white_w, white_h, offset, model=model)
                if self.selected_file_path:
                    self.load_img_with_config(img_type, endian, ctf_w, ctf_h, offset, model=model)

        def on_cancel():
            config_win.destroy()

        tk.Button(config_win, text="OK", command=on_ok).grid(row=10, column=0, padx=5, pady=10)
        tk.Button(config_win, text="Cancel", command=on_cancel).grid(row=10, column=1, padx=5, pady=10)

    def load_img_with_config(self, img_type, endian, width, height, offset, model=None):
        if not self.selected_file_path:
            return

        endian_prefix = "<" if endian == "Little-endian" else ">"
        if img_type == "8-bit Unsigned":
            dtype_str = endian_prefix + "u1"
            bytes_per_pixel = 1
        elif img_type == "16-bit Unsigned":
            dtype_str = endian_prefix + "u2"
            bytes_per_pixel = 2
        elif img_type == "16-bit Signed":
            dtype_str = endian_prefix + "i2"
            bytes_per_pixel = 2
        else:
            messagebox.showerror("오류", "지원하지 않는 이미지 타입입니다.")
            return

        try:
            with open(self.selected_file_path, "rb") as f:
                f.seek(offset)
                raw_data = f.read()

            arr = np.frombuffer(raw_data, dtype=dtype_str)
            expected_pixels = width * height
            if len(arr) < expected_pixels:
                messagebox.showerror("오류", "파일 크기가 (width*height)에 비해 부족합니다.")
                return
            arr = arr[:expected_pixels].reshape((height, width))
            if model is None:
                model = self.current_model or self.model_var.get()
            if self.model_requires_transform(model):
                arr = np.rot90(arr, k=1)
                arr = np.fliplr(arr)
            self.image_array_original = arr

            # 16bit -> 8bit (white is zero)
            if bytes_per_pixel == 2:
                min_val, max_val = arr.min(), arr.max()
                if min_val < max_val:
                    scaled = 255.0 - ((arr - min_val) / (max_val - min_val) * 255.0)
                else:
                    scaled = np.zeros_like(arr, dtype=np.float32)
                arr_8 = scaled.astype(np.uint8)
            else:
                arr_8 = arr.astype(np.uint8)

            pil_img = Image.fromarray(arr_8, mode='L')
            # 밝기/대조 자동 조정 (원본 배열 저장)
            self.image_array_8bit = arr_8.copy()
            self.apply_auto_balance()

            # Canvas 크기에 맞춰 축소
            self.canvas.update_idletasks()
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            img_w, img_h = self.display_image.size
            fit_scale = min(canvas_w / img_w, canvas_h / img_h)
            self.zoom_level = fit_scale if fit_scale < 1.0 else 1.0
            self.offset_x = canvas_w // 2
            self.offset_y = canvas_h // 2

            self.update_canvas_image()
            
            # CTF 이미지 표시 후 안내 메시지
            messagebox.showinfo("안내", "클릭을 두번하여 line을 그려주세요(좌->우)")

        except Exception as e:
            messagebox.showerror("오류", f"이미지 파일을 여는 중 오류가 발생했습니다.\n{e}")

    # ==============================
    # (2-2) White 파일 로드
    # ==============================
    def load_white_file_with_config(self, img_type, endian, width, height, offset, model=None):
        if not self.white_file_path:
            return

        endian_prefix = "<" if endian == "Little-endian" else ">"
        if img_type == "8-bit Unsigned":
            dtype_str = endian_prefix + "u1"
        elif img_type == "16-bit Unsigned":
            dtype_str = endian_prefix + "u2"
        elif img_type == "16-bit Signed":
            dtype_str = endian_prefix + "i2"
        else:
            messagebox.showerror("오류", "지원하지 않는 이미지 타입입니다.")
            return

        try:
            with open(self.white_file_path, "rb") as f:
                f.seek(offset)
                raw_data = f.read()

            arr = np.frombuffer(raw_data, dtype=dtype_str)
            expected_pixels = width * height
            if len(arr) < expected_pixels:
                messagebox.showerror("오류", "파일 크기가 (width*height)에 비해 부족합니다.")
                return
            arr = arr[:expected_pixels].reshape((height, width))
            if model is None:
                model = self.current_model or self.model_var.get()
            if self.model_requires_transform(model):
                arr = np.rot90(arr, k=1)
                arr = np.fliplr(arr)
            self.white_array = arr
            self.white_config = (img_type, endian, width, height, offset)
            
            # White ROI 설정 창은 이미 통합 창에서 처리됨

        except Exception as e:
            messagebox.showerror("오류", f"White 파일을 여는 중 오류가 발생했습니다.\n{e}")


    def compute_white_roi_mean(self):
        """White 파일의 ROI mean 값을 계산"""
        if not self.use_roi_sensitivity:
            return None
        if self.white_array is None or self.white_roi is None:
            return None
        
        x, y, w, h = self.white_roi
        roi_arr = self.white_array[y:y+h, x:x+w]
        return float(roi_arr.mean())
    
    def compute_white_full_mean(self):
        """White 파일의 전체 mean 값을 계산"""
        if self.white_array is None:
            return None
        return float(self.white_array.mean())

    def apply_auto_balance(self):
        """Auto balance 적용 (히스토그램 균등화)"""
        if self.image_array_8bit is None:
            return
        
        arr = self.image_array_8bit.copy().astype(np.float32)
        
        # 중심점이 지정된 경우, 해당 지점 주변 영역만 사용
        if self.auto_balance_center is not None:
            cx, cy = self.auto_balance_center
            h, w = arr.shape
            # 중심점 주변 200x200 영역 사용
            radius = 100
            y1 = max(0, cy - radius)
            y2 = min(h, cy + radius)
            x1 = max(0, cx - radius)
            x2 = min(w, cx + radius)
            roi = arr[y1:y2, x1:x2]
            min_val = float(roi.min())
            max_val = float(roi.max())
        else:
            min_val = float(arr.min())
            max_val = float(arr.max())
        
        # 히스토그램 스트레칭
        if max_val > min_val:
            arr = ((arr - min_val) / (max_val - min_val) * 255.0).clip(0, 255)
        else:
            arr = np.zeros_like(arr)
        
        arr_8 = arr.astype(np.uint8)
        pil_img = Image.fromarray(arr_8, mode='L')
        
        # 추가로 autocontrast 적용
        pil_img = ImageOps.autocontrast(pil_img, cutoff=1)
        self.display_image = pil_img
        self.update_canvas_image()

    def apply_brightness_contrast(self):
        """밝기(L)와 대조(W) 적용"""
        if self.image_array_8bit is None:
            return
        
        arr = self.image_array_8bit.copy().astype(np.float32)
        
        # 밝기 조정 (L)
        # brightness_level: 0.0 (어두움) ~ 1.0 (밝음)
        arr = arr + (self.brightness_level - 0.5) * 255.0
        
        # 대조 조정 (W)
        # contrast_width: 0.1 (낮은 대조) ~ 2.0 (높은 대조)
        center = 128.0
        arr = (arr - center) * self.contrast_width + center
        
        # 클리핑
        arr = arr.clip(0, 255).astype(np.uint8)
        
        pil_img = Image.fromarray(arr, mode='L')
        self.display_image = pil_img
        self.update_canvas_image()

    # ==============================
    # (3) Canvas 표시 갱신
    # ==============================
    def update_canvas_image(self):
        if self.display_image is None:
            return
        w, h = self.display_image.size
        new_w = int(w * self.zoom_level)
        new_h = int(h * self.zoom_level)
        if new_w <= 0 or new_h <= 0:
            return

        resized = self.display_image.resize((new_w, new_h), Image.Resampling.NEAREST)
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, image=self.tk_image, anchor=tk.CENTER)
        self.redraw_points_and_line()

    # ==============================
    # (4) 마우스 이벤트
    # ==============================
    def on_left_button_press(self, event):
        # 이미지 이동을 위한 드래그 시작
        self.is_dragging = False
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.last_offset_x = self.offset_x
        self.last_offset_y = self.offset_y

    def on_left_button_drag(self, event):
        # 좌클릭 드래그로 이미지 이동
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        if abs(dx) > 3 or abs(dy) > 3:
            self.is_dragging = True

        if self.is_dragging:
            self.offset_x = self.last_offset_x + dx
            self.offset_y = self.last_offset_y + dy
            self.update_canvas_image()

    def on_left_button_release(self, event):
        if not self.is_dragging:
            # 클릭 -> 점 찍기
            self.handle_left_click_for_points(event)

    def handle_left_click_for_points(self, event):
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        if img_x is None or img_y is None:
            return

        img_x = int(round(img_x))
        img_y = int(round(img_y))

        if self.point1 is None:
            self.point1 = (img_x, img_y)
        else:
            if self.point2 is None:
                self.point2 = (img_x, img_y)
            else:
                # 이미 두 점 있으면 초기화 후 첫 점 다시
                self.point1 = (img_x, img_y)
                self.point2 = None

        self.redraw_points_and_line()

    def redraw_points_and_line(self):
        if self.line_id is not None:
            self.canvas.delete(self.line_id)
            self.line_id = None
        for pid in self.point_ids:
            self.canvas.delete(pid)
        self.point_ids.clear()

        # point1
        if self.point1 is not None:
            cx, cy = self.image_to_canvas_coords(*self.point1)
            if cx is not None and cy is not None:
                r = 5
                p1 = self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline="yellow", fill="yellow")
                self.point_ids.append(p1)

        # point2
        if self.point2 is not None:
            cx, cy = self.image_to_canvas_coords(*self.point2)
            if cx is not None and cy is not None:
                r = 5
                p2 = self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline="yellow", fill="yellow")
                self.point_ids.append(p2)

        # 선
        if self.point1 is not None and self.point2 is not None:
            c1 = self.image_to_canvas_coords(*self.point1)
            c2 = self.image_to_canvas_coords(*self.point2)
            if c1[0] is not None and c2[0] is not None:
                self.line_id = self.canvas.create_line(
                    c1[0], c1[1], c2[0], c2[1],
                    fill="yellow", width=2
                )

    def on_right_button_press(self, event):
        # 우클릭 드래그 시작 (L, W 조절용)
        self.is_right_dragging = False
        self.right_drag_start_x = event.x
        self.right_drag_start_y = event.y
        self.last_brightness = self.brightness_level
        self.last_contrast = self.contrast_width

    def on_right_button_drag(self, event):
        # 우클릭 드래그로 L, W 조절
        if self.image_array_8bit is None:
            return
        
        dx = event.x - self.right_drag_start_x
        dy = event.y - self.right_drag_start_y
        
        # Ctrl 키 확인
        is_ctrl = (event.state & 0x4) != 0  # Control 키 체크
        unit = 10 if is_ctrl else 5
        
        # 좌우 드래그: W (대조) 조절
        if abs(dx) > 3:
            self.is_right_dragging = True
            # 픽셀 단위로 조절 (단위: 5 또는 10)
            contrast_steps = dx / unit
            self.contrast_width = max(0.1, min(2.0, self.last_contrast + contrast_steps * 0.01))
        
        # 위아래 드래그: L (밝기) 조절
        if abs(dy) > 3:
            self.is_right_dragging = True
            # 픽셀 단위로 조절 (단위: 5 또는 10), 위로 드래그 = 밝게
            brightness_steps = -dy / unit
            self.brightness_level = max(0.0, min(1.0, self.last_brightness + brightness_steps * 0.01))
        
        if self.is_right_dragging:
            self.apply_brightness_contrast()

    def on_right_button_release(self, event):
        if not self.is_right_dragging:
            # 우클릭 -> 초기화
            self.point1 = None
            self.point2 = None
            if self.line_id is not None:
                self.canvas.delete(self.line_id)
                self.line_id = None
            for pid in self.point_ids:
                self.canvas.delete(pid)
            self.point_ids.clear()
        self.is_right_dragging = False

    def on_double_click(self, event):
        # 더블클릭 시 해당 지점에서 auto balance 재적용
        if self.image_array_8bit is None:
            return
        
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        if img_x is None or img_y is None:
            return
        
        img_x = int(round(img_x))
        img_y = int(round(img_y))
        
        # 밝기/대조 초기화
        self.brightness_level = 0.5
        self.contrast_width = 1.0
        
        # 해당 지점을 중심으로 auto balance 재적용
        self.auto_balance_center = (img_x, img_y)
        self.apply_auto_balance()

    def on_mouse_wheel(self, event):
        if event.delta > 0 or event.num == 4:
            self.zoom_level *= 1.1
        elif event.delta < 0 or event.num == 5:
            self.zoom_level /= 1.1

        self.zoom_level = max(0.05, min(self.zoom_level, 20.0))
        self.update_canvas_image()

    # ==============================
    # (5) Line Plot + CTF 계산
    # ==============================
    def show_line_plot(self):
        if self.point1 is None or self.point2 is None:
            messagebox.showinfo("알림", "노란색 선(두 점)을 먼저 지정하세요.")
            return
        if self.image_array_original is None:
            messagebox.showinfo("알림", "이미지가 로드되어 있지 않습니다.")
            return
        
        # 기존 결과 창 닫기
        for win in self.result_windows:
            try:
                win.destroy()
            except:
                pass
        self.result_windows.clear()
        self.close_all_matplotlib_figures()
        self.selection_fig = None

        # 1) 라인 프로파일 계산
        x1, y1 = self.point1
        x2, y2 = self.point2
        profile = bresenham_line_profile(self.image_array_original, x1, y1, x2, y2)
        use_3lp = bool(self.use_3lp_var.get())
        if self.selection_fig is not None:
            try:
                plt.close(self.selection_fig)
            except Exception:
                pass
            self.selection_fig = None

        fig_select, ax_select = plt.subplots()
        self.selection_fig = fig_select

        def close_selection():
            for cid in pan_zoom_ids:
                try:
                    fig_select.canvas.mpl_disconnect(cid)
                except Exception:
                    pass
            try:
                plt.close(fig_select)
            except Exception:
                pass
            if self.selection_fig is fig_select:
                self.selection_fig = None

        ax_select.plot(profile, color='blue')
        ax_select.set_title("Line Profile (Selection)")
        ax_select.set_xlabel("Pixel index")
        ax_select.set_ylabel("Intensity")
        ax_select.set_xlim(0, len(profile))
        ymin, ymax = float(profile.min()), float(profile.max())
        ax_select.set_ylim(ymin, ymax)

        interactor = PlotInteractor(ax_select)
        pan_zoom_ids = []
        pan_zoom_ids.append(fig_select.canvas.mpl_connect("button_press_event", interactor.on_button_press))
        pan_zoom_ids.append(fig_select.canvas.mpl_connect("button_release_event", interactor.on_button_release))
        pan_zoom_ids.append(fig_select.canvas.mpl_connect("motion_notify_event", interactor.on_mouse_move))
        pan_zoom_ids.append(fig_select.canvas.mpl_connect("scroll_event", interactor.on_scroll))

        # SpanSelector로 범위를 지정받는 헬퍼
        def get_span(prompt, color='red'):
            messagebox.showinfo("범위 지정", prompt)
            # 팬/줌 임시 해제
            for cid in pan_zoom_ids:
                fig_select.canvas.mpl_disconnect(cid)

            span_result = {}
            def onselect(xmin, xmax):
                span_result['xmin'] = xmin
                span_result['xmax'] = xmax

            span = SpanSelector(
                ax_select, onselect, 'horizontal', useblit=True,
                props=dict(alpha=0.5, facecolor=color)
            )
            while 'xmin' not in span_result or 'xmax' not in span_result:
                plt.pause(0.1)

            span.disconnect_events()

            # 팬/줌 재연결
            pan_zoom_ids.clear()
            pan_zoom_ids.append(fig_select.canvas.mpl_connect("button_press_event", interactor.on_button_press))
            pan_zoom_ids.append(fig_select.canvas.mpl_connect("button_release_event", interactor.on_button_release))
            pan_zoom_ids.append(fig_select.canvas.mpl_connect("motion_notify_event", interactor.on_mouse_move))
            pan_zoom_ids.append(fig_select.canvas.mpl_connect("scroll_event", interactor.on_scroll))

            return span_result['xmin'], span_result['xmax']

        # 결과 저장용
        #  각 dict에: { 'lp': int, 'max_val': float, 'max_idx': int, 'min_val': float, 'min_idx': int, 'frac': float, 'ctf': float }
        results = []

        # --- (A) 1lp/mm MAX 범위
        xmin_1max, xmax_1max = get_span("좌클릭 드래그로 1lp/mm MAX 범위를 지정하세요", color='red')
        st1 = max(0, int(round(xmin_1max)))
        ed1 = min(len(profile), int(round(xmax_1max)))
        if ed1 <= st1:
            close_selection()
            messagebox.showerror("오류", "1lp/mm MAX 범위가 잘못되었습니다.")
            return
        max_1 = float(np.max(profile[st1:ed1]))
        idx_1max_local = np.argmax(profile[st1:ed1])
        idx_1max = st1 + idx_1max_local

        # --- (B) 1lp/mm MIN 범위 (드래그로 선택)
        xmin_1min, xmax_1min = get_span("좌클릭 드래그로 1lp/mm MIN 범위를 지정하세요", color='blue')
        min_range_1lp_start = max(0, int(round(xmin_1min)))
        min_range_1lp_end = min(len(profile), int(round(xmax_1min)))
        if min_range_1lp_end <= min_range_1lp_start:
            close_selection()
            messagebox.showerror("오류", "1lp/mm MIN 범위가 잘못되었습니다.")
            return
        min_1 = float(np.min(profile[min_range_1lp_start:min_range_1lp_end]))
        idx_1min_local = np.argmin(profile[min_range_1lp_start:min_range_1lp_end])
        idx_1min = min_range_1lp_start + idx_1min_local

        # fraction_1
        denom_1 = max_1 + min_1
        fraction_1 = (max_1 - min_1)/denom_1 if denom_1 != 0 else 0
        ctf_1 = fraction_1*100

        # 1lp/mm 결과 저장
        results.append({
            'lp': 1,
            'max_val': max_1,
            'max_idx': idx_1max,
            'min_val': min_1,
            'min_idx': idx_1min,
            'frac': fraction_1,  # ex) 0.5012
            'ctf': ctf_1         # ex) 50.12
        })

        # --- (C) 2lp/mm MAX 범위
        xmin_2max, xmax_2max = get_span("좌클릭 드래그로 2lp/mm MAX 범위를 지정하세요", color='red')
        st2 = max(0, int(round(xmin_2max)))
        ed2 = min(len(profile), int(round(xmax_2max)))
        if ed2 <= st2:
            close_selection()
            messagebox.showerror("오류", "2lp/mm MAX 범위가 잘못되었습니다.")
            return
        max_2 = float(np.max(profile[st2:ed2]))
        idx_2max_local = np.argmax(profile[st2:ed2])
        idx_2max = st2 + idx_2max_local

        # --- (D) 2lp/mm MIN 범위 (드래그로 선택)
        xmin_2min, xmax_2min = get_span("좌클릭 드래그로 2lp/mm MIN 범위를 지정하세요", color='blue')
        min_st2 = max(0, int(round(xmin_2min)))
        min_ed2 = min(len(profile), int(round(xmax_2min)))
        if min_ed2 <= min_st2:
            close_selection()
            messagebox.showerror("오류", "2lp/mm MIN 범위가 잘못되었습니다.")
            return
        min_2 = float(np.min(profile[min_st2:min_ed2]))
        idx_2min_local = np.argmin(profile[min_st2:min_ed2])
        idx_2min = min_st2 + idx_2min_local
        denom_2 = (max_2 + min_2)
        fraction_2 = (max_2 - min_2)/denom_2 if denom_2 != 0 else 0
        
        ctf_2 = 0
        if fraction_1 != 0:
            ctf_2 = (fraction_2/fraction_1)*100

        results.append({
            'lp': 2,
            'max_val': max_2,
            'max_idx': idx_2max,
            'min_val': min_2,
            'min_idx': idx_2min,
            'frac': fraction_2,  # ex) 0.1973
            'ctf': ctf_2         # ex) 39.37
        })
        if use_3lp:
            # --- (E) 3lp/mm MAX range
            xmin_3max, xmax_3max = get_span("Drag to select 3lp/mm MAX range", color='red')
            st3 = max(0, int(round(xmin_3max)))
            ed3 = min(len(profile), int(round(xmax_3max)))
            if ed3 <= st3:
                close_selection()
                messagebox.showerror("Error", "Invalid 3lp/mm MAX range.")
                return
            max_3 = float(np.max(profile[st3:ed3]))
            idx_3max_local = np.argmax(profile[st3:ed3])
            idx_3max = st3 + idx_3max_local

            # --- (F) 3lp/mm MIN range
            xmin_3min, xmax_3min = get_span("Drag to select 3lp/mm MIN range", color='blue')
            min_st3 = max(0, int(round(xmin_3min)))
            min_ed3 = min(len(profile), int(round(xmax_3min)))
            if min_ed3 <= min_st3:
                close_selection()
                messagebox.showerror("Error", "Invalid 3lp/mm MIN range.")
                return
            min_3 = float(np.min(profile[min_st3:min_ed3]))
            idx_3min_local = np.argmin(profile[min_st3:min_ed3])
            idx_3min = min_st3 + idx_3min_local

            denom_3 = (max_3 + min_3)
            fraction_3 = (max_3 - min_3) / denom_3 if denom_3 != 0 else 0

            ctf_3 = 0
            if fraction_1 != 0:
                ctf_3 = (fraction_3 / fraction_1) * 100

            results.append({
                'lp': 3,
                'max_val': max_3,
                'max_idx': idx_3max,
                'min_val': min_3,
                'min_idx': idx_3min,
                'frac': fraction_3,
                'ctf': ctf_3
            })

        # Selection complete -> close figure
        close_selection()

        # --- (G) Final result figure
        fig_result, ax_result = plt.subplots()

        ax_result.plot(profile, color='blue')
        ax_result.set_title("Line Profile (Final)")
        ax_result.set_xlabel("Pixel index")
        ax_result.set_ylabel("Intensity")
        ax_result.set_xlim(0, len(profile))
        ymin2, ymax2 = float(profile.min()), float(profile.max())
        ax_result.set_ylim(ymin2, ymax2)

        # 각 lp/mm의 MAX, MIN 위치를 scatter
        for r in results:
            # MAX (빨간)
            if r['max_idx'] is not None:
                ax_result.scatter(r['max_idx'], r['max_val'], color='red', marker='o')
                ax_result.annotate(
                    f"{r['lp']} lp/mm MAX",
                    xy=(r['max_idx'], r['max_val']),
                    xytext=(r['max_idx'], r['max_val'] + (ymax2-ymin2)*0.03),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    color='red', fontsize=8
                )
            # MIN (초록)
            if r['min_idx'] is not None:
                ax_result.scatter(r['min_idx'], r['min_val'], color='green', marker='o')
                ax_result.annotate(
                    f"{r['lp']} lp/mm MIN",
                    xy=(r['min_idx'], r['min_val']),
                    xytext=(r['min_idx'], r['min_val'] - (ymax2-ymin2)*0.07),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    color='green', fontsize=8
                )

        fig_result.tight_layout()

        # (H) 결과 테이블 표시 (한 창에 합치기)
        white_roi_mean = self.compute_white_roi_mean()
        white_full_mean = self.compute_white_full_mean()
        self.show_ctf_results(results, white_roi_mean, white_full_mean, fig_result)

    def show_ctf_results(self, results, white_roi_mean=None, white_full_mean=None, fig_result=None):
        """
        lp/mm, MAX, MIN, (MAX-MIN)/(MAX+MIN), CTF, White ROI Mean
        - 1lp/mm: fraction_1 * 100
        - 2lp/mm: fraction_2 / fraction_1 * 100
        - 3lp/mm: fraction_3 / fraction_1 * 100
        - txt 파일은 탭(\t) 구분
        """
        lines = []
        header = "lp/mm\tMAX\tMIN\t(MAX-MIN)/(MAX+MIN)\tCTF"
        if white_roi_mean is not None:
            header += "\tWhite ROI Mean"
        lines.append(header)

        # 먼저 1lp/mm의 fraction_1 구하기
        fraction_1 = 0
        for r in results:
            if r['lp'] == 1:
                fraction_1 = r['frac']
                break

        for r in results:
            lp_val = r['lp']
            max_val = r['max_val']
            min_val = r['min_val']
            frac = r['frac']
            
            # CTF 값 (이미 계산됨)
            ctf_val = r['ctf']
            frac_percent = frac * 100

            line_str = (
                f"{lp_val}\t"
                f"{int(max_val)}\t"
                f"{int(min_val)}\t"
                f"{frac_percent:.1f}%\t"
                f"{ctf_val:.1f}%"
            )
            if white_roi_mean is not None:
                line_str += f"\t{int(white_roi_mean)}"
            lines.append(line_str)

        # 줄 띄우고 추가 정보
        lines.append("")  # 빈 줄
        lines.append("CTF(2lp)\tCTF(3lp)\t전체 감도\tROI 감도")
        
        # CTF 값 찾기
        ctf_2lp_value = None
        ctf_3lp_value = None
        for r in results:
            if r['lp'] == 2:
                ctf_2lp_value = r['ctf']
            elif r['lp'] == 3:
                ctf_3lp_value = r['ctf']
        
        str_2 = f"{ctf_2lp_value:.1f}%" if ctf_2lp_value is not None else "미설정"
        str_3 = f"{ctf_3lp_value:.1f}%" if ctf_3lp_value is not None else "미설정"
        
        ctf_line = f"{str_2}\t{str_3}"
        
        if white_full_mean is not None:
            ctf_line += f"\t{int(white_full_mean)}"
        else:
            ctf_line += "\t미설정"
        if white_roi_mean is not None:
            ctf_line += f"\t{int(white_roi_mean)}"
        else:
            ctf_line += "\t미설정"
        lines.append(ctf_line)

        output_str = "\n".join(lines)

        # 새 창 (Line Plot과 결과를 한 창에)
        result_win = tk.Toplevel(self)
        result_win.title("CTF 결과")
        result_win.geometry("1000x700")
        self.result_windows.append(result_win)

        # 상단에 Line Plot 배치
        if fig_result is not None:
            canvas_fig = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(fig_result, master=result_win)
            canvas_fig.draw()
            canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True, side=tk.TOP, pady=5)

        # 하단에 결과 텍스트 배치
        text_widget = tk.Text(result_win, width=70, height=10)
        text_widget.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM, pady=5)
        text_widget.insert("1.0", output_str)

        # 클립보드
        self.clipboard_clear()
        self.clipboard_append(output_str)

        # txt 파일 저장 (탭 구분)
        now_str = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
        filename = f"ctf({now_str}).txt"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(output_str)
        except Exception as e:
            messagebox.showwarning("경고", f"결과 파일 저장 중 오류가 발생했습니다.\n{e}")

    # ==============================
    # 좌표 변환 (Canvas <-> 이미지)
    # ==============================
    def canvas_to_image_coords(self, cx, cy):
        if self.display_image is None:
            return None, None

        w, h = self.display_image.size
        new_w = int(w * self.zoom_level)
        new_h = int(h * self.zoom_level)

        img_left = self.offset_x - new_w/2
        img_top  = self.offset_y - new_h/2

        img_x = cx - img_left
        img_y = cy - img_top

        if img_x < 0 or img_y < 0 or img_x >= new_w or img_y >= new_h:
            return None, None

        orig_x = img_x / self.zoom_level
        orig_y = img_y / self.zoom_level
        return orig_x, orig_y

    def image_to_canvas_coords(self, ix, iy):
        if self.display_image is None:
            return None, None

        w, h = self.display_image.size
        new_w = int(w * self.zoom_level)
        new_h = int(h * self.zoom_level)

        img_left = self.offset_x - new_w/2
        img_top  = self.offset_y - new_h/2

        cx = img_left + ix * self.zoom_level
        cy = img_top  + iy * self.zoom_level
        return cx, cy


if __name__ == "__main__":
    app = ImageViewer()
    app.mainloop()
