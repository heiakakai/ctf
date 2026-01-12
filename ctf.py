import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk, ImageOps
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import datetime

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
        self.title("IMG 파일 뷰어 (CTF 계산 포함)")

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
        self.common_config = None  # (img_type, endian, width, height, offset)
        
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

        btn_open = tk.Button(top_frame, text="IMG 파일 열기", command=self.on_click_open_file)
        btn_open.pack(side=tk.LEFT, padx=5)

        self.btn_plot = tk.Button(top_frame, text="line plot 그리기", command=self.show_line_plot)
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

    # ==============================
    # (1) 파일 열기 + 설정
    # ==============================
    def on_click_open_file(self):
        # White 파일 선택
        messagebox.showinfo("파일 선택", "White 파일을 열어주세요.")
        white_path = filedialog.askopenfilename(
            title="White IMG 파일 선택",
            filetypes=[("IMG files", "*.img"), ("All files", "*.*")]
        )
        if not white_path:
            return
        self.white_file_path = white_path
        
        # CTF 파일 선택
        messagebox.showinfo("파일 선택", "CTF 파일을 열어주세요.")
        ctf_path = filedialog.askopenfilename(
            title="CTF IMG 파일 선택",
            filetypes=[("IMG files", "*.img"), ("All files", "*.*")]
        )
        if not ctf_path:
            return
        self.selected_file_path = ctf_path
        
        # 두 파일 모두 선택한 후 설정창 띄우기
        self.open_config_window(is_ctf_file=False, is_first=True)

    def open_config_window(self, is_ctf_file=True, is_first=True):
        file_type = "CTF" if is_ctf_file else "White"
        config_win = tk.Toplevel(self)
        config_win.title(f"{file_type} 파일 Import 설정")
        
        # 이미 저장된 설정이 있으면 사용
        if not is_first and self.common_config:
            img_type, endian, width, height, offset = self.common_config
            default_img_type = img_type
            default_endian = endian
            default_width = str(width)
            default_height = str(height)
            default_offset = str(offset)
        else:
            default_img_type = "16-bit Unsigned"
            default_endian = "Little-endian"
            default_width = "3840"
            default_height = "3072"
            default_offset = "0"

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

        tk.Label(config_win, text="Width:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        width_var = tk.StringVar(value=default_width)
        tk.Entry(config_win, textvariable=width_var, width=10).grid(row=2, column=1, padx=5, pady=5, sticky="w")

        tk.Label(config_win, text="Height:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        height_var = tk.StringVar(value=default_height)
        tk.Entry(config_win, textvariable=height_var, width=10).grid(row=3, column=1, padx=5, pady=5, sticky="w")

        tk.Label(config_win, text="Offset to first image (bytes):").grid(row=4, column=0, padx=5, pady=5, sticky="e")
        offset_var = tk.StringVar(value=default_offset)
        tk.Entry(config_win, textvariable=offset_var, width=10).grid(row=4, column=1, padx=5, pady=5, sticky="w")

        # ROI 설정 섹션
        tk.Label(config_win, text="--- ROI 설정 ---", font=("Arial", 9, "bold")).grid(row=5, column=0, columnspan=2, padx=5, pady=10)
        
        tk.Label(config_win, text="ROI X:").grid(row=6, column=0, padx=5, pady=5, sticky="e")
        x_var = tk.StringVar(value="1420")
        tk.Entry(config_win, textvariable=x_var, width=10).grid(row=6, column=1, padx=5, pady=5, sticky="w")

        tk.Label(config_win, text="ROI Y:").grid(row=7, column=0, padx=5, pady=5, sticky="e")
        y_var = tk.StringVar(value="1536")
        tk.Entry(config_win, textvariable=y_var, width=10).grid(row=7, column=1, padx=5, pady=5, sticky="w")

        tk.Label(config_win, text="ROI Width:").grid(row=8, column=0, padx=5, pady=5, sticky="e")
        w_var = tk.StringVar(value="1000")
        tk.Entry(config_win, textvariable=w_var, width=10).grid(row=8, column=1, padx=5, pady=5, sticky="w")

        tk.Label(config_win, text="ROI Height:").grid(row=9, column=0, padx=5, pady=5, sticky="e")
        h_var = tk.StringVar(value="1000")
        tk.Entry(config_win, textvariable=h_var, width=10).grid(row=9, column=1, padx=5, pady=5, sticky="w")

        # ROI 감도 계산 체크박스
        use_roi_var = tk.BooleanVar(value=False)
        tk.Checkbutton(config_win, text="ROI 감도 계산", variable=use_roi_var).grid(row=10, column=0, columnspan=2, padx=5, pady=5)

        def on_ok():
            img_type = image_type_var.get()
            endian = endian_var.get()
            try:
                w = int(width_var.get())
                h = int(height_var.get())
                offset = int(offset_var.get())
                roi_x = int(x_var.get())
                roi_y = int(y_var.get())
                roi_w = int(w_var.get())
                roi_h = int(h_var.get())
            except ValueError:
                messagebox.showerror("오류", "모든 값은 정수로 입력해야 합니다.")
                return
            
            # ROI 범위 검증
            if roi_x < 0 or roi_y < 0 or roi_x + roi_w > w or roi_y + roi_h > h:
                messagebox.showerror("오류", f"ROI가 이미지 범위를 벗어났습니다.\n이미지 크기: {w}x{h}")
                return
            
            config_win.destroy()
            # 설정 저장
            self.common_config = (img_type, endian, w, h, offset)
            self.white_roi = (roi_x, roi_y, roi_w, roi_h)
            self.use_roi_sensitivity = use_roi_var.get()
            
            if is_ctf_file:
                self.load_img_with_config(img_type, endian, w, h, offset)
            else:
                # White 파일과 CTF 파일 모두 로드
                self.load_white_file_with_config(img_type, endian, w, h, offset)
                # White 설정을 그대로 사용하여 CTF 파일 로드
                if self.selected_file_path:
                    self.load_img_with_config(img_type, endian, w, h, offset)

        def on_cancel():
            config_win.destroy()

        tk.Button(config_win, text="확인", command=on_ok).grid(row=11, column=0, padx=5, pady=10)
        tk.Button(config_win, text="취소", command=on_cancel).grid(row=11, column=1, padx=5, pady=10)

    # ==============================
    # (2) 설정값대로 파일 로드
    # ==============================
    def load_img_with_config(self, img_type, endian, width, height, offset):
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
    def load_white_file_with_config(self, img_type, endian, width, height, offset):
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

        # 1) 라인 프로파일 계산
        x1, y1 = self.point1
        x2, y2 = self.point2
        profile = bresenham_line_profile(self.image_array_original, x1, y1, x2, y2)

        fig_select, ax_select = plt.subplots()
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
            plt.close(fig_select)
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
            plt.close(fig_select)
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
            plt.close(fig_select)
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
            plt.close(fig_select)
            messagebox.showerror("오류", "2lp/mm MIN 범위가 잘못되었습니다.")
            return
        min_2 = float(np.min(profile[min_st2:min_ed2]))
        idx_2min_local = np.argmin(profile[min_st2:min_ed2])
        idx_2min = min_st2 + idx_2min_local
        denom_2 = (max_2 + min_2)
        fraction_2 = (max_2 - min_2)/denom_2 if denom_2 != 0 else 0
        # 2lp/mm CTF = fraction_2 / fraction_1 * 100
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

        # 선택 완료 -> figure 닫기
        plt.close(fig_select)

        # --- (F) 최종 결과용 Figure
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

        # (G) 결과 테이블 표시 (한 창에 합치기)
        white_roi_mean = self.compute_white_roi_mean()
        white_full_mean = self.compute_white_full_mean()
        self.show_ctf_results(results, white_roi_mean, white_full_mean, fig_result)

    def show_ctf_results(self, results, white_roi_mean=None, white_full_mean=None, fig_result=None):
        """
        lp/mm, MAX, MIN, (MAX-MIN)/(MAX+MIN), CTF, White ROI Mean
        - 1lp/mm: fraction_1 * 100
        - 2lp/mm: fraction_2 / fraction_1 * 100
        - txt 파일은 탭(\t) 구분
        - CTF는 소수점 첫째 자리, White ROI Mean은 정수
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
            frac = r['frac']  # 1lp/mm은 그대로 fraction, 2lp/mm도 fraction(0.xxx)
            if lp_val == 1:
                # (MAX-MIN)/(MAX+MIN) = fraction * 100
                frac_percent = frac * 100
                ctf_val = r['ctf']  # fraction*100
            else:
                # fraction_n => fraction
                # ctf => fraction_n/fraction_1 * 100
                frac_percent = frac * 100
                ctf_val = r['ctf']

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
        lines.append("CTF\t전체 감도\tROI 감도")
        
        # 2lp/mm CTF 값 찾기
        ctf_2lp_value = None
        for r in results:
            if r['lp'] == 2:
                ctf_2lp_value = r['ctf']
                break
        
        ctf_line = f"{ctf_2lp_value:.1f}%" if ctf_2lp_value is not None else "미설정"
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
